# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Clang-based parser for C/C++ code that extracts code components for docstring generation.

This module uses libclang to parse C/C++ files and identify functions, classes, and methods.
Dependencies are currently left empty for C/C++ components.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .ast_parser import CodeComponent

logger = logging.getLogger(__name__)

try:
    from clang import cindex
except ImportError as exc:
    cindex = None
    _clang_import_error = exc
else:
    _clang_import_error = None


C_EXTENSIONS = {".c", ".h"}
CPP_EXTENSIONS = {".cc", ".cpp", ".cxx", ".hpp", ".hh", ".hxx", ".ipp"}


@dataclass(frozen=True)
class ComponentLocation:
    start_line: int
    start_column: int
    end_line: int
    end_column: int


def _ensure_clang_available() -> None:
    if cindex is None:
        raise RuntimeError(
            "libclang Python bindings are required for C/C++ parsing. "
            "Install clang and the `clang` Python package (e.g., `pip install clang`), "
            "and ensure libclang is discoverable. "
            f"Original error: {_clang_import_error}"
        )


def _file_to_module_path(file_path: str) -> str:
    path, _ = os.path.splitext(file_path)
    return path.replace(os.path.sep, ".")


def _get_clang_args(language: str, extra_args: Optional[Sequence[str]] = None) -> List[str]:
    if language == "c":
        base_args = ["-std=c11"]
    else:
        base_args = ["-std=c++17"]
    if extra_args:
        base_args.extend(extra_args)
    return base_args


def _is_c_cpp_file(filename: str, language: str) -> bool:
    _, ext = os.path.splitext(filename)
    ext = ext.lower()
    if language == "c":
        return ext in C_EXTENSIONS
    return ext in C_EXTENSIONS | CPP_EXTENSIONS


def _cursor_in_file(cursor: cindex.Cursor, file_path: str) -> bool:
    if cursor.location is None or cursor.location.file is None:
        return False
    try:
        return os.path.samefile(str(cursor.location.file), file_path)
    except FileNotFoundError:
        return False


def _extract_source_segment(
    lines: List[str],
    start_line: int,
    start_col: int,
    end_line: int,
    end_col: int,
) -> str:
    if start_line < 1 or end_line < 1:
        return ""
    if start_line > len(lines) or end_line > len(lines):
        return ""
    if start_line == end_line:
        line = lines[start_line - 1]
        return line[start_col - 1:end_col - 1]
    segment_lines = [lines[start_line - 1][start_col - 1:]]
    segment_lines.extend(lines[start_line:end_line - 1])
    segment_lines.append(lines[end_line - 1][:end_col - 1])
    return "\n".join(segment_lines)


def _qualified_name(cursor: cindex.Cursor) -> List[str]:
    parts: List[str] = []
    current = cursor
    while current is not None and current.kind != cindex.CursorKind.TRANSLATION_UNIT:
        if current.spelling:
            if current.kind in {
                cindex.CursorKind.NAMESPACE,
                cindex.CursorKind.CLASS_DECL,
                cindex.CursorKind.STRUCT_DECL,
                cindex.CursorKind.CLASS_TEMPLATE,
                cindex.CursorKind.FUNCTION_DECL,
                cindex.CursorKind.CXX_METHOD,
                cindex.CursorKind.CONSTRUCTOR,
                cindex.CursorKind.DESTRUCTOR,
            }:
                parts.append(current.spelling)
        current = current.semantic_parent
    return list(reversed(parts))


def build_component_id(file_path: str, relative_path: str, cursor: cindex.Cursor) -> str:
    module_path = _file_to_module_path(relative_path)
    name_parts = _qualified_name(cursor)
    if not name_parts:
        return module_path
    return f"{module_path}." + ".".join(name_parts)


def _is_doc_comment_line(line: str) -> bool:
    stripped = line.lstrip()
    return stripped.startswith("///") or stripped.startswith("//!") or stripped.startswith("/**") or stripped.startswith("/*!")


def _find_doc_comment_range(lines: List[str], start_line: int) -> Optional[Tuple[int, int]]:
    line_index = start_line - 2
    if line_index < 0:
        return None
    if lines[line_index].strip() == "":
        return None
    if lines[line_index].lstrip().startswith(("///", "//!")):
        start = line_index
        while start >= 0 and lines[start].lstrip().startswith(("///", "//!")):
            start -= 1
        return start + 1, line_index
    if "*/" in lines[line_index]:
        end = line_index
        start = line_index
        while start >= 0 and "/*" not in lines[start]:
            start -= 1
        if start >= 0 and _is_doc_comment_line(lines[start]):
            return start, end
    if _is_doc_comment_line(lines[line_index]):
        return line_index, line_index
    return None


def _extract_doc_comment(lines: List[str], start_line: int) -> str:
    comment_range = _find_doc_comment_range(lines, start_line)
    if not comment_range:
        return ""
    start, end = comment_range
    comment_lines = lines[start:end + 1]
    cleaned: List[str] = []
    for line in comment_lines:
        stripped = line.strip()
        if stripped.startswith("///") or stripped.startswith("//!"):
            cleaned.append(stripped[3:].lstrip())
        elif stripped.startswith("/**") or stripped.startswith("/*!"):
            cleaned.append(stripped[3:].lstrip())
        elif stripped.startswith("*/"):
            continue
        elif stripped.startswith("*"):
            cleaned.append(stripped[1:].lstrip())
        else:
            cleaned.append(stripped)
    return "\n".join(cleaned).strip()


class ClangDependencyParser:
    """
    Parses C/C++ code to build a dependency graph between code components.

    Currently, dependency relationships are not extracted for C/C++.
    """

    def __init__(
        self,
        repo_path: str,
        language: str = "cpp",
        clang_args: Optional[Sequence[str]] = None,
    ) -> None:
        _ensure_clang_available()
        self.repo_path = os.path.abspath(repo_path)
        self.language = "c" if language == "c" else "cpp"
        self.clang_args = _get_clang_args(self.language, clang_args)
        self.components: Dict[str, CodeComponent] = {}
        self.modules: set[str] = set()
        self.index = cindex.Index.create()

    def parse_repository(self) -> Dict[str, CodeComponent]:
        logger.info(f"Parsing repository at {self.repo_path} for {self.language.upper()} files")
        for root, _, files in os.walk(self.repo_path):
            for filename in files:
                if not _is_c_cpp_file(filename, self.language):
                    continue
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, self.repo_path)
                module_path = _file_to_module_path(relative_path)
                self.modules.add(module_path)
                self._parse_file(file_path, relative_path)
        logger.info(f"Found {len(self.components)} C/C++ code components")
        return self.components

    def _parse_file(self, file_path: str, relative_path: str) -> None:
        try:
            translation_unit = self.index.parse(
                file_path,
                args=self.clang_args,
                options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
        except cindex.TranslationUnitLoadError as exc:
            logger.warning(f"Error parsing {file_path}: {exc}")
            return

        try:
            with open(file_path, "r", encoding="utf-8") as handle:
                lines = handle.read().splitlines()
        except OSError as exc:
            logger.warning(f"Error reading {file_path}: {exc}")
            return

        for cursor in translation_unit.cursor.walk_preorder():
            if not cursor.is_definition():
                continue
            if not _cursor_in_file(cursor, file_path):
                continue

            component_type = self._cursor_component_type(cursor)
            if component_type is None:
                continue

            extent = cursor.extent
            component_id = build_component_id(file_path, relative_path, cursor)
            docstring = _extract_doc_comment(lines, extent.start.line)

            component = CodeComponent(
                id=component_id,
                node=None,
                component_type=component_type,
                file_path=file_path,
                relative_path=relative_path,
                depends_on=set(),
                source_code=_extract_source_segment(
                    lines,
                    extent.start.line,
                    extent.start.column,
                    extent.end.line,
                    extent.end.column,
                ),
                start_line=extent.start.line,
                end_line=extent.end.line,
                has_docstring=bool(docstring),
                docstring=docstring,
            )
            self.components[component_id] = component

    def _cursor_component_type(self, cursor: cindex.Cursor) -> Optional[str]:
        if cursor.kind in {cindex.CursorKind.FUNCTION_DECL, cindex.CursorKind.FUNCTION_TEMPLATE}:
            return "function"
        if cursor.kind in {
            cindex.CursorKind.CXX_METHOD,
            cindex.CursorKind.CONSTRUCTOR,
            cindex.CursorKind.DESTRUCTOR,
        }:
            return "method"
        if cursor.kind in {
            cindex.CursorKind.CLASS_DECL,
            cindex.CursorKind.STRUCT_DECL,
            cindex.CursorKind.CLASS_TEMPLATE,
        }:
            return "class"
        return None

    def save_dependency_graph(self, output_path: str) -> None:
        serializable_components = {
            comp_id: component.to_dict()
            for comp_id, component in self.components.items()
        }
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(serializable_components, handle, indent=2)
        logger.info(f"Saved dependency graph to {output_path}")

    def load_dependency_graph(self, input_path: str) -> Dict[str, CodeComponent]:
        with open(input_path, "r", encoding="utf-8") as handle:
            serialized_components = json.load(handle)
        self.components = {
            comp_id: CodeComponent.from_dict(comp_data)
            for comp_id, comp_data in serialized_components.items()
        }
        logger.info(f"Loaded {len(self.components)} components from {input_path}")
        return self.components

    def find_component_location(self, file_path: str, component_id: str) -> Optional[ComponentLocation]:
        relative_path = os.path.relpath(file_path, self.repo_path)
        try:
            translation_unit = self.index.parse(
                file_path,
                args=self.clang_args,
                options=cindex.TranslationUnit.PARSE_DETAILED_PROCESSING_RECORD,
            )
        except cindex.TranslationUnitLoadError as exc:
            logger.warning(f"Error parsing {file_path} for location lookup: {exc}")
            return None

        for cursor in translation_unit.cursor.walk_preorder():
            if not cursor.is_definition():
                continue
            if not _cursor_in_file(cursor, file_path):
                continue
            if build_component_id(file_path, relative_path, cursor) == component_id:
                extent = cursor.extent
                return ComponentLocation(
                    start_line=extent.start.line,
                    start_column=extent.start.column,
                    end_line=extent.end.line,
                    end_column=extent.end.column,
                )
        return None

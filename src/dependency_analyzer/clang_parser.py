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
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

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


def _resolve_compile_commands_dir(compile_commands_path: Optional[str]) -> Optional[str]:
    if not compile_commands_path:
        return None
    path = os.path.abspath(compile_commands_path)
    if os.path.isdir(path):
        return path
    if os.path.isfile(path) and os.path.basename(path) == "compile_commands.json":
        return os.path.dirname(path)
    return None


def _load_compilation_database(directory: Optional[str]) -> Optional["cindex.CompilationDatabase"]:
    if not directory or cindex is None:
        return None
    if not hasattr(cindex, "CompilationDatabase"):
        logger.warning("libclang bindings do not expose CompilationDatabase; compile_commands.json ignored.")
        return None
    try:
        return cindex.CompilationDatabase.fromDirectory(directory)
    except cindex.CompilationDatabaseError as exc:
        logger.warning(f"Failed to load compilation database from {directory}: {exc}")
        return None


class ClangDependencyParser:
    """
    Parses C/C++ code to build a dependency graph between code components.
    """

    def __init__(
        self,
        repo_path: str,
        language: str = "cpp",
        clang_args: Optional[Sequence[str]] = None,
        include_dirs: Optional[Sequence[str]] = None,
        defines: Optional[Sequence[str]] = None,
        compile_commands_path: Optional[str] = None,
    ) -> None:
        _ensure_clang_available()
        self.repo_path = os.path.abspath(repo_path)
        self.language = "c" if language == "c" else "cpp"
        self.base_args = _get_clang_args(self.language, clang_args)
        self.include_dirs = list(include_dirs or [])
        self.defines = list(defines or [])
        self.compile_commands_dir = _resolve_compile_commands_dir(compile_commands_path)
        self.compilation_db = _load_compilation_database(self.compile_commands_dir)
        self.components: Dict[str, CodeComponent] = {}
        self._component_cursors: Dict[str, cindex.Cursor] = {}
        self._usr_to_component_id: Dict[str, str] = {}
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
        self._build_dependencies()
        logger.info(f"Found {len(self.components)} C/C++ code components")
        return self.components

    def _parse_file(self, file_path: str, relative_path: str) -> None:
        try:
            parse_args = self._get_parse_args(file_path)
            translation_unit = self.index.parse(
                file_path,
                args=parse_args,
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
            usr = cursor.get_usr()
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
            if usr:
                self._usr_to_component_id[usr] = component_id
                self._component_cursors[component_id] = cursor

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

    def _get_parse_args(self, file_path: str) -> List[str]:
        args = list(self.base_args)
        args.extend(self._compile_db_args(file_path))
        args.extend(f"-I{path}" for path in self.include_dirs)
        args.extend(f"-D{definition}" for definition in self.defines)
        return args

    def _compile_db_args(self, file_path: str) -> List[str]:
        if not self.compilation_db:
            return []
        try:
            commands = self.compilation_db.getCompileCommands(file_path)
        except cindex.CompilationDatabaseError as exc:
            logger.warning(f"Error reading compile commands for {file_path}: {exc}")
            return []
        if not commands:
            return []
        command = commands[0]
        args: List[str] = []
        skip_next = False
        for arg in command.arguments:
            if skip_next:
                skip_next = False
                continue
            if arg in {"-c", command.filename}:
                continue
            if arg == "-o":
                skip_next = True
                continue
            if arg.startswith("-o"):
                continue
            args.append(arg)
        return args

    def _build_dependencies(self) -> None:
        if not self._component_cursors:
            return
        for component_id, cursor in self._component_cursors.items():
            depends_on = self._collect_dependencies(cursor)
            self.components[component_id].depends_on = depends_on

    def _collect_dependencies(self, cursor: cindex.Cursor) -> set[str]:
        dependencies: set[str] = set()
        for child in cursor.walk_preorder():
            referenced = child.referenced
            if referenced is None:
                continue
            usr = referenced.get_usr()
            if not usr:
                continue
            target_id = self._usr_to_component_id.get(usr)
            if not target_id:
                continue
            if target_id == self._usr_to_component_id.get(cursor.get_usr(), ""):
                continue
            dependencies.add(target_id)
        return dependencies

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
            parse_args = self._get_parse_args(file_path)
            translation_unit = self.index.parse(
                file_path,
                args=parse_args,
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

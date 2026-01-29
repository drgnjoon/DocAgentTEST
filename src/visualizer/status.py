# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict, Set, Optional, Any, List
from colorama import Fore, Back, Style, init
import sys
import time
import ast
from agent.tool.ast import _get_component_name_from_code


class StatusVisualizer:
    """Visualizes the workflow status of DocAssist agents in the terminal."""
    
    def __init__(
        self,
        show_dependency_tree: bool = False,
        max_tree_depth: int = 2,
        max_tree_nodes: int = 30
    ):
        """Initialize the status visualizer."""
        init()  # Initialize colorama
        self.active_agent = None  # Track only the currently active agent
        self._agent_art = {
            'reader': [
                "┌─────────┐",
                "│ READER  │",
                "└─────────┘"
            ],
            'searcher': [
                "┌─────────┐",
                "│SEARCHER │",
                "└─────────┘"
            ],
            'writer': [
                "┌─────────┐",
                "│ WRITER  │",
                "└─────────┘"
            ],
            'verifier': [
                "┌─────────┐",
                "│VERIFIER │",
                "└─────────┘"
            ]
        }
        self._status_message = ""
        self._current_component = ""
        self._current_file = ""
        self._dependency_components: Optional[Dict[str, Any]] = None
        self._dependency_root_id: Optional[str] = None
        self._show_dependency_tree = show_dependency_tree
        self._max_tree_depth = max_tree_depth
        self._max_tree_nodes = max_tree_nodes
    
    def _clear_screen(self):
        """Clear the terminal screen."""
        sys.stdout.write("\033[2J\033[H")
        sys.stdout.flush()
    
    def _get_agent_color(self, agent: str) -> str:
        """Get the color for an agent based on its state."""
        return Fore.GREEN if agent == self.active_agent else Fore.WHITE
    
    def set_current_component(self, focal_component: str, file_path: str):
        """Set the current component being processed and display its information.
        
        Args:
            focal_component: The code component being processed
            file_path: Relative path to the file containing the component
        """
        # Try to extract the component name from the code
        try:
            self._current_component = _get_component_name_from_code(focal_component)
        except:
            # If parsing fails, just use a generic name
            self._current_component = "unknown component"
        
        self._current_file = file_path
        self._display_component_info()

    def set_dependency_context(self, component_id: str, components: Dict[str, Any]) -> None:
        """Set the dependency graph context for the currently processed component."""
        self._dependency_components = components
        self._dependency_root_id = component_id
    
    def _display_component_info(self):
        """Display information about the current component being processed."""
        # print(f"\n{Fore.CYAN}Currently Processing:{Style.RESET_ALL}")
        print(f"Component: {self._current_component}")
        print(f"File: {self._current_file}\n")

    def _format_component_label(self, component_id: str) -> str:
        """Format a dependency component label for tree display."""
        component = None
        if self._dependency_components:
            component = self._dependency_components.get(component_id)
        parts = component_id.split('.')
        if component and component.component_type == "method" and len(parts) > 2:
            name = f"{parts[-2]}.{parts[-1]}"
        else:
            name = parts[-1] if parts else component_id
        comp_type = component.component_type.capitalize() if component else "Component"
        return f"{comp_type} '{name}'"

    def _build_dependency_tree_lines(self) -> List[str]:
        """Build tree lines for the current component's dependencies."""
        if not self._show_dependency_tree:
            return []
        if not self._dependency_components or not self._dependency_root_id:
            return []

        lines: List[str] = ["Dependency tree:"]
        visited: Set[str] = set()
        node_count = 0

        def walk(node_id: str, prefix: str, depth: int, is_last: bool) -> None:
            nonlocal node_count
            if node_count >= self._max_tree_nodes:
                return

            component = self._dependency_components.get(node_id) if self._dependency_components else None
            label = self._format_component_label(node_id)
            if depth == 0:
                lines.append(label)
            else:
                connector = "└─" if is_last else "├─"
                lines.append(f"{prefix}{connector} {label}")
            node_count += 1

            if node_id in visited:
                if depth != 0:
                    lines[-1] += " (cycle)"
                return
            visited.add(node_id)

            if depth >= self._max_tree_depth or not component:
                return

            deps = sorted(dep_id for dep_id in component.depends_on if dep_id in self._dependency_components)
            if not deps:
                return

            next_prefix = prefix + ("   " if is_last else "│  ")
            for idx, dep_id in enumerate(deps):
                if node_count >= self._max_tree_nodes:
                    lines.append(f"{next_prefix}└─ ...")
                    return
                walk(dep_id, next_prefix, depth + 1, idx == len(deps) - 1)

        walk(self._dependency_root_id, "", 0, True)
        if node_count >= self._max_tree_nodes:
            lines.append(f"... (showing first {self._max_tree_nodes} nodes)")
        return lines
    
    def update(self, active_agent: str, status_message: str = ""):
        """Update the visualization with the current active agent and status.
        
        Args:
            active_agent: Name of the currently active agent
            status_message: Current status message to display
        """
        self.active_agent = active_agent  # Update the single active agent
        self._status_message = status_message
        self._clear_screen()
        
        # Build the visualization
        lines = []
        
        # Add header
        # lines.append(f"{Fore.CYAN}DocAssist Workflow Status{Style.RESET_ALL}")
        # lines.append("")
        
        # Display current component info if available
        if self._current_component and self._current_file:
            lines.append(f"Processing: {self._current_component}")
            lines.append(f"File: {self._current_file}")
            lines.append("")

        dependency_lines = self._build_dependency_tree_lines()
        if dependency_lines:
            lines.extend(dependency_lines)
            lines.append("")
        
        # Input arrow to Reader
        # lines.append("     Input")
        # lines.append("       ↓")
        
        # First row: Reader and Searcher with loop
        for i in range(3):
            line = (f"{self._get_agent_color('reader')}{self._agent_art['reader'][i]}"
                   f"  ←→  "
                   f"{self._get_agent_color('searcher')}{self._agent_art['searcher'][i]}"
                   f"{Style.RESET_ALL}")
            lines.append(line)
        
        # Arrow from Reader to Writer
        # lines.append("       ↓")
        
        # Second row: Writer
        for i in range(3):
            line = (f"    {self._get_agent_color('writer')}{self._agent_art['writer'][i]}{Style.RESET_ALL}")
            lines.append(line)
        
        # Arrow from Writer to Verifier
        # lines.append("       ↓")
        
        # Third row: Verifier with output
        for i in range(3):
            if i == 1:
                line = (f"    {self._get_agent_color('verifier')}{self._agent_art['verifier'][i]}{Style.RESET_ALL}  →  Output")
            else:
                line = (f"    {self._get_agent_color('verifier')}{self._agent_art['verifier'][i]}{Style.RESET_ALL}")
            lines.append(line)
        
        # # Feedback arrows from Verifier
        # lines.append("       ↑")
        # lines.append("    ↗  ↑")
        
        # Add status message
        if self._status_message:
            lines.append("")
            lines.append(f"{Fore.YELLOW}Status: {self._status_message}{Style.RESET_ALL}")
        
        # Print the visualization
        print("\n".join(lines))
        sys.stdout.flush()
    
    def reset(self):
        """Reset the visualization state."""
        self.active_agent = None
        self._status_message = ""
        self._current_component = ""
        self._current_file = ""
        self._clear_screen()

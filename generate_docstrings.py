#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates
"""
Docstring Generator with Dependency-Based Ordering

This script generates docstrings for Python code components (functions, classes, methods)
using a DFS-based approach that starts from components with no dependencies.

Key features:
1. Parses Python code to identify components and their dependencies
2. Builds a dependency graph where A→B means "A depends on B"
3. Performs DFS traversal starting from components with no dependencies
4. Processes dependencies before the components that depend on them
5. Ensures classes depend on their methods, not vice versa
6. Skips __init__ methods as they typically don't need separate docstrings
7. Provides visual representation of progress in the terminal

Usage:
    python generate_docstrings.py --repo-path PATH --config-path PATH [--test-mode]
"""

import os
import sys
import time
import ast
import json
import argparse
import logging
import random
from pathlib import Path
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
import tiktoken  # Add this import for token counting

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("docstring_generator")

# Import dependency analyzer modules
from src.dependency_analyzer import (
    CodeComponent, 
    DependencyParser, 
    dependency_first_dfs, 
    build_graph_from_components
)
from src.visualizer import ProgressVisualizer
from src.agent.orchestrator import Orchestrator


def generate_test_docstring(component: CodeComponent) -> str:
    """
    Generate a placeholder docstring for test mode.
    
    Args:
        component: The code component to generate a placeholder docstring for.
        
    Returns:
        A placeholder docstring based on the component type.
    """
    comp_type = component.component_type
    name = component.id.split(".")[-1]
    
    if comp_type == "function":
        return f"""
        Test docstring for function '{name}'.
        
        This is a placeholder docstring generated in test mode.
        In normal mode, this would be replaced with an AI-generated docstring.
        
        Args:
            arg1: Description of first argument
            arg2: Description of second argument
            
        Returns:
            Description of return value
        """
    elif comp_type == "class":
        return f"""
        Test docstring for class '{name}'.
        
        This is a placeholder docstring generated in test mode.
        In normal mode, this would be replaced with an AI-generated docstring.
        
        Attributes:
            attr1: Description of first attribute
            attr2: Description of second attribute
        """
    elif comp_type == "method":
        class_name = component.id.split(".")[-2]
        return f"""
        Test docstring for method '{name}' in class '{class_name}'.
        
        This is a placeholder docstring generated in test mode.
        In normal mode, this would be replaced with an AI-generated docstring.
        
        Args:
            arg1: Description of first argument
            arg2: Description of second argument
            
        Returns:
            Description of return value
        """
    else:
        return f"""
        Test docstring for {comp_type} '{name}'.
        
        This is a placeholder docstring generated in test mode.
        """


def generate_docstring_for_component(component: CodeComponent, orchestrator: Optional[Orchestrator], test_mode: str = 'none',
                                     dependency_graph: Optional[Dict[str, List[str]]] = None) -> str:
    """
    Generate a docstring for a single component.
    
    Args:
        component: The component to generate a docstring for.
        orchestrator: The orchestrator instance.
        test_mode: The test mode to use.
        dependency_graph: Optional dependency graph.
        
    Returns:
        The generated docstring.
    """

    # do not use try/except here, we want to fail if there is an error
    if not orchestrator:
        return ""
    
    file_path = component.file_path
    
    # Get the component code
    component_code = component.source_code
    
    # Estimate token count of the focal component
    encoding = tiktoken.get_encoding("cl100k_base")  # Default OpenAI encoding
    token_consume_focal = len(encoding.encode(component_code))
    
    # Skip if the component is too large (> 10000 tokens)
    if token_consume_focal > 10000:
        # truncate the component code to 10000 tokens
        component_code = encoding.decode(encoding.encode(component_code)[:10000])
    
    # Parse the file
    with open(file_path, "r", encoding="utf-8") as f:
        file_content = f.read()
    
    ast_tree = ast.parse(file_content)
    ast_node = None
    
    # Locate the AST node for the component
    component_parts = component.id.split(".")
    component_name = component_parts[-1]
    
    if component.component_type == "function":
        # Find top-level function
        for node in ast.iter_child_nodes(ast_tree):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) 
                    and node.name == component_name):
                ast_node = node
                break
            
    elif component.component_type == "class":
        # Find class
        for node in ast.iter_child_nodes(ast_tree):
            if isinstance(node, ast.ClassDef) and node.name == component_name:
                ast_node = node
                break
            
    elif component.component_type == "method":
        # Find method inside class
        class_name, method_name = component_parts[-2:]
        for node in ast.iter_child_nodes(ast_tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) 
                            and item.name == method_name):
                        ast_node = item
                        break
                break
    
    try:
        # Pass component.id as the focal_node_dependency_path
        docstring = orchestrator.process(
            focal_component=component_code,
            file_path=file_path,
            ast_node=ast_node,
            ast_tree=ast_tree,
            dependency_graph=dependency_graph,
            focal_node_dependency_path=component.id,
            token_consume_focal=token_consume_focal  # Pass token count to orchestrator
        )
        return docstring
    except Exception as e:
        print(f"Error generating docstring for {component.id}: {str(e)}")
        return ""


def set_docstring_in_file(file_path: str, component: CodeComponent, docstring: str) -> bool:
    """
    Update a Python file with a newly generated docstring for a component.
    
    Args:
        file_path: Path to the file to update.
        component: The component to update with a docstring.
        docstring: The docstring to insert.
        
    Returns:
        True if successful, False otherwise.
    """
    # Do not use Try/Except here, we want to fail if there is an error
    # Read the file
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    
    # Parse the file
    tree = ast.parse(source)
    
    # Find the component in the parsed AST
    component_node = None
    component_parts = component.id.split(".")
    component_name = component_parts[-1]
    
    if component.component_type == "function":
        # Find top-level function
        for node in ast.iter_child_nodes(tree):
            if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) 
                    and node.name == component_name):
                component_node = node
                break
                
    elif component.component_type == "class":
        # Find class
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name == component_name:
                component_node = node
                break
                
    elif component.component_type == "method":
        # Find method inside class
        class_name, method_name = component_parts[-2:]
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if (isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) 
                            and item.name == method_name):
                        component_node = item
                        break
                break
    
    if not component_node:
        logger.error(f"Could not find component {component.id} in {file_path}")
        return False
    
    # Set the docstring
    set_node_docstring(component_node, docstring)
    
    # Unparse the AST back to source code
    if hasattr(ast, "unparse"):
        new_source = ast.unparse(tree)
    else:
        try:
            import astor
            new_source = astor.to_source(tree)
        except ImportError:
            logger.error(
                "Error: You need to install 'astor' or use Python 3.9+ to unparse the AST. "
                f"Skipping file: {file_path}"
            )
            return False
    
    # Write back to the file
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_source)
    
    return True


def set_node_docstring(node: ast.AST, docstring: str):
    """
    Safely set or update the docstring on an AST node (ClassDef, FunctionDef, etc.).
    Also adjusts indentation relative to the node's existing indentation level,
    ensuring both the opening and closing triple quotes are properly aligned.

    Args:
        node: The AST node to modify (ClassDef, FunctionDef, etc.).
        docstring: The new docstring (as a plain string) to insert.
    """
    import textwrap
    
    # 1. Strip leading/trailing empty lines in the provided docstring
    #    to avoid spurious blank lines.
    stripped_docstring = docstring.strip('\n')
    if not stripped_docstring:
        # If empty or all whitespace, provide a placeholder
        stripped_docstring = "No docstring provided."

    # 2. Dedent possible indentation in docstring (so a multiline docstring
    #    doesn't carry undesired left margins).
    dedented = textwrap.dedent(stripped_docstring)

    # 3. Determine how many spaces to indent for doc lines plus triple quotes.
    existing_indent = getattr(node, 'col_offset', 0)
    doc_indent_str = ' ' * (existing_indent + 4)

    # 4. Build the final string: 
    #    - Start with a newline (so triple quotes appear on a new line).
    #    - Indent all docstring lines.
    #    - End with a newline+same indentation (so the closing triple quotes
    #      line also has the doc_indent_str).
    prepared_docstring = (
        "\n"
        + textwrap.indent(dedented, doc_indent_str)
        + "\n"
        + doc_indent_str
    )

    # 5. Create an AST Expr node to store this docstring as a constant.
    docstring_node = ast.Expr(value=ast.Constant(value=prepared_docstring, kind=None))

    # If there's no body, just make one containing our new docstring.
    if not hasattr(node, "body") or not isinstance(node.body, list) or len(node.body) == 0:
        node.body = [docstring_node]
    else:
        # If the first statement is an existing docstring, replace it;
        # otherwise, insert the new docstring as the first statement.
        first_stmt = node.body[0]
        if (
            isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)
        ):
            node.body[0] = docstring_node
        else:
            node.body.insert(0, docstring_node)


def main():
    """
    Main entry point for the docstring generation script with flexible component ordering.
    
    The script supports different ordering modes through the --order-mode flag:
    - 'topo' (default): Dependency-based ordering using a DFS-based approach:
        1. If A depends on B, the graph has an edge A→B (meaning "A depends on B")
        2. Root nodes (nodes with no dependencies) are processed first
        3. Dependencies are always processed before the components that depend on them
        4. This ensures proper docstring generation order
    - 'random_node': Randomly shuffles all Python components, ignoring dependencies
    - 'random_file': Processes files in random order, but preserves component order within files
    
    Class methods are processed before the classes that depend on them (not vice versa) in 'topo' mode,
    ensuring proper docstring generation order. Special __init__ methods are skipped as
    they typically don't need separate docstrings.
    
    The script provides options to skip or overwrite existing docstrings:
    - By default, components with existing docstrings are skipped
    - With --overwrite-docstrings flag, existing docstrings will be overwritten
    - This behavior can also be configured in the config.yaml file under docstring_options.overwrite_docstrings
    
    Web interface integration:
    - With --enable-web flag, the script enables integration with the web UI
    - This allows visualization of the docstring generation process in a web browser
    - Run the web UI separately using the run_web_ui.py script
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Generate docstrings for Python components in dependency order.'
    )
    parser.add_argument(
        '--repo-path', 
        type=str, 
        default='data/raw_test_repo',
        help='Path to the repository (default: data/raw_test_repo)'
    )
    parser.add_argument(
        '--config-path', 
        type=str, 
        default='config/agent_config.yaml',
        help='Path to the configuration file (default: config/agent_config.yaml)'
    )
    parser.add_argument(
        '--test-mode',
        type=str,
        choices=['placeholder', 'context_print', 'none'],
        default='none',
        help='Test mode to run: "placeholder" for placeholder docstrings (no LLM calls), "context_print" to print context before writer calls, "none" for normal operation'
    )
    parser.add_argument(
        '--order-mode',
        type=str,
        choices=['topo', 'random_node', 'random_file'],
        default='topo',
        help='Order mode for docstring generation: "topo" follows dependency order (default), "random_node" selects random Python nodes, "random_file" processes files in random order'
    )
    parser.add_argument(
        '--enable-web',
        action='store_true',
        help='Enable integration with the web interface'
    )
    parser.add_argument(
        '--overwrite-docstrings',
        action='store_true',
        help='Overwrite existing docstrings instead of skipping them (default: False)'
    )
    parser.add_argument(
        '--visual-details',
        action='store_true',
        help='Show dependency tree details in the agent status view'
    )
    parser.add_argument(
        '--visual-tree-depth',
        type=int,
        default=2,
        help='Max dependency depth to render in the tree view (default: 2)'
    )
    parser.add_argument(
        '--visual-tree-limit',
        type=int,
        default=30,
        help='Max dependency nodes to render in the tree view (default: 30)'
    )
    
    args = parser.parse_args()
    repo_path = args.repo_path
    config_path = args.config_path
    test_mode = args.test_mode
    order_mode = args.order_mode
    overwrite_docstrings = args.overwrite_docstrings
    
    # Create output directory for dependency graph
    output_dir = os.path.join("output", "dependency_graphs")
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract repository name from path for creating a unique filename
    repo_name = os.path.basename(os.path.normpath(repo_path))
    # Create a sanitized version of the repo name (remove special characters)
    sanitized_repo_name = ''.join(c if c.isalnum() else '_' for c in repo_name)
    dependency_graph_path = os.path.join(output_dir, f"{sanitized_repo_name}_dependency_graph.json")
    
    # Initialize the orchestrator for docstring generation
    orchestrator = None
    
    # Initialize orchestrator unless we're in placeholder test mode
    if test_mode != 'placeholder':
        logger.info(f"Initializing orchestrator with config: {config_path}")
        # Pass the test_mode to the orchestrator if it's "context_print"
        orchestrator_test_mode = test_mode if test_mode != 'none' else None
        visual_options = None
        if args.visual_details:
            visual_options = {
                "show_dependency_tree": True,
                "max_tree_depth": args.visual_tree_depth,
                "max_tree_nodes": args.visual_tree_limit
            }
        orchestrator = Orchestrator(
            repo_path=repo_path,
            config_path=config_path,
            test_mode=orchestrator_test_mode,
            visual_options=visual_options
        )
        
        # Check if the overwrite_docstrings option is in the config file
        # If it's there, it overrides the command-line argument
        if hasattr(orchestrator, 'config'):
            docstring_options = orchestrator.config.get('docstring_options', {})
            config_overwrite = docstring_options.get('overwrite_docstrings')
            if config_overwrite is not None:
                overwrite_docstrings = config_overwrite
                logger.info(f"Using config file setting for overwrite_docstrings: {overwrite_docstrings}")
    else:
        logger.info("Running in PLACEHOLDER TEST MODE with placeholder docstrings (no LLM calls)")
    
    # Parse the repository to build the dependency graph
    logger.info(f"Parsing repository: {repo_path}")
    parser = DependencyParser(repo_path)
    components = parser.parse_repository()
    
    # Save the dependency graph for future reference
    parser.save_dependency_graph(dependency_graph_path)
    logger.info(f"Dependency graph saved to: {dependency_graph_path}")
    
    # Build the graph for traversal
    graph = build_graph_from_components(components)
    
    # Create a dependency graph in the format expected by the orchestrator
    # Dictionary mapping component paths to their dependencies
    dependency_graph = {}
    for component_id, deps in graph.items():
        dependency_graph[component_id] = list(deps)
    
    # Perform DFS-based traversal
    logger.info("Performing DFS traversal on the dependency graph (starting from nodes with no dependencies)")
    sorted_components = dependency_first_dfs(graph)
    logger.info(f"Sorted {len(sorted_components)} components for processing")
    
    # Apply the selected ordering mode
    if order_mode == 'random_node':
        # Randomly shuffle all components
        logger.info("Using random node ordering mode - shuffling all components")
        random.shuffle(sorted_components)
    elif order_mode == 'random_file':
        # Group components by file path
        logger.info("Using random file ordering mode - processing files in random order")
        # Group components by file
        file_to_components = defaultdict(list)
        for component_id in sorted_components:
            component = components.get(component_id)
            if component:
                file_to_components[component.file_path].append(component_id)
        
        # Randomly shuffle the file order but maintain the order of components within each file
        file_paths = list(file_to_components.keys())
        random.shuffle(file_paths)
        
        # Create a new ordering based on randomly shuffled files
        sorted_components = []
        for file_path in file_paths:
            sorted_components.extend(file_to_components[file_path])
    else:
        # Default to topological order (already set in sorted_components)
        logger.info("Using topological ordering mode - processing components based on dependencies")
    
    # Check if web interface is enabled
    if args.enable_web:
        try:
            from src.visualizer.web_bridge import patch_visualizers
            logger.info("Web interface integration enabled")
            patch_visualizers()
        except ImportError as e:
            logger.warning(f"Failed to enable web interface integration: {e}")
            logger.warning("Make sure you have installed the required web dependencies")

    # Initialize the progress visualizer
    visualizer = ProgressVisualizer(components, sorted_components)
    visualizer.initialize()
    
    # Show dependency statistics
    visualizer.show_dependency_stats()
    
    # Process components in order determined by DFS traversal
    for component_id in sorted_components:
        component = components.get(component_id)
        if not component:
            logger.warning(f"Component {component_id} not found in parsed components")
            continue
        
        # Skip __init__ methods as they don't need docstrings
        if component.component_type == "method" and component_id.endswith(".__init__"):
            logger.info(f"Skipping {component_id} - __init__ methods don't need docstrings")
            visualizer.update(component_id, "completed")
            continue
        
        # compute the length of docstring if exists (using white space as delimiter)
        docstring_length = len(component.docstring.split()) if component.has_docstring else 0
        # Skip components that already have docstrings (unless overwrite_docstrings is True)
        if component.has_docstring and not overwrite_docstrings and docstring_length > 10:
            logger.info(f"Skipping {component_id} - already has docstring")
            visualizer.update(component_id, "completed")
            continue
        elif component.has_docstring and overwrite_docstrings:
            logger.info(f"Overwriting existing docstring for {component_id}")
        
        # Update the visualizer
        visualizer.update(component_id, "processing")
        
        # Log the component type
        comp_type = component.component_type
        logger.info(f"Processing {comp_type}: {component_id}")
        
        # Generate the docstring
        logger.info(f"Generating docstring for {component_id}")
        if orchestrator and hasattr(orchestrator.visualizer, "set_dependency_context"):
            orchestrator.visualizer.set_dependency_context(component_id, components)
        docstring = generate_docstring_for_component(component, orchestrator, test_mode, dependency_graph)
        
        # Update the file with the new docstring
        file_path = component.file_path
        success = set_docstring_in_file(file_path, component, docstring)
        
        if success:
            logger.info(f"Successfully updated docstring for {component_id}")
            visualizer.update(component_id, "completed")
        else:
            logger.error(f"Failed to update docstring for {component_id}")
            visualizer.update(component_id, "error")
        
        # Re-parse the file in case the line numbers changed due to docstring insertion
        # This is only necessary if there are more components from the same file
        same_file_components = [
            comp_id for comp_id in sorted_components 
            if comp_id != component_id and components[comp_id].file_path == file_path
        ]
        
        if same_file_components:
            logger.info(f"Re-parsing file {file_path} for updated line numbers")
            parser = DependencyParser(repo_path)
            updated_components = parser.parse_repository()
            
            # Update the components dictionary with new line numbers
            for comp_id, comp in updated_components.items():
                if comp_id in components:
                    components[comp_id] = comp
    
    # Finalize the visualization
    visualizer.finalize()
    
    # Create a more descriptive mode message based on the test mode
    if test_mode == 'placeholder':
        mode_str = "PLACEHOLDER TEST MODE (no LLM calls)"
    elif test_mode == 'context_print':
        mode_str = "CONTEXT PRINT TEST MODE (with context debugging)"
    else:
        mode_str = "normal mode"
    
    # Add ordering mode to the completion message
    order_mode_str = {
        'topo': 'topological ordering',
        'random_node': 'random node ordering',
        'random_file': 'random file ordering'
    }.get(order_mode, 'unknown ordering')
    
    logger.info(f"Docstring generation complete ({mode_str}, {order_mode_str})")
    
    # Print usage statistics for LLM providers if available
    if orchestrator:
        try:
            # Access the rate limiters from agents
            rate_limiters = []
            
            for agent_name in ['reader', 'writer', 'verifier']:
                agent = getattr(orchestrator, agent_name, None)
                if agent and hasattr(agent, 'llm') and hasattr(agent.llm, 'rate_limiter'):
                    rate_limiters.append(agent.llm.rate_limiter)
            
            # Print statistics for each rate limiter
            if rate_limiters:
                logger.info("=" * 50)
                logger.info("TOKEN USAGE AND COST STATISTICS")
                logger.info("=" * 50)
                
                for limiter in rate_limiters:
                    limiter.print_usage_stats()
                
                # Calculate total cost across all limiters
                total_cost = sum(limiter.total_cost for limiter in rate_limiters)
                logger.info("=" * 50)
                logger.info(f"TOTAL COST: ${total_cost:.6f}")
                logger.info("=" * 50)
        except Exception as e:
            logger.warning(f"Could not print token usage statistics: {e}")


if __name__ == "__main__":
    main() 

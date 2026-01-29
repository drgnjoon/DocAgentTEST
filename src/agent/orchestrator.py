# Copyright (c) Meta Platforms, Inc. and affiliates
from typing import Dict, Any, Optional, List
import time
from .base import BaseAgent
from .reader import Reader
from .searcher import Searcher
from .writer import Writer
from .verifier import Verifier
from visualizer import StatusVisualizer
import re
import yaml
import ast
import tiktoken

# Dummy visualizer class that mimics StatusVisualizer but does nothing
class DummyVisualizer:
    """A no-op visualizer that implements the same interface as StatusVisualizer but does nothing."""
    
    def __init__(self, *args, **kwargs):
        """Do nothing."""
        pass

    def reset(self):
        """Do nothing."""
        pass
    
    def set_current_component(self, component, file_path):
        """Do nothing."""
        pass

    def set_dependency_context(self, component_id, components):
        """Do nothing."""
        pass
    
    def update(self, agent_name, status):
        """Do nothing."""
        pass

class Orchestrator(BaseAgent):
    """Agent responsible for managing the workflow between all other agents."""
    
    def __init__(
        self,
        repo_path: str,
        config_path: Optional[str] = None,
        test_mode: Optional[str] = None,
        visual_options: Optional[Dict[str, Any]] = None
    ):
        """Initialize the Orchestrator agent and its sub-agents.
        
        Args:
            repo_path: Path to the repository being analyzed
            config_path: Optional path to the configuration file
            test_mode: Optional test mode to run only specific components. Values: "reader_searcher", "context_print" or None
            visual_options: Optional settings for the status visualizer
        """
        super().__init__("Orchestrator")
        self.repo_path = repo_path
        self.context = ""
        self.test_mode = test_mode
        
        # Load configuration
        self.config = {}
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Get flow control parameters with defaults
        flow_config = self.config.get('flow_control', {})
        self.max_reader_search_attempts = flow_config.get('max_reader_search_attempts', 4)
        self.max_verifier_rejections = flow_config.get('max_verifier_rejections', 3)
        self.status_sleep_time = flow_config.get('status_sleep_time', 3)
        
        # Check model type for context constraints
        llm_config = self.config.get('llm', {})
        self.model_type = llm_config.get('type', 'openai')
        
        # Add max_input_tokens to config for context length constraint
        if 'max_input_tokens' not in self.config:
            self.config['max_input_tokens'] = llm_config.get('max_input_tokens', 10000)
        
        # Initialize visualization - use dummy visualizer for "context_print" test mode
        if test_mode == "context_print":
            self.visualizer = DummyVisualizer()
        else:
            visual_options = visual_options or {}
            self.visualizer = StatusVisualizer(**visual_options)
        
        # Initialize all sub-agents
        self.reader = Reader(config_path=config_path)
        self.searcher = Searcher(repo_path, config_path=config_path)
        
        # Only initialize writer and verifier if not in reader_searcher test mode
        if test_mode != "reader_searcher":
            self.writer = Writer(config_path=config_path)
            self.verifier = Verifier(config_path=config_path)

    def _parse_verifier_response(self, response: str) -> Dict[str, Any]:
        """Parse the verifier's XML response into a structured format.
        
        Args:
            response: The XML response from the verifier
            
        Returns:
            Dictionary containing parsed verification results with structure:
            {
                'needs_revision': bool,
                'needs_context': bool,
                'suggestion': str,
                'context_suggestion': str
            }
        """
        result = {
            'needs_revision': False,
            'needs_context': False,
            'suggestion': '',
            'context_suggestion': ''
        }
        
        # Parse NEED_REVISION
        need_revision_match = re.search(r'<NEED_REVISION>(.*?)</NEED_REVISION>', response, re.DOTALL)
        if need_revision_match:
            result['needs_revision'] = need_revision_match.group(1).strip().lower() == 'true'
            
            if result['needs_revision']:
                # Parse MORE_CONTEXT
                more_context_match = re.search(r'<MORE_CONTEXT>(.*?)</MORE_CONTEXT>', response, re.DOTALL)
                if more_context_match:
                    result['needs_context'] = more_context_match.group(1).strip().lower() == 'true'
                    
                    if result['needs_context']:
                        # Extract context suggestion
                        context_suggestion_match = re.search(r'<SUGGESTION_CONTEXT>(.*?)</SUGGESTION_CONTEXT>', response, re.DOTALL)
                        if context_suggestion_match:
                            result['context_suggestion'] = context_suggestion_match.group(1).strip()
                    else:
                        # Extract improvement suggestion
                        suggestion_match = re.search(r'<SUGGESTION>(.*?)</SUGGESTION>', response, re.DOTALL)
                        if suggestion_match:
                            result['suggestion'] = suggestion_match.group(1).strip()
        
        return result

    def process(
        self,
        focal_component: str,
        file_path: str,
        ast_node: ast.AST = None,
        ast_tree: ast.AST = None,
        dependency_graph: Dict[str, List[str]] = None,
        focal_node_dependency_path: str = None,
        token_consume_focal: int = 0
    ) -> str:
        """Process a docstring generation request through the entire agent workflow.
        
        Args:
            focal_component: The code component needing a docstring (full code snippet)
            file_path: Path to the file containing the component (Only input relative file path to the belonged repo!)
            ast_node: Optional AST node representing the focal component
            ast_tree: Optional AST tree for the entire file
            
        Returns:
            The generated and verified docstring, or reader response in test mode
        """
        # Reset visualization and set current component
        self.visualizer.reset()
        self.visualizer.set_current_component(focal_component, file_path)
        # context should be reset to empty string
        self.context = ""
        # Initialize attempt counters
        reader_search_attempts = 0
        verifier_rejection_count = 0
        
        while True:
            # Step 1: Reader determines if more context is needed
            self.visualizer.update('reader', "Analyzing code component...")
            reader_response = self.reader.process(
                focal_component,
                self.context
            )
            # add reader_response to reader's memory (assistant)
            self.reader.add_to_memory("assistant", reader_response)
            
            # Step 2: Check if more information is needed
            match = re.search(r'<INFO_NEED>(.*?)</INFO_NEED>', reader_response, re.DOTALL)
            needs_info = match and match.group(1).strip().lower() == 'true'
            
            if needs_info and reader_search_attempts < self.max_reader_search_attempts:
                reader_search_attempts += 1
                self.visualizer.update('reader', f"Need more information (attempt {reader_search_attempts}/{self.max_reader_search_attempts}), ask Searcher to search additional context...")
                if self.test_mode != "context_print":
                    time.sleep(self.status_sleep_time)
                # Use Searcher to gather more information
                self.visualizer.update('searcher', "Searching for additional context...")
                if self.test_mode != "context_print":
                    time.sleep(self.status_sleep_time)
                search_results = self.searcher.process(reader_response, ast_node, ast_tree, dependency_graph, focal_node_dependency_path)
                self._update_context(search_results, token_consume_focal)
                # Refresh reader's memory with new context
                self.reader.refresh_memory([
                    {"role": "system", "content": self.reader.system_prompt},
                    {"role": "user", "content": f"Current context:\n{self.context}"}
                ])
                self.visualizer.update('reader', "Search complete, Context updated, restarting analysis...")
                if self.test_mode != "context_print":
                    time.sleep(self.status_sleep_time)
                continue
            elif needs_info:
                self.visualizer.update('reader', f"Max search attempts ({self.max_reader_search_attempts}) reached, proceeding with current context...")
                if self.test_mode != "context_print":
                    time.sleep(self.status_sleep_time)

            self.visualizer.update('reader', "No additional context needed, starting docstring generation...")
            if self.test_mode != "context_print":
                time.sleep(self.status_sleep_time)
            
            # If in reader_searcher test mode, return after context gathering
            if self.test_mode == "reader_searcher":
                return reader_response
            
            while True:  # Inner loop for writer-verifier cycle
                # Step 3: When enough context is gathered, use Writer to generate docstring
                self.visualizer.update('writer', "Generating docstring...")
                
                # Print context if in context_print test mode
                if self.test_mode == "context_print":
                    print("\n=== CONTEXT BEFORE WRITER CALL ===")
                    print(self.context)
                    print("=== END OF CONTEXT ===\n")
                
                docstring = self.writer.process(
                    focal_component,
                    self.context
                )
                # assert docstring is not empty
                # add writer_response to writer's memory (assistant)
                self.writer.add_to_memory("assistant", docstring)

                # Step 4: Use Verifier to check the quality
                self.visualizer.update('verifier', "Verifying docstring quality...")
                verification_response = self.verifier.process(
                    focal_component,
                    docstring,
                    self.context
                )
                
                # Step 5: Parse and process verification results
                verification_result = self._parse_verifier_response(verification_response)
                
                if not verification_result['needs_revision'] or verifier_rejection_count >= self.max_verifier_rejections:
                    if verifier_rejection_count >= self.max_verifier_rejections:
                        self.visualizer.update('verifier', f"Max rejection attempts ({self.max_verifier_rejections}) reached, accepting current docstring.")
                    else:
                        self.visualizer.update('verifier', "Docstring generated successfully! No need for revision.")
                    if self.test_mode != "context_print":
                        time.sleep(self.status_sleep_time)
                    return docstring
                # if needs_revision is true, then needs_context is true
                else:
                    verifier_rejection_count += 1
                    # clean verifier's memory
                    self.verifier.clear_memory()
                    if verification_result['needs_context'] and reader_search_attempts < self.max_reader_search_attempts:
                        self.visualizer.update('verifier', f"Need more context (rejection {verifier_rejection_count}/{self.max_verifier_rejections}), hands back to reader...")
                        if self.test_mode != "context_print":
                            time.sleep(self.status_sleep_time)
                        # Add context suggestion to reader's memory and break inner loop to get more context
                        self.reader.add_to_memory(
                            "user",
                            f"Additional context needed: {verification_result['context_suggestion']}"
                        )

                        # clean writer's and verifier's memory
                        self.writer.clear_memory()
                        
                        break  # Break inner loop to return to reader-searcher cycle
                    else:
                        self.visualizer.update('verifier', f"Content is not good enough (rejection {verifier_rejection_count}/{self.max_verifier_rejections}), hands back to writer...")
                        if self.test_mode != "context_print":
                            time.sleep(self.status_sleep_time)
                        # Add improvement suggestion to writer's memory and continue inner loop
                        self.writer.add_to_memory(
                            "user",
                            f"Please improve the docstring based on this suggestion: {verification_result['suggestion']}"
                        )
                        # Continue inner loop to generate new docstring

    def _update_context(self, search_results: Dict[str, Any], token_consume_focal: int) -> None:
        """Update the context with new search results by merging content within existing XML tags.
        
        Args:
            search_results: Dictionary containing new context information structured as:
                {
                    'internal': {
                        'calls': {
                            'class': {'class1': 'content1', ...},
                            'function': {'func1': 'content1', ...},
                            'method': {'method1': 'content1', ...},
                        },
                        'called_by': ['code snippet1', ...]
                    },
                    'external': {
                        'query1': 'result1',
                        'query2': 'result2'
                    }
                }
        """
        if not self.context:
            # Initialize empty context structure if none exists
            self.context = """<CONTEXT>
<INTERNAL_INFO>
<CLASS>
</CLASS>
<FUNCTION>
</FUNCTION>
<METHOD>
</METHOD>
<CALL_BY>
</CALL_BY>
</INTERNAL_INFO>
<EXTERNAL_RETRIEVAL_INFO>
</EXTERNAL_RETRIEVAL_INFO>
</CONTEXT>"""

        if 'internal' in search_results:
            internal_info = search_results['internal']
            
            # Handle calls (class, function, method)
            if 'calls' in internal_info:
                calls = internal_info['calls']
                
                # Helper function to safely update XML content
                def update_xml_section(tag: str, content_list: list) -> None:
                    if not content_list:
                        return
                    pattern = f'<{tag}>(.*?)</{tag}>'
                    match = re.search(pattern, self.context, re.DOTALL)
                    if not match:
                        # If pattern doesn't exist, something is wrong with context structure
                        return
                    existing_text = match.group(1).strip()
                    new_content = existing_text + "\n" + "\n".join(content_list) if existing_text else "\n".join(content_list)
                    # Escape backslashes in new_content to prevent regex interpretation issues
                    new_content = new_content.replace('\\', '\\\\')
                    self.context = re.sub(pattern, f'<{tag}>\n{new_content}\n</{tag}>', self.context, flags=re.DOTALL)
                
                # Update class calls
                if 'class' in calls:
                    class_content = [f"<{class_name}>{content}</{class_name}>" for class_name, content in calls['class'].items()]
                    update_xml_section('CLASS', class_content)

                # Update function calls
                if 'function' in calls:
                    func_content = [f"<{func_name}>{content}</{func_name}>" for func_name, content in calls['function'].items()]
                    update_xml_section('FUNCTION', func_content)

                # Update method calls
                if 'method' in calls:
                    method_content = [f"<{method_name}>{content}</{method_name}>" for method_name, content in calls['method'].items()]
                    update_xml_section('METHOD', method_content)

            # Update called_by
            if 'called_by' in internal_info:
                called_by_content = internal_info['called_by']
                update_xml_section('CALL_BY', called_by_content)

        # Update external info
        if 'external' in search_results:
            external_content = []
            for query, result in search_results['external'].items():
                external_content.append(f"<QUERY>{query}</QUERY>")
                external_content.append(f"<r>{result}</r>")
            update_xml_section('EXTERNAL_RETRIEVAL_INFO', external_content) 
        
        # Apply context length constraint for all models
        if hasattr(self, 'config') and 'max_input_tokens' in self.config:
            max_input_tokens = self.config.get('max_input_tokens', 10000)
        else:
            max_input_tokens = 10000  # Default fallback
            
        self._constrain_context_length(max_input_tokens=max_input_tokens, token_consume_focal=token_consume_focal)
    
    def _constrain_context_length(self, max_input_tokens: int = 10000, token_consume_focal: int = 0) -> None:
        """Constrain context length for models by truncating the longest component.
        
        Args:
            max_input_tokens: Maximum number of tokens allowed in the input context
            token_consume_focal: Number of tokens consumed by the focal component itself
        """
        try:
            # Use tiktoken to count tokens
            encoding = tiktoken.get_encoding("cl100k_base")  # Using a common encoding
            current_tokens = len(encoding.encode(self.context))
            
            # Check if we need to truncate considering both context and focal component tokens
            if current_tokens + token_consume_focal <= max_input_tokens:
                return  # No need to truncate
            
            # Find the XML section with the most tokens to truncate
            component_tokens = {}
            components = [
                ('CODE_CONTEXT', r'<CODE_CONTEXT>(.*?)</CODE_CONTEXT>'),
                ('FOCAL_COMPONENT', r'<FOCAL_COMPONENT>(.*?)</FOCAL_COMPONENT>'),
                ('RELATED_COMPONENTS', r'<RELATED_COMPONENTS>(.*?)</RELATED_COMPONENTS>'),
                ('FOCAL_DEPENDENCIES', r'<FOCAL_DEPENDENCIES>(.*?)</FOCAL_DEPENDENCIES>'),
                ('EXTERNAL_RETRIEVAL_INFO', r'<EXTERNAL_RETRIEVAL_INFO>(.*?)</EXTERNAL_RETRIEVAL_INFO>')
            ]
            
            for name, pattern in components:
                match = re.search(pattern, self.context, re.DOTALL)
                if match:
                    content = match.group(1)
                    tokens = len(encoding.encode(content))
                    component_tokens[name] = (content, tokens)
            
            # Find the component with the most tokens
            if not component_tokens:
                return  # No components found
                
            longest_component = max(component_tokens.items(), key=lambda x: x[1][1])
            component_name = longest_component[0]
            content = longest_component[1][0]
            component_token_count = longest_component[1][1]
            
            # Calculate tokens to remove, considering focal component
            tokens_to_remove = current_tokens + token_consume_focal - max_input_tokens
            
            if tokens_to_remove <= 0:
                return  # No need to truncate
                
            # Print information about truncation
            print(f"Truncating {component_name}: removing {tokens_to_remove} tokens from {component_token_count} tokens. Current total: {current_tokens} tokens")
                
            if tokens_to_remove >= component_token_count:
                # If removing the entire component isn't enough, we'll just remove it and deal with the rest later
                new_content = ""
            else:
                # Truncate the content by removing tokens from the end
                encoded_content = encoding.encode(content)
                truncated_encoded = encoded_content[:-tokens_to_remove]
                new_content = encoding.decode(truncated_encoded)
            
            # Update the context with truncated content
            pattern = f'<{component_name}>(.*?)</{component_name}>'
            self.context = re.sub(pattern, f'<{component_name}>\n{new_content}\n</{component_name}>', self.context, flags=re.DOTALL)
            
        except Exception as e:
            print(f"Error constraining context length: {e}") 
        

"""
Example custom rules plugin for IntelliRefactor

Demonstrates how to create custom analysis and refactoring rules using the hook system.
This plugin shows various ways to extend IntelliRefactor's functionality through hooks.
"""

import ast
import os
from typing import Dict, List, Any, Optional
from pathlib import Path

from ..plugin_interface import AnalysisPlugin, PluginMetadata, PluginType
from ..hook_system import HookSystem, HookType, HookPriority


class CustomRulesPlugin(AnalysisPlugin):
    """
    Example plugin demonstrating custom analysis and refactoring rules.
    
    This plugin implements several custom rules:
    1. Detect long parameter lists (analysis rule)
    2. Identify complex conditional expressions (analysis rule)
    3. Find potential constant extraction opportunities (refactoring rule)
    4. Detect naming convention violations (analysis rule)
    """
    
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="custom_rules_example",
            version="1.0.0",
            description="Example plugin demonstrating custom analysis and refactoring rules",
            author="IntelliRefactor Team",
            plugin_type=PluginType.ANALYSIS,
            dependencies=[],
            config_schema={
                "max_parameters": {"type": "integer", "default": 5, "description": "Maximum allowed parameters"},
                "max_complexity": {"type": "integer", "default": 10, "description": "Maximum cyclomatic complexity"},
                "naming_conventions": {
                    "type": "object",
                    "default": {
                        "class_pattern": "^[A-Z][a-zA-Z0-9]*$",
                        "function_pattern": "^[a-z_][a-z0-9_]*$",
                        "constant_pattern": "^[A-Z_][A-Z0-9_]*$"
                    }
                }
            }
        )
    
    def initialize(self) -> bool:
        """Initialize the plugin and register hooks."""
        try:
            # Get hook system instance (would be injected by plugin manager)
            self.hook_system = getattr(self, '_hook_system', None)
            if not self.hook_system:
                self.logger.warning("Hook system not available, creating local instance")
                self.hook_system = HookSystem()
            
            # Register our custom hooks
            self._register_hooks()
            
            self.logger.info("Custom rules plugin initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize custom rules plugin: {e}")
            return False
    
    def _register_hooks(self) -> None:
        """Register all custom hooks."""
        # Pre-analysis hook to set up custom context
        self.hook_system.register_hook(
            hook_type=HookType.PRE_FILE_ANALYSIS,
            callback=self._pre_analysis_hook,
            name="custom_rules_pre_analysis",
            priority=HookPriority.HIGH,
            plugin_name=self.metadata.name,
            description="Set up custom analysis context"
        )
        
        # Post-analysis hook to add custom analysis results
        self.hook_system.register_hook(
            hook_type=HookType.POST_FILE_ANALYSIS,
            callback=self._post_analysis_hook,
            name="custom_rules_post_analysis",
            priority=HookPriority.NORMAL,
            plugin_name=self.metadata.name,
            description="Add custom analysis results"
        )
        
        # Custom hook for parameter analysis
        self.hook_system.register_hook(
            hook_type=HookType.CUSTOM,
            callback=self._analyze_parameters,
            name="analyze_parameters",
            priority=HookPriority.NORMAL,
            plugin_name=self.metadata.name,
            description="Analyze function parameters",
            custom_key="analyze_parameters"
        )
        
        # Custom hook for complexity analysis
        self.hook_system.register_hook(
            hook_type=HookType.CUSTOM,
            callback=self._analyze_complexity,
            name="analyze_complexity",
            priority=HookPriority.NORMAL,
            plugin_name=self.metadata.name,
            description="Analyze code complexity",
            custom_key="analyze_complexity"
        )
        
        # Custom hook for naming convention analysis
        self.hook_system.register_hook(
            hook_type=HookType.CUSTOM,
            callback=self._analyze_naming,
            name="analyze_naming",
            priority=HookPriority.NORMAL,
            plugin_name=self.metadata.name,
            description="Analyze naming conventions",
            custom_key="analyze_naming"
        )
        
        # Refactoring opportunity detection hook
        self.hook_system.register_hook(
            hook_type=HookType.POST_OPPORTUNITY_DETECTION,
            callback=self._detect_refactoring_opportunities,
            name="custom_refactoring_opportunities",
            priority=HookPriority.NORMAL,
            plugin_name=self.metadata.name,
            description="Detect custom refactoring opportunities"
        )
    
    def _pre_analysis_hook(self, file_path: str, content: str, context: Dict[str, Any]) -> None:
        """Pre-analysis hook to set up custom context."""
        self.logger.debug(f"Pre-analysis hook for {file_path}")
        
        # Add custom context for our analysis
        context['custom_rules'] = {
            'file_path': file_path,
            'issues': [],
            'opportunities': [],
            'metrics': {}
        }
    
    def _post_analysis_hook(self, file_path: str, content: str, 
                           context: Dict[str, Any], results: Dict[str, Any]) -> None:
        """Post-analysis hook to add custom analysis results."""
        self.logger.debug(f"Post-analysis hook for {file_path}")
        
        if 'custom_rules' in context:
            # Add our custom analysis results to the main results
            results['custom_rules'] = context['custom_rules']
    
    def _analyze_parameters(self, file_path: str, ast_tree: ast.AST, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze function parameters for long parameter lists."""
        max_params = self.config.get('max_parameters', 5)
        issues = []
        
        class ParameterAnalyzer(ast.NodeVisitor):
            def visit_FunctionDef(self, node):
                param_count = len(node.args.args)
                if param_count > max_params:
                    issues.append({
                        'type': 'long_parameter_list',
                        'severity': 'warning',
                        'line': node.lineno,
                        'column': node.col_offset,
                        'function': node.name,
                        'parameter_count': param_count,
                        'max_allowed': max_params,
                        'message': f"Function '{node.name}' has {param_count} parameters (max: {max_params})"
                    })
                self.generic_visit(node)
        
        analyzer = ParameterAnalyzer()
        analyzer.visit(ast_tree)
        
        return {'parameter_issues': issues}
    
    def _analyze_complexity(self, file_path: str, ast_tree: ast.AST, 
                           context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze cyclomatic complexity."""
        max_complexity = self.config.get('max_complexity', 10)
        issues = []
        
        class ComplexityAnalyzer(ast.NodeVisitor):
            def __init__(self):
                self.complexity = 1  # Base complexity
                self.function_name = None
                self.function_line = None
            
            def visit_FunctionDef(self, node):
                old_complexity = self.complexity
                old_name = self.function_name
                old_line = self.function_line
                
                self.complexity = 1  # Reset for this function
                self.function_name = node.name
                self.function_line = node.lineno
                
                self.generic_visit(node)
                
                if self.complexity > max_complexity:
                    issues.append({
                        'type': 'high_complexity',
                        'severity': 'warning',
                        'line': node.lineno,
                        'column': node.col_offset,
                        'function': node.name,
                        'complexity': self.complexity,
                        'max_allowed': max_complexity,
                        'message': f"Function '{node.name}' has complexity {self.complexity} (max: {max_complexity})"
                    })
                
                # Restore previous state
                self.complexity = old_complexity
                self.function_name = old_name
                self.function_line = old_line
            
            def visit_If(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_While(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_For(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_ExceptHandler(self, node):
                self.complexity += 1
                self.generic_visit(node)
            
            def visit_With(self, node):
                self.complexity += 1
                self.generic_visit(node)
        
        analyzer = ComplexityAnalyzer()
        analyzer.visit(ast_tree)
        
        return {'complexity_issues': issues}
    
    def _analyze_naming(self, file_path: str, ast_tree: ast.AST, 
                       context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze naming conventions."""
        import re
        
        naming_config = self.config.get('naming_conventions', {})
        class_pattern = naming_config.get('class_pattern', r'^[A-Z][a-zA-Z0-9]*$')
        function_pattern = naming_config.get('function_pattern', r'^[a-z_][a-z0-9_]*$')
        constant_pattern = naming_config.get('constant_pattern', r'^[A-Z_][A-Z0-9_]*$')
        
        issues = []
        
        class NamingAnalyzer(ast.NodeVisitor):
            def visit_ClassDef(self, node):
                if not re.match(class_pattern, node.name):
                    issues.append({
                        'type': 'naming_convention',
                        'severity': 'info',
                        'line': node.lineno,
                        'column': node.col_offset,
                        'element_type': 'class',
                        'name': node.name,
                        'expected_pattern': class_pattern,
                        'message': f"Class '{node.name}' doesn't follow naming convention"
                    })
                self.generic_visit(node)
            
            def visit_FunctionDef(self, node):
                if not re.match(function_pattern, node.name):
                    issues.append({
                        'type': 'naming_convention',
                        'severity': 'info',
                        'line': node.lineno,
                        'column': node.col_offset,
                        'element_type': 'function',
                        'name': node.name,
                        'expected_pattern': function_pattern,
                        'message': f"Function '{node.name}' doesn't follow naming convention"
                    })
                self.generic_visit(node)
            
            def visit_Assign(self, node):
                # Check for constants (all uppercase assignments at module level)
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        if name.isupper() and not re.match(constant_pattern, name):
                            issues.append({
                                'type': 'naming_convention',
                                'severity': 'info',
                                'line': node.lineno,
                                'column': node.col_offset,
                                'element_type': 'constant',
                                'name': name,
                                'expected_pattern': constant_pattern,
                                'message': f"Constant '{name}' doesn't follow naming convention"
                            })
                self.generic_visit(node)
        
        analyzer = NamingAnalyzer()
        analyzer.visit(ast_tree)
        
        return {'naming_issues': issues}
    
    def _detect_refactoring_opportunities(self, analysis_results: Dict[str, Any], 
                                        opportunities: List[Dict[str, Any]]) -> None:
        """Detect custom refactoring opportunities."""
        # Add opportunities based on our custom analysis
        if 'custom_rules' in analysis_results:
            custom_results = analysis_results['custom_rules']
            
            # Convert issues to refactoring opportunities
            for issue in custom_results.get('issues', []):
                if issue['type'] == 'long_parameter_list':
                    opportunities.append({
                        'type': 'extract_parameter_object',
                        'priority': 'medium',
                        'description': f"Extract parameter object for function '{issue['function']}'",
                        'location': {
                            'file': custom_results['file_path'],
                            'line': issue['line'],
                            'column': issue['column']
                        },
                        'metadata': {
                            'function_name': issue['function'],
                            'parameter_count': issue['parameter_count']
                        }
                    })
                
                elif issue['type'] == 'high_complexity':
                    opportunities.append({
                        'type': 'extract_method',
                        'priority': 'high',
                        'description': f"Extract methods from complex function '{issue['function']}'",
                        'location': {
                            'file': custom_results['file_path'],
                            'line': issue['line'],
                            'column': issue['column']
                        },
                        'metadata': {
                            'function_name': issue['function'],
                            'complexity': issue['complexity']
                        }
                    })
    
    def analyze_project(self, project_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze entire project with custom rules."""
        results = {
            'plugin': self.metadata.name,
            'version': self.metadata.version,
            'files_analyzed': 0,
            'total_issues': 0,
            'issues_by_type': {},
            'files': {}
        }
        
        # Analyze all Python files in the project
        for root, dirs, files in os.walk(project_path):
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        file_results = self.analyze_file(file_path, content, context)
                        results['files'][file_path] = file_results
                        results['files_analyzed'] += 1
                        
                        # Aggregate statistics
                        if 'custom_rules' in file_results:
                            file_issues = file_results['custom_rules'].get('issues', [])
                            results['total_issues'] += len(file_issues)
                            
                            for issue in file_issues:
                                issue_type = issue['type']
                                results['issues_by_type'][issue_type] = results['issues_by_type'].get(issue_type, 0) + 1
                    
                    except Exception as e:
                        self.logger.error(f"Error analyzing file {file_path}: {e}")
        
        return results
    
    def analyze_file(self, file_path: str, content: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single file with custom rules."""
        try:
            # Parse the file
            ast_tree = ast.parse(content, filename=file_path)
            
            # Execute pre-analysis hooks
            self.hook_system.execute_hooks(HookType.PRE_FILE_ANALYSIS, file_path, content, context)
            
            # Run our custom analysis hooks
            param_results = self.hook_system.execute_custom_hooks('analyze_parameters', file_path, ast_tree, context)
            complexity_results = self.hook_system.execute_custom_hooks('analyze_complexity', file_path, ast_tree, context)
            naming_results = self.hook_system.execute_custom_hooks('analyze_naming', file_path, ast_tree, context)
            
            # Collect all issues
            all_issues = []
            for result_list in [param_results, complexity_results, naming_results]:
                for result in result_list:
                    if result:
                        for key, issues in result.items():
                            if isinstance(issues, list):
                                all_issues.extend(issues)
            
            # Update context with our results
            if 'custom_rules' in context:
                context['custom_rules']['issues'] = all_issues
            
            # Execute post-analysis hooks
            results = {'analyzed_by': self.metadata.name}
            self.hook_system.execute_hooks(HookType.POST_FILE_ANALYSIS, file_path, content, context, results)
            
            return results
            
        except SyntaxError as e:
            self.logger.warning(f"Syntax error in {file_path}: {e}")
            return {'error': f'Syntax error: {e}'}
        except Exception as e:
            self.logger.error(f"Error analyzing {file_path}: {e}")
            return {'error': str(e)}
    
    def cleanup(self) -> None:
        """Clean up plugin resources."""
        if hasattr(self, 'hook_system'):
            # Unregister our hooks
            hooks_to_remove = [
                'custom_rules_pre_analysis',
                'custom_rules_post_analysis',
                'analyze_parameters',
                'analyze_complexity',
                'analyze_naming',
                'custom_refactoring_opportunities'
            ]
            
            for hook_name in hooks_to_remove:
                self.hook_system.unregister_hook(hook_name)
        
        self.logger.info("Custom rules plugin cleaned up")
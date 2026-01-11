"""
Optional Dependencies Analyzer for expert refactoring analysis.

Analyzes feature flags and conditional imports to understand
optional dependencies and execution modes.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


class OptionalDependenciesAnalyzer:
    """Analyzes optional dependencies and feature flags."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)

    def analyze_optional_dependencies(self, module_ast: ast.Module) -> Dict[str, any]:
        """
        Analyze optional dependencies and feature flags.
        
        Args:
            module_ast: Parsed AST of the target module
            
        Returns:
            Dictionary with optional dependency analysis
        """
        logger.info("Analyzing optional dependencies and feature flags...")
        
        # Find conditional imports
        conditional_imports = self._find_conditional_imports(module_ast)
        
        # Find feature flags
        feature_flags = self._find_feature_flags(module_ast)
        
        # Find optional module usage
        optional_modules = self._find_optional_module_usage(module_ast)
        
        # Map dependencies to code branches
        dependency_branches = self._map_dependency_branches(module_ast, conditional_imports, feature_flags)
        
        # Generate test scenarios
        test_scenarios = self._generate_test_scenarios(conditional_imports, feature_flags, optional_modules)
        
        return {
            'conditional_imports': conditional_imports,
            'feature_flags': feature_flags,
            'optional_modules': optional_modules,
            'dependency_branches': dependency_branches,
            'test_scenarios': test_scenarios
        }

    def export_detailed_optional_dependencies(self, analysis: Dict[str, any]) -> Dict[str, any]:
        """
        Export detailed optional dependencies data as requested by experts.
        
        Returns:
            Dictionary with feature flags map and test plans
        """
        # Create feature flags map
        feature_flags_map = {}
        for flag in analysis['feature_flags']:
            flag_name = flag['name']
            feature_flags_map[flag_name] = {
                'type': flag['type'],
                'default_value': flag['default_value'],
                'dependent_methods': flag['dependent_methods'],
                'dependent_branches': flag['dependent_branches'],
                'usage_locations': flag['usage_locations']
            }
        
        # Create module availability map
        module_availability_map = {}
        for module in analysis['optional_modules']:
            module_name = module['name']
            module_availability_map[module_name] = {
                'import_method': module['import_method'],
                'fallback_behavior': module['fallback_behavior'],
                'dependent_methods': module['dependent_methods'],
                'usage_patterns': module['usage_patterns']
            }
        
        # Generate dual-mode test plans
        dual_mode_tests = []
        for scenario in analysis['test_scenarios']:
            dual_mode_tests.append({
                'scenario_name': scenario['name'],
                'available_mode': {
                    'setup': scenario['available_setup'],
                    'expected_behavior': scenario['available_behavior'],
                    'test_methods': scenario['available_tests']
                },
                'unavailable_mode': {
                    'setup': scenario['unavailable_setup'],
                    'expected_behavior': scenario['unavailable_behavior'],
                    'test_methods': scenario['unavailable_tests']
                }
            })
        
        return {
            'feature_flags_map': feature_flags_map,
            'module_availability_map': module_availability_map,
            'dual_mode_test_plans': dual_mode_tests,
            'execution_modes': self._identify_execution_modes(analysis),
            'recommendations': self._generate_optional_dependency_recommendations(analysis)
        }

    def _find_conditional_imports(self, module_ast: ast.Module) -> List[Dict[str, any]]:
        """Find imports that are conditional (try/except, if statements)."""
        conditional_imports = []
        
        for node in ast.walk(module_ast):
            if isinstance(node, ast.Try):
                # Check if try block contains imports
                for stmt in node.body:
                    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                        import_info = self._extract_import_info(stmt)
                        import_info.update({
                            'condition_type': 'try_except',
                            'line': getattr(node, 'lineno', 0),
                            'fallback_behavior': self._extract_except_behavior(node.handlers)
                        })
                        conditional_imports.append(import_info)
            
            elif isinstance(node, ast.If):
                # Check if if block contains imports
                for stmt in node.body:
                    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                        import_info = self._extract_import_info(stmt)
                        import_info.update({
                            'condition_type': 'if_statement',
                            'line': getattr(node, 'lineno', 0),
                            'condition': self._extract_condition(node.test)
                        })
                        conditional_imports.append(import_info)
        
        return conditional_imports

    def _find_feature_flags(self, module_ast: ast.Module) -> List[Dict[str, any]]:
        """Find feature flags and configuration variables."""
        feature_flags = []
        
        # Look for assignments that might be feature flags
        for node in ast.walk(module_ast):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        var_name = target.id
                        if self._is_likely_feature_flag(var_name):
                            flag_info = {
                                'name': var_name,
                                'type': self._determine_flag_type(node.value),
                                'default_value': self._extract_default_value(node.value),
                                'line': getattr(node, 'lineno', 0),
                                'dependent_methods': self._find_flag_usage(module_ast, var_name),
                                'dependent_branches': self._find_flag_branches(module_ast, var_name),
                                'usage_locations': self._find_flag_usage_locations(module_ast, var_name)
                            }
                            feature_flags.append(flag_info)
        
        return feature_flags

    def _find_optional_module_usage(self, module_ast: ast.Module) -> List[Dict[str, any]]:
        """Find usage of optionally imported modules."""
        optional_modules = []
        
        # Common optional modules in the domain
        known_optional = [
            'advanced_attacks', 'operation_logger', 'UnifiedAttackDispatcher',
            'SNIExtractor', 'pcap_metadata', 'AttackRegistry'
        ]
        
        for module_name in known_optional:
            usage_info = self._analyze_module_usage(module_ast, module_name)
            if usage_info['is_used']:
                optional_modules.append(usage_info)
        
        return optional_modules

    def _map_dependency_branches(self, module_ast: ast.Module, conditional_imports: List[Dict], feature_flags: List[Dict]) -> List[Dict[str, any]]:
        """Map dependencies to specific code branches."""
        branches = []
        
        # Map conditional import branches
        for imp in conditional_imports:
            branch_info = {
                'type': 'conditional_import',
                'dependency': imp['module_name'],
                'condition': imp.get('condition', 'import_available'),
                'affected_methods': self._find_methods_using_import(module_ast, imp['symbols']),
                'branch_complexity': self._calculate_branch_complexity(module_ast, imp['symbols'])
            }
            branches.append(branch_info)
        
        # Map feature flag branches
        for flag in feature_flags:
            branch_info = {
                'type': 'feature_flag',
                'dependency': flag['name'],
                'condition': f"{flag['name']} == {flag['default_value']}",
                'affected_methods': flag['dependent_methods'],
                'branch_complexity': len(flag['dependent_branches'])
            }
            branches.append(branch_info)
        
        return branches

    def _generate_test_scenarios(self, conditional_imports: List[Dict], feature_flags: List[Dict], optional_modules: List[Dict]) -> List[Dict[str, any]]:
        """Generate test scenarios for different dependency states."""
        scenarios = []
        
        # Scenarios for conditional imports
        for imp in conditional_imports:
            scenario = {
                'name': f"test_{imp['module_name']}_availability",
                'available_setup': f"Mock {imp['module_name']} as available",
                'available_behavior': f"Should use {imp['module_name']} functionality",
                'available_tests': [f"test_with_{imp['module_name']}_available"],
                'unavailable_setup': f"Mock {imp['module_name']} import to fail",
                'unavailable_behavior': imp.get('fallback_behavior', 'Should use fallback behavior'),
                'unavailable_tests': [f"test_with_{imp['module_name']}_unavailable"]
            }
            scenarios.append(scenario)
        
        # Scenarios for feature flags
        for flag in feature_flags:
            scenario = {
                'name': f"test_{flag['name']}_modes",
                'available_setup': f"Set {flag['name']} = True",
                'available_behavior': f"Should enable {flag['name']} functionality",
                'available_tests': [f"test_{flag['name']}_enabled"],
                'unavailable_setup': f"Set {flag['name']} = False",
                'unavailable_behavior': f"Should disable {flag['name']} functionality",
                'unavailable_tests': [f"test_{flag['name']}_disabled"]
            }
            scenarios.append(scenario)
        
        return scenarios

    def _identify_execution_modes(self, analysis: Dict[str, any]) -> List[Dict[str, any]]:
        """Identify different execution modes based on dependencies."""
        modes = []
        
        # Basic mode (no optional dependencies)
        modes.append({
            'name': 'basic_mode',
            'description': 'Basic functionality without optional dependencies',
            'required_setup': 'Disable all optional imports and flags',
            'expected_behavior': 'Core functionality should work'
        })
        
        # Full mode (all optional dependencies)
        modes.append({
            'name': 'full_mode',
            'description': 'Full functionality with all optional dependencies',
            'required_setup': 'Enable all optional imports and flags',
            'expected_behavior': 'All advanced features should be available'
        })
        
        # Mixed modes based on specific combinations
        feature_flags = analysis.get('feature_flags', [])
        if len(feature_flags) > 1:
            modes.append({
                'name': 'mixed_mode',
                'description': 'Partial functionality with some optional dependencies',
                'required_setup': 'Enable subset of optional features',
                'expected_behavior': 'Should gracefully handle mixed availability'
            })
        
        return modes

    def _generate_optional_dependency_recommendations(self, analysis: Dict[str, any]) -> List[str]:
        """Generate recommendations for handling optional dependencies."""
        recommendations = []
        
        conditional_imports = analysis.get('conditional_imports', [])
        feature_flags = analysis.get('feature_flags', [])
        optional_modules = analysis.get('optional_modules', [])
        
        if conditional_imports:
            recommendations.append(f"Test {len(conditional_imports)} conditional imports in both available/unavailable states")
        
        if feature_flags:
            recommendations.append(f"Create test matrix for {len(feature_flags)} feature flag combinations")
        
        if optional_modules:
            recommendations.append(f"Verify graceful degradation when {len(optional_modules)} optional modules are unavailable")
        
        # Specific recommendations
        if len(feature_flags) > 3:
            recommendations.append("Consider reducing feature flag complexity - too many conditional paths")
        
        if len(conditional_imports) > 5:
            recommendations.append("High number of conditional imports - consider dependency injection pattern")
        
        return recommendations

    def _extract_import_info(self, import_node: ast.stmt) -> Dict[str, any]:
        """Extract information from an import statement."""
        if isinstance(import_node, ast.Import):
            return {
                'module_name': import_node.names[0].name,
                'symbols': [alias.name for alias in import_node.names],
                'import_type': 'import'
            }
        elif isinstance(import_node, ast.ImportFrom):
            return {
                'module_name': import_node.module or '',
                'symbols': [alias.name for alias in import_node.names],
                'import_type': 'from_import'
            }
        return {}

    def _extract_except_behavior(self, handlers: List[ast.ExceptHandler]) -> str:
        """Extract behavior from except handlers."""
        if not handlers:
            return "No fallback behavior"
        
        # Simple heuristic - look for assignments or function calls in except block
        for handler in handlers:
            if handler.body:
                first_stmt = handler.body[0]
                if isinstance(first_stmt, ast.Assign):
                    return "Assigns fallback value"
                elif isinstance(first_stmt, ast.Expr) and isinstance(first_stmt.value, ast.Call):
                    return "Calls fallback function"
        
        return "Has fallback behavior"

    def _extract_condition(self, test_node: ast.expr) -> str:
        """Extract condition from if statement."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(test_node)
            else:
                return "condition"
        except Exception:
            return "condition"

    def _is_likely_feature_flag(self, var_name: str) -> bool:
        """Check if a variable name is likely a feature flag."""
        flag_indicators = [
            'enable', 'disable', 'use_', 'has_', 'is_', 'allow_',
            'advanced', 'debug', 'verbose', 'strict', 'fallback'
        ]
        
        var_lower = var_name.lower()
        return any(indicator in var_lower for indicator in flag_indicators)

    def _determine_flag_type(self, value_node: ast.expr) -> str:
        """Determine the type of a feature flag."""
        if isinstance(value_node, ast.Constant):
            if isinstance(value_node.value, bool):
                return 'boolean'
            elif isinstance(value_node.value, str):
                return 'string'
            elif isinstance(value_node.value, (int, float)):
                return 'numeric'
        elif isinstance(value_node, ast.Name):
            if value_node.id in ['True', 'False']:
                return 'boolean'
            elif value_node.id == 'None':
                return 'optional'
        
        return 'unknown'

    def _extract_default_value(self, value_node: ast.expr) -> any:
        """Extract the default value of a feature flag."""
        if isinstance(value_node, ast.Constant):
            return value_node.value
        elif isinstance(value_node, ast.Name):
            if value_node.id == 'True':
                return True
            elif value_node.id == 'False':
                return False
            elif value_node.id == 'None':
                return None
        
        return 'unknown'

    def _find_flag_usage(self, module_ast: ast.Module, flag_name: str) -> List[str]:
        """Find methods that use a specific feature flag."""
        using_methods = []
        
        for node in ast.walk(module_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if flag is used in this method
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and child.id == flag_name:
                        using_methods.append(node.name)
                        break
        
        return using_methods

    def _find_flag_branches(self, module_ast: ast.Module, flag_name: str) -> List[Dict[str, any]]:
        """Find code branches that depend on a feature flag."""
        branches = []
        
        for node in ast.walk(module_ast):
            if isinstance(node, ast.If):
                # Check if condition involves the flag
                condition_str = self._extract_condition(node.test)
                if flag_name in condition_str:
                    branch_info = {
                        'line': getattr(node, 'lineno', 0),
                        'condition': condition_str,
                        'branch_size': len(node.body),
                        'has_else': len(node.orelse) > 0
                    }
                    branches.append(branch_info)
        
        return branches

    def _find_flag_usage_locations(self, module_ast: ast.Module, flag_name: str) -> List[Dict[str, any]]:
        """Find all locations where a flag is used."""
        locations = []
        
        for node in ast.walk(module_ast):
            if isinstance(node, ast.Name) and node.id == flag_name:
                location = {
                    'line': getattr(node, 'lineno', 0),
                    'context': self._get_usage_context(node)
                }
                locations.append(location)
        
        return locations

    def _get_usage_context(self, node: ast.Name) -> str:
        """Get context information about flag usage."""
        # This is a simplified version - in practice you'd walk up the AST
        return f"usage at line {getattr(node, 'lineno', 0)}"

    def _analyze_module_usage(self, module_ast: ast.Module, module_name: str) -> Dict[str, any]:
        """Analyze how an optional module is used."""
        usage_info = {
            'name': module_name,
            'is_used': False,
            'import_method': 'unknown',
            'fallback_behavior': 'unknown',
            'dependent_methods': [],
            'usage_patterns': []
        }
        
        # Check if module is imported or used
        for node in ast.walk(module_ast):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if self._import_references_module(node, module_name):
                    usage_info['is_used'] = True
                    usage_info['import_method'] = 'direct_import'
            elif isinstance(node, ast.Name) and node.id == module_name:
                usage_info['is_used'] = True
                usage_info['usage_patterns'].append({
                    'line': getattr(node, 'lineno', 0),
                    'context': 'name_reference'
                })
        
        return usage_info

    def _import_references_module(self, import_node: ast.stmt, module_name: str) -> bool:
        """Check if an import statement references a specific module."""
        if isinstance(import_node, ast.Import):
            return any(alias.name == module_name for alias in import_node.names)
        elif isinstance(import_node, ast.ImportFrom):
            return (import_node.module == module_name or 
                    any(alias.name == module_name for alias in import_node.names))
        return False

    def _find_methods_using_import(self, module_ast: ast.Module, symbols: List[str]) -> List[str]:
        """Find methods that use imported symbols."""
        using_methods = []
        
        for node in ast.walk(module_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Check if any symbol is used in this method
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and child.id in symbols:
                        using_methods.append(node.name)
                        break
        
        return using_methods

    def _calculate_branch_complexity(self, module_ast: ast.Module, symbols: List[str]) -> int:
        """Calculate complexity of branches using specific symbols."""
        complexity = 0
        
        for node in ast.walk(module_ast):
            if isinstance(node, ast.If):
                # Check if this branch uses any of the symbols
                uses_symbols = False
                for child in ast.walk(node):
                    if isinstance(child, ast.Name) and child.id in symbols:
                        uses_symbols = True
                        break
                
                if uses_symbols:
                    complexity += 1
        
        return complexity
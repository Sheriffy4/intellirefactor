"""
Dependency Interface Analyzer for expert refactoring analysis.

Analyzes external dependencies and their interfaces to understand
the "стыки" (boundaries) between modules for safe refactoring.
"""

from __future__ import annotations

import ast
import importlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from ..models import (
    DependencyInterface,
    InterfaceUsage,
    RiskLevel,
)

logger = logging.getLogger(__name__)


class DependencyInterfaceAnalyzer:
    """Analyzes interfaces of external dependencies."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)

    def extract_dependency_interfaces(self, module_ast: ast.Module) -> List[DependencyInterface]:
        """
        Extract interfaces of all external dependencies.
        
        Args:
            module_ast: Parsed AST of the target module
            
        Returns:
            List of DependencyInterface objects
        """
        logger.info("Extracting dependency interfaces...")
        
        # First, find all imports
        imports = self._extract_imports(module_ast)
        
        # Then analyze the interface of each dependency
        interfaces = []
        for import_info in imports:
            interface = self._analyze_dependency_interface(import_info, module_ast)
            if interface:
                interfaces.append(interface)
        
        logger.info(f"Extracted {len(interfaces)} dependency interfaces")
        return interfaces

    def _extract_imports(self, module_ast: ast.Module) -> List[Dict[str, Any]]:
        """Extract all import statements from the module."""
        imports = []
        
        for node in ast.walk(module_ast):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append({
                        'type': 'import',
                        'module': alias.name,
                        'alias': alias.asname,
                        'line': getattr(node, 'lineno', 0),
                        'node': node
                    })
            
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                level = getattr(node, 'level', 0)
                
                for alias in node.names:
                    imports.append({
                        'type': 'from_import',
                        'module': module_name,
                        'name': alias.name,
                        'alias': alias.asname,
                        'level': level,
                        'line': getattr(node, 'lineno', 0),
                        'node': node
                    })
        
        return imports

    def _analyze_dependency_interface(self, import_info: Dict[str, Any], module_ast: ast.Module) -> Optional[DependencyInterface]:
        """Analyze the interface of a single dependency."""
        module_name = import_info['module']
        import_type = import_info['type']
        
        # Skip relative imports and standard library for now
        if import_info.get('level', 0) > 0:
            return None
        
        if self._is_standard_library(module_name):
            # Still analyze standard library but with lower criticality
            criticality = RiskLevel.LOW
        else:
            criticality = RiskLevel.MEDIUM
        
        # Find what methods/attributes are actually used
        used_methods, used_attributes = self._find_usage_in_ast(import_info, module_ast)
        
        # Try to introspect the actual module interface
        interface_info = self._introspect_module_interface(module_name, import_info)
        
        # Determine import style
        import_style = import_type
        if import_type == 'from_import':
            import_style = f"from {module_name} import {import_info['name']}"
        else:
            import_style = f"import {module_name}"
        
        interface = DependencyInterface(
            module_name=module_name,
            used_methods=used_methods,
            used_attributes=used_attributes,
            import_style=import_style,
            criticality=criticality,
            version_constraints=interface_info.get('version')
        )
        
        return interface

    def _is_standard_library(self, module_name: str) -> bool:
        """Check if a module is part of the Python standard library."""
        stdlib_modules = {
            'os', 'sys', 'json', 'logging', 'datetime', 'time', 'random',
            'collections', 'itertools', 'functools', 'operator', 'typing',
            'pathlib', 'shutil', 'subprocess', 'threading', 'multiprocessing',
            'asyncio', 'concurrent', 'queue', 'socket', 'urllib', 'http',
            'email', 'html', 'xml', 'csv', 'sqlite3', 'pickle', 'base64',
            'hashlib', 'hmac', 'secrets', 'uuid', 'enum', 'dataclasses',
            'contextlib', 'warnings', 'traceback', 'inspect', 'ast', 're',
            'string', 'textwrap', 'unicodedata', 'codecs', 'io', 'tempfile',
            'glob', 'fnmatch', 'linecache', 'fileinput', 'stat', 'filecmp',
            'tarfile', 'zipfile', 'gzip', 'bz2', 'lzma', 'zlib'
        }
        
        # Check if it's a known stdlib module or starts with stdlib prefix
        base_module = module_name.split('.')[0]
        return base_module in stdlib_modules

    def _find_usage_in_ast(self, import_info: Dict[str, Any], module_ast: ast.Module) -> Tuple[List[str], List[str]]:
        """Find actual usage of imported symbols in the AST."""
        used_methods = []
        used_attributes = []
        
        import_type = import_info['type']
        module_name = import_info['module']
        
        if import_type == 'import':
            # For "import module", look for module.attribute usage
            alias = import_info.get('alias') or module_name
            used_methods, used_attributes = self._find_attribute_usage(module_ast, alias)
        
        elif import_type == 'from_import':
            # For "from module import name", look for direct usage of name
            imported_name = import_info['name']
            alias = import_info.get('alias') or imported_name
            
            # Find direct usage of the imported name
            for node in ast.walk(module_ast):
                if isinstance(node, ast.Name) and node.id == alias:
                    # This is usage of the imported symbol
                    context = self._get_usage_context(node)
                    if context == 'call':
                        used_methods.append(imported_name)
                    else:
                        used_attributes.append(imported_name)
                
                elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == alias:
                    # This is attribute access on the imported symbol
                    used_attributes.append(f"{imported_name}.{node.attr}")
        
        return list(set(used_methods)), list(set(used_attributes))

    def _find_attribute_usage(self, module_ast: ast.Module, module_alias: str) -> Tuple[List[str], List[str]]:
        """Find attribute usage for imported modules."""
        used_methods = []
        used_attributes = []
        
        for node in ast.walk(module_ast):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == module_alias:
                    # This is module.attribute usage
                    context = self._get_usage_context(node)
                    if context == 'call':
                        used_methods.append(node.attr)
                    else:
                        used_attributes.append(node.attr)
        
        return list(set(used_methods)), list(set(used_attributes))

    def _get_usage_context(self, node: ast.AST) -> str:
        """Determine how a symbol is being used (call, assignment, etc.)."""
        # This is a simplified context detection
        # In a real implementation, we'd walk up the AST to find the parent context
        return 'call'  # Default assumption

    def _introspect_module_interface(self, module_name: str, import_info: Dict[str, Any]) -> Dict[str, Any]:
        """Try to introspect the actual module to get interface information."""
        interface_info = {}
        
        try:
            # Try to import the module
            module = importlib.import_module(module_name)
            
            # Get version if available
            if hasattr(module, '__version__'):
                interface_info['version'] = module.__version__
            
            # Get public interface
            if hasattr(module, '__all__'):
                interface_info['public_interface'] = module.__all__
            else:
                # Fallback: get all non-private attributes
                interface_info['public_interface'] = [
                    name for name in dir(module) 
                    if not name.startswith('_')
                ]
            
            # Get docstring
            if hasattr(module, '__doc__') and module.__doc__:
                interface_info['docstring'] = module.__doc__[:200]  # First 200 chars
            
        except ImportError as e:
            logger.warning(f"Could not import {module_name}: {e}")
            interface_info['import_error'] = str(e)
        except Exception as e:
            logger.warning(f"Error introspecting {module_name}: {e}")
            interface_info['introspection_error'] = str(e)
        
        return interface_info

    def analyze_interface_usage(self, interfaces: List[DependencyInterface]) -> InterfaceUsage:
        """
        Analyze patterns in dependency interface usage.
        
        Args:
            interfaces: List of dependency interfaces
            
        Returns:
            InterfaceUsage analysis
        """
        logger.info("Analyzing interface usage patterns...")
        
        critical_interfaces = []
        unused_imports = []
        potential_violations = []
        
        for interface in interfaces:
            # Determine criticality based on usage
            total_usage = len(interface.used_methods) + len(interface.used_attributes)
            
            if total_usage == 0:
                unused_imports.append(interface.module_name)
            elif total_usage > 5 or interface.criticality == RiskLevel.HIGH:
                critical_interfaces.append(interface)
            
            # Check for potential interface violations
            if interface.module_name.startswith('_'):
                potential_violations.append(f"Using private module: {interface.module_name}")
            
            for method in interface.used_methods:
                if method.startswith('_'):
                    potential_violations.append(f"Using private method: {interface.module_name}.{method}")
            
            for attr in interface.used_attributes:
                if attr.startswith('_'):
                    potential_violations.append(f"Using private attribute: {interface.module_name}.{attr}")
        
        usage = InterfaceUsage(
            total_dependencies=len(interfaces),
            critical_interfaces=critical_interfaces,
            unused_imports=unused_imports,
            potential_violations=potential_violations
        )
        
        logger.info(f"Interface usage: {len(critical_interfaces)} critical, {len(unused_imports)} unused")
        return usage

    def detect_interface_violations(self, usage: InterfaceUsage) -> List[str]:
        """
        Detect potential interface violations.
        
        Args:
            usage: InterfaceUsage analysis
            
        Returns:
            List of violation descriptions
        """
        violations = []
        
        # Add violations from usage analysis
        violations.extend(usage.potential_violations)
        
        # Check for other violation patterns
        for interface in usage.critical_interfaces:
            # Check for version constraints
            if not interface.version_constraints:
                violations.append(f"No version constraint for critical dependency: {interface.module_name}")
            
            # Check for too many used methods (tight coupling)
            if len(interface.used_methods) > 10:
                violations.append(f"High coupling with {interface.module_name}: {len(interface.used_methods)} methods used")
        
        return violations

    def suggest_interface_improvements(self, interfaces: List[DependencyInterface]) -> List[str]:
        """
        Suggest improvements for dependency interfaces.
        
        Args:
            interfaces: List of dependency interfaces
            
        Returns:
            List of improvement suggestions
        """
        suggestions = []
        
        # Analyze usage patterns
        usage = self.analyze_interface_usage(interfaces)
        
        # Suggest removing unused imports
        if usage.unused_imports:
            suggestions.append(f"Remove unused imports: {', '.join(usage.unused_imports)}")
        
        # Suggest interface abstractions for high-coupling dependencies
        for interface in usage.critical_interfaces:
            if len(interface.used_methods) > 8:
                suggestions.append(
                    f"Consider creating an adapter/facade for {interface.module_name} "
                    f"(uses {len(interface.used_methods)} methods)"
                )
        
        # Suggest version pinning for critical dependencies
        for interface in usage.critical_interfaces:
            if not interface.version_constraints and not self._is_standard_library(interface.module_name):
                suggestions.append(f"Pin version for critical dependency: {interface.module_name}")
        
        return suggestions

    def analyze_import_dependencies(self, module_ast: ast.Module) -> Dict[str, any]:
        """
        Analyze import-level dependencies and cycles.
        
        Args:
            module_ast: Parsed AST of the target module
            
        Returns:
            Dictionary with import dependency analysis
        """
        logger.info("Analyzing import dependencies and cycles...")
        
        # Extract all imports
        imports = self._extract_imports(module_ast)
        
        # Categorize imports
        external_imports = []
        internal_imports = []
        
        for import_info in imports:
            module_name = import_info['module']
            if self._is_internal_module(module_name):
                internal_imports.append(import_info)
            else:
                external_imports.append(import_info)
        
        # Detect import cycles (simplified - would need full project analysis for real cycles)
        import_cycles = self._detect_import_cycles(internal_imports)
        
        return {
            'external_imports': [
                {
                    'module': imp['module'],
                    'type': imp['type'],
                    'line': imp['line'],
                    'is_standard_library': self._is_standard_library(imp['module'])
                }
                for imp in external_imports
            ],
            'internal_imports': [
                {
                    'module': imp['module'],
                    'type': imp['type'],
                    'line': imp['line']
                }
                for imp in internal_imports
            ],
            'cycles': import_cycles
        }

    def extract_external_dependency_contracts(self, module_ast: ast.Module) -> Dict[str, any]:
        """
        Extract contracts of external dependencies as requested by Expert 2.
        
        Args:
            module_ast: Parsed AST of the target module
            
        Returns:
            Dictionary with external dependency contracts
        """
        logger.info("Extracting external dependency contracts...")
        
        # Get dependency interfaces
        interfaces = self.extract_dependency_interfaces(module_ast)
        
        contracts = {}
        
        for interface in interfaces:
            module_name = interface.module_name
            
            # Skip standard library for detailed contract analysis
            if self._is_standard_library(module_name):
                continue
            
            contract_info = self._extract_dependency_contract(interface, module_name)
            if contract_info:
                contracts[module_name] = contract_info
        
        return {
            'dependency_contracts': contracts,
            'contract_summary': {
                'total_external_dependencies': len([i for i in interfaces if not self._is_standard_library(i.module_name)]),
                'contracts_extracted': len(contracts),
                'high_risk_dependencies': [
                    name for name, contract in contracts.items()
                    if contract.get('risk_level') == 'high'
                ]
            },
            'recommendations': self._generate_contract_recommendations(contracts)
        }

    def _is_internal_module(self, module_name: str) -> bool:
        """Check if a module is internal to the project."""
        # Simple heuristic - internal modules don't have dots or are relative
        return '.' not in module_name or module_name.startswith('.')

    def _detect_import_cycles(self, internal_imports: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Detect import cycles (simplified version)."""
        cycles = []
        
        # This is a simplified version - real cycle detection would require
        # analyzing the entire project's import graph
        
        # For now, just detect potential cycles based on naming patterns
        module_names = [imp['module'] for imp in internal_imports]
        
        # Look for modules that might import each other
        for i, module1 in enumerate(module_names):
            for j, module2 in enumerate(module_names[i+1:], i+1):
                if self._might_have_cycle(module1, module2):
                    cycles.append({
                        'modules': [module1, module2],
                        'type': 'potential_cycle',
                        'description': f'Potential cycle between {module1} and {module2}'
                    })
        
        return cycles

    def _might_have_cycle(self, module1: str, module2: str) -> bool:
        """Check if two modules might have a circular dependency."""
        # Simple heuristic based on naming patterns
        if module1 in module2 or module2 in module1:
            return True
        
        # Check for common patterns that might indicate cycles
        common_patterns = [
            ('handler', 'dispatcher'),
            ('client', 'server'),
            ('parser', 'builder')
        ]
        
        for pattern1, pattern2 in common_patterns:
            if (pattern1 in module1.lower() and pattern2 in module2.lower()) or \
               (pattern2 in module1.lower() and pattern1 in module2.lower()):
                return True
        
        return False

    def _extract_dependency_contract(self, interface: DependencyInterface, module_name: str) -> Optional[Dict[str, any]]:
        """Extract contract information for a dependency."""
        contract = {
            'module_name': module_name,
            'used_methods': interface.used_methods,
            'used_attributes': interface.used_attributes,
            'import_style': interface.import_style,
            'risk_level': 'medium',  # Default
            'contract_details': {}
        }
        
        # Analyze specific known dependencies
        if 'AttackRegistry' in module_name:
            contract['contract_details'] = {
                'description': 'Registry containing attack handlers',
                'key_methods': ['get_attack', 'list_attacks', 'validate_attack'],
                'expected_structure': 'Dict[str, AttackHandler]',
                'critical_invariants': [
                    'attacks dictionary must be populated',
                    'priority.value must be accessible',
                    'handler must be callable'
                ]
            }
            contract['risk_level'] = 'high'
        
        elif 'ParameterNormalizer' in module_name:
            contract['contract_details'] = {
                'description': 'Parameter normalization and validation',
                'key_methods': ['normalize', 'validate'],
                'expected_structure': 'Class with normalize method',
                'critical_invariants': [
                    'normalize() must return Dict[str, Any]',
                    'Input validation must occur',
                    'Output format must be consistent'
                ]
            }
            contract['risk_level'] = 'high'
        
        elif 'SNIExtractor' in module_name:
            contract['contract_details'] = {
                'description': 'SNI extraction from TLS packets',
                'key_methods': ['extract_sni', 'is_tls_packet'],
                'expected_structure': 'Class with extraction methods',
                'critical_invariants': [
                    'extract_sni() returns Optional[str]',
                    'is_tls_packet() returns bool',
                    'Must handle malformed packets gracefully'
                ]
            }
            contract['risk_level'] = 'medium'
        
        elif 'operation_logger' in module_name:
            contract['contract_details'] = {
                'description': 'Operation logging functionality',
                'key_methods': ['log_operation', 'debug', 'info', 'error'],
                'expected_structure': 'Logger-like interface',
                'critical_invariants': [
                    'Must accept message and level parameters',
                    'Must handle formatting gracefully',
                    'Should not raise exceptions on logging errors'
                ]
            }
            contract['risk_level'] = 'low'
        
        elif 'save_pcap_metadata' in module_name:
            contract['contract_details'] = {
                'description': 'PCAP metadata saving functionality',
                'key_methods': ['save_pcap_metadata'],
                'expected_structure': 'Function accepting metadata dict',
                'critical_invariants': [
                    'Must accept dict with required fields',
                    'Should handle file I/O errors gracefully',
                    'Metadata format must be preserved'
                ]
            }
            contract['risk_level'] = 'low'
        
        else:
            # Generic contract for unknown dependencies
            contract['contract_details'] = {
                'description': f'External dependency: {module_name}',
                'key_methods': interface.used_methods,
                'expected_structure': 'Unknown - requires investigation',
                'critical_invariants': [
                    'API compatibility must be maintained',
                    'Error handling must be preserved'
                ]
            }
        
        return contract

    def _generate_contract_recommendations(self, contracts: Dict[str, any]) -> List[str]:
        """Generate recommendations for dependency contracts."""
        recommendations = []
        
        high_risk_count = len([c for c in contracts.values() if c.get('risk_level') == 'high'])
        
        if high_risk_count > 0:
            recommendations.append(f"Document and test {high_risk_count} high-risk dependency contracts")
        
        # Specific recommendations for known dependencies
        for module_name, contract in contracts.items():
            if 'AttackRegistry' in module_name:
                recommendations.append("Create mock AttackRegistry for testing with proper attack structure")
            elif 'ParameterNormalizer' in module_name:
                recommendations.append("Verify ParameterNormalizer.normalize() output format in tests")
            elif 'SNIExtractor' in module_name:
                recommendations.append("Test SNIExtractor with various TLS packet formats")
        
        recommendations.extend([
            "Create integration tests that verify dependency contracts",
            "Document expected behavior of external dependencies",
            "Consider dependency injection for easier testing"
        ])
        
        return recommendations
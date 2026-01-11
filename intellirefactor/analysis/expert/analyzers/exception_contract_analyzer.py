"""
Exception Contract Analyzer for expert refactoring analysis.

Analyzes exception handling patterns to understand what exceptions
can be raised by each method and under what conditions.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ExceptionContractAnalyzer:
    """Analyzes exception contracts and handling patterns."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)

    def analyze_exception_contracts(self, module_ast: ast.Module) -> Dict[str, any]:
        """
        Analyze exception contracts for all methods.
        
        Args:
            module_ast: Parsed AST of the target module
            
        Returns:
            Dictionary with exception contract analysis
        """
        logger.info("Analyzing exception contracts...")
        
        # Extract all exception types defined in the module
        defined_exceptions = self._extract_defined_exceptions(module_ast)
        
        # Analyze each method for exceptions it can raise
        method_exceptions = {}
        exception_handlers = {}
        
        for node in ast.walk(module_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_name = node.name
                
                # Analyze what exceptions this method can raise
                raised_exceptions = self._analyze_method_exceptions(node)
                method_exceptions[method_name] = raised_exceptions
                
                # Analyze exception handling in this method
                handlers = self._analyze_exception_handlers(node)
                if handlers:
                    exception_handlers[method_name] = handlers
        
        # Build exception propagation map
        propagation_map = self._build_exception_propagation_map(method_exceptions, module_ast)
        
        # Generate exception contract summary
        contract_summary = self._generate_contract_summary(method_exceptions, exception_handlers)
        
        return {
            "defined_exceptions": defined_exceptions,
            "method_exceptions": method_exceptions,
            "exception_handlers": exception_handlers,
            "propagation_map": propagation_map,
            "contract_summary": contract_summary,
            "refactoring_risks": self._assess_exception_refactoring_risks(method_exceptions, exception_handlers)
        }

    def _extract_defined_exceptions(self, module_ast: ast.Module) -> List[Dict[str, any]]:
        """Extract exception classes defined in the module."""
        exceptions = []
        
        for node in ast.walk(module_ast):
            if isinstance(node, ast.ClassDef):
                # Check if it's an exception class
                is_exception = False
                base_classes = []
                
                for base in node.bases:
                    if isinstance(base, ast.Name):
                        base_name = base.id
                        base_classes.append(base_name)
                        if 'Error' in base_name or 'Exception' in base_name:
                            is_exception = True
                
                if is_exception:
                    exceptions.append({
                        "name": node.name,
                        "line": getattr(node, 'lineno', 0),
                        "base_classes": base_classes,
                        "docstring": ast.get_docstring(node)
                    })
        
        return exceptions

    def _analyze_method_exceptions(self, method_node: ast.FunctionDef) -> Dict[str, any]:
        """Analyze what exceptions a method can raise."""
        exceptions_info = {
            "explicit_raises": [],
            "implicit_raises": [],
            "conditions": [],
            "fallback_patterns": []
        }
        
        # Find explicit raise statements
        for node in ast.walk(method_node):
            if isinstance(node, ast.Raise):
                exception_info = self._analyze_raise_statement(node)
                if exception_info:
                    exceptions_info["explicit_raises"].append(exception_info)
        
        # Find implicit exception sources (calls that can raise)
        for node in ast.walk(method_node):
            if isinstance(node, ast.Call):
                implicit_exceptions = self._analyze_call_exceptions(node)
                exceptions_info["implicit_raises"].extend(implicit_exceptions)
        
        # Analyze conditions that lead to exceptions
        conditions = self._analyze_exception_conditions(method_node)
        exceptions_info["conditions"] = conditions
        
        # Look for fallback patterns
        fallbacks = self._analyze_fallback_patterns(method_node)
        exceptions_info["fallback_patterns"] = fallbacks
        
        return exceptions_info

    def _analyze_raise_statement(self, raise_node: ast.Raise) -> Optional[Dict[str, any]]:
        """Analyze a single raise statement."""
        if not raise_node.exc:
            return {"type": "re-raise", "line": getattr(raise_node, 'lineno', 0)}
        
        exception_info = {
            "line": getattr(raise_node, 'lineno', 0),
            "type": None,
            "message": None,
            "condition": None
        }
        
        # Extract exception type
        if isinstance(raise_node.exc, ast.Call):
            if isinstance(raise_node.exc.func, ast.Name):
                exception_info["type"] = raise_node.exc.func.id
                
                # Extract message if available
                if raise_node.exc.args:
                    first_arg = raise_node.exc.args[0]
                    if isinstance(first_arg, ast.Constant):
                        exception_info["message"] = first_arg.value
                    elif isinstance(first_arg, ast.JoinedStr):
                        exception_info["message"] = "f-string message"
        
        elif isinstance(raise_node.exc, ast.Name):
            exception_info["type"] = raise_node.exc.id
        
        return exception_info

    def _analyze_call_exceptions(self, call_node: ast.Call) -> List[Dict[str, any]]:
        """Analyze what exceptions a function call might raise."""
        implicit_exceptions = []
        
        # Common patterns that can raise exceptions
        if isinstance(call_node.func, ast.Attribute):
            attr_name = call_node.func.attr
            
            # Dictionary access patterns
            if attr_name in ['get', '__getitem__']:
                # dict.get() is safe, dict['key'] can raise KeyError
                if attr_name == '__getitem__':
                    implicit_exceptions.append({
                        "type": "KeyError",
                        "line": getattr(call_node, 'lineno', 0),
                        "source": "dictionary access",
                        "implicit": True
                    })
            
            # List/array access
            elif attr_name in ['pop', 'remove']:
                implicit_exceptions.append({
                    "type": "IndexError/ValueError",
                    "line": getattr(call_node, 'lineno', 0),
                    "source": f"list.{attr_name}()",
                    "implicit": True
                })
        
        elif isinstance(call_node.func, ast.Name):
            func_name = call_node.func.id
            
            # Built-in functions that can raise
            if func_name in ['int', 'float']:
                implicit_exceptions.append({
                    "type": "ValueError",
                    "line": getattr(call_node, 'lineno', 0),
                    "source": f"{func_name}() conversion",
                    "implicit": True
                })
        
        return implicit_exceptions

    def _analyze_exception_conditions(self, method_node: ast.FunctionDef) -> List[Dict[str, any]]:
        """Analyze conditions that lead to exceptions."""
        conditions = []
        
        # Look for if statements that contain raise
        for node in ast.walk(method_node):
            if isinstance(node, ast.If):
                # Check if this if block contains a raise
                has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(node))
                if has_raise:
                    condition_info = {
                        "line": getattr(node, 'lineno', 0),
                        "condition": self._extract_condition_text(node.test),
                        "raises_in_block": True
                    }
                    conditions.append(condition_info)
        
        return conditions

    def _extract_condition_text(self, test_node: ast.AST) -> str:
        """Extract readable text from a condition node."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(test_node)
            else:
                # Fallback for older Python versions
                if isinstance(test_node, ast.Compare):
                    return "comparison condition"
                elif isinstance(test_node, ast.Name):
                    return f"variable: {test_node.id}"
                else:
                    return "complex condition"
        except Exception:
            return "unparseable condition"

    def _analyze_fallback_patterns(self, method_node: ast.FunctionDef) -> List[Dict[str, any]]:
        """Analyze fallback patterns in exception handling."""
        fallbacks = []
        
        # Look for try-except blocks with fallback logic
        for node in ast.walk(method_node):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    # Analyze what happens in the except block
                    fallback_info = {
                        "line": getattr(handler, 'lineno', 0),
                        "exception_type": None,
                        "fallback_action": None
                    }
                    
                    # Extract exception type
                    if handler.type:
                        if isinstance(handler.type, ast.Name):
                            fallback_info["exception_type"] = handler.type.id
                        elif isinstance(handler.type, ast.Attribute):
                            fallback_info["exception_type"] = handler.type.attr
                    
                    # Analyze fallback action
                    if handler.body:
                        first_stmt = handler.body[0]
                        if isinstance(first_stmt, ast.Return):
                            fallback_info["fallback_action"] = "return_fallback"
                        elif isinstance(first_stmt, ast.Raise):
                            fallback_info["fallback_action"] = "re_raise"
                        elif isinstance(first_stmt, ast.Pass):
                            fallback_info["fallback_action"] = "ignore"
                        else:
                            fallback_info["fallback_action"] = "custom_handling"
                    
                    fallbacks.append(fallback_info)
        
        return fallbacks

    def _analyze_exception_handlers(self, method_node: ast.FunctionDef) -> List[Dict[str, any]]:
        """Analyze exception handlers in a method."""
        handlers = []
        
        for node in ast.walk(method_node):
            if isinstance(node, ast.ExceptHandler):
                handler_info = {
                    "line": getattr(node, 'lineno', 0),
                    "exception_type": None,
                    "variable_name": node.name,
                    "handling_strategy": None
                }
                
                # Extract exception type
                if node.type:
                    if isinstance(node.type, ast.Name):
                        handler_info["exception_type"] = node.type.id
                    elif isinstance(node.type, ast.Attribute):
                        handler_info["exception_type"] = node.type.attr
                
                # Analyze handling strategy
                if node.body:
                    if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                        handler_info["handling_strategy"] = "ignore"
                    elif any(isinstance(stmt, ast.Raise) for stmt in node.body):
                        handler_info["handling_strategy"] = "re_raise"
                    elif any(isinstance(stmt, ast.Return) for stmt in node.body):
                        handler_info["handling_strategy"] = "return_fallback"
                    else:
                        handler_info["handling_strategy"] = "custom_handling"
                
                handlers.append(handler_info)
        
        return handlers

    def _build_exception_propagation_map(self, method_exceptions: Dict, module_ast: ast.Module) -> Dict[str, List[str]]:
        """Build a map of how exceptions propagate through method calls."""
        propagation_map = {}
        
        # For each method, find what other methods it calls
        for node in ast.walk(module_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_name = node.name
                called_methods = []
                
                # Find method calls within this method
                for call_node in ast.walk(node):
                    if isinstance(call_node, ast.Call):
                        if isinstance(call_node.func, ast.Name):
                            called_methods.append(call_node.func.id)
                        elif isinstance(call_node.func, ast.Attribute):
                            if isinstance(call_node.func.value, ast.Name) and call_node.func.value.id == 'self':
                                called_methods.append(call_node.func.attr)
                
                propagation_map[method_name] = called_methods
        
        return propagation_map

    def _generate_contract_summary(self, method_exceptions: Dict, exception_handlers: Dict) -> List[Dict[str, any]]:
        """Generate a summary of exception contracts for public methods."""
        contracts = []
        
        for method_name, exceptions_info in method_exceptions.items():
            if not method_name.startswith('_'):  # Public methods only
                contract = {
                    "method": method_name,
                    "can_raise": [],
                    "conditions": [],
                    "has_fallback": False,
                    "safety_level": "unknown"
                }
                
                # Collect all exception types this method can raise
                exception_types = set()
                
                for exc in exceptions_info["explicit_raises"]:
                    if exc.get("type"):
                        exception_types.add(exc["type"])
                
                for exc in exceptions_info["implicit_raises"]:
                    if exc.get("type"):
                        exception_types.add(exc["type"])
                
                contract["can_raise"] = list(exception_types)
                
                # Extract conditions
                contract["conditions"] = [
                    cond["condition"] for cond in exceptions_info["conditions"]
                ]
                
                # Check for fallback patterns
                contract["has_fallback"] = len(exceptions_info["fallback_patterns"]) > 0
                
                # Assess safety level
                if not exception_types:
                    contract["safety_level"] = "safe"
                elif contract["has_fallback"]:
                    contract["safety_level"] = "safe_with_fallback"
                elif len(exception_types) <= 2:
                    contract["safety_level"] = "moderate"
                else:
                    contract["safety_level"] = "high_risk"
                
                contracts.append(contract)
        
        return contracts

    def export_detailed_exception_contracts(self, analysis: Dict[str, any]) -> Dict[str, any]:
        """
        Export detailed exception contract data as requested by experts.
        
        Returns:
            Dictionary with exception contract details
        """
        contracts = analysis.get('contract_summary', [])
        
        # Organize contracts by method
        contracts_by_method = {}
        total_exceptions = 0
        
        for contract in contracts:
            method_name = contract['method']
            contracts_by_method[method_name] = {
                'exceptions_raised': contract['can_raise'],
                'conditions': contract['conditions'],
                'safety_level': contract['safety_level'],
                'has_fallback': contract['has_fallback']
            }
            total_exceptions += len(contract['can_raise'])
        
        # Create exception type summary
        exception_types = set()
        for contract in contracts:
            exception_types.update(contract['can_raise'])
        
        return {
            'exception_contracts': contracts_by_method,
            'summary': {
                'total_methods_analyzed': len(contracts),
                'total_exceptions_identified': total_exceptions,
                'unique_exception_types': list(exception_types),
                'methods_with_exceptions': len([c for c in contracts if c['can_raise']])
            },
            'recommendations': self._generate_exception_recommendations(contracts)
        }

    def _generate_exception_recommendations(self, contracts: List[Dict[str, any]]) -> List[str]:
        """Generate recommendations for exception handling."""
        recommendations = []
        
        methods_without_exceptions = [c for c in contracts if not c['can_raise']]
        if methods_without_exceptions:
            recommendations.append(f"Document exception behavior for {len(methods_without_exceptions)} methods")
        
        high_risk_methods = [c for c in contracts if c['safety_level'] == 'high_risk']
        if high_risk_methods:
            recommendations.append(f"Review exception handling in {len(high_risk_methods)} high-risk methods")
        
        return recommendations

    def _assess_exception_refactoring_risks(self, method_exceptions: Dict, exception_handlers: Dict) -> List[str]:
        """Assess risks related to exception handling during refactoring."""
        risks = []
        
        # Check for methods with many exception types
        for method_name, exceptions_info in method_exceptions.items():
            exception_count = len(exceptions_info["explicit_raises"]) + len(exceptions_info["implicit_raises"])
            if exception_count > 5:
                risks.append(f"Method {method_name} has {exception_count} potential exception sources - high complexity")
        
        # Check for bare except handlers
        for method_name, handlers in exception_handlers.items():
            for handler in handlers:
                if handler["exception_type"] is None:
                    risks.append(f"Method {method_name} has bare except handler - may hide errors during refactoring")
        
        # Check for methods without exception handling
        methods_without_handlers = set(method_exceptions.keys()) - set(exception_handlers.keys())
        if len(methods_without_handlers) > 5:
            risks.append(f"{len(methods_without_handlers)} methods have no exception handling - refactoring may introduce unhandled exceptions")
        
        return risks
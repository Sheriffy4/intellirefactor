"""
Behavioral Contract Analyzer for expert refactoring analysis.

Extracts and analyzes behavioral contracts from docstrings, comments,
and code patterns to ensure refactoring preserves semantics.
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional

from ..models import BehavioralContract

logger = logging.getLogger(__name__)


class BehavioralContractAnalyzer:
    """Analyzes behavioral contracts and invariants in code."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)
        
        # Patterns for extracting contracts from docstrings
        self.contract_patterns = {
            'preconditions': [
                r'(?:precondition|requires?|assumes?|expects?)[:\s]+(.*?)(?:\n|$)',
                r'(?:args?|parameters?)[:\s]+.*?(?:must|should|requires?)\s+(.*?)(?:\n|$)',
                r'@pre[:\s]+(.*?)(?:\n|$)',
            ],
            'postconditions': [
                r'(?:postcondition|ensures?|returns?|guarantees?)[:\s]+(.*?)(?:\n|$)',
                r'(?:returns?)[:\s]+.*?(?:will|shall|guarantees?)\s+(.*?)(?:\n|$)',
                r'@post[:\s]+(.*?)(?:\n|$)',
            ],
            'invariants': [
                r'(?:invariant|maintains?|preserves?)[:\s]+(.*?)(?:\n|$)',
                r'@invariant[:\s]+(.*?)(?:\n|$)',
            ],
            'side_effects': [
                r'(?:side.?effects?|modifies?|changes?|mutates?)[:\s]+(.*?)(?:\n|$)',
                r'(?:warning|note)[:\s]+.*?(?:modifies?|changes?|mutates?)\s+(.*?)(?:\n|$)',
            ],
            'performance': [
                r'(?:complexity|performance|time|space)[:\s]+.*?O\([^)]+\)',
                r'(?:runs? in|takes?|complexity)[:\s]+(.*?)(?:\n|$)',
            ],
            'errors': [
                r'(?:raises?|throws?|errors?)[:\s]+(.*?)(?:\n|$)',
                r'(?:may raise|can throw)[:\s]+(.*?)(?:\n|$)',
            ]
        }

    def extract_contracts_from_docstrings(self, module_ast: ast.Module) -> List[BehavioralContract]:
        """
        Extract behavioral contracts from method docstrings.
        
        Args:
            module_ast: Parsed AST of the target module
            
        Returns:
            List of BehavioralContract objects
        """
        logger.info("Extracting behavioral contracts from docstrings...")
        
        contracts = []
        
        for node in ast.walk(module_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                contract = self._extract_method_contract(node)
                if contract:
                    contracts.append(contract)
        
        logger.info(f"Extracted {len(contracts)} behavioral contracts")
        return contracts

    def _extract_method_contract(self, func_node: ast.FunctionDef) -> Optional[BehavioralContract]:
        """Extract contract from a single method."""
        # Get method name and class context
        method_name = func_node.name
        class_name = self._get_containing_class(func_node)
        
        # Extract docstring
        docstring = self._get_docstring(func_node)
        if not docstring:
            # Try to infer contracts from code patterns
            return self._infer_implicit_contracts(func_node, method_name, class_name)
        
        # Parse docstring for contracts
        contract = BehavioralContract(
            method_name=method_name,
            class_name=class_name
        )
        
        # Extract different types of contracts
        contract.preconditions = self._extract_contract_type(docstring, 'preconditions')
        contract.postconditions = self._extract_contract_type(docstring, 'postconditions')
        contract.invariants = self._extract_contract_type(docstring, 'invariants')
        contract.side_effects = self._extract_contract_type(docstring, 'side_effects')
        contract.performance_constraints = self._extract_contract_type(docstring, 'performance')
        contract.error_conditions = self._extract_contract_type(docstring, 'errors')
        
        # Only return if we found some contracts
        if any([
            contract.preconditions,
            contract.postconditions,
            contract.invariants,
            contract.side_effects,
            contract.performance_constraints,
            contract.error_conditions
        ]):
            return contract
        
        # Fallback to implicit contract inference
        return self._infer_implicit_contracts(func_node, method_name, class_name)

    def _get_containing_class(self, func_node: ast.FunctionDef) -> Optional[str]:
        """Get the name of the class containing this method."""
        # This is a simplified approach - in a real implementation,
        # we'd need to walk up the AST tree to find the containing class
        # For now, we'll use a heuristic based on the first parameter
        if func_node.args.args and func_node.args.args[0].arg == 'self':
            # This is likely a method - we'd need more context to get the class name
            # For now, return a placeholder
            return "UnknownClass"
        return None

    def _get_docstring(self, func_node: ast.FunctionDef) -> Optional[str]:
        """Extract docstring from a function node."""
        if (func_node.body 
            and isinstance(func_node.body[0], ast.Expr)
            and isinstance(func_node.body[0].value, ast.Constant)
            and isinstance(func_node.body[0].value.value, str)):
            return func_node.body[0].value.value
        return None

    def _extract_contract_type(self, docstring: str, contract_type: str) -> List[str]:
        """Extract a specific type of contract from docstring."""
        contracts = []
        patterns = self.contract_patterns.get(contract_type, [])
        
        for pattern in patterns:
            matches = re.finditer(pattern, docstring, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                contract_text = match.group(1).strip()
                if contract_text and contract_text not in contracts:
                    contracts.append(contract_text)
        
        return contracts

    def _infer_implicit_contracts(self, func_node: ast.FunctionDef, method_name: str, class_name: Optional[str]) -> Optional[BehavioralContract]:
        """Infer implicit contracts from code patterns."""
        contract = BehavioralContract(
            method_name=method_name,
            class_name=class_name
        )
        
        # Analyze function body for implicit contracts
        for node in ast.walk(func_node):
            # Look for assertions (explicit preconditions/postconditions)
            if isinstance(node, ast.Assert):
                assertion_text = self._get_assertion_text(node)
                if assertion_text:
                    # Heuristic: assertions at the beginning are preconditions
                    # assertions at the end are postconditions
                    if self._is_early_in_function(node, func_node):
                        contract.preconditions.append(f"assert {assertion_text}")
                    else:
                        contract.postconditions.append(f"assert {assertion_text}")
            
            # Look for raise statements (error conditions)
            elif isinstance(node, ast.Raise):
                error_text = self._get_raise_text(node)
                if error_text:
                    contract.error_conditions.append(error_text)
            
            # Look for attribute modifications (side effects)
            elif isinstance(node, ast.Assign):
                side_effect = self._analyze_assignment_side_effects(node)
                if side_effect:
                    contract.side_effects.append(side_effect)
        
        # Infer contracts from method name patterns
        name_contracts = self._infer_from_method_name(method_name)
        contract.preconditions.extend(name_contracts.get('preconditions', []))
        contract.postconditions.extend(name_contracts.get('postconditions', []))
        contract.side_effects.extend(name_contracts.get('side_effects', []))
        
        # Only return if we inferred some contracts
        if any([
            contract.preconditions,
            contract.postconditions,
            contract.invariants,
            contract.side_effects,
            contract.error_conditions
        ]):
            return contract
        
        return None

    def _get_assertion_text(self, assert_node: ast.Assert) -> Optional[str]:
        """Get text representation of an assertion."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(assert_node.test)
            else:
                # Fallback for older Python versions
                return "assertion condition"
        except Exception:
            return None

    def _is_early_in_function(self, node: ast.AST, func_node: ast.FunctionDef) -> bool:
        """Check if a node appears early in the function (likely a precondition)."""
        # Simple heuristic: check if the node appears in the first 25% of the function
        try:
            node_line = getattr(node, 'lineno', 0)
            func_start = getattr(func_node, 'lineno', 0)
            func_end = getattr(func_node, 'end_lineno', func_start + 10)
            
            func_length = func_end - func_start
            early_threshold = func_start + (func_length * 0.25)
            
            return node_line <= early_threshold
        except Exception:
            return True  # Default to treating as precondition

    def _get_raise_text(self, raise_node: ast.Raise) -> Optional[str]:
        """Get text representation of a raise statement."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(raise_node)
            else:
                # Try to get exception type
                if raise_node.exc and isinstance(raise_node.exc, ast.Call):
                    if isinstance(raise_node.exc.func, ast.Name):
                        return f"raises {raise_node.exc.func.id}"
                return "raises exception"
        except Exception:
            return None

    def _analyze_assignment_side_effects(self, assign_node: ast.Assign) -> Optional[str]:
        """Analyze assignment for side effects."""
        try:
            # Look for self.attribute assignments (instance variable modifications)
            for target in assign_node.targets:
                if isinstance(target, ast.Attribute):
                    if isinstance(target.value, ast.Name) and target.value.id == 'self':
                        return f"modifies self.{target.attr}"
                elif isinstance(target, ast.Name):
                    # Global variable modification
                    return f"modifies {target.id}"
        except Exception:
            pass
        return None

    def _infer_from_method_name(self, method_name: str) -> Dict[str, List[str]]:
        """Infer contracts from method naming patterns."""
        contracts = {
            'preconditions': [],
            'postconditions': [],
            'side_effects': []
        }
        
        name_lower = method_name.lower()
        
        # Getter methods
        if name_lower.startswith(('get_', 'fetch_', 'retrieve_', 'find_')):
            contracts['postconditions'].append("returns requested data or None")
            contracts['side_effects'].append("no side effects (read-only)")
        
        # Setter methods
        elif name_lower.startswith(('set_', 'update_', 'modify_')):
            contracts['preconditions'].append("value parameter must be valid")
            contracts['side_effects'].append("modifies object state")
        
        # Validation methods
        elif name_lower.startswith(('validate_', 'check_', 'verify_')):
            contracts['postconditions'].append("returns True if valid, False otherwise")
            contracts['side_effects'].append("no side effects (validation only)")
        
        # Creation methods
        elif name_lower.startswith(('create_', 'make_', 'build_', 'generate_')):
            contracts['postconditions'].append("returns newly created object")
            contracts['side_effects'].append("may modify system state")
        
        # Deletion methods
        elif name_lower.startswith(('delete_', 'remove_', 'destroy_')):
            contracts['preconditions'].append("target must exist")
            contracts['postconditions'].append("target no longer exists")
            contracts['side_effects'].append("permanently modifies system state")
        
        # Processing methods
        elif name_lower.startswith(('process_', 'handle_', 'execute_')):
            contracts['preconditions'].append("input must be valid")
            contracts['postconditions'].append("processing completed successfully")
            contracts['side_effects'].append("may have various side effects")
        
        return contracts

    def infer_implicit_contracts(self, method_ast: ast.FunctionDef) -> List[BehavioralContract]:
        """
        Infer implicit contracts from code patterns.
        
        Args:
            method_ast: AST node of the method to analyze
            
        Returns:
            List of inferred contracts
        """
        # This method provides a public interface for implicit contract inference
        contract = self._infer_implicit_contracts(method_ast, method_ast.name, None)
        return [contract] if contract else []

    def generate_formal_specification(self, contracts: List[BehavioralContract]) -> Dict[str, any]:
        """
        Generate formal specification from behavioral contracts.
        
        Args:
            contracts: List of behavioral contracts
            
        Returns:
            Formal specification dictionary
        """
        logger.info("Generating formal specification...")
        
        specification = {
            'methods': {},
            'class_invariants': [],
            'global_constraints': []
        }
        
        for contract in contracts:
            method_key = f"{contract.class_name}.{contract.method_name}" if contract.class_name else contract.method_name
            
            specification['methods'][method_key] = {
                'preconditions': contract.preconditions,
                'postconditions': contract.postconditions,
                'side_effects': contract.side_effects,
                'error_conditions': contract.error_conditions,
                'performance_constraints': contract.performance_constraints
            }
            
            # Class-level invariants
            if contract.invariants:
                specification['class_invariants'].extend(contract.invariants)
        
        logger.info(f"Generated formal specification for {len(contracts)} methods")
        return specification
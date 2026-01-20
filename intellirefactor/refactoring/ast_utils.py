"""AST analysis utilities for refactoring operations.

This module provides helper functions for AST traversal, analysis, and code extraction.
Extracted from AutoRefactor to reduce god class complexity.
"""

from __future__ import annotations

import ast
from typing import Set, Tuple, Optional


def find_largest_top_level_class(tree: ast.Module) -> Tuple[Optional[ast.ClassDef], int]:
    """Find the class with the most methods in a module.
    
    Args:
        tree: AST module to analyze
        
    Returns:
        Tuple of (class node, method count) or (None, 0) if no classes found
    """
    main_class: Optional[ast.ClassDef] = None
    max_methods = 0
    
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            methods = [
                n
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            if len(methods) > max_methods:
                max_methods = len(methods)
                main_class = node
                
    return main_class, max_methods


def collect_module_level_names(tree: ast.Module) -> Set[str]:
    """Collect all module-level names (variables, functions, classes).
    
    Args:
        tree: AST module to analyze
        
    Returns:
        Set of names defined at module level
    """
    names: Set[str] = set()
    
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    names.add(t.id)
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.ClassDef):
            names.add(node.name)
            
    return names

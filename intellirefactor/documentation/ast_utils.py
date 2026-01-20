"""
Shared AST helpers for documentation generators.

Why:
Some documentation modules referenced `intellirefactor.unified.documentation`,
but that module does not exist. These helpers provide the required functionality
in a stable, local place.
"""

from __future__ import annotations

import ast
from typing import List, Optional


def extract_docstring(node: ast.AST) -> Optional[str]:
    """
    Robust docstring extraction for ClassDef / FunctionDef / Module.
    """
    try:
        ds = ast.get_docstring(node, clean=True)
        if ds:
            ds = ds.strip()
        return ds or None
    except Exception:
        # Fallback for very custom AST nodes / edge-cases
        try:
            body = getattr(node, "body", None)
            if not body:
                return None
            first = body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(getattr(first, "value", None), ast.Constant)
                and isinstance(getattr(first.value, "value", None), str)
            ):
                return str(first.value.value).strip() or None
        except Exception:
            return None
    return None


def assess_class_complexity(node: ast.ClassDef, methods: List[str]) -> str:
    """
    Simple, deterministic class complexity assessment for documentation.

    NOTE: This is intentionally lightweight; it's used for docs rendering,
    not for core refactoring decisions.
    """
    method_count = len(methods or [])
    base_count = len(getattr(node, "bases", []) or [])

    # A few quick heuristics
    if method_count > 20 or base_count > 3:
        return "High"
    if method_count > 10 or base_count > 1:
        return "Medium"
    return "Low"

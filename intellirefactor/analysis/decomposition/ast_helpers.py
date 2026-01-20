"""
AST Utility Helpers

Shared AST manipulation functions for block extraction and analysis.
"""

from __future__ import annotations

import ast
from typing import List


_BUILTIN_TYPES = {
    "list",
    "dict",
    "set",
    "tuple",
    "str",
    "bytes",
    "int",
    "float",
    "bool",
}


def looks_like_class_ref(ref: str) -> bool:
    """
    Heuristic: accept class/type references where last segment starts with uppercase.

    Examples:
      - "QualityAnalyzer" -> True
      - "analysis.QualityAnalyzer" -> True
      - "create_analyzer" -> False
      - "typing.Optional" -> False
    """
    if not ref:
        return False
    last = ref.split(".")[-1]
    return (last and last[0].isupper()) or (last in _BUILTIN_TYPES)


def call_key_from_ast(func: ast.AST) -> str:
    """
    Generate normalized call key from AST func node.

    FIX: keep dot even for complex receivers, so callers like x[i].append(...)
    become "<subscript>.append" (dynamic_attribute), not "append" (not_found).
    """
    if isinstance(func, ast.Name):
        return func.id

    if isinstance(func, ast.Attribute):
        base = func.value
        if isinstance(base, ast.Call):
            base = base.func

        if isinstance(base, ast.Subscript):
            base_s = call_key_from_ast(base.value) or "<subscript>"
        elif isinstance(base, (ast.Constant, ast.BinOp, ast.UnaryOp, ast.Compare)):
            base_s = "<expr>"
        else:
            try:
                base_s = call_key_from_ast(base) or "<expr>"
            except Exception:
                base_s = "<expr>"

        return f"{base_s}.{func.attr}"

    try:
        return ast.unparse(func)
    except Exception:
        return ""


def attr_path_from_ast(node: ast.AST) -> str:
    """
    Extract dotted attribute path from AST node.

    Examples:
      - ast.Name("x") -> "x"
      - ast.Attribute(ast.Name("self"), "logger") -> "self.logger"
      - ast.Attribute(ast.Attribute(ast.Name("self"), "ctx"), "store") -> "self.ctx.store"
    """
    parts: List[str] = []
    cur = node
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
    else:
        return ""
    return ".".join(reversed(parts))


def calculate_cyclomatic_complexity(node: ast.AST) -> int:
    """
    Calculate cyclomatic complexity for an AST node.

    Counts decision points: if, while, for, with, except, boolean operators, etc.
    """
    complexity = 1
    for child in ast.walk(node):
        if isinstance(
            child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith)
        ):
            complexity += 1
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += max(1, len(getattr(child, "values", [])) - 1)
        elif isinstance(child, ast.IfExp):
            complexity += 1
        elif isinstance(
            child, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        ):
            complexity += 1
            for gen in child.generators:
                complexity += len(getattr(gen, "ifs", []))
    return complexity

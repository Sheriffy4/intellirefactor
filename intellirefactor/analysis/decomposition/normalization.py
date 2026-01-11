"""
Unified AST normalization for fingerprints.

Goal:
- Normalize variable names to reduce noise
- Preserve API surface:
  - keep called function names
  - keep attribute names (method/property)
  - normalize receiver object names (obj.save vs db.save should match)
"""

from __future__ import annotations

import ast


class APIPreservingASTNormalizer(ast.NodeTransformer):
    def __init__(self):
        self._var_counter = 0
        self._var_mapping: dict[str, str] = {}
        self._in_call_func = False

    def visit(self, node):
        # wipe positions for stable dump
        for attr in ("lineno", "col_offset", "end_lineno", "end_col_offset"):
            if hasattr(node, attr):
                setattr(node, attr, 0)
        return super().visit(node)

    def visit_Call(self, node: ast.Call):
        prev = self._in_call_func
        self._in_call_func = True
        node.func = self.visit(node.func)
        self._in_call_func = prev

        node.args = [self.visit(a) for a in node.args]
        node.keywords = [self.visit(k) for k in node.keywords]
        return node

    def visit_Name(self, node: ast.Name):
        if node.id in ("True", "False", "None", "self", "cls"):
            return node

        # preserve callee name in foo(...)
        if self._in_call_func and isinstance(node.ctx, ast.Load):
            return node

        if node.id not in self._var_mapping:
            self._var_mapping[node.id] = f"var_{self._var_counter}"
            self._var_counter += 1

        node.id = self._var_mapping[node.id]
        return node

    def visit_Attribute(self, node: ast.Attribute):
        # preserve attr, normalize receiver
        node.value = self.visit(node.value)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not (node.name.startswith("__") and node.name.endswith("__")):
            node.name = "func"
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if not (node.name.startswith("__") and node.name.endswith("__")):
            node.name = "func"
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        node.name = "Class"
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        # FIX: bool is subclass of int -> handle it BEFORE int/float
        if isinstance(node.value, bool) or node.value is None:
            return node
        if isinstance(node.value, str):
            node.value = "STRING"
        elif isinstance(node.value, (int, float)):
            node.value = 0
        return node


def normalize_for_hash(node: ast.AST) -> ast.AST:
    return APIPreservingASTNormalizer().visit(node)
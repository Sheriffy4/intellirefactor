"""
Unified AST normalization for fingerprints.

Goal:
- Normalize variable/argument names to reduce noise
- Preserve API surface:
  - keep called function names (foo(...))
  - keep attribute names (.save/.items/etc)
  - normalize receiver object names (obj.save vs db.save should match)
"""

from __future__ import annotations

import ast
from contextlib import contextmanager


class APIPreservingASTNormalizer(ast.NodeTransformer):
    def __init__(self):
        self._counter = 0
        self._mapping: dict[str, str] = {}
        self._preserve_name_depth = 0  # preserve only the *callee* Name in Call.func

    @contextmanager
    def _preserve_names(self):
        self._preserve_name_depth += 1
        try:
            yield
        finally:
            self._preserve_name_depth -= 1

    def _map(self, name: str, *, prefix: str = "var") -> str:
        if name not in self._mapping:
            self._mapping[name] = f"{prefix}_{self._counter}"
            self._counter += 1
        return self._mapping[name]

    def visit_Call(self, node: ast.Call):
        # Preserve only direct Name as callee: foo(...)
        if isinstance(node.func, ast.Name):
            with self._preserve_names():
                node.func = self.visit(node.func)
        else:
            # For Attribute callee (obj.save), we want receiver normalized, attribute preserved
            node.func = self.visit(node.func)

        node.args = [self.visit(a) for a in node.args]
        node.keywords = [self.visit(k) for k in node.keywords]
        return node

    def visit_Name(self, node: ast.Name):
        if node.id in ("True", "False", "None", "self", "cls"):
            return node

        # preserve only callee Name in foo(...)
        if self._preserve_name_depth and isinstance(node.ctx, ast.Load):
            return node

        node.id = self._map(node.id, prefix="var")
        return node

    def visit_arg(self, node: ast.arg):
        if node.arg in ("self", "cls"):
            node.annotation = None
            return node
        node.arg = self._map(node.arg, prefix="arg")
        node.annotation = None
        return node

    def visit_Attribute(self, node: ast.Attribute):
        # preserve .attr, normalize receiver
        node.value = self.visit(node.value)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        if not (node.name.startswith("__") and node.name.endswith("__")):
            node.name = "func"
        node.returns = None
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        if not (node.name.startswith("__") and node.name.endswith("__")):
            node.name = "func"
        node.returns = None
        return self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        node.name = "Class"
        return self.generic_visit(node)

    def visit_Constant(self, node: ast.Constant):
        # bool is subclass of int -> handle first
        if isinstance(node.value, bool) or node.value is None:
            return node
        if isinstance(node.value, str):
            node.value = "STRING"
        elif isinstance(node.value, (int, float)):
            node.value = 0
        return node


def normalize_for_hash(node: ast.AST) -> ast.AST:
    node2 = APIPreservingASTNormalizer().visit(node)
    ast.fix_missing_locations(node2)
    return node2
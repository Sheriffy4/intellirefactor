"""
AST Visitor Classes

Visitor classes for traversing and extracting information from AST nodes.
"""

from __future__ import annotations

import ast
from typing import Dict, List, Set, Union


class FunctionVisitor(ast.NodeVisitor):
    """
    AST visitor to extract functions with proper class context.

    Uses stacks to track current class context and function nesting.
    """

    def __init__(
        self,
        file_path: str,
        module_name: str,
        source: str,
        file_imports: List[str],
        module_type_hints: Dict[str, str],
        extractor,
    ):
        self.file_path = file_path
        self.module_name = module_name
        self.source = source
        self.file_imports = file_imports
        self.module_type_hints = module_type_hints or {}
        self.extractor = extractor

        self.blocks: List = []  # List[FunctionalBlock]
        self.class_stack: List[str] = []
        self.function_stack: List[str] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.class_stack.append(node.name)
        self.generic_visit(node)
        self.class_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        is_nested = len(self.function_stack) > 0

        # do not extract nested defs unless configured
        if is_nested and not self.extractor.config.extract_nested_functions:
            return

        self._process_function(node, is_nested)

        self.function_stack.append(node.name)
        if self.extractor.config.extract_nested_functions:
            self.generic_visit(node)
        self.function_stack.pop()

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        is_nested = len(self.function_stack) > 0

        if is_nested and not self.extractor.config.extract_nested_functions:
            return

        self._process_function(node, is_nested)

        self.function_stack.append(node.name)
        if self.extractor.config.extract_nested_functions:
            self.generic_visit(node)
        self.function_stack.pop()

    def _process_function(
        self,
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        is_nested: bool = False,
    ) -> None:
        # keep full nesting for classes: Outer.Inner.method
        if self.class_stack:
            class_prefix = ".".join(self.class_stack)
            qualname = f"{class_prefix}.{node.name}"
        else:
            qualname = node.name

        block = self.extractor._function_to_block(
            node=node,
            file_path=self.file_path,
            module_name=self.module_name,
            source=self.source,
            qualname=qualname,
            file_imports=self.file_imports,
            module_type_hints=self.module_type_hints,
            is_nested=is_nested,
        )
        if block:
            self.blocks.append(block)


class AssignedNameVisitor(ast.NodeVisitor):
    """
    Collect names assigned in the current function scope.

    Does not descend into nested defs/classes/lambdas/comprehensions.
    """

    __slots__ = ("root", "names")

    def __init__(self, root: Union[ast.FunctionDef, ast.AsyncFunctionDef]):
        self.root = root
        self.names: Set[str] = set()

    def visit_FunctionDef(self, node):
        if node is self.root:
            self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node):
        if node is self.root:
            self.generic_visit(node)

    def visit_ClassDef(self, node):
        return

    def visit_Lambda(self, node):
        return

    def visit_ListComp(self, node):
        return

    def visit_SetComp(self, node):
        return

    def visit_DictComp(self, node):
        return

    def visit_GeneratorExp(self, node):
        return

    def visit_Assign(self, node):
        for t in node.targets:
            self._collect_target(t)

    def visit_AnnAssign(self, node):
        self._collect_target(node.target)

    def visit_AugAssign(self, node):
        self._collect_target(node.target)

    def visit_For(self, node):
        self._collect_target(node.target)
        for s in node.body + node.orelse:
            self.visit(s)

    def visit_AsyncFor(self, node):
        self._collect_target(node.target)
        for s in node.body + node.orelse:
            self.visit(s)

    def visit_With(self, node):
        for item in node.items:
            if item.optional_vars:
                self._collect_target(item.optional_vars)
        for s in node.body:
            self.visit(s)

    def visit_AsyncWith(self, node):
        for item in node.items:
            if item.optional_vars:
                self._collect_target(item.optional_vars)
        for s in node.body:
            self.visit(s)

    def visit_ExceptHandler(self, node):
        if node.name:
            self.names.add(node.name)
        for s in node.body:
            self.visit(s)

    def visit_NamedExpr(self, node):
        self._collect_target(node.target)

    def _collect_target(self, t):
        if isinstance(t, ast.Name):
            self.names.add(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            for e in t.elts:
                self._collect_target(e)

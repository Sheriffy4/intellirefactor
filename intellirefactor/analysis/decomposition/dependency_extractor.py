"""
Dependency Extractor

Extracts dependencies (calls, imports, globals, literals) from AST nodes.
"""

from __future__ import annotations

import ast
import logging
from typing import List, Optional, Tuple, Union

from .ast_helpers import call_key_from_ast

logger = logging.getLogger(__name__)


class DependencyExtractor:
    """
    Extracts dependencies from function AST nodes.

    Identifies:
    - Function calls
    - Import statements
    - Global variable references
    - String literals
    """

    def __init__(self, config=None):
        """
        Initialize dependency extractor.

        Args:
            config: DecompositionConfig with settings like exclude_docstrings_from_literals
        """
        self.config = config
        self.logger = logger

    def extract_from_node(
        self,
        func_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        source: str,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Extract dependencies from AST node (preferred method).

        Returns:
            (calls, imports_used, globals_used, literals)
        """
        try:
            visitor = _DependencyVisitor(self, func_node)
            visitor.visit(func_node)
            return (
                visitor.calls,
                visitor.imports_used,
                visitor.globals_used,
                visitor.literals,
            )
        except Exception as e:
            self.logger.debug(f"Failed to extract dependencies from AST: {e}")
            return self.extract_from_source(source, func_node)

    def extract_from_source(
        self,
        source: str,
        func_node: Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef]] = None,
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Fallback method: extract dependencies by parsing function source.

        Returns:
            (calls, imports_used, globals_used, literals)
        """
        calls: List[str] = []
        imports_used: List[str] = []
        globals_used: List[str] = []
        literals: List[str] = []

        try:
            import textwrap

            dedented_source = textwrap.dedent(source)

            if not dedented_source.strip():
                return calls, imports_used, globals_used, literals

            tree = ast.parse(dedented_source)

            function_docstring = None
            if (
                func_node
                and self.config
                and self.config.exclude_docstrings_from_literals
            ):
                function_docstring = ast.get_docstring(func_node)

            tree_docstring = None
            if (
                self.config
                and self.config.exclude_docstrings_from_literals
                and tree.body
            ):
                first_node = tree.body[0]
                if isinstance(first_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    tree_docstring = ast.get_docstring(first_node)

            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    call_key = call_key_from_ast(node.func)
                    if call_key:
                        calls.append(call_key)

                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports_used.append(alias.name)

                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports_used.append(node.module)
                        for alias in node.names:
                            if alias.name != "*":
                                imports_used.append(f"{node.module}.{alias.name}")

                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    if len(node.value) > 2:
                        if self.config and self.config.exclude_docstrings_from_literals:
                            if (
                                function_docstring and node.value == function_docstring
                            ) or (tree_docstring and node.value == tree_docstring):
                                continue
                        literals.append(node.value)

                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    if node.id.isupper():
                        globals_used.append(node.id)

        except Exception as e:
            self.logger.debug(f"Failed to extract dependencies: {e}")

        return calls, imports_used, globals_used, literals

    def calculate_used_imports(
        self, file_imports: List[str], raw_calls: List[str]
    ) -> List[str]:
        """
        Evidence-based import usage detection based on raw_calls.

        Args:
            file_imports: List of imports available in the file
            raw_calls: List of call keys extracted from the function

        Returns:
            List of imports that are actually used
        """
        used_imports: List[str] = []
        raw_calls_set = set(raw_calls)

        for imp in file_imports:
            parts = imp.split(".")
            head = parts[0]
            tail = parts[-1]

            is_used = False
            if tail in raw_calls_set:
                is_used = True
            elif any(call.startswith(head + ".") for call in raw_calls_set):
                is_used = True
            elif head in raw_calls_set:
                is_used = True
            elif any(
                call.startswith(tail + ".") or call.endswith("." + tail)
                for call in raw_calls_set
            ):
                is_used = True

            if is_used:
                used_imports.append(imp)

        return used_imports


# -----------------------------
# Dependency visitor
# -----------------------------


class _DependencyVisitor(ast.NodeVisitor):
    """
    Collect dependencies from the function scope WITHOUT descending into nested defs/classes/lambdas.
    """

    __slots__ = (
        "extractor",
        "root",
        "calls",
        "imports_used",
        "globals_used",
        "literals",
        "_docstring",
    )

    def __init__(
        self,
        extractor: DependencyExtractor,
        root: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ):
        self.extractor = extractor
        self.root = root
        self.calls: List[str] = []
        self.imports_used: List[str] = []
        self.globals_used: List[str] = []
        self.literals: List[str] = []

        config = extractor.config
        self._docstring = (
            ast.get_docstring(root)
            if config and config.exclude_docstrings_from_literals
            else None
        )

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

    def visit_Call(self, node):
        call_key = call_key_from_ast(node.func)
        if call_key:
            self.calls.append(call_key)
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.imports_used.append(alias.name)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports_used.append(node.module)
            for alias in node.names:
                if alias.name != "*":
                    self.imports_used.append(f"{node.module}.{alias.name}")

    def visit_Constant(self, node):
        if isinstance(node.value, str) and len(node.value) > 2:
            if not (self._docstring and node.value == self._docstring):
                self.literals.append(node.value)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load) and node.id.isupper():
            self.globals_used.append(node.id)

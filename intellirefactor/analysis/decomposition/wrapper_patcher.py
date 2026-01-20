"""
Wrapper Patcher Module

Handles the generation and application of wrapper patches for functional decomposition.
Wraps existing function/method bodies with delegation calls to unified symbols.

Extracted from DecompositionAnalyzer god class to improve modularity.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import List, Optional

from .models import FunctionalBlock


class WrapperPatcher:
    """
    Generates and applies wrapper patches that replace function bodies
    with delegation calls to unified symbols.
    """

    def __init__(
        self,
        wrapper_marker: str,
        safe_decorators_allowlist: set,
        ast_utils_module,
    ):
        """
        Initialize the WrapperPatcher.

        Args:
            wrapper_marker: Comment marker to identify auto-generated wrappers
            safe_decorators_allowlist: Set of decorator names that are safe to preserve
            ast_utils_module: Reference to ast_utils module for AST operations
        """
        self._WRAPPER_MARKER = wrapper_marker
        self._SAFE_DECORATORS_ALLOWLIST = safe_decorators_allowlist
        self.ast_utils = ast_utils_module

    def apply_wrapper_patch(
        self,
        *,
        source_code: str,
        block: FunctionalBlock,
        unified_module: str,
        unified_symbol: str,
        package_name: str,
    ) -> str:
        """
        Apply a wrapper patch to replace a function body with a delegation call.

        Args:
            source_code: Original source code
            block: Functional block to wrap
            unified_module: Target module path for unified symbol
            unified_symbol: Name of the unified symbol to delegate to
            package_name: Package name for import statement

        Returns:
            Modified source code with wrapper applied

        Raises:
            RuntimeError: If node cannot be located or has unsupported decorators
        """
        tree = ast.parse(source_code)
        node = self.ast_utils.find_def_node(tree, block.qualname, lineno=block.lineno)
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise RuntimeError(f"Cannot locate callable node for {block.qualname}")

        dec_names = [self.ast_utils.get_decorator_name(d) for d in (node.decorator_list or [])]
        dec_bad = [d for d in dec_names if d and d not in self._SAFE_DECORATORS_ALLOWLIST]
        if dec_bad:
            raise RuntimeError(f"Unsupported decorators: {dec_bad}")

        body_start_lineno: Optional[int] = None
        if node.body:
            if (
                isinstance(node.body[0], ast.Expr)
                and isinstance(getattr(node.body[0], "value", None), ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                body_start_lineno = getattr(node.body[0], "end_lineno", node.body[0].lineno) + 1
            else:
                body_start_lineno = node.body[0].lineno

        end_lineno = getattr(node, "end_lineno", None)
        if not body_start_lineno or not end_lineno:
            return source_code

        lines = source_code.splitlines(True)
        indent = " " * (node.col_offset + 4)
        mod_dotted = self._module_dotted_from_target_module(unified_module)
        alias = f"__ir_unified_{unified_symbol}"
        call_args = self._build_call_arguments(node.args)

        wrapper_lines: List[str] = []
        wrapper_lines.append(f"{indent}# {self._WRAPPER_MARKER}\n")
        wrapper_lines.append(f"{indent}from {package_name}.{mod_dotted} import {unified_symbol} as {alias}\n")
        if isinstance(node, ast.AsyncFunctionDef):
            wrapper_lines.append(f"{indent}return await {alias}({call_args})\n")
        else:
            wrapper_lines.append(f"{indent}return {alias}({call_args})\n")

        new_lines = lines[: body_start_lineno - 1] + wrapper_lines + lines[end_lineno:]
        return "".join(new_lines)

    def _module_dotted_from_target_module(self, target_module: str) -> str:
        """
        Convert a target module path to dotted notation.

        Args:
            target_module: Module path (e.g., "path/to/module.py")

        Returns:
            Dotted module name (e.g., "path.to.module")
        """
        s = target_module.replace("\\", "/").strip()
        if s.endswith(".py"):
            s = s[:-3]
        return ".".join([p for p in s.split("/") if p])

    def _build_call_arguments(self, args: ast.arguments) -> str:
        """
        Build call arguments string from function signature.

        Args:
            args: AST arguments node

        Returns:
            String representation of call arguments
        """
        from .unified_symbol_generator import build_call_arguments
        return build_call_arguments(args)

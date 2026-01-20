"""
Function Signature Builder

Utilities for building and extracting function signatures from AST nodes.
"""

from __future__ import annotations

import ast
from typing import List, Tuple, Union


class SignatureBuilder:
    """
    Builds function signatures and extracts signature tokens from AST nodes.
    """

    @staticmethod
    def build_signature(node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        """
        Build function signature from AST node.

        Returns a string like: "func_name(arg1: int, arg2: str, *args, **kwargs) -> ReturnType"
        """
        parts: List[str] = []

        def fmt_arg(arg: ast.arg) -> str:
            s = arg.arg
            if arg.annotation:
                try:
                    s += f": {ast.unparse(arg.annotation)}"
                except Exception:
                    pass
            return s

        posonly = [fmt_arg(a) for a in getattr(node.args, "posonlyargs", [])]
        regular = [fmt_arg(a) for a in node.args.args]
        kwonly = [fmt_arg(a) for a in node.args.kwonlyargs]

        if posonly:
            parts.extend(posonly)
            parts.append("/")

        parts.extend(regular)

        if node.args.vararg:
            parts.append("*" + fmt_arg(node.args.vararg))
        elif kwonly:
            parts.append("*")

        parts.extend(kwonly)

        if node.args.kwarg:
            parts.append("**" + fmt_arg(node.args.kwarg))

        ret = ""
        if node.returns:
            try:
                ret = f" -> {ast.unparse(node.returns)}"
            except Exception:
                ret = ""

        return f"{node.name}({', '.join(parts)}){ret}"

    @staticmethod
    def extract_signature_tokens(
        node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    ) -> Tuple[List[str], List[str]]:
        """
        Extract input/output tokens from signature.

        Returns:
            (inputs, outputs) where:
            - inputs: list of argument names and type annotations
            - outputs: list of return type annotations
        """
        inputs: List[str] = []
        outputs: List[str] = []

        all_args: List[ast.arg] = []
        all_args.extend(getattr(node.args, "posonlyargs", []))
        all_args.extend(node.args.args)
        all_args.extend(node.args.kwonlyargs)

        for idx, arg in enumerate(all_args):
            if idx == 0 and arg.arg in {"self", "cls"}:
                continue
            inputs.append(arg.arg)
            if arg.annotation:
                try:
                    inputs.append(ast.unparse(arg.annotation))
                except Exception:
                    pass

        if node.args.vararg:
            inputs.append("*" + node.args.vararg.arg)
        if node.args.kwarg:
            inputs.append("**" + node.args.kwarg.arg)

        if node.returns:
            try:
                outputs.append(ast.unparse(node.returns))
            except Exception:
                pass

        return inputs, outputs

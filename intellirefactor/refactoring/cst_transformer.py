"""
refactoring/cst_transformer.py

Safe code transformations using LibCST to preserve formatting and comments.

Notes:
- LibCST preserves formatting, comments, and trivia better than AST-based rewriting.
- This module focuses on safe extraction/removal operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Set, Union

import libcst as cst


class MethodExtractorTransformer(cst.CSTTransformer):
    """
    Extracts methods from a class and removes them from the original source.

    Collects removed nodes in `extracted_nodes` so the caller can reuse them
    (e.g., generate an interface or a new component implementation).
    """

    def __init__(self, methods_to_remove: Set[str]) -> None:
        self.methods_to_remove = methods_to_remove
        self.extracted_nodes: List[cst.FunctionDef] = []

    def leave_FunctionDef(
        self,
        original_node: cst.FunctionDef,
        updated_node: cst.FunctionDef,
    ) -> Union[cst.FunctionDef, cst.RemovalSentinel]:
        if original_node.name.value in self.methods_to_remove:
            self.extracted_nodes.append(original_node)
            return cst.RemoveFromParent()
        return updated_node


class InterfaceGenerator:
    """
    Generates interface code using CST construction.

    Implementation note:
    - For typing.Protocol, it's idiomatic to use method stubs with `...` body.
    - We avoid forcing @abstractmethod because Protocols are structural.
    """

    @staticmethod
    def generate_interface(component_name: str, methods: Sequence[cst.FunctionDef]) -> str:
        """
        Build a Protocol interface for extracted methods.

        Args:
            component_name: Name of the component (e.g. "UserService").
            methods: Extracted concrete methods as LibCST FunctionDef nodes.

        Returns:
            A Python module source code string defining `I{component_name}` Protocol.
        """
        interface_methods: List[cst.FunctionDef] = []

        for method in methods:
            stub_body = cst.IndentedBlock(
                body=[
                    cst.SimpleStatementLine(
                        body=[cst.Expr(value=cst.Ellipsis())]
                    )
                ]
            )

            # Keep decorators that affect call style (staticmethod/classmethod/property),
            # but still stub the body.
            interface_methods.append(method.with_changes(body=stub_body))

        class_def = cst.ClassDef(
            name=cst.Name(f"I{component_name}"),
            bases=[cst.Arg(value=cst.Name("Protocol"))],
            body=cst.IndentedBlock(body=list(interface_methods)),
            # LibCST expects EmptyLine nodes here (not raw Comment nodes).
            leading_lines=[
                cst.EmptyLine(),
                cst.EmptyLine(comment=cst.Comment("# Auto-generated interface by IntelliRefactor")),
            ],
        )

        module = cst.Module(
            body=[
                cst.SimpleStatementLine(
                    body=[
                        cst.ImportFrom(
                            module=cst.Name("typing"),
                            names=[
                                cst.ImportAlias(name=cst.Name("Protocol")),
                                cst.ImportAlias(name=cst.Name("Any")),
                            ],
                        )
                    ]
                ),
                class_def,
            ]
        )
        return module.code


@dataclass(frozen=True)
class ExtractedMethodsResult:
    """Result of safe method extraction."""

    modified_source: str
    extracted_nodes: List[cst.FunctionDef]


def safe_extract_methods(source_code: str, methods_to_extract: Sequence[str]) -> ExtractedMethodsResult:
    """
    Safely extract methods preserving comments and formatting.

    Args:
        source_code: Original Python module code.
        methods_to_extract: Method names to remove from the source.

    Returns:
        ExtractedMethodsResult with modified_source and extracted CST nodes.

    Raises:
        libcst.ParserSyntaxError: If source_code is not valid Python syntax.
    """
    source_tree = cst.parse_module(source_code)
    transformer = MethodExtractorTransformer(set(methods_to_extract))
    modified_tree = source_tree.visit(transformer)

    return ExtractedMethodsResult(
        modified_source=modified_tree.code,
        extracted_nodes=transformer.extracted_nodes,
    )
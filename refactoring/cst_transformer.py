"""
refactoring/cst_transformer.py
Safe code transformations using LibCST to preserve formatting and comments.
"""

import libcst as cst
from typing import List, Optional, Union, Set


class MethodExtractorTransformer(cst.CSTTransformer):
    """
    Extracts methods from a class and removes them from the original source.
    Used to prepare the original file after extraction.
    """
    def __init__(self, methods_to_remove: Set[str]):
        self.methods_to_remove = methods_to_remove
        self.extracted_nodes: List[cst.FunctionDef] = []

    def leave_FunctionDef(self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef) -> Union[cst.FunctionDef, cst.RemovalSentinel]:
        if original_node.name.value in self.methods_to_remove:
            # Save the node for potential use elsewhere
            self.extracted_nodes.append(original_node)
            # Remove from the tree
            return cst.RemoveFromParent()
        return updated_node


class InterfaceGenerator:
    """Generates interface code using CST construction."""
    
    @staticmethod
    def generate_interface(component_name: str, methods: List[cst.FunctionDef]) -> str:
        # Create abstract methods from the extracted concrete methods
        abstract_methods = []
        for method in methods:
            # Strip body, add @abstractmethod decorator, replace body with ...
            new_body = cst.IndentedBlock(body=[cst.SimpleStatementLine(body=[cst.Expr(value=cst.Ellipsis())])])
            
            # Create decorator
            decorator = cst.Decorator(decorator=cst.Name("abstractmethod"))
            
            # Create new function definition
            new_method = method.with_changes(
                body=new_body,
                decorators=[decorator]
            )
            abstract_methods.append(new_method)

        # Build the class structure
        class_def = cst.ClassDef(
            name=cst.Name(f"I{component_name}"),
            bases=[cst.Arg(value=cst.Name("Protocol"))],
            body=cst.IndentedBlock(body=abstract_methods),
            leading_lines=[
                cst.EmptyLine(), 
                cst.Comment("# Auto-generated interface by IntelliRefactor")
            ]
        )
        
        # Add imports
        module = cst.Module(
            body=[
                cst.SimpleStatementLine(body=[cst.ImportFrom(
                    module=cst.Name("typing"),
                    names=[cst.ImportAlias(name=cst.Name("Protocol")), cst.ImportAlias(name=cst.Name("Any"))]
                )]),
                cst.SimpleStatementLine(body=[cst.ImportFrom(
                    module=cst.Name("abc"),
                    names=[cst.ImportAlias(name=cst.Name("abstractmethod"))]
                )]),
                class_def
            ]
        )
        
        return module.code


def safe_extract_methods(source_code: str, methods_to_extract: List[str]) -> dict:
    """
    Safely extracts methods preserving comments.
    Returns modified source code and extracted method code.
    """
    source_tree = cst.parse_module(source_code)
    transformer = MethodExtractorTransformer(set(methods_to_extract))
    
    # Transform original code (remove methods)
    modified_tree = source_tree.visit(transformer)
    
    return {
        "modified_source": modified_tree.code,
        "extracted_nodes": transformer.extracted_nodes
    }
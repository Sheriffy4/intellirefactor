"""Facade generation for maintaining backward compatibility.

This module provides the FacadeBuilder class that generates facade classes
which delegate to extracted components while maintaining the original API.

The facade pattern allows refactored code to maintain backward compatibility
by providing a thin wrapper that delegates method calls to specialized components.
"""

from __future__ import annotations

import ast
import textwrap
from typing import List, Dict, Tuple, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .auto_refactor import MethodInfo, RefactoringPlan


class FacadeBuilder:
    """Builds facade classes that delegate to extracted components.
    
    The FacadeBuilder generates a facade class that:
    - Maintains the original class name and public API
    - Delegates method calls to specialized components
    - Initializes a DI container for component management
    - Preserves non-method class members (attributes, nested classes)
    
    Attributes:
        container_attr: Name of the DI container attribute
        components_attr: Name of the components dictionary attribute
        interface_prefix: Prefix for interface class names
    
    Example:
        >>> builder = FacadeBuilder("_container", "_components", "I")
        >>> facade_code = builder.create_facade(
        ...     "MyClass",
        ...     ["DataService", "ValidationService"],
        ...     method_map,
        ...     plan,
        ...     code_generator,
        ...     get_group_name_func
        ... )
    """
    
    def __init__(
        self,
        container_attr: str,
        components_attr: str,
        interface_prefix: str
    ):
        """Initialize FacadeBuilder.
        
        Args:
            container_attr: Attribute name for DI container
            components_attr: Attribute name for components dictionary
            interface_prefix: Prefix for interface class names (e.g., "I")
        """
        self._container_attr = container_attr
        self._components_attr = components_attr
        self._interface_prefix = interface_prefix
    
    def create_facade(
        self,
        original_class_name: str,
        components: List[str],
        method_map: Dict[str, Tuple[str, bool, "MethodInfo"]],
        plan: "RefactoringPlan",
        code_generator,
        get_group_name_func
    ) -> str:
        """Generate facade class code that delegates to components.
        
        Args:
            original_class_name: Name of the original class
            components: List of component class names
            method_map: Mapping of method names to (component, is_async, MethodInfo)
            plan: Refactoring plan with cached content and tree
            code_generator: Code generator instance for extracting code
            get_group_name_func: Function to get group name from component name
            
        Returns:
            Complete facade class source code as string
        """
        content = plan._cached_content or ""
        tree = plan._cached_tree
        if not content or not tree:
            return content

        original_lines = content.splitlines()

        # Find the main class node
        main_class_node: Optional[ast.ClassDef] = None
        start_idx = 0
        end_idx = len(original_lines)

        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == original_class_name:
                main_class_node = node
                start_idx = node.lineno - 1
                end_idx = getattr(node, "end_lineno", end_idx)
                break

        if main_class_node is None:
            return content

        # Build facade code
        out: List[str] = []
        out.extend(original_lines[:start_idx])

        # Add imports
        out.append("")
        facade_imports = code_generator.generate_facade_import_block(
            components, code_generator.output_directory
        )
        out.extend(facade_imports)
        out.append("")

        # Add class definition
        out.append(f"class {original_class_name}:")
        out.append(
            '    """Facade maintaining backward compatibility (auto-generated)."""'
        )
        out.append("")

        # Process class body
        for node in main_class_node.body:
            # Preserve class attributes and nested classes
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.ClassDef)):
                code = code_generator.extract_node_code(node, content)
                if code:
                    out.append(textwrap.indent(textwrap.dedent(code), "    "))
                    out.append("")
                continue

            # Process methods
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name

                if name == "__init__":
                    self.create_enhanced_init_method(
                        node, content, out, components,
                        code_generator, get_group_name_func
                    )
                elif name in method_map:
                    comp, _is_async, mi = method_map[name]
                    self.create_active_delegation_method(
                        mi, comp, out, code_generator, get_group_name_func
                    )
                else:
                    # Keep unextracted methods as-is
                    code = code_generator.extract_method_code(node, content)
                    if code:
                        out.append(textwrap.indent(textwrap.dedent(code), "    "))
                        out.append("")

        # Add remaining module content after class
        if end_idx < len(original_lines):
            out.append("")
            out.extend(original_lines[end_idx:])

        return "\n".join(out)
    
    def create_enhanced_init_method(
        self,
        init_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        content: str,
        out_lines: List[str],
        components: List[str],
        code_generator,
        get_group_name_func
    ) -> None:
        """Generate enhanced __init__ method with DI container setup.
        
        Args:
            init_node: AST node for __init__ method
            content: Original file content
            out_lines: Output lines list to append to
            components: List of component class names
            code_generator: Code generator for extracting method code
            get_group_name_func: Function to get group name from component
        """
        # Extract and add original __init__ code
        init_code = code_generator.extract_method_code(init_node, content)
        ded = textwrap.dedent(init_code).splitlines()

        for line in ded:
            out_lines.append(f"    {line}")
        out_lines.append("")
        
        # Add DI container initialization
        out_lines.append(
            "        # [AutoRefactor] Initialize DI container and owner-proxy components"
        )
        out_lines.append(
            f"        self.{self._container_attr} = DIContainer.create_default()"
        )
        out_lines.append(f"        self.{self._components_attr} = {{}}")
        out_lines.append("")

        # Initialize each component
        for comp in components:
            group = get_group_name_func(comp)
            iface = f"{self._interface_prefix}{comp}"
            out_lines.append(
                f"        self.{self._components_attr}[{group!r}] = "
                f"self.{self._container_attr}.get({iface}, owner=self)"
            )

        out_lines.append("")
    
    def create_active_delegation_method(
        self,
        method: "MethodInfo",
        component: str,
        out_lines: List[str],
        code_generator,
        get_group_name_func
    ) -> None:
        """Generate delegation method that forwards calls to component.
        
        Args:
            method: MethodInfo for the method to delegate
            component: Component class name to delegate to
            out_lines: Output lines list to append to
            code_generator: Code generator for method signatures
            get_group_name_func: Function to get group name from component
        """
        group = get_group_name_func(component)

        # Add decorators
        for dec in method.node.decorator_list:
            dec_str = code_generator.unparse_safe(dec, "")
            if dec_str:
                out_lines.append(f"    @{dec_str}")

        # Generate method signature and delegation
        sig = code_generator.generate_method_signature(method)
        call_args = code_generator.build_call_arguments(method.node.args)
        await_kw = "await " if method.is_async else ""

        out_lines.append(f"    {sig}")
        out_lines.append(
            f'        """Delegates to {component}.{method.name} (auto-generated)."""'
        )
        out_lines.append(
            f"        return {await_kw}self.{self._components_attr}[{group!r}].{method.name}({call_args})"
        )
        out_lines.append("")

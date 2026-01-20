"""
Code generation utilities for AutoRefactor.

This module provides the CodeGenerator class that handles all code generation
tasks for the refactoring system, including:

- Interface generation (abstract base classes)
- Component implementation generation (concrete classes)
- DI container generation (dependency injection)
- Package initialization files
- Facade class generation

The CodeGenerator uses AST manipulation and template-based generation to create
syntactically correct Python code that maintains the original functionality while
improving structure and modularity.

Classes:
    CodeGenerator: Main code generation engine

Example:
    >>> generator = CodeGenerator(
    ...     interface_prefix='I',
    ...     component_template='Service'
    ... )
    >>> interface_code = generator.generate_interface('Storage', methods)
    >>> impl_code = generator.generate_component_implementation(
    ...     'StorageService', 'storage', public_methods, private_methods, plan
    ... )
"""

from __future__ import annotations

import ast
import re
import textwrap
from typing import Dict, FrozenSet, List, Optional, Set, Tuple, Union

# Constants for safe globals
BUILTINS: FrozenSet[str] = frozenset(
    {
        "print",
        "len",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sorted",
        "reversed",
        "min",
        "max",
        "sum",
        "abs",
        "round",
        "isinstance",
        "issubclass",
        "hasattr",
        "getattr",
        "setattr",
        "delattr",
        "type",
        "id",
        "hash",
        "repr",
        "str",
        "int",
        "float",
        "bool",
        "list",
        "dict",
        "set",
        "tuple",
        "open",
        "input",
        "format",
        "chr",
        "ord",
        "hex",
        "oct",
        "bin",
        "all",
        "any",
        "iter",
        "next",
        "callable",
        "super",
        "property",
        "classmethod",
        "staticmethod",
    }
)

BUILTIN_TYPES: FrozenSet[str] = frozenset(
    {
        "str",
        "int",
        "float",
        "bool",
        "bytes",
        "list",
        "dict",
        "set",
        "tuple",
        "None",
        "Any",
        "List",
        "Dict",
        "Set",
        "Tuple",
        "Optional",
        "Union",
        "Callable",
        "Type",
        "Sequence",
        "Mapping",
        "Iterable",
        "Iterator",
    }
)

SAFE_COMPONENT_GLOBALS: FrozenSet[str] = frozenset({"logger"})


class CodeGenerator:
    """Generates Python code for refactored components."""

    def __init__(
        self,
        interface_prefix: str = "I",
        component_template: str = "Service",
    ):
        """
        Initialize CodeGenerator.

        Args:
            interface_prefix: Prefix for interface names (e.g., "I")
            component_template: Suffix for component names (e.g., "Service")
        """
        self.interface_prefix = interface_prefix
        self.component_template = component_template

    def unparse_safe(self, node: Optional[ast.AST], default: str = "Any") -> str:
        """
        Safely unparse an AST node to string.

        Args:
            node: AST node to unparse
            default: Default value if unparsing fails

        Returns:
            Unparsed string or default
        """
        if node is None:
            return default
        try:
            return ast.unparse(node)
        except Exception:
            return default

    def format_arg(self, arg: ast.arg) -> str:
        """
        Format a function argument with type annotation.

        Args:
            arg: AST argument node

        Returns:
            Formatted argument string
        """
        if arg.annotation:
            return f"{arg.arg}: {self.unparse_safe(arg.annotation, 'Any')}"
        return f"{arg.arg}: Any"

    def generate_method_signature(self, method_info) -> str:
        """
        Generate method signature from MethodInfo.

        Args:
            method_info: MethodInfo object containing method details

        Returns:
            Complete method signature string
        """
        from .auto_refactor import DecoratorType

        node = method_info.node
        args = node.args
        parts: List[str] = []

        # Determine first parameter based on decorator
        if method_info.decorator_type == DecoratorType.STATICMETHOD:
            first_param: Optional[str] = None
        elif method_info.decorator_type == DecoratorType.CLASSMETHOD:
            first_param = "cls"
        else:
            first_param = "self"

        # Handle positional-only arguments (Python 3.8+)
        posonlyargs = getattr(args, "posonlyargs", [])
        for i, a in enumerate(posonlyargs):
            if i == 0 and a.arg in ("self", "cls"):
                continue
            parts.append(self.format_arg(a))
        if posonlyargs:
            non_self = [a for a in posonlyargs if a.arg not in ("self", "cls")]
            if non_self:
                parts.append("/")

        # Handle regular arguments with defaults
        num_defaults = len(args.defaults)
        num_args = len(args.args)

        for i, a in enumerate(args.args):
            if i == 0 and a.arg in ("self", "cls"):
                continue

            arg_str = self.format_arg(a)
            default_idx = i - (num_args - num_defaults)
            if 0 <= default_idx < num_defaults:
                arg_str += f" = {self.unparse_safe(args.defaults[default_idx], '...')}"
            parts.append(arg_str)

        # Handle *args
        if args.vararg:
            var = f"*{args.vararg.arg}"
            if args.vararg.annotation:
                var += f": {self.unparse_safe(args.vararg.annotation, 'Any')}"
            parts.append(var)
        elif args.kwonlyargs:
            parts.append("*")

        # Handle keyword-only arguments
        for i, a in enumerate(args.kwonlyargs):
            arg_str = self.format_arg(a)
            if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
                arg_str += f" = {self.unparse_safe(args.kw_defaults[i], '...')}"
            parts.append(arg_str)

        # Handle **kwargs
        if args.kwarg:
            kw = f"**{args.kwarg.arg}"
            if args.kwarg.annotation:
                kw += f": {self.unparse_safe(args.kwarg.annotation, 'Any')}"
            parts.append(kw)

        args_str = ", ".join(parts)
        return_type = self.unparse_safe(node.returns, "Any") if node.returns else "Any"
        prefix = "async " if method_info.is_async else ""

        if first_param:
            return f"{prefix}def {node.name}({first_param}{', ' if args_str else ''}{args_str}) -> {return_type}:"
        return f"{prefix}def {node.name}({args_str}) -> {return_type}:"

    def build_call_arguments(self, args: ast.arguments) -> str:
        """
        Build argument list for method delegation.

        Args:
            args: AST arguments object

        Returns:
            Comma-separated argument list for calling
        """
        parts: List[str] = []

        # Positional-only arguments
        posonlyargs = getattr(args, "posonlyargs", [])
        for i, a in enumerate(posonlyargs):
            if i == 0 and a.arg in ("self", "cls"):
                continue
            parts.append(a.arg)

        # Regular arguments (skip self/cls)
        start_idx = 0
        if args.args and args.args[0].arg in ("self", "cls"):
            start_idx = 1
        for a in args.args[start_idx:]:
            parts.append(a.arg)

        # *args
        if args.vararg:
            parts.append(f"*{args.vararg.arg}")

        # Keyword-only arguments
        for a in args.kwonlyargs:
            parts.append(f"{a.arg}={a.arg}")

        # **kwargs
        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")

        return ", ".join(parts)

    def generate_import_statement(
        self,
        import_info,
        needed_names: Set[str],
        level_adjustment: int = 1,
    ) -> Optional[str]:
        """
        Generate import statement for needed names.

        Args:
            import_info: ImportInfo object
            needed_names: Set of names that are needed
            level_adjustment: Adjustment for relative import level

        Returns:
            Import statement string or None if no names needed
        """
        provided = import_info.get_all_names()
        if provided.isdisjoint(needed_names):
            return None

        node = import_info.node
        if isinstance(node, ast.Import):
            return ast.unparse(node)

        new_level = import_info.level + (
            level_adjustment if import_info.is_relative else 0
        )

        new_names: List[ast.alias] = []
        for orig, alias in import_info.names.items():
            use = alias if alias else orig
            if use in needed_names or orig in needed_names:
                new_names.append(ast.alias(name=orig, asname=alias))

        if not new_names:
            return None

        new_node = ast.ImportFrom(
            module=import_info.module, names=new_names, level=new_level
        )
        return ast.unparse(new_node)

    def extract_node_code(
        self, node: ast.AST, content: str, start_line: Optional[int] = None
    ) -> str:
        """
        Extract source code for an AST node.

        Args:
            node: AST node
            content: Full source code
            start_line: Optional starting line (0-indexed)

        Returns:
            Extracted code string
        """
        lines = content.splitlines()
        if start_line is None:
            start_line = getattr(node, "lineno", 1) - 1

        end = getattr(node, "end_lineno", None)
        if end is not None:
            return "\n".join(lines[start_line:end])

        return "\n".join(lines[start_line:])

    def extract_method_code(
        self, method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef], content: str
    ) -> str:
        """
        Extract method code including decorators.

        Args:
            method_node: Method AST node
            content: Full source code

        Returns:
            Method code string
        """
        start = method_node.lineno - 1
        if method_node.decorator_list:
            start = min(d.lineno for d in method_node.decorator_list) - 1
        return self.extract_node_code(method_node, content, start_line=start)

    def generate_owner_proxy_base(self) -> str:
        """
        Generate base class for owner-proxy pattern.

        Returns:
            Complete base.py file content
        """
        return "\n".join(
            [
                "from __future__ import annotations",
                "",
                '"""Owner-proxy base utilities for generated components."""',
                "",
                "from typing import Any",
                "",
                "",
                "class OwnerProxyMixin:",
                '    """Mixin that forwards attribute access to an owning facade object."""',
                "",
                "    __slots__ = ('_owner',)",
                "",
                "    def __init__(self, owner: Any) -> None:",
                "        object.__setattr__(self, '_owner', owner)",
                "",
                "    def __getattr__(self, name: str) -> Any:",
                "        return getattr(object.__getattribute__(self, '_owner'), name)",
                "",
                "    def __setattr__(self, name: str, value: Any) -> None:",
                "        if name == '_owner':",
                "            object.__setattr__(self, name, value)",
                "            return",
                "        setattr(object.__getattribute__(self, '_owner'), name, value)",
                "",
                "    def __delattr__(self, name: str) -> None:",
                "        if name == '_owner':",
                "            raise AttributeError('cannot delete _owner')",
                "        delattr(object.__getattribute__(self, '_owner'), name)",
                "",
            ]
        )

    def generate_component_implementation(
        self,
        component_name: str,
        group_name: str,
        public_methods: List,
        private_methods: List,
        plan,
    ) -> str:
        """
        Generate component implementation file.

        Args:
            component_name: Name of the component class
            group_name: Group name for the component
            public_methods: List of public MethodInfo objects
            private_methods: List of private MethodInfo objects
            plan: RefactoringPlan with cached content

        Returns:
            Complete implementation file content
        """
        content = plan._cached_content or ""

        # Collect used names
        used: Set[str] = set()
        for m in public_methods + private_methods:
            used |= set(m.used_names)

        used -= SAFE_COMPONENT_GLOBALS
        needed_imports = used - plan._module_level_names - BUILTINS - BUILTIN_TYPES

        lines: List[str] = [
            "from __future__ import annotations",
            "",
            f'"""Implementation of {self.interface_prefix}{component_name}."""',
            "",
            "import logging",
            "from typing import Any",
            "",
            "from .base import OwnerProxyMixin",
        ]

        # Add necessary imports
        added: Set[str] = set()
        for imp in plan._imports:
            stmt = self.generate_import_statement(
                imp, needed_imports, level_adjustment=1
            )
            if stmt and stmt not in added:
                added.add(stmt)
                lines.append(stmt)

        lines += [
            "",
            f"from .interfaces import {self.interface_prefix}{component_name}",
            "",
            "logger = logging.getLogger(__name__)",
            "",
            "",
            f"class {component_name}(OwnerProxyMixin, {self.interface_prefix}{component_name}):",
            f'    """Implementation of {self.interface_prefix}{component_name} using owner-proxy state."""',
            "",
            "    def __init__(self, owner: Any) -> None:",
            "        super().__init__(owner)",
            "",
        ]

        # Add methods
        for m in public_methods + private_methods:
            code = self.extract_method_code(m.node, content)
            if code:
                lines.append(textwrap.indent(textwrap.dedent(code), "    "))
                lines.append("")

        return "\n".join(lines)

    def generate_interface(self, component_name: str, public_methods: List) -> str:
        """
        Generate interface class for a component.

        Args:
            component_name: Name of the component
            public_methods: List of public MethodInfo objects

        Returns:
            Interface class code
        """
        from .auto_refactor import DecoratorType

        iface = f"{self.interface_prefix}{component_name}"
        if not public_methods:
            return ""

        lines = [
            f"class {iface}(ABC):",
            f'    """Interface for {component_name.lower()} operations."""',
            "",
        ]

        for m in public_methods:
            if m.decorator_type == DecoratorType.STATICMETHOD:
                lines.append("    @staticmethod")
            elif m.decorator_type == DecoratorType.CLASSMETHOD:
                lines.append("    @classmethod")
            elif m.decorator_type == DecoratorType.PROPERTY:
                lines.append("    @property")

            lines.append("    @abstractmethod")
            sig = self.generate_method_signature(m)
            lines += [
                f"    {sig}",
                '        """TODO: Add documentation."""',
                "        ...",
                "",
            ]
        return "\n".join(lines)

    def generate_interfaces_file(self, interface_classes: List[str], plan) -> str:
        """
        Generate interfaces.py file with all interfaces.

        Args:
            interface_classes: List of interface class code strings
            plan: RefactoringPlan (reserved for future use)

        Returns:
            Complete interfaces.py file content
        """
        lines: List[str] = [
            "from __future__ import annotations",
            "",
            '"""Auto-generated interfaces."""',
            "",
            "from abc import ABC, abstractmethod",
            "from typing import Any",
            "",
        ]
        lines.append(
            "\n\n".join(interface_classes)
            if interface_classes
            else "# No interfaces generated\n"
        )
        return "\n".join(lines)

    def generate_di_container(self, components: List[str], get_group_name_func) -> str:
        """
        Generate DI container file.

        Args:
            components: List of component class names
            get_group_name_func: Function to convert component name to group name

        Returns:
            Complete container.py file content
        """
        lines: List[str] = [
            "from __future__ import annotations",
            "",
            '"""Dependency Injection Container (auto-generated)."""',
            "",
            "import inspect",
            "from typing import Any, Dict, Tuple, Type, TypeVar",
            "",
            "T = TypeVar('T')",
            "",
        ]

        for comp in components:
            group = get_group_name_func(comp)
            lines.append(
                f"from .{group}_{self.component_template.lower()} import {comp}"
            )
            lines.append(f"from .interfaces import {self.interface_prefix}{comp}")

        lines += [
            "",
            "",
            "class DIContainer:",
            '    """Simple DI container for generated components."""',
            "",
            "    def __init__(self) -> None:",
            "        self._services: Dict[Type, Type] = {}",
            "        self._singletons: Dict[Tuple[Type, int], Any] = {}",
            "",
            "    @classmethod",
            "    def create_default(cls) -> 'DIContainer':",
            "        c = cls()",
        ]

        for comp in components:
            iface = f"{self.interface_prefix}{comp}"
            lines.append(f"        c.register({iface}, {comp})")

        lines += [
            "        return c",
            "",
            "    def register(self, interface: Type, implementation: Type) -> None:",
            "        self._services[interface] = implementation",
            "",
            "    def get(self, interface: Type[T], owner: Any) -> T:",
            "        if owner is None:",
            "            raise ValueError('owner is required for owner-proxy components')",
            "",
            "        key = (interface, id(owner))",
            "        if key in self._singletons:",
            "            return self._singletons[key]",
            "",
            "        if interface not in self._services:",
            "            raise ValueError(f'Service {interface} not registered')",
            "",
            "        impl = self._services[interface]",
            "        try:",
            "            sig = inspect.signature(impl.__init__)",
            "            params = list(sig.parameters.values())",
            "            accepts_owner_kw = any(p.name == 'owner' for p in params[1:])",
            "            instance = impl(owner=owner) if accepts_owner_kw else impl(owner)",
            "        except TypeError as e:",
            "            raise TypeError(f'Failed to construct {impl}: {e}')",
            "",
            "        self._singletons[key] = instance",
            "        return instance",
            "",
        ]

        return "\n".join(lines)

    def generate_package_init(
        self, components: List[str], has_interfaces: bool, get_group_name_func
    ) -> str:
        """
        Generate __init__.py for components package.

        Args:
            components: List of component class names
            has_interfaces: Whether interfaces exist
            get_group_name_func: Function to convert component name to group name

        Returns:
            Complete __init__.py file content
        """
        lines = ['"""Auto-generated components package."""', ""]
        lines.append("from .container import DIContainer")
        lines.append("")
        for comp in components:
            group = get_group_name_func(comp)
            lines.append(
                f"from .{group}_{self.component_template.lower()} import {comp}"
            )
        if has_interfaces:
            lines.append("")
            lines.append("from .interfaces import *  # noqa: F401,F403")

        exports = ["DIContainer"] + components
        lines.append("")
        lines.append(f"__all__ = {exports!r}")
        lines.append("")
        return "\n".join(lines)

    def generate_facade_import_block(
        self, components: List[str], output_directory: str
    ) -> List[str]:
        """
        Generate robust import block for facade.

        Args:
            components: List of component class names
            output_directory: Name of the components directory

        Returns:
            List of import statement lines
        """
        pkg = output_directory
        iface_names = ", ".join(f"{self.interface_prefix}{c}" for c in components)

        return [
            "# Auto-generated imports for refactored components",
            "try:",
            f"    from .{pkg}.container import DIContainer",
            f"    from .{pkg}.interfaces import {iface_names}",
            "except ImportError as e:",
            "    # Fallback for different import contexts",
            "    try:",
            f"        from {pkg}.container import DIContainer",
            f"        from {pkg}.interfaces import {iface_names}",
            "    except ImportError:",
            "        # If components are not available, create stub implementations",
            "        import logging",
            "        logger = logging.getLogger(__name__)",
            "        logger.warning(f'Component imports failed: {e}. Using stub implementations.')",
            "        ",
            "        class DIContainer:",
            "            @classmethod",
            "            def create_default(cls): return cls()",
            "            def get(self, interface, owner): return None",
            "        ",
            f"        # Stub interfaces for {iface_names}",
            "        "
            + "\n        ".join(
                [f"class {self.interface_prefix}{c}: pass" for c in components]
            ),
        ]

    @staticmethod
    def component_class_name(group_name: str, component_template: str) -> str:
        """
        Convert group name to component class name.

        Args:
            group_name: Group name (e.g., "console", "misc_1")
            component_template: Template suffix (e.g., "Service")

        Returns:
            Component class name (e.g., "ConsoleService", "Misc1Service")
        """
        parts: List[str] = []
        buf = ""
        for ch in group_name:
            if ch.isalnum():
                buf += ch
            else:
                if buf:
                    parts.append(buf)
                    buf = ""
        if buf:
            parts.append(buf)

        camel = "".join(p[:1].upper() + p[1:] for p in parts if p)
        return f"{camel}{component_template}"

    @staticmethod
    def get_group_name(component_name: str, component_template: str) -> str:
        """
        Convert component class name to group/module name.

        Args:
            component_name: Component class name (e.g., "ConsoleService")
            component_template: Template suffix (e.g., "Service")

        Returns:
            Group name (e.g., "console")
        """
        suffix = component_template
        base = (
            component_name[: -len(suffix)]
            if component_name.endswith(suffix)
            else component_name
        )

        # CamelCase -> snake_case
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", base)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        # Insert '_' between letters and digits: misc1 -> misc_1
        s3 = re.sub(r"(?<=\D)(?=\d)", "_", s2)
        return s3.lower()

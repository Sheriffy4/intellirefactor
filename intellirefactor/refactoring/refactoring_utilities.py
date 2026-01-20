"""
Reusable Refactoring Components and Utilities.

This module provides reusable components and utilities that can be used
to automate common refactoring operations based on proven refactoring patterns.

Key improvements made:
- Safer defaults in dataclasses (no mutable default / None pitfalls)
- More reliable method removal using AST line ranges instead of heuristics
- Correct imports for generated interfaces/implementations
- Better naming (snake_case modules), PEP 8, and clearer docstrings
"""

from __future__ import annotations

import ast
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _to_snake_case(name: str) -> str:
    """
    Convert CamelCase/PascalCase to snake_case.

    Examples:
        UserService -> user_service
        XMLParser -> xml_parser
    """
    s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


@dataclass
class RefactoringResult:
    """Result of a refactoring operation."""

    success: bool
    files_created: List[str]
    files_modified: List[str]
    files_deleted: List[str]
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)


class RefactoringUtility(ABC):
    """Base class for refactoring utilities."""

    @abstractmethod
    def can_apply(self, file_path: str, **kwargs) -> bool:
        """Check if this utility can be applied to the given file."""
        raise NotImplementedError

    @abstractmethod
    def apply(self, file_path: str, **kwargs) -> RefactoringResult:
        """Apply the refactoring to the given file."""
        raise NotImplementedError

    @abstractmethod
    def get_description(self) -> str:
        """Get a description of what this utility does."""
        raise NotImplementedError


class ComponentExtractor(RefactoringUtility):
    """Utility for extracting components from monolithic classes."""

    def __init__(self) -> None:
        self.interface_template = '''"""
{interface_description}
"""
from typing import Any, Protocol

class {interface_name}(Protocol):
    """Interface for {component_description}."""

{interface_methods}
'''

        self.implementation_template = '''"""
{implementation_description}
"""
from __future__ import annotations

from .i_{component_module} import {interface_name}

class {class_name}({interface_name}):
    """Implementation of {interface_name}."""

    def __init__(self{constructor_params}):
{constructor_body}

{methods}
'''

    def can_apply(self, file_path: str, **kwargs) -> bool:
        """Check if component extraction can be applied."""
        try:
            content = Path(file_path).read_text(encoding="utf-8-sig")
            tree = ast.parse(content)

            classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
            if not classes:
                return False

            # Heuristic: any large class in a large file.
            for cls in classes:
                methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)]
                if len(methods) > 5 and len(content.splitlines()) > 500:
                    return True
            return False

        except Exception as e:
            logger.exception(
                "Error checking if component extraction can be applied: %s", e
            )
            return False

    def apply(self, file_path: str, **kwargs) -> RefactoringResult:
        """
        Extract specified methods into a new component.

        kwargs:
            component_name: str (default: "ExtractedComponent")
            methods: List[str] method names to extract (required)
            target_directory: Path | str (default: parent directory of file_path)
        """
        try:
            component_name = str(kwargs.get("component_name", "ExtractedComponent"))
            methods_to_extract = list(kwargs.get("methods", []))
            target_directory = Path(
                kwargs.get("target_directory", Path(file_path).parent)
            )

            if not methods_to_extract:
                return RefactoringResult(
                    success=False,
                    files_created=[],
                    files_modified=[],
                    files_deleted=[],
                    error_message="No methods specified for extraction",
                )

            target_directory.mkdir(parents=True, exist_ok=True)

            content = Path(file_path).read_text(encoding="utf-8-sig")
            tree = ast.parse(content)

            interface_code, implementation_code = self._extract_component(
                tree=tree,
                content=content,
                component_name=component_name,
                methods_to_extract=methods_to_extract,
            )

            component_module = _to_snake_case(component_name)

            interface_path = target_directory / f"i_{component_module}.py"
            impl_path = target_directory / f"{component_module}.py"

            interface_path.write_text(interface_code, encoding="utf-8")
            impl_path.write_text(implementation_code, encoding="utf-8")

            modified_content = self._modify_original_file(
                content=content,
                component_name=component_name,
                methods_to_extract=methods_to_extract,
            )
            Path(file_path).write_text(modified_content, encoding="utf-8")

            return RefactoringResult(
                success=True,
                files_created=[str(interface_path), str(impl_path)],
                files_modified=[file_path],
                files_deleted=[],
                warnings=[
                    "Please review extracted component boundaries and update tests"
                ],
            )

        except Exception as e:
            logger.exception("Error applying component extraction: %s", e)
            return RefactoringResult(
                success=False,
                files_created=[],
                files_modified=[],
                files_deleted=[],
                error_message=str(e),
            )

    def _extract_component(
        self,
        tree: ast.AST,
        content: str,
        component_name: str,
        methods_to_extract: List[str],
    ) -> Tuple[str, str]:
        """Extract component interface and implementation."""
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        if not classes:
            raise ValueError("No classes found in file")

        # Prefer the largest class as the primary candidate.
        main_class = max(classes, key=lambda c: len(getattr(c, "body", [])))

        interface_methods: List[str] = []
        implementation_methods: List[str] = []

        for node in main_class.body:
            if isinstance(node, ast.FunctionDef) and node.name in methods_to_extract:
                interface_methods.append(self._create_interface_method(node, content))
                implementation_methods.append(
                    self._extract_method_implementation(node, content)
                )

        if not interface_methods:
            raise ValueError(
                "None of the specified methods were found in the target class"
            )

        component_module = _to_snake_case(component_name)
        interface_name = f"I{component_name}"

        interface_code = self.interface_template.format(
            interface_description=f"Interface for {component_name} component.",
            interface_name=interface_name,
            component_description=component_name.lower(),
            interface_methods="\n".join(interface_methods).rstrip() + "\n",
        )

        implementation_code = self.implementation_template.format(
            implementation_description=f"Implementation of {component_name} component.",
            component_module=component_module,
            class_name=component_name,
            interface_name=interface_name,
            constructor_params="",
            constructor_body="        pass",
            methods="\n".join(implementation_methods).rstrip() + "\n",
        )

        return interface_code, implementation_code

    def _create_interface_method(
        self, method_node: ast.FunctionDef, content: str
    ) -> str:
        """
        Create an interface method signature from an AST FunctionDef.

        We keep parameter names and basic argument kinds; return type defaults to Any.
        """
        args = method_node.args

        parts: List[str] = ["self"]

        # posonlyargs (py3.8+)
        for a in getattr(args, "posonlyargs", []):
            parts.append(a.arg)
        if getattr(args, "posonlyargs", []):
            parts.append("/")

        # normal args (skip self if present in AST)
        norm_args = [a.arg for a in args.args]
        if norm_args and norm_args[0] == "self":
            norm_args = norm_args[1:]
        parts.extend(norm_args)

        if args.vararg:
            parts.append(f"*{args.vararg.arg}")
        elif args.kwonlyargs:
            parts.append("*")

        for a in args.kwonlyargs:
            parts.append(a.arg)

        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")

        args_str = ", ".join([p for p in parts if p not in ("", None)])

        # Return type: try to preserve if possible (Python 3.9+: ast.unparse)
        return_type = "Any"
        if method_node.returns is not None and hasattr(ast, "unparse"):
            try:
                return_type = ast.unparse(method_node.returns)
            except Exception:
                return_type = "Any"

        return (
            f"    def {method_node.name}({args_str}) -> {return_type}:\n"
            f'        """Method {method_node.name}."""\n'
            f"        ...\n"
        )

    def _extract_method_implementation(
        self, method_node: ast.FunctionDef, content: str
    ) -> str:
        """Extract original method source by line range."""
        lines = content.splitlines()
        start_line = method_node.lineno - 1

        # Include decorators if present.
        decorator_lines = [
            getattr(d, "lineno", method_node.lineno) for d in method_node.decorator_list
        ]
        start_line = min([start_line] + [dl - 1 for dl in decorator_lines if dl])

        end_line = getattr(method_node, "end_lineno", None)
        if not end_line:
            # Fallback: take a reasonable slice if end_lineno is absent.
            end_line = min(len(lines), start_line + 25)

        method_lines = lines[start_line:end_line]
        return "\n".join(method_lines) + "\n"

    def _modify_original_file(
        self, content: str, component_name: str, methods_to_extract: List[str]
    ) -> str:
        """
        Modify original file:
        - remove extracted methods by exact AST ranges
        - insert correct imports for new component + interface

        This is still simplified (does not rewrite call sites), but is *much* safer than
        regex-based skipping.
        """
        tree = ast.parse(content)
        lines = content.splitlines()

        # Collect ranges to delete (inclusive in AST, slice-exclusive in Python).
        ranges: List[Tuple[int, int]] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name in methods_to_extract:
                start = node.lineno
                end = getattr(node, "end_lineno", node.lineno)

                # include decorators
                for d in node.decorator_list:
                    if hasattr(d, "lineno") and d.lineno:
                        start = min(start, d.lineno)

                ranges.append((start, end))

        # Delete from bottom to top to keep indexes stable.
        for start, end in sorted(ranges, key=lambda x: x[0], reverse=True):
            del lines[start - 1 : end]

        component_module = _to_snake_case(component_name)
        import_lines = [
            f"from .{component_module} import {component_name}",
            f"from .i_{component_module} import I{component_name}",
        ]

        # Insert after existing imports / module docstring.
        insert_at = 0
        in_docstring = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if i == 0 and (stripped.startswith('"""') or stripped.startswith("'''")):
                in_docstring = True
            if in_docstring:
                if stripped.endswith('"""') or stripped.endswith("'''"):
                    in_docstring = False
                insert_at = i + 1
                continue

            if stripped.startswith(("import ", "from ")):
                insert_at = i + 1
                continue

            if stripped and not stripped.startswith("#"):
                break

        # Avoid duplicate imports.
        existing = {line.strip() for line in lines}
        for imp in import_lines:
            if imp not in existing:
                lines.insert(insert_at, imp)
                insert_at += 1

        return "\n".join(lines) + "\n"

    def get_description(self) -> str:
        return "Extracts components from monolithic classes following SRP"


class ConfigurationSplitter(RefactoringUtility):
    """Utility for splitting monolithic configuration classes."""

    def __init__(self) -> None:
        self.domain_config_template = '''"""
{domain_name} configuration.
"""
from dataclasses import dataclass
from typing import Optional

@dataclass
class {class_name}:
    """{domain_name} configuration settings."""

{fields}
'''

        self.main_config_template = '''"""
Main configuration that composes domain-specific configurations.
"""
from dataclasses import dataclass
{imports}

@dataclass
class {main_class_name}:
    """Main configuration composed of domain-specific configs."""

{domain_fields}
'''

    def can_apply(self, file_path: str, **kwargs) -> bool:
        """Check if configuration splitting can be applied."""
        try:
            if (
                "config" not in file_path.lower()
                and "settings" not in file_path.lower()
            ):
                return False

            content = Path(file_path).read_text(encoding="utf-8-sig")
            tree = ast.parse(content)

            # Look for a dataclass class with many annotated fields.
            for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
                if not self._is_dataclass(cls):
                    continue
                fields = self._extract_dataclass_fields(cls, content)
                if len(fields) >= 10:
                    return True

            return False

        except Exception as e:
            logger.exception("Error checking if config splitting can be applied: %s", e)
            return False

    def apply(self, file_path: str, **kwargs) -> RefactoringResult:
        """
        Split monolithic configuration into domain-specific configs.

        kwargs:
            domains: Dict[str, List[str]]  domain_name -> list of field names (required)
            target_directory: Path | str (default: parent directory)
        """
        try:
            domains: Dict[str, List[str]] = dict(kwargs.get("domains", {}))
            target_directory = Path(
                kwargs.get("target_directory", Path(file_path).parent)
            )
            target_directory.mkdir(parents=True, exist_ok=True)

            if not domains:
                return RefactoringResult(
                    success=False,
                    files_created=[],
                    files_modified=[],
                    files_deleted=[],
                    error_message="No domains specified for configuration split",
                )

            content = Path(file_path).read_text(encoding="utf-8-sig")
            config_fields = self._parse_config_fields(content)

            created_files: List[str] = []
            imports: List[str] = []
            domain_fields: List[str] = []

            for domain_name, field_names in domains.items():
                domain_fields_code: List[str] = []
                for field_name in field_names:
                    if field_name in config_fields:
                        domain_fields_code.append(f"    {config_fields[field_name]}")

                class_name = f"{domain_name.title()}Config"

                domain_config_code = self.domain_config_template.format(
                    domain_name=domain_name.title(),
                    class_name=class_name,
                    fields=(
                        "\n".join(domain_fields_code)
                        if domain_fields_code
                        else "    pass"
                    ),
                )

                domain_file_path = target_directory / f"{domain_name}_config.py"
                domain_file_path.write_text(domain_config_code, encoding="utf-8")

                created_files.append(str(domain_file_path))
                imports.append(f"from .{domain_name}_config import {class_name}")
                domain_fields.append(f"    {domain_name}: {class_name}")

            main_config_code = self.main_config_template.format(
                main_class_name="MainConfig",
                imports="\n".join(imports),
                domain_fields="\n".join(domain_fields) if domain_fields else "    pass",
            )

            main_config_path = target_directory / "main_config.py"
            main_config_path.write_text(main_config_code, encoding="utf-8")
            created_files.append(str(main_config_path))

            return RefactoringResult(
                success=True,
                files_created=created_files,
                files_modified=[],
                files_deleted=[],
                warnings=[
                    "Please update imports and usage of the original configuration"
                ],
            )

        except Exception as e:
            logger.exception("Error applying configuration splitting: %s", e)
            return RefactoringResult(
                success=False,
                files_created=[],
                files_modified=[],
                files_deleted=[],
                error_message=str(e),
            )

    def _is_dataclass(self, cls: ast.ClassDef) -> bool:
        """Detect @dataclass decorator."""
        for d in cls.decorator_list:
            if isinstance(d, ast.Name) and d.id == "dataclass":
                return True
            if isinstance(d, ast.Attribute) and d.attr == "dataclass":
                return True
        return False

    def _extract_dataclass_fields(
        self, cls: ast.ClassDef, content: str
    ) -> Dict[str, str]:
        """Extract field definitions from a dataclass class body."""
        fields: Dict[str, str] = {}
        for node in cls.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                # Best-effort source segment; fallback to type-only.
                seg = ast.get_source_segment(content, node)
                if seg:
                    # seg already includes indentation; normalize to a single "name: type = default" string
                    fields[name] = seg.strip()
                else:
                    fields[name] = f"{name}: Optional[str] = None"
        return fields

    def _parse_config_fields(self, content: str) -> Dict[str, str]:
        """
        Parse configuration fields from content using AST.

        Returns:
            dict: field_name -> "field_name: Type = default" or "field_name: Type"
        """
        tree = ast.parse(content)
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        for cls in classes:
            if self._is_dataclass(cls):
                return self._extract_dataclass_fields(cls, content)
        return {}

    def get_description(self) -> str:
        return "Splits monolithic configuration dataclasses into domain-specific configurations"


class DependencyInjectionIntroducer(RefactoringUtility):
    """Utility for introducing dependency injection patterns."""

    def __init__(self) -> None:
        self.interface_template = '''"""
Interface for {service_name}.
"""
from typing import Any, Protocol

class I{service_name}(Protocol):
    """Interface for {service_name} service."""

    def process(self, data: Any) -> Any:
        """Process data."""
        ...
'''

        # Improved DI container:
        # - supports singleton/transient/factory lifetimes
        # - avoids caching transients
        # - safer resolution: resolves only if annotation is registered
        self.container_template = '''"""
Dependency injection container.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional, Type, TypeVar
import inspect

T = TypeVar("T")


class Lifetime(str, Enum):
    SINGLETON = "singleton"
    TRANSIENT = "transient"
    FACTORY = "factory"


@dataclass(frozen=True)
class Registration:
    provider: Any  # Type[T] or Callable[[], T]
    lifetime: Lifetime


class DIContainer:
    """Simple dependency injection container."""

    def __init__(self) -> None:
        self._registrations: Dict[Type[Any], Registration] = {}
        self._singletons: Dict[Type[Any], Any] = {}

    def register_singleton(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a singleton service."""
        self._registrations[interface] = Registration(provider=implementation, lifetime=Lifetime.SINGLETON)

    def register_transient(self, interface: Type[T], implementation: Type[T]) -> None:
        """Register a transient service."""
        self._registrations[interface] = Registration(provider=implementation, lifetime=Lifetime.TRANSIENT)

    def register_factory(self, interface: Type[T], factory: Callable[[], T]) -> None:
        """Register a factory for creating services."""
        self._registrations[interface] = Registration(provider=factory, lifetime=Lifetime.FACTORY)

    def get(self, interface: Type[T]) -> T:
        """Get a service instance."""
        if interface in self._singletons:
            return self._singletons[interface]

        reg = self._registrations.get(interface)
        if reg is None:
            raise ValueError(f"Service {interface} not registered")

        if reg.lifetime == Lifetime.FACTORY:
            instance = reg.provider()
            # Factory usually behaves like singleton in many containers; keep as singleton here.
            self._singletons[interface] = instance
            return instance

        implementation: Type[T] = reg.provider

        # Constructor injection: only resolve dependencies that are registered.
        sig = inspect.signature(implementation.__init__)
        kwargs: Dict[str, Any] = {}
        for name, param in sig.parameters.items():
            if name == "self":
                continue
            ann = param.annotation
            if ann is inspect.Parameter.empty:
                continue
            if isinstance(ann, str):
                # Forward refs are not resolved in this simple container.
                continue
            if ann in self._registrations:
                kwargs[name] = self.get(ann)

        instance = implementation(**kwargs)

        if reg.lifetime == Lifetime.SINGLETON:
            self._singletons[interface] = instance

        return instance

    @classmethod
    def create_default(cls, config: Any = None) -> "DIContainer":
        """Create container with default registrations."""
        container = cls()
        # Default registrations would go here.
        return container
'''

    def can_apply(self, file_path: str, **kwargs) -> bool:
        """Check if dependency injection can be applied."""
        try:
            content = Path(file_path).read_text(encoding="utf-8-sig")
            # Heuristic: self.x = SomeClass(...)
            return bool(re.search(r"self\.\w+\s*=\s*\w+\(", content))
        except Exception as e:
            logger.exception("Error checking if DI can be applied: %s", e)
            return False

    def apply(self, file_path: str, **kwargs) -> RefactoringResult:
        """
        Introduce DI patterns:
        - generates interfaces for given services
        - generates container.py
        - inserts interface imports into the file (simplified)
        """
        try:
            services = list(kwargs.get("services", []))
            target_directory = Path(
                kwargs.get("target_directory", Path(file_path).parent)
            )
            target_directory.mkdir(parents=True, exist_ok=True)

            created_files: List[str] = []

            for service_name in services:
                interface_code = self._create_service_interface(service_name)
                interface_path = (
                    target_directory / f"i_{_to_snake_case(service_name)}.py"
                )
                interface_path.write_text(interface_code, encoding="utf-8")
                created_files.append(str(interface_path))

            container_path = target_directory / "container.py"
            container_path.write_text(self.container_template, encoding="utf-8")
            created_files.append(str(container_path))

            self._modify_for_di(file_path, services)

            return RefactoringResult(
                success=True,
                files_created=created_files,
                files_modified=[file_path],
                files_deleted=[],
                warnings=[
                    "Please review constructor parameters and update service registrations"
                ],
            )

        except Exception as e:
            logger.exception("Error applying dependency injection: %s", e)
            return RefactoringResult(
                success=False,
                files_created=[],
                files_modified=[],
                files_deleted=[],
                error_message=str(e),
            )

    def _create_service_interface(self, service_name: str) -> str:
        """Create interface for a service."""
        return self.interface_template.format(service_name=service_name)

    def _modify_for_di(self, file_path: str, services: List[str]) -> None:
        """Insert interface imports into the file (simplified, text-based)."""
        content = Path(file_path).read_text(encoding="utf-8-sig")
        lines = content.splitlines()

        imports_to_add = [
            f"from .i_{_to_snake_case(service)} import I{service}"
            for service in services
        ]

        # Find last import line
        insert_at = 0
        for i, line in enumerate(lines):
            if line.strip().startswith(("import ", "from ")):
                insert_at = i + 1

        existing = {line.strip() for line in lines}
        for imp in imports_to_add:
            if imp not in existing:
                lines.insert(insert_at, imp)
                insert_at += 1

        Path(file_path).write_text("\n".join(lines) + "\n", encoding="utf-8-sig")

    def get_description(self) -> str:
        return (
            "Introduces dependency injection patterns with interfaces and a container"
        )


class FacadeCreator(RefactoringUtility):
    """Utility for creating facade patterns for backward compatibility."""

    def __init__(self) -> None:
        self.facade_template = '''"""
Facade for backward compatibility.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
{imports}

class {facade_class_name}:
    """Facade that maintains backward compatibility while using new architecture."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize facade with backward-compatible configuration."""
        new_config = self._convert_config(config)
        self.container = DIContainer.create_default(new_config)
{service_initializations}

{facade_methods}

    def _convert_config(self, old_config: Optional[Dict[str, Any]]) -> Any:
        """Convert old configuration format to new structure."""
        return old_config or {{}}
'''

    def can_apply(self, file_path: str, **kwargs) -> bool:
        """Facade is typically applied after major refactoring."""
        return bool(kwargs.get("major_refactoring_completed", False))

    def apply(self, file_path: str, **kwargs) -> RefactoringResult:
        """
        Create facade for backward compatibility.

        kwargs:
            facade_class_name: str
            services: List[str] service base names (e.g. ["UserService"])
            methods: List[str] facade method names
            target_directory: Path | str
        """
        try:
            facade_class_name = str(
                kwargs.get("facade_class_name", "BackwardCompatibleFacade")
            )
            services = list(kwargs.get("services", []))
            methods = list(kwargs.get("methods", []))
            target_directory = Path(
                kwargs.get("target_directory", Path(file_path).parent)
            )
            target_directory.mkdir(parents=True, exist_ok=True)

            imports = ["from .container import DIContainer"]
            service_inits: List[str] = []

            for service in services:
                imports.append(f"from .i_{_to_snake_case(service)} import I{service}")
                service_inits.append(
                    f"        self.{_to_snake_case(service)} = self.container.get(I{service})"
                )

            facade_methods = [self._create_facade_method(m) for m in methods]

            facade_code = self.facade_template.format(
                facade_class_name=facade_class_name,
                imports="\n".join(imports),
                service_initializations=(
                    "\n".join(service_inits) if service_inits else "        pass"
                ),
                facade_methods="\n".join(facade_methods).rstrip() + "\n",
            )

            facade_path = target_directory / f"{_to_snake_case(facade_class_name)}.py"
            facade_path.write_text(facade_code, encoding="utf-8")

            return RefactoringResult(
                success=True,
                files_created=[str(facade_path)],
                files_modified=[],
                files_deleted=[],
                warnings=[
                    "Please review facade method implementations and test backward compatibility"
                ],
            )

        except Exception as e:
            logger.exception("Error creating facade: %s", e)
            return RefactoringResult(
                success=False,
                files_created=[],
                files_modified=[],
                files_deleted=[],
                error_message=str(e),
            )

    def _create_facade_method(self, method_name: str) -> str:
        """Create a facade method stub."""
        return f'''    def {method_name}(self, *args: Any, **kwargs: Any) -> Any:
        """Facade method for {method_name}."""
        # Delegate to appropriate internal service(s)
        raise NotImplementedError
'''

    def get_description(self) -> str:
        return "Creates facade pattern for maintaining backward compatibility"


class RefactoringUtilityRegistry:
    """Registry of available refactoring utilities."""

    def __init__(self) -> None:
        self.utilities: Dict[str, RefactoringUtility] = {}
        self._register_default_utilities()

    def _register_default_utilities(self) -> None:
        self.register("component_extractor", ComponentExtractor())
        self.register("configuration_splitter", ConfigurationSplitter())
        self.register("dependency_injection", DependencyInjectionIntroducer())
        self.register("facade_creator", FacadeCreator())

    def register(self, name: str, utility: RefactoringUtility) -> None:
        """Register a refactoring utility."""
        if not name:
            raise ValueError("Utility name must be non-empty")
        self.utilities[name] = utility
        logger.info("Registered refactoring utility: %s", name)

    def get_utility(self, name: str) -> Optional[RefactoringUtility]:
        """Get a refactoring utility by name."""
        return self.utilities.get(name)

    def get_applicable_utilities(
        self, file_path: str, **kwargs
    ) -> List[Tuple[str, RefactoringUtility]]:
        """Get utilities that can be applied to the given file."""
        applicable: List[Tuple[str, RefactoringUtility]] = []
        for name, utility in self.utilities.items():
            try:
                if utility.can_apply(file_path, **kwargs):
                    applicable.append((name, utility))
            except Exception as e:
                logger.exception("Utility %s failed in can_apply(): %s", name, e)
        return applicable

    def list_utilities(self) -> Dict[str, str]:
        """List all registered utilities with their descriptions."""
        return {name: util.get_description() for name, util in self.utilities.items()}


_utility_registry: Optional[RefactoringUtilityRegistry] = None


def get_utility_registry() -> RefactoringUtilityRegistry:
    """Get the global utility registry."""
    global _utility_registry
    if _utility_registry is None:
        _utility_registry = RefactoringUtilityRegistry()
    return _utility_registry


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    registry = get_utility_registry()

    logger.info("Available refactoring utilities:")
    for name, desc in registry.list_utilities().items():
        logger.info("  %s: %s", name, desc)

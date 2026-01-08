"""
Auto Refactor - Automatic refactoring based on a knowledge base.

Safety-first refactoring tool:
- Detects God Objects (largest class with many methods).
- Groups public methods into responsibilities (keywords + optional cohesion clustering).
- Extracts methods into owner-proxy components (components operate on facade state).
- Generates a Facade class maintaining original public API and delegating to components.
- Optional automatic import/logging normalization using codebase analysis + learned patterns.

Important notes:
- Python 3.9+ required (uses ast.unparse and end_lineno).
- Extraction is conservative and skips methods with dangerous patterns by default.
"""

from __future__ import annotations

import ast
import json
import logging
import re
import shutil
import sys
import textwrap
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, NamedTuple, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)

# -----------------------------
# Optional integrations
# -----------------------------
try:
    from ..knowledge.import_fixing_patterns import ImportFixingPatterns, CodebaseAnalysisPatterns
    from ..knowledge.self_learning_patterns import get_learning_system

    IMPORT_FIXING_AVAILABLE = True
    SELF_LEARNING_AVAILABLE = True
except ImportError:
    IMPORT_FIXING_AVAILABLE = False
    SELF_LEARNING_AVAILABLE = False
    logger.warning("Import fixing patterns and/or self-learning not available")

# -----------------------------
# Constants / Defaults
# -----------------------------

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
        "True",
        "False",
        "None",
        "Ellipsis",
        "NotImplemented",
        "Exception",
        "BaseException",
        "ValueError",
        "TypeError",
        "KeyError",
        "AttributeError",
        "ImportError",
        "RuntimeError",
        "StopIteration",
        "OSError",
        "IOError",
        "FileNotFoundError",
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
        "Generator",
        "Coroutine",
        "Awaitable",
        "AsyncIterator",
        "AsyncGenerator",
        "ClassVar",
        "Final",
        "Literal",
        "TypeVar",
        "Generic",
        "Protocol",
        "FrozenSet",
        "DefaultDict",
        "Counter",
        "Deque",
        "ChainMap",
        "NamedTuple",
        "TypedDict",
        "Annotated",
        "Self",
        "Never",
        "NoReturn",
        "object",
        "type",
        "Exception",
        "BaseException",
    }
)

SAFE_COMPONENT_GLOBALS: FrozenSet[str] = frozenset({"logger"})

DEFAULT_COHESION_STOP_FEATURES: FrozenSet[str] = frozenset(
    {"config", "logger", "settings", "options", "args", "kwargs", "env", "context", "state", "data"}
)

# -----------------------------
# Types
# -----------------------------


class DecoratorType(Enum):
    NONE = auto()
    STATICMETHOD = auto()
    CLASSMETHOD = auto()
    PROPERTY = auto()


class DangerousPattern(Enum):
    SUPER = "super()"
    SELF_DICT = "self.__dict__"
    VARS_SELF = "vars(self)"
    TYPE_SELF = "type(self)"
    ISINSTANCE_SELF = "isinstance(self, ...)"
    SELF_CLASS = "self.__class__ / __class__"
    BARE_SELF = "bare self usage"
    MODULE_LEVEL_DEP = "module-level dependency"


class ImportInfo(NamedTuple):
    node: Union[ast.Import, ast.ImportFrom]
    names: Dict[str, Optional[str]]  # original_name -> alias (or None)
    is_relative: bool
    level: int
    module: Optional[str]

    def get_all_names(self) -> Set[str]:
        provided: Set[str] = set()
        for orig, alias in self.names.items():
            provided.add(alias if alias else orig)
        return provided


class MethodInfo(NamedTuple):
    node: Union[ast.FunctionDef, ast.AsyncFunctionDef]
    name: str
    is_async: bool
    decorator_type: DecoratorType
    called_methods: FrozenSet[str]
    used_attributes: FrozenSet[str]
    used_names: FrozenSet[str]
    dangerous_reasons: FrozenSet[str]
    module_level_deps: FrozenSet[str]
    bare_self_used: bool


@dataclass
class RefactoringPlan:
    target_file: str
    target_class_name: str
    transformations: List[str]
    extracted_components: List[str]
    new_files: List[str]
    estimated_effort: float
    risk_level: str
    backup_required: bool = True

    # cached data
    _cached_content: Optional[str] = field(default=None, repr=False)
    _cached_tree: Optional[ast.Module] = field(default=None, repr=False)
    _imports: List[ImportInfo] = field(default_factory=list, repr=False)
    _module_level_names: Set[str] = field(default_factory=set, repr=False)

    # extraction details
    _method_groups: Dict[str, List[MethodInfo]] = field(default_factory=dict, repr=False)
    _private_methods_by_group: Dict[str, List[MethodInfo]] = field(default_factory=dict, repr=False)
    _unextracted_methods: List[MethodInfo] = field(default_factory=list, repr=False)
    _init_method: Optional[MethodInfo] = field(default=None, repr=False)
    _dunder_methods: List[MethodInfo] = field(default_factory=list, repr=False)

    _dangerous_methods: Dict[str, Set[str]] = field(default_factory=dict, repr=False)


# -----------------------------
# AST analysis helpers
# -----------------------------


class ParentTrackingVisitor(ast.NodeVisitor):
    """NodeVisitor that keeps a stack to infer parent context."""

    def __init__(self) -> None:
        self._stack: List[ast.AST] = []

    def visit(self, node: ast.AST) -> Any:
        self._stack.append(node)
        try:
            return super().visit(node)
        finally:
            self._stack.pop()

    @property
    def parent(self) -> Optional[ast.AST]:
        if len(self._stack) < 2:
            return None
        return self._stack[-2]


class DependencyAnalyzer(ParentTrackingVisitor):
    """Extract method dependencies and dangerous patterns."""

    def __init__(self) -> None:
        super().__init__()
        self.read_attributes: Set[str] = set()
        self.written_attributes: Set[str] = set()
        self.deleted_attributes: Set[str] = set()
        self.called_methods: Set[str] = set()
        self.used_names: Set[str] = set()
        self.dangerous: Set[str] = set()
        self.bare_self_used: bool = False

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id == "super":
            self.dangerous.add(DangerousPattern.SUPER.value)

        if isinstance(node.func, ast.Name) and node.args:
            if node.func.id == "vars" and isinstance(node.args[0], ast.Name) and node.args[0].id == "self":
                self.dangerous.add(DangerousPattern.VARS_SELF.value)
            if node.func.id == "type" and isinstance(node.args[0], ast.Name) and node.args[0].id == "self":
                self.dangerous.add(DangerousPattern.TYPE_SELF.value)
            if node.func.id == "isinstance" and isinstance(node.args[0], ast.Name) and node.args[0].id == "self":
                self.dangerous.add(DangerousPattern.ISINSTANCE_SELF.value)

        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self"
        ):
            self.called_methods.add(node.func.attr)

        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id == "self":
            if node.attr == "__dict__":
                self.dangerous.add(DangerousPattern.SELF_DICT.value)
            if node.attr == "__class__":
                self.dangerous.add(DangerousPattern.SELF_CLASS.value)

            if isinstance(node.ctx, ast.Load):
                self.read_attributes.add(node.attr)
            elif isinstance(node.ctx, ast.Store):
                self.written_attributes.add(node.attr)
            elif isinstance(node.ctx, ast.Del):
                self.deleted_attributes.add(node.attr)
            return

        if isinstance(node.value, ast.Name) and node.value.id == "__class__":
            self.dangerous.add(DangerousPattern.SELF_CLASS.value)

        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            if node.id not in ("self", "cls"):
                self.used_names.add(node.id)
            elif node.id == "self":
                p = self.parent
                if not (isinstance(p, ast.Attribute) and p.value is node):
                    self.bare_self_used = True
                    self.dangerous.add(DangerousPattern.BARE_SELF.value)

        self.generic_visit(node)

    def get_state_usage(self) -> FrozenSet[str]:
        return frozenset(self.read_attributes | self.written_attributes | self.deleted_attributes)


def analyze_method(
    method: Union[ast.FunctionDef, ast.AsyncFunctionDef],
    module_level_names: Set[str],
    *,
    allow_bare_self: bool,
    allow_dangerous: bool,
    allow_module_level_deps: bool,
    decorated_extract_allowed: bool,
) -> MethodInfo:
    is_async = isinstance(method, ast.AsyncFunctionDef)

    decorator_type = DecoratorType.NONE
    for dec in method.decorator_list:
        if isinstance(dec, ast.Name):
            if dec.id == "staticmethod":
                decorator_type = DecoratorType.STATICMETHOD
                break
            if dec.id == "classmethod":
                decorator_type = DecoratorType.CLASSMETHOD
                break
            if dec.id == "property":
                decorator_type = DecoratorType.PROPERTY
                break

    analyzer = DependencyAnalyzer()
    analyzer.visit(method)

    filtered_names = frozenset(analyzer.used_names - BUILTINS - BUILTIN_TYPES)

    module_level_deps = (set(filtered_names) & set(module_level_names)) - set(SAFE_COMPONENT_GLOBALS)

    dangerous_reasons: Set[str] = set(analyzer.dangerous)
    if module_level_deps:
        dangerous_reasons.add(DangerousPattern.MODULE_LEVEL_DEP.value)

    if decorator_type != DecoratorType.NONE and not decorated_extract_allowed:
        dangerous_reasons.add(f"decorated:{decorator_type.name.lower()}")

    if analyzer.bare_self_used and not allow_bare_self:
        dangerous_reasons.add(DangerousPattern.BARE_SELF.value)

    # allow_dangerous / allow_module_level_deps flags are enforced by adding reasons above;
    # they control whether reasons block extraction in higher-level logic.

    return MethodInfo(
        node=method,
        name=method.name,
        is_async=is_async,
        decorator_type=decorator_type,
        called_methods=frozenset(analyzer.called_methods),
        used_attributes=analyzer.get_state_usage(),
        used_names=filtered_names,
        dangerous_reasons=frozenset(dangerous_reasons),
        module_level_deps=frozenset(module_level_deps),
        bare_self_used=analyzer.bare_self_used,
    )


class ImportCollector(ast.NodeVisitor):
    def __init__(self) -> None:
        self.imports: List[ImportInfo] = []

    def visit_Import(self, node: ast.Import) -> None:
        names: Dict[str, Optional[str]] = {}
        for alias in node.names:
            base = alias.name.split(".")[0]
            names[base] = alias.asname
        self.imports.append(ImportInfo(node=node, names=names, is_relative=False, level=0, module=None))

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        names: Dict[str, Optional[str]] = {}
        for alias in node.names:
            names[alias.name] = alias.asname
        self.imports.append(
            ImportInfo(
                node=node,
                names=names,
                is_relative=node.level > 0,
                level=node.level,
                module=node.module,
            )
        )


# -----------------------------
# AutoRefactor implementation
# -----------------------------


class AutoRefactor:
    def __init__(self, config: Optional[Any] = None):
        if config is not None and hasattr(config, "__dict__") and not isinstance(config, dict):
            self.config: Dict[str, Any] = {
                "safety_level": getattr(config, "safety_level", "moderate"),
                "auto_apply": getattr(config, "auto_apply", False),
                "backup_enabled": getattr(config, "backup_enabled", True),
                "validation_required": getattr(config, "validation_required", True),
            }
        else:
            self.config = config if isinstance(config, dict) else {}

        require_39 = bool(self.config.get("require_python39", True))
        if require_39 and sys.version_info < (3, 9):
            raise RuntimeError("AutoRefactor requires Python 3.9+ (uses ast.unparse).")

        self.output_directory = self.config.get("output_directory", "components")
        self.component_template = self.config.get("component_template", "Service")
        self.interface_prefix = self.config.get("interface_prefix", "I")
        self.preserve_original = bool(self.config.get("preserve_original", True))
        self.facade_suffix = self.config.get("facade_suffix", "_refactored")

        self.reserved_prefix = self.config.get("reserved_prefix", "__ar_")
        self._container_attr = f"{self.reserved_prefix}container"
        self._components_attr = f"{self.reserved_prefix}components"

        self.extract_decorated_public_methods = bool(self.config.get("extract_decorated_public_methods", False))
        self.extract_private_methods = bool(self.config.get("extract_private_methods", True))
        self.keep_private_methods_in_facade = bool(self.config.get("keep_private_methods_in_facade", True))

        self.skip_methods_with_module_level_deps = bool(self.config.get("skip_methods_with_module_level_deps", True))
        self.skip_methods_with_bare_self_usage = bool(self.config.get("skip_methods_with_bare_self_usage", True))
        self.skip_methods_with_dangerous_patterns = bool(self.config.get("skip_methods_with_dangerous_patterns", True))

        self.responsibility_keywords: Dict[str, List[str]] = self.config.get(
            "responsibility_keywords",
            {
                "console": ["print", "log", "display", "show", "console", "output", "render", "table", "progress"],
                "validation": ["validate", "verify", "check", "ensure", "is_valid", "normalize"],
                "analysis": ["analyze", "parse", "examine", "inspect", "scan", "stats", "metric"],
                "export": ["export", "dump", "serialize", "write", "save", "json", "csv"],
                "storage": ["load", "read", "file", "path", "store", "cache", "persist", "db"],
                "network": ["request", "connect", "http", "api", "fetch", "download", "send", "receive"],
                "config": ["config", "setting", "option", "setup", "init", "env"],
                "utility": ["util", "helper", "format", "convert", "transform", "build"],
            },
        )

        self.cohesion_cluster_other = bool(self.config.get("cohesion_cluster_other", True))
        self.cohesion_similarity_threshold = float(self.config.get("cohesion_similarity_threshold", 0.30))
        self.cohesion_stop_features: FrozenSet[str] = frozenset(
            self.config.get("cohesion_stop_features", list(DEFAULT_COHESION_STOP_FEATURES))
        )

        self.god_class_threshold = self._validate_positive_int(
            self.config.get("god_class_threshold", 10),
            "god_class_threshold",
            10,
        )
        self.min_methods_for_extraction = self._validate_positive_int(
            self.config.get("min_methods_for_extraction", 2),
            "min_methods_for_extraction",
            2,
        )

        self.effort_per_component = float(self.config.get("effort_per_component", 2.5))
        self.base_effort = float(self.config.get("base_effort", 4.0))

        self._created_files: List[Path] = []
        self._original_filepath: Optional[Path] = None
        self._backup_path: Optional[Path] = None

        self._codebase_analysis: Optional[Dict[str, Any]] = None
        if IMPORT_FIXING_AVAILABLE:
            self._analyze_codebase_standards()

    def _validate_positive_int(self, value: Any, name: str, default: int) -> int:
        try:
            v = int(value)
            return v if v > 0 else default
        except (TypeError, ValueError):
            logger.debug("Invalid int for %s=%r, using default=%d", name, value, default)
            return default

    def _analyze_codebase_standards(self) -> None:
        """Analyze codebase (limited) to infer logging/import conventions."""
        try:
            current_dir = Path.cwd()
            python_files = list(current_dir.rglob("*.py"))[:50]

            if not python_files:
                logger.warning("No Python files found for codebase analysis")
                return

            file_contents: List[str] = []
            for py_file in python_files:
                try:
                    file_contents.append(py_file.read_text(encoding="utf-8-sig"))
                except Exception as e:
                    logger.debug("Failed to read %s: %s", py_file, e)

            if not file_contents:
                logger.warning("No readable Python files found for analysis")
                return

            logging_standard = CodebaseAnalysisPatterns.recommend_logging_standard(file_contents)
            existing_modules = CodebaseAnalysisPatterns.find_existing_modules(file_contents)

            self._codebase_analysis = {
                "logging_standard": logging_standard,
                "existing_modules": set(existing_modules),
                "analyzed_files": len(file_contents),
            }

            logger.info(
                "Analyzed %d files, recommended logging standard: %s",
                len(file_contents),
                logging_standard,
            )

        except Exception as e:
            logger.warning("Failed to analyze codebase standards: %s", e)
            self._codebase_analysis = None

    # -----------------------------
    # Public API
    # -----------------------------

    def analyze_god_object(self, filepath: Path) -> RefactoringPlan:
        logger.info("Analyzing %s", filepath)

        try:
            content = filepath.read_text(encoding="utf-8-sig")
            tree = ast.parse(content)
        except Exception as e:
            logger.error("Failed to read/parse %s: %s", filepath, e)
            return self._create_empty_plan(filepath)

        main_class, max_methods = self._find_largest_top_level_class(tree)
        if not main_class or max_methods < self.god_class_threshold:
            logger.info("No God Object found (max methods: %s)", max_methods)
            return self._create_empty_plan(filepath)

        import_collector = ImportCollector()
        import_collector.visit(tree)
        module_level_names = self._collect_module_level_names(tree)

        (
            method_groups,
            private_by_group,
            unextracted,
            init_method,
            dunder_methods,
            dangerous_methods,
        ) = self._group_methods_by_responsibility(main_class, module_level_names)

        extracted_components: List[str] = []
        new_files: List[str] = []
        transformations: List[str] = []

        for group_name, methods in method_groups.items():
            extractable_public = [m for m in methods if self._is_public_extractable(m)]
            if len(extractable_public) >= self.min_methods_for_extraction:
                comp_name = self._component_class_name(group_name)
                extracted_components.append(comp_name)
                new_files.append(
                    str(Path(self.output_directory) / f"{group_name}_{self.component_template.lower()}.py")
                )

                priv_count = len(private_by_group.get(group_name, []))
                transformations.append(
                    f"Extract {len(extractable_public)} public + {priv_count} private methods to {comp_name}"
                )

        if not extracted_components:
            return self._create_empty_plan(filepath)

        if unextracted:
            remaining_public = [m for m in unextracted if not m.name.startswith("_")]
            if remaining_public:
                transformations.append(f"Keep {len(remaining_public)} methods in facade")

        estimated_effort = len(extracted_components) * self.effort_per_component + self.base_effort
        risk_level = self._assess_risk(extracted_components, method_groups, private_by_group, dangerous_methods)

        return RefactoringPlan(
            target_file=str(filepath),
            target_class_name=main_class.name,
            transformations=transformations,
            extracted_components=extracted_components,
            new_files=new_files,
            estimated_effort=estimated_effort,
            risk_level=risk_level,
            _cached_content=content,
            _cached_tree=tree,
            _imports=import_collector.imports,
            _module_level_names=module_level_names,
            _method_groups=method_groups,
            _private_methods_by_group=private_by_group,
            _unextracted_methods=unextracted,
            _init_method=init_method,
            _dunder_methods=dunder_methods,
            _dangerous_methods=dangerous_methods,
        )

    def refactor_project(
        self,
        project_path: Union[str, Path],
        strategy: Optional[str] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        _ = strategy  # reserved for future extensions
        _ = dry_run   # currently only analysis is performed here

        project_path = Path(project_path)
        results: Dict[str, Any] = {
            "success": True,
            "operations_applied": 0,
            "changes": [],
            "errors": [],
            "warnings": [],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        if not project_path.exists():
            results["success"] = False
            results["errors"].append(f"Project path does not exist: {project_path}")
            return results

        try:
            for p in project_path.rglob("*.py"):
                try:
                    plan = self.analyze_god_object(p)
                    if plan.transformations:
                        results["changes"].append(
                            {
                                "file": str(p),
                                "target_class": plan.target_class_name,
                                "transformations": plan.transformations,
                                "new_files": plan.new_files,
                                "estimated_effort": plan.estimated_effort,
                                "risk_level": plan.risk_level,
                            }
                        )
                except Exception as e:
                    results["warnings"].append(f"Failed to analyze {p}: {e}")

            results["operations_applied"] = len(results["changes"])
            return results
        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
            return results

    def apply_opportunity(self, opportunity: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
        try:
            description = opportunity.get("description", "Unknown opportunity")
            priority = opportunity.get("priority", 0)
            filepath = opportunity.get("filepath")

            logger.info("Applying opportunity: %s (Priority: %s)", description, priority)

            if not filepath:
                return {"success": False, "error": "No filepath specified", "operations_applied": 0, "changes_made": []}

            fp = Path(filepath) if isinstance(filepath, str) else filepath
            if not fp.exists():
                return {"success": False, "error": f"File not found: {fp}", "operations_applied": 0, "changes_made": []}

            plan = self.analyze_god_object(fp)
            if not plan.transformations:
                return {"success": True, "message": "No refactoring needed", "operations_applied": 0, "changes_made": []}

            results = self.execute_refactoring(fp, plan, dry_run=dry_run)
            return {
                "success": results.get("success", False),
                "message": results.get("message", ""),
                "operations_applied": len(plan.transformations),
                "changes_made": results.get("files_created", []) + results.get("files_modified", []),
                "validation_results": results.get("validation", {}),
                "errors": results.get("errors", []),
                "warnings": results.get("warnings", []),
            }

        except Exception as e:
            logger.error("Failed to apply opportunity: %s", e)
            return {"success": False, "error": str(e), "operations_applied": 0, "changes_made": []}

    def execute_refactoring(self, filepath: Path, plan: RefactoringPlan, dry_run: bool = True) -> Dict[str, Any]:
        results: Dict[str, Any] = {
            "success": False,
            "files_created": [],
            "files_modified": [],
            "backup_created": None,
            "errors": [],
            "warnings": [],
        }

        if not plan.transformations or not plan.extracted_components:
            results["success"] = True
            return results

        self._original_filepath = filepath
        output_dir = filepath.parent / self.output_directory

        try:
            if dry_run:
                return self._execute_dry_run(filepath, plan, output_dir, results)

            if plan.backup_required and bool(self.config.get("backup_enabled", True)):
                backup_path = filepath.parent / f"{filepath.stem}.backup_{int(time.time())}{filepath.suffix}"
                shutil.copy2(filepath, backup_path)
                self._backup_path = backup_path
                results["backup_created"] = str(backup_path)
                logger.info("Backup created: %s", backup_path)

            with self._atomic_write_session():
                output_dir.mkdir(parents=True, exist_ok=True)

                base_code = self._generate_owner_proxy_base()
                self._write_validated(output_dir / "base.py", base_code, results)

                interface_classes: List[str] = []
                method_map: Dict[str, Tuple[str, bool, MethodInfo]] = {}

                for group_name, methods in plan._method_groups.items():
                    extractable_public = [m for m in methods if self._is_public_extractable(m)]
                    if len(extractable_public) < self.min_methods_for_extraction:
                        continue

                    component_name = self._component_class_name(group_name)

                    private_methods = plan._private_methods_by_group.get(group_name, []) if self.extract_private_methods else []

                    for m in extractable_public:
                        method_map[m.name] = (component_name, m.is_async, m)

                    iface_code = self._generate_interface(component_name, extractable_public)
                    if iface_code:
                        interface_classes.append(iface_code)

                    impl_code = self._generate_component_implementation(
                        component_name=component_name,
                        group_name=group_name,
                        public_methods=extractable_public,
                        private_methods=private_methods,
                        plan=plan,
                    )
                    impl_file = output_dir / f"{group_name}_{self.component_template.lower()}.py"
                    self._write_validated(impl_file, impl_code, results)

                # IMPORTANT: interfaces.py must exist if components import it.
                iface_content = self._generate_interfaces_file(interface_classes, plan)
                self._write_validated(output_dir / "interfaces.py", iface_content, results)

                container_code = self._generate_di_container(plan.extracted_components)
                self._write_validated(output_dir / "container.py", container_code, results)

                init_code = self._generate_package_init(plan.extracted_components, has_interfaces=True)
                self._write_validated(output_dir / "__init__.py", init_code, results)

                facade_code = self._create_facade(plan.target_class_name, plan.extracted_components, method_map, plan)

                if self.preserve_original:
                    facade_file = filepath.with_name(f"{filepath.stem}{self.facade_suffix}{filepath.suffix}")
                    self._write_validated(facade_file, facade_code, results)
                    results["warnings"].append(f"Original preserved. New facade: {facade_file.name}")
                    self._validate_refactored_file(facade_file, results)
                else:
                    self._write_file_direct(filepath, facade_code, results)
                    results["files_modified"].append(str(filepath))
                    results["warnings"].append(f"Original replaced. Backup: {results['backup_created']}")
                    self._validate_refactored_file(filepath, results)

            results["success"] = len(results["errors"]) == 0
            return results

        except Exception as e:
            results["errors"].append(f"Refactoring failed: {e}")
            logger.exception("Refactoring failed")
            return results

    # -----------------------------
    # Grouping / Extraction policy
    # -----------------------------

    def _find_largest_top_level_class(self, tree: ast.Module) -> Tuple[Optional[ast.ClassDef], int]:
        main_class: Optional[ast.ClassDef] = None
        max_methods = 0
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if len(methods) > max_methods:
                    max_methods = len(methods)
                    main_class = node
        return main_class, max_methods

    def _collect_module_level_names(self, tree: ast.Module) -> Set[str]:
        names: Set[str] = set()
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    if isinstance(t, ast.Name):
                        names.add(t.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                names.add(node.name)
        return names

    def _is_public_extractable(self, m: MethodInfo) -> bool:
        return (not m.name.startswith("_")) and (len(m.dangerous_reasons) == 0)

    def _group_methods_by_responsibility(
        self, class_node: ast.ClassDef, module_level_names: Set[str]
    ) -> Tuple[
        Dict[str, List[MethodInfo]],
        Dict[str, List[MethodInfo]],
        List[MethodInfo],
        Optional[MethodInfo],
        List[MethodInfo],
        Dict[str, Set[str]],
    ]:
        init_method: Optional[MethodInfo] = None
        dunder_methods: List[MethodInfo] = []
        all_methods: List[MethodInfo] = []

        allow_bare_self = not self.skip_methods_with_bare_self_usage
        allow_dangerous = not self.skip_methods_with_dangerous_patterns
        allow_module_level = not self.skip_methods_with_module_level_deps

        dangerous_methods: Dict[str, Set[str]] = {}

        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                info = analyze_method(
                    node,
                    module_level_names=module_level_names,
                    allow_bare_self=allow_bare_self,
                    allow_dangerous=allow_dangerous,
                    allow_module_level_deps=allow_module_level,
                    decorated_extract_allowed=self.extract_decorated_public_methods,
                )

                if info.dangerous_reasons:
                    dangerous_methods[info.name] = set(info.dangerous_reasons)

                if node.name == "__init__":
                    init_method = info
                elif node.name.startswith("__") and node.name.endswith("__"):
                    dunder_methods.append(info)
                else:
                    all_methods.append(info)

        public_methods = [m for m in all_methods if not m.name.startswith("_")]
        private_methods = [m for m in all_methods if m.name.startswith("_") and not m.name.startswith("__")]

        groups: Dict[str, List[MethodInfo]] = {k: [] for k in self.responsibility_keywords}
        other: List[MethodInfo] = []

        for m in public_methods:
            name = m.name.lower()
            scores: Dict[str, int] = {}
            for group, words in self.responsibility_keywords.items():
                score = sum(1 for w in words if w in name)
                if score:
                    scores[group] = score
            if scores:
                best = max(scores, key=lambda k: scores[k])
                groups[best].append(m)
            else:
                other.append(m)

        if other:
            if self.cohesion_cluster_other and len(other) >= self.min_methods_for_extraction * 2:
                clusters = self._cluster_methods_by_cohesion(other, self.cohesion_similarity_threshold)
                for idx, cluster in enumerate(clusters, start=1):
                    groups[f"misc_{idx}"] = cluster
            else:
                groups["other"] = other

        groups = {k: v for k, v in groups.items() if v}

        extract_groups: Set[str] = set()
        for g, ms in groups.items():
            extractable = [m for m in ms if self._is_public_extractable(m)]
            if len(extractable) >= self.min_methods_for_extraction:
                extract_groups.add(g)

        unextracted: List[MethodInfo] = []
        for g in list(groups.keys()):
            if g not in extract_groups:
                unextracted.extend(groups[g])
                del groups[g]

        private_by_group = self._assign_private_to_groups(groups, private_methods)

        assigned_private: Set[str] = set()
        for ms in private_by_group.values():
            assigned_private.update(m.name for m in ms)

        for pm in private_methods:
            if pm.name not in assigned_private:
                unextracted.append(pm)

        return groups, private_by_group, unextracted, init_method, dunder_methods, dangerous_methods

    def _cluster_methods_by_cohesion(self, methods: List[MethodInfo], threshold: float) -> List[List[MethodInfo]]:
        feats: Dict[str, Set[str]] = {}
        for m in methods:
            f = set(m.used_attributes) | set(m.called_methods)
            f -= set(self.cohesion_stop_features)
            feats[m.name] = f

        def jaccard(a: Set[str], b: Set[str]) -> float:
            u = a | b
            return 0.0 if not u else (len(a & b) / len(u))

        names = [m.name for m in methods]
        adj: Dict[str, Set[str]] = {n: set() for n in names}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if jaccard(feats[a], feats[b]) >= threshold:
                    adj[a].add(b)
                    adj[b].add(a)

        seen: Set[str] = set()
        clusters: List[List[str]] = []

        for n in names:
            if n in seen:
                continue
            stack = [n]
            comp: List[str] = []
            seen.add(n)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in adj[cur]:
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            clusters.append(comp)

        clusters.sort(key=len, reverse=True)
        by_name = {m.name: m for m in methods}
        return [[by_name[n] for n in cluster] for cluster in clusters]

    def _assign_private_to_groups(
        self,
        public_groups: Dict[str, List[MethodInfo]],
        private_methods: List[MethodInfo],
    ) -> Dict[str, List[MethodInfo]]:
        private_names = {m.name for m in private_methods}
        private_by_name = {m.name: m for m in private_methods}

        def safe_private(m: MethodInfo) -> bool:
            return len(m.dangerous_reasons) == 0

        result: Dict[str, Set[str]] = {g: set() for g in public_groups}

        for g, methods in public_groups.items():
            for m in methods:
                called_priv = set(m.called_methods) & private_names
                for pn in called_priv:
                    pm = private_by_name.get(pn)
                    if pm and safe_private(pm):
                        result[g].add(pn)

        changed = True
        limit = len(private_methods) + 1
        it = 0
        while changed and it < limit:
            it += 1
            changed = False
            for g, assigned in result.items():
                new_calls: Set[str] = set()
                for pn in list(assigned):
                    pm = private_by_name.get(pn)
                    if not pm:
                        continue
                    trans = (set(pm.called_methods) & private_names) - assigned
                    for t in trans:
                        tm = private_by_name.get(t)
                        if tm and safe_private(tm):
                            new_calls.add(t)
                if new_calls:
                    assigned |= new_calls
                    changed = True

        return {g: [private_by_name[n] for n in sorted(ns)] for g, ns in result.items() if ns}

    def _assess_risk(
        self,
        components: List[str],
        method_groups: Dict[str, List[MethodInfo]],
        private_by_group: Dict[str, List[MethodInfo]],
        dangerous_methods: Dict[str, Set[str]],
    ) -> str:
        if len(components) > 8:
            return "high"

        dangerous_count = len(dangerous_methods)

        all_private_names: Set[str] = set()
        for ms in private_by_group.values():
            all_private_names.update(m.name for m in ms)

        cross_deps = 0
        for g, ms in method_groups.items():
            group_priv = {m.name for m in private_by_group.get(g, [])}
            for m in ms:
                external = (set(m.called_methods) & all_private_names) - group_priv
                cross_deps += len(external)

        if dangerous_count > 10 or cross_deps > 6 or len(components) > 5:
            return "high"
        if dangerous_count > 0 or cross_deps > 0 or len(components) > 2:
            return "medium"
        return "low"

    # -----------------------------
    # Naming helpers (CRITICAL for module imports)
    # -----------------------------

    def _component_class_name(self, group_name: str) -> str:
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
        return f"{camel}{self.component_template}"

    def _get_group_name(self, component_name: str) -> str:
        """
        Convert component class name to a stable group/module key.

        Must match filenames produced elsewhere:
          group_name + "_" + component_template.lower() + ".py"

        Example:
          Misc1Service -> misc_1
          ConsoleService -> console
        """
        suffix = self.component_template
        base = component_name[:-len(suffix)] if component_name.endswith(suffix) else component_name

        # CamelCase -> snake_case
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", base)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        # Insert '_' between letters and digits: misc1 -> misc_1
        s3 = re.sub(r"(?<=\D)(?=\d)", "_", s2)
        return s3.lower()

    # -----------------------------
    # Code generation
    # -----------------------------

    def _unparse_safe(self, node: Optional[ast.AST], default: str = "Any") -> str:
        if node is None:
            return default
        try:
            return ast.unparse(node)
        except Exception:
            return default

    def _format_arg(self, arg: ast.arg) -> str:
        if arg.annotation:
            return f"{arg.arg}: {self._unparse_safe(arg.annotation, 'Any')}"
        return f"{arg.arg}: Any"

    def _generate_method_signature(self, method: MethodInfo) -> str:
        node = method.node
        args = node.args
        parts: List[str] = []

        if method.decorator_type == DecoratorType.STATICMETHOD:
            first_param: Optional[str] = None
        elif method.decorator_type == DecoratorType.CLASSMETHOD:
            first_param = "cls"
        else:
            first_param = "self"

        posonlyargs = getattr(args, "posonlyargs", [])
        for i, a in enumerate(posonlyargs):
            if i == 0 and a.arg in ("self", "cls"):
                continue
            parts.append(self._format_arg(a))
        if posonlyargs:
            non_self = [a for a in posonlyargs if a.arg not in ("self", "cls")]
            if non_self:
                parts.append("/")

        num_defaults = len(args.defaults)
        num_args = len(args.args)

        for i, a in enumerate(args.args):
            if i == 0 and a.arg in ("self", "cls"):
                continue

            arg_str = self._format_arg(a)
            default_idx = i - (num_args - num_defaults)
            if 0 <= default_idx < num_defaults:
                arg_str += f" = {self._unparse_safe(args.defaults[default_idx], '...')}"
            parts.append(arg_str)

        if args.vararg:
            var = f"*{args.vararg.arg}"
            if args.vararg.annotation:
                var += f": {self._unparse_safe(args.vararg.annotation, 'Any')}"
            parts.append(var)
        elif args.kwonlyargs:
            parts.append("*")

        for i, a in enumerate(args.kwonlyargs):
            arg_str = self._format_arg(a)
            if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
                arg_str += f" = {self._unparse_safe(args.kw_defaults[i], '...')}"
            parts.append(arg_str)

        if args.kwarg:
            kw = f"**{args.kwarg.arg}"
            if args.kwarg.annotation:
                kw += f": {self._unparse_safe(args.kwarg.annotation, 'Any')}"
            parts.append(kw)

        args_str = ", ".join(parts)
        return_type = self._unparse_safe(node.returns, "Any") if node.returns else "Any"
        prefix = "async " if method.is_async else ""

        if first_param:
            return f"{prefix}def {node.name}({first_param}{', ' if args_str else ''}{args_str}) -> {return_type}:"
        return f"{prefix}def {node.name}({args_str}) -> {return_type}:"

    def _build_call_arguments(self, args: ast.arguments) -> str:
        parts: List[str] = []
        posonlyargs = getattr(args, "posonlyargs", [])
        for i, a in enumerate(posonlyargs):
            if i == 0 and a.arg in ("self", "cls"):
                continue
            parts.append(a.arg)

        start_idx = 0
        if args.args and args.args[0].arg in ("self", "cls"):
            start_idx = 1
        for a in args.args[start_idx:]:
            parts.append(a.arg)

        if args.vararg:
            parts.append(f"*{args.vararg.arg}")

        for a in args.kwonlyargs:
            parts.append(f"{a.arg}={a.arg}")

        if args.kwarg:
            parts.append(f"**{args.kwarg.arg}")

        return ", ".join(parts)

    def _generate_import_statement(self, import_info: ImportInfo, needed_names: Set[str], level_adjustment: int = 1) -> Optional[str]:
        provided = import_info.get_all_names()
        if provided.isdisjoint(needed_names):
            return None

        node = import_info.node
        if isinstance(node, ast.Import):
            return ast.unparse(node)

        new_level = import_info.level + (level_adjustment if import_info.is_relative else 0)

        new_names: List[ast.alias] = []
        for orig, alias in import_info.names.items():
            use = alias if alias else orig
            if use in needed_names or orig in needed_names:
                new_names.append(ast.alias(name=orig, asname=alias))

        if not new_names:
            return None

        new_node = ast.ImportFrom(module=import_info.module, names=new_names, level=new_level)
        return ast.unparse(new_node)

    def _extract_node_code(self, node: ast.AST, content: str, start_line: Optional[int] = None) -> str:
        lines = content.splitlines()
        if start_line is None:
            start_line = getattr(node, "lineno", 1) - 1

        end = getattr(node, "end_lineno", None)
        if end is not None:
            return "\n".join(lines[start_line:end])

        return "\n".join(lines[start_line:])

    def _extract_method_code(self, method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef], content: str) -> str:
        start = method_node.lineno - 1
        if method_node.decorator_list:
            start = min(d.lineno for d in method_node.decorator_list) - 1
        return self._extract_node_code(method_node, content, start_line=start)

    def _generate_owner_proxy_base(self) -> str:
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

    def _generate_component_implementation(
        self,
        *,
        component_name: str,
        group_name: str,
        public_methods: List[MethodInfo],
        private_methods: List[MethodInfo],
        plan: RefactoringPlan,
    ) -> str:
        content = plan._cached_content or ""

        used: Set[str] = set()
        for m in public_methods + private_methods:
            used |= set(m.used_names)

        used -= set(SAFE_COMPONENT_GLOBALS)
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

        added: Set[str] = set()
        for imp in plan._imports:
            stmt = self._generate_import_statement(imp, needed_imports, level_adjustment=1)
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

        for m in public_methods + private_methods:
            code = self._extract_method_code(m.node, content)
            if code:
                lines.append(textwrap.indent(textwrap.dedent(code), "    "))
                lines.append("")

        return "\n".join(lines)

    def _generate_interface(self, component_name: str, public_methods: List[MethodInfo]) -> str:
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
            sig = self._generate_method_signature(m)
            lines += [
                f"    {sig}",
                '        """TODO: Add documentation."""',
                "        ...",
                "",
            ]
        return "\n".join(lines)

    def _generate_interfaces_file(self, interface_classes: List[str], plan: RefactoringPlan) -> str:
        _ = plan  # reserved for future type import propagation
        lines: List[str] = [
            "from __future__ import annotations",
            "",
            '"""Auto-generated interfaces."""',
            "",
            "from abc import ABC, abstractmethod",
            "from typing import Any",
            "",
        ]
        lines.append("\n\n".join(interface_classes) if interface_classes else "# No interfaces generated\n")
        return "\n".join(lines)

    def _generate_di_container(self, components: List[str]) -> str:
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
            group = self._get_group_name(comp)
            lines.append(f"from .{group}_{self.component_template.lower()} import {comp}")
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

    def _generate_package_init(self, components: List[str], has_interfaces: bool) -> str:
        lines = ['"""Auto-generated components package."""', ""]
        lines.append("from .container import DIContainer")
        lines.append("")
        for comp in components:
            group = self._get_group_name(comp)
            lines.append(f"from .{group}_{self.component_template.lower()} import {comp}")
        if has_interfaces:
            lines.append("")
            lines.append("from .interfaces import *  # noqa: F401,F403")

        exports = ["DIContainer"] + components
        lines.append("")
        lines.append(f"__all__ = {exports!r}")
        lines.append("")
        return "\n".join(lines)

    # -----------------------------
    # Facade generation
    # -----------------------------

    def _create_facade(
        self,
        original_class_name: str,
        components: List[str],
        method_map: Dict[str, Tuple[str, bool, MethodInfo]],
        plan: RefactoringPlan,
    ) -> str:
        content = plan._cached_content or ""
        tree = plan._cached_tree
        if not content or not tree:
            return content

        original_lines = content.splitlines()

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

        out: List[str] = []
        out.extend(original_lines[:start_idx])

        out.append("")
        out.extend(self._generate_facade_import_block(components))
        out.append("")

        out.append(f"class {original_class_name}:")
        out.append('    """Facade maintaining backward compatibility (auto-generated)."""')
        out.append("")

        for node in main_class_node.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.ClassDef)):
                code = self._extract_node_code(node, content)
                if code:
                    out.append(textwrap.indent(textwrap.dedent(code), "    "))
                    out.append("")
                continue

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = node.name

                if name == "__init__":
                    self._create_enhanced_init_method(node, content, out, components)
                elif name in method_map:
                    comp, _is_async, mi = method_map[name]
                    self._create_active_delegation_method(mi, comp, out)
                else:
                    code = self._extract_method_code(node, content)
                    if code:
                        out.append(textwrap.indent(textwrap.dedent(code), "    "))
                        out.append("")

        if end_idx < len(original_lines):
            out.append("")
            out.extend(original_lines[end_idx:])

        return "\n".join(out)

    def _generate_facade_import_block(self, components: List[str]) -> List[str]:
        pkg = self.output_directory
        iface_names = ", ".join(f"{self.interface_prefix}{c}" for c in components)
        return [
            "try:",
            f"    from .{pkg}.container import DIContainer",
            f"    from .{pkg}.interfaces import {iface_names}",
            "except ImportError:  # pragma: no cover",
            f"    from {pkg}.container import DIContainer",
            f"    from {pkg}.interfaces import {iface_names}",
        ]

    def _create_enhanced_init_method(
        self,
        init_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        content: str,
        out_lines: List[str],
        components: List[str],
    ) -> None:
        init_code = self._extract_method_code(init_node, content)
        ded = textwrap.dedent(init_code).splitlines()

        for line in ded:
            out_lines.append(f"    {line}")
        out_lines.append("")
        out_lines.append("        # [AutoRefactor] Initialize DI container and owner-proxy components")
        out_lines.append(f"        self.{self._container_attr} = DIContainer.create_default()")
        out_lines.append(f"        self.{self._components_attr} = {{}}")
        out_lines.append("")

        for comp in components:
            group = self._get_group_name(comp)
            iface = f"{self.interface_prefix}{comp}"
            out_lines.append(
                f"        self.{self._components_attr}[{group!r}] = "
                f"self.{self._container_attr}.get({iface}, owner=self)"
            )

        out_lines.append("")

    def _create_active_delegation_method(self, method: MethodInfo, component: str, out_lines: List[str]) -> None:
        group = self._get_group_name(component)

        for dec in method.node.decorator_list:
            dec_str = self._unparse_safe(dec, "")
            if dec_str:
                out_lines.append(f"    @{dec_str}")

        sig = self._generate_method_signature(method)
        call_args = self._build_call_arguments(method.node.args)
        await_kw = "await " if method.is_async else ""

        out_lines.append(f"    {sig}")
        out_lines.append(f'        """Delegates to {component}.{method.name} (auto-generated)."""')
        out_lines.append(
            f"        return {await_kw}self.{self._components_attr}[{group!r}].{method.name}({call_args})"
        )
        out_lines.append("")

    # -----------------------------
    # Dry run / Validation / IO safety
    # -----------------------------

    def _execute_dry_run(self, filepath: Path, plan: RefactoringPlan, output_dir: Path, results: Dict[str, Any]) -> Dict[str, Any]:
        if not plan.extracted_components:
            results["success"] = True
            return results

        interface_classes: List[str] = []
        method_map: Dict[str, Tuple[str, bool, MethodInfo]] = {}

        base_code = self._generate_owner_proxy_base()
        if self._validate_syntax(base_code, output_dir / "base.py", results):
            results["files_created"].append(str(output_dir / "base.py"))

        for group_name, methods in plan._method_groups.items():
            extractable_public = [m for m in methods if self._is_public_extractable(m)]
            if len(extractable_public) < self.min_methods_for_extraction:
                continue

            comp_name = self._component_class_name(group_name)
            private_methods = plan._private_methods_by_group.get(group_name, []) if self.extract_private_methods else []

            for m in extractable_public:
                method_map[m.name] = (comp_name, m.is_async, m)

            iface = self._generate_interface(comp_name, extractable_public)
            if iface:
                interface_classes.append(iface)

            impl = self._generate_component_implementation(
                component_name=comp_name,
                group_name=group_name,
                public_methods=extractable_public,
                private_methods=private_methods,
                plan=plan,
            )
            impl_file = output_dir / f"{group_name}_{self.component_template.lower()}.py"
            if self._validate_syntax(impl, impl_file, results):
                results["files_created"].append(str(impl_file))

        iface_content = self._generate_interfaces_file(interface_classes, plan)
        if self._validate_syntax(iface_content, output_dir / "interfaces.py", results):
            results["files_created"].append(str(output_dir / "interfaces.py"))

        container = self._generate_di_container(plan.extracted_components)
        if self._validate_syntax(container, output_dir / "container.py", results):
            results["files_created"].append(str(output_dir / "container.py"))

        init_code = self._generate_package_init(plan.extracted_components, has_interfaces=True)
        if self._validate_syntax(init_code, output_dir / "__init__.py", results):
            results["files_created"].append(str(output_dir / "__init__.py"))

        facade_code = self._create_facade(plan.target_class_name, plan.extracted_components, method_map, plan)
        facade_file = filepath.with_name(f"{filepath.stem}{self.facade_suffix}{filepath.suffix}")
        if self._validate_syntax(facade_code, facade_file, results):
            results["files_created"].append(str(facade_file))
            self._validate_generated_facade(facade_code, facade_file, results)

        results["success"] = len(results["errors"]) == 0
        return results

    def _validate_syntax(self, code: str, path: Path, results: Dict[str, Any]) -> bool:
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            results["errors"].append(f"Invalid syntax in {path.name} at line {e.lineno}: {e.msg}")
            return False

    def _validate_generated_facade(self, code: str, path: Path, results: Dict[str, Any]) -> None:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        results["errors"].append(
                            f"{path.name}: nested function {child.name} inside __init__ (likely generation bug)"
                        )

    def _validate_refactored_file(self, filepath: Path, results: Dict[str, Any]) -> bool:
        try:
            if not filepath.exists():
                results["errors"].append(f"Refactored file not found: {filepath}")
                return False

            content = filepath.read_text(encoding="utf-8-sig")
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                results["errors"].append(f"Syntax error in {filepath.name}: {e}")
                return False

            self._validate_generated_facade(content, filepath, results)

            import_errors = self._validate_generated_package_files(filepath, tree)
            if import_errors:
                results["errors"].extend([f"{filepath.name}: {msg}" for msg in import_errors])
                return False

            results.setdefault("validation", {})[str(filepath)] = "PASSED"
            return True

        except Exception as e:
            results["errors"].append(f"Validation error for {filepath}: {e}")
            return False

    def _validate_generated_package_files(self, facade_path: Path, tree: ast.AST) -> List[str]:
        """
        Validate that facade imports point to files that exist (real run).

        We can’t fully import modules safely here, but we can at least check that
        components/{container.py,interfaces.py,__init__.py} are present if imported.
        """
        errors: List[str] = []
        output_dir = facade_path.parent / self.output_directory

        # If output dir doesn't exist, skip (e.g. in some dry-run contexts).
        if not output_dir.exists():
            return errors

        expected = {
            "container.py": output_dir / "container.py",
            "interfaces.py": output_dir / "interfaces.py",
            "__init__.py": output_dir / "__init__.py",
        }

        imports_components_pkg = False
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.endswith(f"{self.output_directory}.container") or node.module.endswith(f"{self.output_directory}.interfaces"):
                    imports_components_pkg = True
                    break

        if imports_components_pkg:
            for name, path in expected.items():
                if not path.exists():
                    errors.append(f"Missing generated file {self.output_directory}/{name}")

        return errors

    @contextmanager
    def _atomic_write_session(self):
        self._created_files = []
        try:
            yield
        except Exception:
            self._rollback()
            raise

    def _rollback(self) -> None:
        logger.warning("Rolling back changes...")
        for p in reversed(self._created_files):
            try:
                if p.exists() and p != self._original_filepath:
                    p.unlink()
            except Exception as e:
                logger.error("Failed to remove %s: %s", p, e)

        if self._backup_path and self._backup_path.exists() and self._original_filepath:
            try:
                shutil.copy2(self._backup_path, self._original_filepath)
                logger.info("Restored from backup: %s", self._original_filepath)
            except Exception as e:
                logger.error("Failed to restore backup: %s", e)

        self._created_files = []

    def _write_validated(self, path: Path, content: str, results: Dict[str, Any]) -> None:
        try:
            ast.parse(content)
        except SyntaxError as e:
            msg = f"Cannot write {path.name}: syntax error at line {e.lineno}: {e.msg}"
            results["errors"].append(msg)
            raise ValueError(msg)

        path.write_text(content, encoding="utf-8")
        self._created_files.append(path)
        results["files_created"].append(str(path))

    def _write_file_direct(self, path: Path, content: str, results: Dict[str, Any]) -> None:
        if IMPORT_FIXING_AVAILABLE:
            content = self._apply_automatic_fixes(content, path)
        ast.parse(content)
        path.write_text(content, encoding="utf-8")

    def _apply_automatic_fixes(self, content: str, path: Path) -> str:
        """Apply learned patterns + import/logging fixes to generated code."""
        original_content = content
        try:
            new_content = content

            if SELF_LEARNING_AVAILABLE:
                learning_system = get_learning_system()
                new_content = learning_system.apply_learned_patterns(new_content, str(path))

            if IMPORT_FIXING_AVAILABLE:
                new_content = ImportFixingPatterns.apply_all_fixes(new_content)

            if self._codebase_analysis:
                logging_standard = self._codebase_analysis.get("logging_standard", "logger")

                if logging_standard == "logger":
                    # Replace LOG = logging.getLogger(...) -> logger = ...
                    new_content = re.sub(r"\bLOG\s*=\s*logging\.getLogger", "logger = logging.getLogger", new_content)
                    new_content = re.sub(r"\bLOG\.", "logger.", new_content)

                existing_modules = self._codebase_analysis.get("existing_modules", set())
                lines = new_content.splitlines()
                fixed_lines: List[str] = []

                for line in lines:
                    if "from core.cli" in line:
                        fixed_lines.append(f"# Removed non-existent import: {line.strip()}")
                        continue

                    import_match = re.match(r"\s*from\s+(core\.[^\s]+)\s+import\s+", line)
                    if import_match:
                        module = import_match.group(1)
                        if module not in existing_modules:
                            fixed_lines.append("# Import fallback for missing module")
                            fixed_lines.append("try:")
                            fixed_lines.append(f"    {line}")
                            fixed_lines.append("except ImportError:")
                            fixed_lines.append("    pass  # Module not available")
                            continue

                    fixed_lines.append(line)

                new_content = "\n".join(fixed_lines) + ("\n" if not new_content.endswith("\n") else "")

            if new_content != original_content:
                logger.info("Applied automatic fixes to %s", path.name)

            return new_content

        except Exception as e:
            logger.warning("Failed to apply automatic fixes to %s: %s", path.name, e)
            return original_content

    def learn_from_manual_corrections(self, original_file: Path, corrected_files: List[Path], description: str = "") -> Dict[str, Any]:
        if not SELF_LEARNING_AVAILABLE:
            return {"success": False, "error": "Self-learning system not available", "patterns_extracted": 0}

        try:
            learning_system = get_learning_system()
            patterns = learning_system.analyze_manual_corrections(
                str(original_file),
                [str(f) for f in corrected_files],
                description,
            )
            stats = learning_system.get_pattern_statistics()
            logger.info("Learning completed: extracted %d patterns", len(patterns))

            return {
                "success": True,
                "patterns_extracted": len(patterns),
                "pattern_names": [p.name for p in patterns],
                "total_patterns": stats.get("total_patterns"),
                "learning_sessions": stats.get("learning_sessions"),
                "codebase_standards": stats.get("codebase_standards"),
            }

        except Exception as e:
            logger.error("Failed to learn from manual corrections: %s", e)
            return {"success": False, "error": str(e), "patterns_extracted": 0}

    def _create_empty_plan(self, filepath: Path) -> RefactoringPlan:
        return RefactoringPlan(
            target_file=str(filepath),
            target_class_name="",
            transformations=[],
            extracted_components=[],
            new_files=[],
            estimated_effort=0.0,
            risk_level="low",
        )


# -----------------------------
# Factory + CLI
# -----------------------------


def create_auto_refactor(config: Optional[Dict[str, Any]] = None) -> AutoRefactor:
    """Factory for AutoRefactor."""
    return AutoRefactor(config)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Automatic refactoring of God Objects (owner-proxy components)")
    parser.add_argument("file", help="Path to file for refactoring")
    parser.add_argument("--dry-run", action="store_true", help="Validate without writing files")
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument("--output-dir", default="components", help="Output directory")
    parser.add_argument("--replace-original", action="store_true", help="Replace original (DANGEROUS)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(levelname)s: %(message)s")

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return 1

    config: Dict[str, Any] = {}
    if args.config:
        cfg = Path(args.config)
        if cfg.exists():
            try:
                config = json.loads(cfg.read_text(encoding="utf-8-sig"))
            except Exception as e:
                print(f"Warning: Failed to load config: {e}")

    config["output_directory"] = args.output_dir
    config["preserve_original"] = not args.replace_original

    refactor = create_auto_refactor(config)
    plan = refactor.analyze_god_object(filepath)

    print(f"Analyzing: {filepath}")
    print("\nPlan:")
    print(f"  Target: {plan.target_class_name or '(none)'}")
    print(f"  Components: {len(plan.extracted_components)}")
    print(f"  Risk: {plan.risk_level}")

    if plan.transformations:
        print("\nTransformations:")
        for t in plan.transformations:
            print(f"  - {t}")
    else:
        print("\nNo refactoring needed.")
        return 0

    if args.dry_run:
        results = refactor.execute_refactoring(filepath, plan, dry_run=True)
        if results["errors"]:
            print("\nValidation FAILED:")
            for e in results["errors"]:
                print(f"  - {e}")
            return 1
        print("\nValidation PASSED")
        print(f"Would create {len(results['files_created'])} files")
        return 0

    if input("\nExecute? (y/N): ").lower() != "y":
        return 0

    results = refactor.execute_refactoring(filepath, plan, dry_run=False)
    if results["success"]:
        print(f"\nSuccess! Created {len(results['files_created'])} files")
        if results["backup_created"]:
            print(f"Backup: {results['backup_created']}")
        if results["warnings"]:
            print("\nWarnings:")
            for w in results["warnings"]:
                print(f"  - {w}")
        return 0

    print("\nFailed:")
    for e in results["errors"]:
        print(f"  - {e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
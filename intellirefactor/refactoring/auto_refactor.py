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

# Import configuration manager
from .config_manager import RefactorConfig
from .method_analyzer import MethodAnalyzer
from .code_generator import CodeGenerator
from .plan_builder import PlanBuilder
from .validator import CodeValidator
from .executor import RefactoringExecutor
from .ast_utils import find_largest_top_level_class, collect_module_level_names
from .facade_builder import FacadeBuilder
from .code_fixer import CodeFixer

# Optional integrations
try:
    from ..analysis.contextual_analyzer_integration import ContextualAnalyzerIntegration

    CONTEXTUAL_ANALYSIS_AVAILABLE = True
except ImportError:
    CONTEXTUAL_ANALYSIS_AVAILABLE = False

try:
    from ..analysis.enhanced_method_grouping import EnhancedMethodGrouping

    ENHANCED_GROUPING_AVAILABLE = True
except ImportError:
    ENHANCED_GROUPING_AVAILABLE = False

try:
    from ..knowledge.import_fixing_patterns import ImportFixingPatterns
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
    _method_groups: Dict[str, List[MethodInfo]] = field(
        default_factory=dict, repr=False
    )
    _private_methods_by_group: Dict[str, List[MethodInfo]] = field(
        default_factory=dict, repr=False
    )
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
            if (
                node.func.id == "vars"
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "self"
            ):
                self.dangerous.add(DangerousPattern.VARS_SELF.value)
            if (
                node.func.id == "type"
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "self"
            ):
                self.dangerous.add(DangerousPattern.TYPE_SELF.value)
            if (
                node.func.id == "isinstance"
                and isinstance(node.args[0], ast.Name)
                and node.args[0].id == "self"
            ):
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
        return frozenset(
            self.read_attributes | self.written_attributes | self.deleted_attributes
        )


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

    module_level_deps = (set(filtered_names) & set(module_level_names)) - set(
        SAFE_COMPONENT_GLOBALS
    )

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
        self.imports.append(
            ImportInfo(node=node, names=names, is_relative=False, level=0, module=None)
        )

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
        # Use ConfigManager to parse and validate configuration
        self.cfg = RefactorConfig.from_dict(config)

        # Legacy attribute access for backward compatibility
        self.config = config if isinstance(config, dict) else {}

        # Expose commonly used config attributes for backward compatibility
        self.output_directory = self.cfg.output_directory
        self.component_template = self.cfg.component_template
        self.interface_prefix = self.cfg.interface_prefix
        self.preserve_original = self.cfg.preserve_original
        self.facade_suffix = self.cfg.facade_suffix

        self.reserved_prefix = self.cfg.reserved_prefix
        self._container_attr = self.cfg.container_attr
        self._components_attr = self.cfg.components_attr

        self.extract_decorated_public_methods = (
            self.cfg.extract_decorated_public_methods
        )
        self.extract_private_methods = self.cfg.extract_private_methods
        self.keep_private_methods_in_facade = self.cfg.keep_private_methods_in_facade

        self.skip_methods_with_module_level_deps = (
            self.cfg.skip_methods_with_module_level_deps
        )
        self.skip_methods_with_bare_self_usage = (
            self.cfg.skip_methods_with_bare_self_usage
        )
        self.skip_methods_with_dangerous_patterns = (
            self.cfg.skip_methods_with_dangerous_patterns
        )

        self.responsibility_keywords = self.cfg.responsibility_keywords
        self.cohesion_cluster_other = self.cfg.cohesion_cluster_other
        self.cohesion_similarity_threshold = self.cfg.cohesion_similarity_threshold
        self.cohesion_stop_features = self.cfg.cohesion_stop_features

        self.god_class_threshold = self.cfg.god_class_threshold
        self.min_methods_for_extraction = self.cfg.min_methods_for_extraction

        self.effort_per_component = self.cfg.effort_per_component
        self.base_effort = self.cfg.base_effort

        # Runtime state
        self._created_files: List[Path] = []
        self._original_filepath: Optional[Path] = None
        self._backup_path: Optional[Path] = None

        # Codebase analysis from config
        self._codebase_analysis = self.cfg.codebase_analysis

        # Initialize contextual analyzer
        self._contextual_analyzer: Optional[ContextualAnalyzerIntegration] = None
        if CONTEXTUAL_ANALYSIS_AVAILABLE and not self.cfg.disable_contextual_analysis:
            if self.cfg.analysis_results_dir:
                self._contextual_analyzer = ContextualAnalyzerIntegration(
                    self.cfg.analysis_results_dir
                )
            else:
                self._contextual_analyzer = ContextualAnalyzerIntegration()
            logger.info("Contextual analyzer integration enabled")
        elif self.cfg.disable_contextual_analysis:
            logger.info("Contextual analyzer disabled by configuration")

        # Initialize enhanced method grouping
        self._enhanced_grouping: Optional[EnhancedMethodGrouping] = None
        if ENHANCED_GROUPING_AVAILABLE:
            self._enhanced_grouping = EnhancedMethodGrouping()
            logger.info("Enhanced method grouping enabled")

        # Initialize method analyzer
        self._method_analyzer = MethodAnalyzer(
            responsibility_keywords=self.responsibility_keywords,
            cohesion_cluster_other=self.cohesion_cluster_other,
            cohesion_similarity_threshold=self.cohesion_similarity_threshold,
            cohesion_stop_features=self.cohesion_stop_features,
            min_methods_for_extraction=self.min_methods_for_extraction,
        )

        # Initialize code generator
        self._code_generator = CodeGenerator(
            interface_prefix=self.interface_prefix,
            component_template=self.component_template,
        )

        # Initialize plan builder
        self._plan_builder = PlanBuilder(
            output_directory=self.output_directory,
            component_template=self.component_template,
            min_methods_for_extraction=self.min_methods_for_extraction,
            effort_per_component=self.effort_per_component,
            base_effort=self.base_effort,
        )

        # Initialize validator
        self._validator = CodeValidator(
            output_directory=self.output_directory,
        )

        # Initialize executor
        self._executor = RefactoringExecutor(
            backup_enabled=self.cfg.backup_enabled,
            preserve_original=self.preserve_original,
            facade_suffix=self.facade_suffix,
        )

        # Initialize facade builder
        self._facade_builder = FacadeBuilder(
            container_attr=self._container_attr,
            components_attr=self._components_attr,
            interface_prefix=self.interface_prefix,
        )

        # Initialize code fixer
        self._code_fixer = CodeFixer(
            codebase_analysis=self._codebase_analysis,
        )

        # Initialize project refactorer
        from .project_refactorer import ProjectRefactorer
        self._project_refactorer = ProjectRefactorer(self)

    def _validate_positive_int(self, value: Any, name: str, default: int) -> int:
        """Legacy method - delegates to RefactorConfig."""
        return RefactorConfig._validate_positive_int(value, name, default)

    def _analyze_codebase_standards(self) -> None:
        """Legacy method - now handled by RefactorConfig."""
        self._codebase_analysis = RefactorConfig._analyze_codebase_standards()

    # -----------------------------
    # Public API
    # -----------------------------

    def analyze_god_object(self, filepath: Path) -> RefactoringPlan:
        logger.info("Analyzing %s", filepath)

        try:
            content = filepath.read_text(encoding="utf-8-sig")
            tree = ast.parse(content)
            # Кешируем контент для использования в других методах
            self._cached_content = content
            self._original_filepath = filepath
        except Exception as e:
            logger.error("Failed to read/parse %s: %s", filepath, e)
            return self._create_empty_plan(filepath)

        main_class, max_methods = self._find_largest_top_level_class(tree)
        if not main_class or max_methods < self.god_class_threshold:
            logger.info("No God Object found (max methods: %s)", max_methods)
            return self._create_empty_plan(filepath)

        # Попытка использовать контекстный анализ
        contextual_data = None
        if self._contextual_analyzer:
            try:
                project_path = self._find_project_root(filepath)
                contextual_data = self._contextual_analyzer.load_analysis_for_file(
                    filepath, project_path
                )
                if contextual_data:
                    logger.info(
                        "Contextual analysis data available, but checking quality..."
                    )
                    # Проверяем качество контекстного анализа
                    contextual_groups = (
                        self._contextual_analyzer.extract_method_groups_from_context(
                            contextual_data, main_class.name
                        )
                    )
                    if (
                        len(contextual_groups) < 3
                    ):  # Если контекстный анализ дает мало групп
                        logger.warning(
                            f"Contextual analysis gives only {len(contextual_groups)} groups, using enhanced grouping instead"
                        )
                        contextual_data = None  # Игнорируем слабый контекстный анализ
                    else:
                        logger.info(
                            "Using contextual analysis data for enhanced refactoring"
                        )
                else:
                    logger.info(
                        "No contextual analysis data found, using enhanced grouping"
                    )
            except Exception as e:
                logger.warning(f"Failed to load contextual analysis: {e}")
                contextual_data = None

        import_collector = ImportCollector()
        import_collector.visit(tree)
        module_level_names = self._collect_module_level_names(tree)

        # Используем контекстную группировку если доступна
        if contextual_data:
            method_groups = (
                self._contextual_analyzer.extract_method_groups_from_context(
                    contextual_data, main_class.name
                )
            )
            # Преобразуем в формат MethodInfo
            method_groups = self._convert_contextual_groups_to_method_info(
                method_groups, main_class, module_level_names
            )
        elif self._enhanced_grouping:
            # Используем улучшенную группировку
            logger.info("Using enhanced method grouping for better refactoring")
            method_groups = self._use_enhanced_grouping(main_class, module_level_names)
        else:
            # Стандартная группировка
            (
                method_groups,
                private_by_group,
                unextracted,
                init_method,
                dunder_methods,
                dangerous_methods,
            ) = self._group_methods_by_responsibility(main_class, module_level_names)

        # Если использовали контекстную группировку, нужно получить остальные данные
        if contextual_data and method_groups:
            (
                _,
                private_by_group,
                unextracted,
                init_method,
                dunder_methods,
                dangerous_methods,
            ) = self._group_methods_by_responsibility(main_class, module_level_names)
        elif self._enhanced_grouping and not contextual_data:
            # Используем улучшенную группировку только если нет контекстных данных
            # Получаем остальные данные стандартным способом
            (
                _,
                private_by_group,
                unextracted,
                init_method,
                dunder_methods,
                dangerous_methods,
            ) = self._group_methods_by_responsibility(main_class, module_level_names)
        elif not contextual_data:
            # Уже получили все данные выше
            pass

        # Delegate plan building to PlanBuilder
        return self._plan_builder.build_plan(
            filepath=filepath,
            main_class=main_class,
            content=content,
            tree=tree,
            import_collector=import_collector,
            module_level_names=module_level_names,
            method_groups=method_groups,
            private_by_group=private_by_group,
            unextracted=unextracted,
            init_method=init_method,
            dunder_methods=dunder_methods,
            dangerous_methods=dangerous_methods,
            component_class_name_func=self._component_class_name,
            assess_risk_func=self._method_analyzer.assess_risk,
        )

    def _find_project_root(self, filepath: Path) -> Path:
        """Delegate to PlanBuilder."""
        return PlanBuilder.find_project_root(filepath)

    def _convert_contextual_groups_to_method_info(
        self,
        contextual_groups: Dict[str, List[str]],
        main_class: ast.ClassDef,
        module_level_names: Set[str],
    ) -> Dict[str, List[MethodInfo]]:
        """Delegate to PlanBuilder."""
        return PlanBuilder.convert_contextual_groups_to_method_info(
            contextual_groups,
            main_class,
            module_level_names,
            analyze_method,
            self.skip_methods_with_bare_self_usage,
            self.skip_methods_with_dangerous_patterns,
            self.skip_methods_with_module_level_deps,
            self.extract_decorated_public_methods,
        )

    def _create_empty_plan(self, filepath: Path) -> RefactoringPlan:
        """Delegate to PlanBuilder."""
        return self._plan_builder._create_empty_plan(filepath)

    def _get_cached_content(self) -> str:
        """Get cached content or read from file.
        
        Returns:
            File content as string, or empty string if unavailable
        """
        if hasattr(self, "_cached_content") and self._cached_content:
            return self._cached_content
        elif hasattr(self, "_original_filepath") and self._original_filepath:
            return self._original_filepath.read_text(encoding="utf-8")
        else:
            logger.warning("No cached content available for enhanced grouping")
            return ""

    def _analyze_methods_enhanced(
        self, main_class: ast.ClassDef, content: str
    ) -> List[Any]:
        """Analyze methods using enhanced grouping.
        
        Args:
            main_class: Class AST node to analyze
            content: Source code content
            
        Returns:
            List of enhanced method analysis results
        """
        enhanced_methods = []
        for node in main_class.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("__"):  # Exclude dunder methods
                    enhanced_method = self._enhanced_grouping.analyze_method_enhanced(
                        node, content
                    )
                    enhanced_methods.append(enhanced_method)
        return enhanced_methods

    def _convert_enhanced_groups_to_method_info(
        self,
        enhanced_groups: Dict[str, List[str]],
        main_class: ast.ClassDef,
        module_level_names: Set[str],
    ) -> Dict[str, List[MethodInfo]]:
        """Convert enhanced groups to MethodInfo format.
        
        Args:
            enhanced_groups: Groups from enhanced analysis
            main_class: Class AST node
            module_level_names: Module-level symbol names
            
        Returns:
            Dictionary mapping group names to MethodInfo lists
        """
        method_groups = {}
        
        for group_name, method_names in enhanced_groups.items():
            group_methods = []
            
            for method_name in method_names:
                # Find corresponding AST node
                for node in main_class.body:
                    if (
                        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and node.name == method_name
                    ):
                        method_info = analyze_method(
                            node,
                            module_level_names=module_level_names,
                            allow_bare_self=not self.skip_methods_with_bare_self_usage,
                            allow_dangerous=not self.skip_methods_with_dangerous_patterns,
                            allow_module_level_deps=not self.skip_methods_with_module_level_deps,
                            decorated_extract_allowed=self.extract_decorated_public_methods,
                        )
                        group_methods.append(method_info)
                        break

            if group_methods:
                # Generate meaningful group name
                component_name = self._enhanced_grouping._generate_component_name(
                    group_name, []
                )
                clean_group_name = (
                    component_name.lower()
                    .replace("service", "")
                    .replace("handler", "")
                    .replace("processor", "")
                )
                method_groups[clean_group_name] = group_methods
        
        return method_groups

    def _use_enhanced_grouping(
        self, main_class: ast.ClassDef, module_level_names: Set[str]
    ) -> Dict[str, List[MethodInfo]]:
        """Use enhanced method grouping system.
        
        Args:
            main_class: Class AST node to analyze
            module_level_names: Module-level symbol names
            
        Returns:
            Dictionary mapping group names to MethodInfo lists
        """
        # Get cached content
        content = self._get_cached_content()
        if not content:
            return {}

        # Analyze methods with enhanced analysis
        enhanced_methods = self._analyze_methods_enhanced(main_class, content)

        # Group methods with enhanced logic
        enhanced_groups = self._enhanced_grouping.group_methods_enhanced(
            enhanced_methods
        )

        # Generate enhanced responsibility keywords
        enhanced_keywords = (
            self._enhanced_grouping.generate_enhanced_responsibility_keywords(
                enhanced_groups, enhanced_methods
            )
        )

        # Update responsibility keywords for better component naming
        self.responsibility_keywords.update(enhanced_keywords)

        logger.info(
            f"Enhanced grouping created {len(enhanced_groups)} groups with "
            f"{sum(len(methods) for methods in enhanced_groups.values())} methods"
        )

        # Convert to MethodInfo format for compatibility
        return self._convert_enhanced_groups_to_method_info(
            enhanced_groups, main_class, module_level_names
        )

    def refactor_project(
        self,
        project_path: Union[str, Path],
        strategy: Optional[str] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Delegate to ProjectRefactorer."""
        return self._project_refactorer.refactor_project(
            project_path, strategy, dry_run
        )

    def apply_opportunity(
        self, opportunity: Dict[str, Any], dry_run: bool = False
    ) -> Dict[str, Any]:
        """Delegate to ProjectRefactorer."""
        return self._project_refactorer.apply_opportunity(opportunity, dry_run)

    def execute_refactoring(
        self, filepath: Path, plan: RefactoringPlan, dry_run: bool = True
    ) -> Dict[str, Any]:
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
        # Создаем уникальную папку для каждого файла: filename_components
        file_stem = filepath.stem
        unique_output_dir = f"{file_stem}_{self.output_directory}"
        output_dir = filepath.parent / unique_output_dir

        try:
            if dry_run:
                return self._execute_dry_run(filepath, plan, output_dir, results)

            if plan.backup_required and bool(self.config.get("backup_enabled", True)):
                backup_path = (
                    filepath.parent
                    / f"{filepath.stem}.backup_{int(time.time())}{filepath.suffix}"
                )
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
                    extractable_public = [
                        m for m in methods if self._is_public_extractable(m)
                    ]
                    if len(extractable_public) < self.min_methods_for_extraction:
                        continue

                    component_name = self._component_class_name(group_name)

                    private_methods = (
                        plan._private_methods_by_group.get(group_name, [])
                        if self.extract_private_methods
                        else []
                    )

                    for m in extractable_public:
                        method_map[m.name] = (component_name, m.is_async, m)

                    iface_code = self._generate_interface(
                        component_name, extractable_public
                    )
                    if iface_code:
                        interface_classes.append(iface_code)

                    impl_code = self._generate_component_implementation(
                        component_name=component_name,
                        group_name=group_name,
                        public_methods=extractable_public,
                        private_methods=private_methods,
                        plan=plan,
                    )
                    impl_file = (
                        output_dir
                        / f"{group_name}_{self.component_template.lower()}.py"
                    )
                    self._write_validated(impl_file, impl_code, results)

                # IMPORTANT: interfaces.py must exist if components import it.
                iface_content = self._generate_interfaces_file(interface_classes, plan)
                self._write_validated(
                    output_dir / "interfaces.py", iface_content, results
                )

                container_code = self._generate_di_container(plan.extracted_components)
                self._write_validated(
                    output_dir / "container.py", container_code, results
                )

                init_code = self._generate_package_init(
                    plan.extracted_components, has_interfaces=True
                )
                self._write_validated(output_dir / "__init__.py", init_code, results)

                facade_code = self._create_facade(
                    plan.target_class_name, plan.extracted_components, method_map, plan
                )

                if self.preserve_original:
                    facade_file = filepath.with_name(
                        f"{filepath.stem}{self.facade_suffix}{filepath.suffix}"
                    )
                    self._write_validated(facade_file, facade_code, results)
                    results["warnings"].append(
                        f"Original preserved. New facade: {facade_file.name}"
                    )
                    self._validate_refactored_file(facade_file, results)
                else:
                    self._write_file_direct(filepath, facade_code, results)
                    results["files_modified"].append(str(filepath))
                    results["warnings"].append(
                        f"Original replaced. Backup: {results['backup_created']}"
                    )
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

    def _find_largest_top_level_class(
        self, tree: ast.Module
    ) -> Tuple[Optional[ast.ClassDef], int]:
        """Delegate to ast_utils module."""
        return find_largest_top_level_class(tree)

    def _collect_module_level_names(self, tree: ast.Module) -> Set[str]:
        """Delegate to ast_utils module."""
        return collect_module_level_names(tree)

    def _is_public_extractable(self, m: MethodInfo) -> bool:
        """Delegate to MethodAnalyzer."""
        return self._method_analyzer.is_public_extractable(m)

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
        """Delegate to MethodAnalyzer with backward-compatible signature."""
        init_method: Optional[MethodInfo] = None
        dunder_methods: List[MethodInfo] = []
        all_methods: List[MethodInfo] = []

        allow_bare_self = not self.skip_methods_with_bare_self_usage
        allow_dangerous = not self.skip_methods_with_dangerous_patterns
        allow_module_level = not self.skip_methods_with_module_level_deps

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

                if node.name == "__init__":
                    init_method = info
                elif node.name.startswith("__") and node.name.endswith("__"):
                    dunder_methods.append(info)
                else:
                    all_methods.append(info)

        # Delegate to MethodAnalyzer
        groups, private_by_group, unextracted, dangerous_methods = (
            self._method_analyzer.group_methods_by_responsibility(
                class_node, all_methods, init_method, dunder_methods
            )
        )

        return (
            groups,
            private_by_group,
            unextracted,
            init_method,
            dunder_methods,
            dangerous_methods,
        )

    def _cluster_methods_by_cohesion(
        self, methods: List[MethodInfo], threshold: float
    ) -> List[List[MethodInfo]]:
        """Delegate to MethodAnalyzer."""
        return self._method_analyzer.cluster_methods_by_cohesion(methods, threshold)

    def _assign_private_to_groups(
        self,
        public_groups: Dict[str, List[MethodInfo]],
        private_methods: List[MethodInfo],
    ) -> Dict[str, List[MethodInfo]]:
        """Delegate to MethodAnalyzer."""
        return self._method_analyzer.assign_private_to_groups(
            public_groups, private_methods
        )

    def _assess_risk(
        self,
        components: List[str],
        method_groups: Dict[str, List[MethodInfo]],
        private_by_group: Dict[str, List[MethodInfo]],
        dangerous_methods: Dict[str, Set[str]],
    ) -> str:
        """Delegate to MethodAnalyzer."""
        return self._method_analyzer.assess_risk(
            components, method_groups, private_by_group, dangerous_methods
        )

    # -----------------------------
    # Naming helpers (CRITICAL for module imports)
    # -----------------------------

    def _component_class_name(self, group_name: str) -> str:
        """Delegate to CodeGenerator."""
        return CodeGenerator.component_class_name(group_name, self.component_template)

    def _get_group_name(self, component_name: str) -> str:
        """Delegate to CodeGenerator."""
        return CodeGenerator.get_group_name(component_name, self.component_template)

    # -----------------------------
    # Code generation (delegated to CodeGenerator)
    # -----------------------------

    def _unparse_safe(self, node: Optional[ast.AST], default: str = "Any") -> str:
        """Delegate to CodeGenerator."""
        return self._code_generator.unparse_safe(node, default)

    def _format_arg(self, arg: ast.arg) -> str:
        """Delegate to CodeGenerator."""
        return self._code_generator.format_arg(arg)

    def _generate_method_signature(self, method: MethodInfo) -> str:
        """Delegate to CodeGenerator."""
        return self._code_generator.generate_method_signature(method)

    def _build_call_arguments(self, args: ast.arguments) -> str:
        """Delegate to CodeGenerator."""
        return self._code_generator.build_call_arguments(args)

    def _generate_import_statement(
        self, import_info: ImportInfo, needed_names: Set[str], level_adjustment: int = 1
    ) -> Optional[str]:
        """Delegate to CodeGenerator."""
        return self._code_generator.generate_import_statement(
            import_info, needed_names, level_adjustment
        )

    def _extract_node_code(
        self, node: ast.AST, content: str, start_line: Optional[int] = None
    ) -> str:
        """Delegate to CodeGenerator."""
        return self._code_generator.extract_node_code(node, content, start_line)

    def _extract_method_code(
        self, method_node: Union[ast.FunctionDef, ast.AsyncFunctionDef], content: str
    ) -> str:
        """Delegate to CodeGenerator."""
        return self._code_generator.extract_method_code(method_node, content)

    def _generate_owner_proxy_base(self) -> str:
        """Delegate to CodeGenerator."""
        return self._code_generator.generate_owner_proxy_base()

    def _generate_component_implementation(
        self,
        *,
        component_name: str,
        group_name: str,
        public_methods: List[MethodInfo],
        private_methods: List[MethodInfo],
        plan: RefactoringPlan,
    ) -> str:
        """Delegate to CodeGenerator."""
        return self._code_generator.generate_component_implementation(
            component_name, group_name, public_methods, private_methods, plan
        )

    def _generate_interface(
        self, component_name: str, public_methods: List[MethodInfo]
    ) -> str:
        """Delegate to CodeGenerator."""
        return self._code_generator.generate_interface(component_name, public_methods)

    def _generate_interfaces_file(
        self, interface_classes: List[str], plan: RefactoringPlan
    ) -> str:
        """Delegate to CodeGenerator."""
        return self._code_generator.generate_interfaces_file(interface_classes, plan)

    def _generate_di_container(self, components: List[str]) -> str:
        """Delegate to CodeGenerator."""
        return self._code_generator.generate_di_container(
            components, self._get_group_name
        )

    def _generate_package_init(
        self, components: List[str], has_interfaces: bool
    ) -> str:
        """Delegate to CodeGenerator."""
        return self._code_generator.generate_package_init(
            components, has_interfaces, self._get_group_name
        )

    def _generate_facade_import_block(self, components: List[str]) -> List[str]:
        """Delegate to CodeGenerator."""
        return self._code_generator.generate_facade_import_block(
            components, self.output_directory
        )

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
        """Delegate to FacadeBuilder."""
        return self._facade_builder.create_facade(
            original_class_name,
            components,
            method_map,
            plan,
            self._code_generator,
            self._get_group_name,
        )

    def _create_enhanced_init_method(
        self,
        init_node: Union[ast.FunctionDef, ast.AsyncFunctionDef],
        content: str,
        out_lines: List[str],
        components: List[str],
    ) -> None:
        """Delegate to FacadeBuilder."""
        return self._facade_builder.create_enhanced_init_method(
            init_node,
            content,
            out_lines,
            components,
            self._code_generator,
            self._get_group_name,
        )

    def _create_active_delegation_method(
        self, method: MethodInfo, component: str, out_lines: List[str]
    ) -> None:
        """Delegate to FacadeBuilder."""
        return self._facade_builder.create_active_delegation_method(
            method,
            component,
            out_lines,
            self._code_generator,
            self._get_group_name,
        )

    # -----------------------------
    # Dry run / Validation / IO safety
    # -----------------------------

    def _execute_dry_run(
        self,
        filepath: Path,
        plan: RefactoringPlan,
        output_dir: Path,
        results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Delegate to RefactoringExecutor."""
        return self._executor.execute_dry_run(
            filepath=filepath,
            plan=plan,
            output_dir=output_dir,
            results=results,
            code_generator=self._code_generator,
            validator=self._validator,
            component_class_name_func=self._component_class_name,
            get_group_name_func=self._get_group_name,
            is_public_extractable_func=self._is_public_extractable,
            min_methods_for_extraction=self.min_methods_for_extraction,
            extract_private_methods=self.extract_private_methods,
        )

    def _validate_syntax(self, code: str, path: Path, results: Dict[str, Any]) -> bool:
        """Delegate to CodeValidator."""
        return self._validator.validate_syntax(code, path, results)

    def _validate_generated_facade(
        self, code: str, path: Path, results: Dict[str, Any]
    ) -> None:
        """Delegate to CodeValidator."""
        return self._validator.validate_generated_facade(code, path, results)

    def _validate_refactored_file(
        self, filepath: Path, results: Dict[str, Any]
    ) -> bool:
        """Delegate to CodeValidator."""
        return self._validator.validate_refactored_file(filepath, results)

    def _validate_generated_package_files(
        self, facade_path: Path, tree: ast.AST
    ) -> List[str]:
        """Delegate to CodeValidator."""
        return self._validator.validate_generated_package_files(facade_path, tree)



    @contextmanager
    def _atomic_write_session(self):
        """Delegate to RefactoringExecutor."""
        with self._executor.atomic_write_session():
            yield

    def _rollback(self) -> None:
        """Delegate to RefactoringExecutor."""
        self._executor.rollback()

    def _write_validated(
        self, path: Path, content: str, results: Dict[str, Any]
    ) -> None:
        """Delegate to RefactoringExecutor."""
        self._executor.write_validated(path, content, results, self._validator)

    def _write_file_direct(
        self, path: Path, content: str, results: Dict[str, Any]
    ) -> None:
        """Delegate to RefactoringExecutor."""
        self._executor.write_file_direct(
            path, content, results, self._apply_automatic_fixes
        )

    def _apply_automatic_fixes(self, content: str, path: Path) -> str:
        """Delegate to CodeFixer."""
        return self._code_fixer.apply_automatic_fixes(content, path)

    def learn_from_manual_corrections(
        self, original_file: Path, corrected_files: List[Path], description: str = ""
    ) -> Dict[str, Any]:
        """Delegate to CodeFixer."""
        return self._code_fixer.learn_from_manual_corrections(
            original_file, corrected_files, description
        )

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

    parser = argparse.ArgumentParser(
        description="Automatic refactoring of God Objects (owner-proxy components)"
    )
    parser.add_argument("file", help="Path to file for refactoring")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate without writing files"
    )
    parser.add_argument("--config", help="Path to JSON config file")
    parser.add_argument("--output-dir", default="components", help="Output directory")
    parser.add_argument(
        "--replace-original", action="store_true", help="Replace original (DANGEROUS)"
    )
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

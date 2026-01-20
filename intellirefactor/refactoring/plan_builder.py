"""
Refactoring plan construction for AutoRefactor.

This module provides the PlanBuilder class that constructs RefactoringPlan
objects from analysis results. The plan builder:

- Estimates refactoring effort based on component count
- Assesses risk levels (low, moderate, high)
- Determines backup requirements
- Generates transformation descriptions
- Lists new files to be created

The PlanBuilder separates plan construction logic from the main AutoRefactor
class, improving testability and maintainability.

Classes:
    PlanBuilder: Constructs RefactoringPlan objects from analysis results

Example:
    >>> builder = PlanBuilder(
    ...     output_directory='components',
    ...     component_template='Service',
    ...     min_methods_for_extraction=2
    ... )
    >>> plan = builder.build_plan(
    ...     filepath=Path('myclass.py'),
    ...     main_class=class_node,
    ...     content=source_code,
    ...     tree=ast_tree,
    ...     import_collector=imports,
    ...     method_groups=groups
    ... )
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class PlanBuilder:
    """
    Builds RefactoringPlan objects from analysis results.

    This class is responsible for constructing comprehensive refactoring plans
    that include effort estimation, risk assessment, and file generation lists.

    The plan builder:
    - Calculates estimated effort based on component count
    - Assesses risk level (low/moderate/high) based on complexity
    - Determines if backup is required
    - Generates list of transformations to be performed
    - Lists all new files that will be created

    Attributes:
        output_directory: Directory name for generated components
        component_template: Template suffix for component class names
        min_methods_for_extraction: Minimum methods required for extraction
        effort_per_component: Estimated effort hours per component
        base_effort: Base effort hours for refactoring setup

    Example:
        >>> builder = PlanBuilder(
        ...     output_directory='components',
        ...     min_methods_for_extraction=2,
        ...     effort_per_component=2.5
        ... )
        >>> plan = builder.build_plan(filepath, main_class, content, tree, ...)
        >>> print(f"Effort: {plan.estimated_effort}h, Risk: {plan.risk_level}")
    """

    def __init__(
        self,
        output_directory: str = "components",
        component_template: str = "Service",
        min_methods_for_extraction: int = 1,
        effort_per_component: float = 2.5,
        base_effort: float = 4.0,
    ):
        """
        Initialize PlanBuilder.

        Args:
            output_directory: Directory name for generated components
            component_template: Template suffix for component names
            min_methods_for_extraction: Minimum methods required for extraction
            effort_per_component: Effort hours per component
            base_effort: Base effort hours
        """
        self.output_directory = output_directory
        self.component_template = component_template
        self.min_methods_for_extraction = min_methods_for_extraction
        self.effort_per_component = effort_per_component
        self.base_effort = base_effort

    def build_plan(
        self,
        filepath: Path,
        main_class: ast.ClassDef,
        content: str,
        tree: ast.Module,
        import_collector,
        module_level_names: Set[str],
        method_groups: Dict[str, List],
        private_by_group: Dict[str, List],
        unextracted: List,
        init_method: Optional,
        dunder_methods: List,
        dangerous_methods: Dict[str, Set[str]],
        component_class_name_func,
        assess_risk_func,
    ):
        """
        Build RefactoringPlan from analysis results.

        Args:
            filepath: Path to the file being analyzed
            main_class: AST node of the main class
            content: Source code content
            tree: AST tree
            import_collector: ImportCollector instance
            module_level_names: Set of module-level names
            method_groups: Grouped methods by responsibility
            private_by_group: Private methods assigned to groups
            unextracted: Methods not extracted
            init_method: __init__ method if present
            dunder_methods: List of dunder methods
            dangerous_methods: Methods with dangerous patterns
            component_class_name_func: Function to generate component class names
            assess_risk_func: Function to assess refactoring risk

        Returns:
            RefactoringPlan instance
        """
        from .auto_refactor import RefactoringPlan

        extracted_components: List[str] = []
        new_files: List[str] = []
        transformations: List[str] = []

        # Build component list and transformations
        for group_name, methods in method_groups.items():
            extractable_public = [m for m in methods if self._is_public_extractable(m)]
            if len(extractable_public) >= self.min_methods_for_extraction:
                comp_name = component_class_name_func(group_name)
                extracted_components.append(comp_name)
                new_files.append(
                    str(
                        Path(self.output_directory)
                        / f"{group_name}_{self.component_template.lower()}.py"
                    )
                )

                priv_count = len(private_by_group.get(group_name, []))
                transformations.append(
                    f"Extract {len(extractable_public)} public + {priv_count} private methods to {comp_name}"
                )

        # Check if any components to extract
        if not extracted_components:
            return self._create_empty_plan(filepath)

        # Add transformation for remaining methods
        if unextracted:
            remaining_public = [m for m in unextracted if not m.name.startswith("_")]
            if remaining_public:
                transformations.append(
                    f"Keep {len(remaining_public)} methods in facade"
                )

        # Calculate effort and risk
        estimated_effort = (
            len(extracted_components) * self.effort_per_component + self.base_effort
        )
        risk_level = assess_risk_func(
            extracted_components, method_groups, private_by_group, dangerous_methods
        )

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

    def _is_public_extractable(self, method) -> bool:
        """
        Check if method is public and extractable.

        Args:
            method: MethodInfo object

        Returns:
            True if method is public and has no dangerous patterns
        """
        return (not method.name.startswith("_")) and (
            len(method.dangerous_reasons) == 0
        )

    def _create_empty_plan(self, filepath: Path):
        """
        Create an empty RefactoringPlan.

        Args:
            filepath: Path to the file

        Returns:
            Empty RefactoringPlan
        """
        from .auto_refactor import RefactoringPlan

        return RefactoringPlan(
            target_file=str(filepath),
            target_class_name="",
            transformations=[],
            extracted_components=[],
            new_files=[],
            estimated_effort=0.0,
            risk_level="low",
        )

    @staticmethod
    def find_project_root(filepath: Path) -> Path:
        """
        Find project root directory.

        Args:
            filepath: Starting file path

        Returns:
            Path to project root
        """
        current = filepath.parent if filepath.is_file() else filepath

        # Walk up directory tree
        while current != current.parent:
            # Look for project markers
            markers = [
                ".git",
                "pyproject.toml",
                "setup.py",
                "requirements.txt",
                "intellirefactor.json",
            ]
            if any((current / marker).exists() for marker in markers):
                logger.info(f"Found project root: {current}")
                return current
            current = current.parent

        # Fallback to current working directory
        cwd = Path.cwd()
        logger.info(f"Using current working directory as project root: {cwd}")
        return cwd

    @staticmethod
    def convert_contextual_groups_to_method_info(
        contextual_groups: Dict[str, List[str]],
        main_class: ast.ClassDef,
        module_level_names: Set[str],
        analyze_method_func,
        skip_bare_self: bool,
        skip_dangerous: bool,
        skip_module_level: bool,
        extract_decorated: bool,
    ) -> Dict[str, List]:
        """
        Convert contextual groups to MethodInfo format.

        Args:
            contextual_groups: Groups from contextual analyzer
            main_class: AST node of the class
            module_level_names: Set of module-level names
            analyze_method_func: Function to analyze methods
            skip_bare_self: Whether to skip methods with bare self
            skip_dangerous: Whether to skip dangerous methods
            skip_module_level: Whether to skip methods with module-level deps
            extract_decorated: Whether to extract decorated methods

        Returns:
            Dictionary mapping group names to MethodInfo lists
        """
        # First get all methods of the class
        all_methods = {}
        for node in main_class.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("__"):  # Exclude dunder methods
                    info = analyze_method_func(
                        node,
                        module_level_names=module_level_names,
                        allow_bare_self=not skip_bare_self,
                        allow_dangerous=not skip_dangerous,
                        allow_module_level_deps=not skip_module_level,
                        decorated_extract_allowed=extract_decorated,
                    )
                    all_methods[node.name] = info

        # Convert groups
        method_groups = {}
        for group_name, method_names in contextual_groups.items():
            group_methods = []
            for method_name in method_names:
                if method_name in all_methods:
                    group_methods.append(all_methods[method_name])
            if group_methods:
                method_groups[group_name] = group_methods

        return method_groups

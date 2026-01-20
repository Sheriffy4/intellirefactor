"""
Refactoring module for IntelliRefactor

Provides intelligent refactoring capabilities including:
- Automated refactoring opportunity detection
- Safe code transformations
- Refactoring utilities and primitives
- Quality assessment and validation
"""

__all__ = [
    "IntelligentRefactoringSystem",
    "AutoRefactor",
    "RefactoringUtility",
    "RefactoringUtilityRegistry",
    "ComponentExtractor",
    "ConfigurationSplitter",
    "DependencyInjectionIntroducer",
    "FacadeCreator",
    # Utility modules
    "find_largest_top_level_class",
    "collect_module_level_names",
    "CodeValidator",
    "FacadeBuilder",
    "CodeFixer",
    "ProjectRefactorer",
]


def __getattr__(name: str):
    if name == "IntelligentRefactoringSystem":
        from .intelligent_refactoring_system import IntelligentRefactoringSystem

        return IntelligentRefactoringSystem

    if name == "AutoRefactor":
        from .auto_refactor import AutoRefactor

        return AutoRefactor

    if name in {"find_largest_top_level_class", "collect_module_level_names"}:
        from .ast_utils import find_largest_top_level_class, collect_module_level_names

        return {
            "find_largest_top_level_class": find_largest_top_level_class,
            "collect_module_level_names": collect_module_level_names,
        }[name]

    if name == "CodeValidator":
        from .validator import CodeValidator

        return CodeValidator

    if name == "FacadeBuilder":
        from .facade_builder import FacadeBuilder

        return FacadeBuilder

    if name == "CodeFixer":
        from .code_fixer import CodeFixer

        return CodeFixer

    if name == "ProjectRefactorer":
        from .project_refactorer import ProjectRefactorer

        return ProjectRefactorer

    if name in {
        "RefactoringUtility",
        "RefactoringUtilityRegistry",
        "ComponentExtractor",
        "ConfigurationSplitter",
        "DependencyInjectionIntroducer",
        "FacadeCreator",
    }:
        from .refactoring_utilities import (  # type: ignore
            RefactoringUtility,
            RefactoringUtilityRegistry,
            ComponentExtractor,
            ConfigurationSplitter,
            DependencyInjectionIntroducer,
            FacadeCreator,
        )

        return {
            "RefactoringUtility": RefactoringUtility,
            "RefactoringUtilityRegistry": RefactoringUtilityRegistry,
            "ComponentExtractor": ComponentExtractor,
            "ConfigurationSplitter": ConfigurationSplitter,
            "DependencyInjectionIntroducer": DependencyInjectionIntroducer,
            "FacadeCreator": FacadeCreator,
        }[name]

    raise AttributeError(name)

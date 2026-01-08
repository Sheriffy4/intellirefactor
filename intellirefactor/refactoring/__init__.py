"""
Refactoring module for IntelliRefactor

Provides intelligent refactoring capabilities including:
- Automated refactoring opportunity detection
- Safe code transformations
- Refactoring utilities and primitives
- Quality assessment and validation
"""

from .intelligent_refactoring_system import IntelligentRefactoringSystem
from .refactoring_utilities import (
    RefactoringUtility,
    RefactoringUtilityRegistry,
    ComponentExtractor,
    ConfigurationSplitter,
    DependencyInjectionIntroducer,
    FacadeCreator,
)
from .auto_refactor import AutoRefactor

__all__ = [
    "IntelligentRefactoringSystem",
    "AutoRefactor",
    "RefactoringUtility",
    "RefactoringUtilityRegistry",
    "ComponentExtractor",
    "ConfigurationSplitter",
    "DependencyInjectionIntroducer",
    "FacadeCreator",
]

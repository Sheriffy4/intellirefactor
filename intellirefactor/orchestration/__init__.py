"""
Orchestration module for IntelliRefactor

Provides workflow orchestration and coordination including:
- Multi-step refactoring workflows
- Dependency management between operations
- Validation and reporting
- Rollback and recovery capabilities
"""

__all__ = [
    "GlobalRefactoringOrchestrator",
    "RefactoringValidator",
    "RefactoringReporter",
]


def __getattr__(name: str):
    if name == "RefactoringReporter":
        from .refactoring_reporter import RefactoringReporter
        return RefactoringReporter
    if name == "GlobalRefactoringOrchestrator":
        from .global_refactoring_orchestrator import GlobalRefactoringOrchestrator
        return GlobalRefactoringOrchestrator
    if name == "RefactoringValidator":
        from .refactoring_validator import RefactoringValidator
        return RefactoringValidator
    raise AttributeError(name)

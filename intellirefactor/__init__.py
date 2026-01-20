"""
IntelliRefactor - Intelligent Project Analysis and Refactoring System

A standalone system for automated analysis of code duplication, project structure
assessment, and intelligent refactoring capabilities for Python projects.

Extracted from the recon project to provide reusable refactoring intelligence
for any Python codebase.
"""

__version__ = "0.1.0"
__author__ = "IntelliRefactor Team"
__email__ = "contact@intellirefactor.dev"

# Only expose version by default - everything else is lazy loaded
__all__ = ["__version__", "__author__", "__email__"]


def __getattr__(name):
    """Lazy loading of main API classes to prevent heavy imports at module level."""
    if name in {"IntelliRefactor", "AnalysisResult", "RefactoringResult"}:
        from .api import IntelliRefactor, AnalysisResult, RefactoringResult
        return {
            "IntelliRefactor": IntelliRefactor,
            "AnalysisResult": AnalysisResult,
            "RefactoringResult": RefactoringResult,
        }[name]
    
    if name in {"IntelliRefactorConfig", "SafetyLevel"}:
        from .config import IntelliRefactorConfig, SafetyLevel
        return {
            "IntelliRefactorConfig": IntelliRefactorConfig,
            "SafetyLevel": SafetyLevel,
        }[name]
    
    if name in {"ProjectAnalyzer", "FileAnalyzer"}:
        from .analysis import ProjectAnalyzer, FileAnalyzer
        return {
            "ProjectAnalyzer": ProjectAnalyzer,
            "FileAnalyzer": FileAnalyzer,
        }[name]
    
    if name in {"IntelligentRefactoringSystem", "AutoRefactor"}:
        from .refactoring import IntelligentRefactoringSystem, AutoRefactor
        return {
            "IntelligentRefactoringSystem": IntelligentRefactoringSystem,
            "AutoRefactor": AutoRefactor,
        }[name]
    
    if name == "KnowledgeManager":
        from .knowledge import KnowledgeManager
        return KnowledgeManager
    
    if name in {"GlobalRefactoringOrchestrator", "RefactoringValidator", "RefactoringReporter"}:
        from .orchestration import (
            GlobalRefactoringOrchestrator,
            RefactoringValidator,
            RefactoringReporter,
        )
        return {
            "GlobalRefactoringOrchestrator": GlobalRefactoringOrchestrator,
            "RefactoringValidator": RefactoringValidator,
            "RefactoringReporter": RefactoringReporter,
        }[name]
    
    raise AttributeError(f"module 'intellirefactor' has no attribute '{name}'")

__all__ = [
    # Main API
    "IntelliRefactor",
    "AnalysisResult",
    "RefactoringResult",
    "IntelliRefactorConfig",
    "SafetyLevel",
    # Core components (for advanced usage)
    "ProjectAnalyzer",
    "FileAnalyzer",
    "IntelligentRefactoringSystem",
    "AutoRefactor",
    "KnowledgeManager",
    "GlobalRefactoringOrchestrator",
    "RefactoringValidator",
    "RefactoringReporter",
]

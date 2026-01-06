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

# Main API classes
from .api import IntelliRefactor, AnalysisResult, RefactoringResult
from .config import IntelliRefactorConfig, SafetyLevel

# Core component classes (for advanced usage)
from .analysis import ProjectAnalyzer, FileAnalyzer
from .refactoring import IntelligentRefactoringSystem, AutoRefactor
from .knowledge import KnowledgeManager
from .orchestration import GlobalRefactoringOrchestrator, RefactoringValidator, RefactoringReporter

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
    "RefactoringReporter"
]
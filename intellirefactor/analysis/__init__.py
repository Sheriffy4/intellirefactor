"""
Analysis module for IntelliRefactor

Provides project and file analysis capabilities including:
- Project structure analysis
- Code metrics calculation
- Duplicate code detection
- Refactoring opportunity identification
"""

from .project_analyzer import ProjectAnalyzer
from .file_analyzer import FileAnalyzer
from .metrics_analyzer import MetricsAnalyzer

__all__ = ["ProjectAnalyzer", "FileAnalyzer", "MetricsAnalyzer"]

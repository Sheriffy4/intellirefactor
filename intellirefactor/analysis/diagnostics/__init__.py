"""
Diagnostics subsystem for IntelliRefactor.

Contains user-facing diagnostic helpers: rich error categorization,
environment inspection, and remediation suggestions.
"""

from .error_handler import (  # noqa: F401
    ErrorCategory,
    ErrorSeverity,
    AnalysisError,
    AnalysisErrorHandler,
    ErrorReporter,
    to_foundation_severity,
)

__all__ = [
    "ErrorCategory",
    "ErrorSeverity",
    "AnalysisError",
    "AnalysisErrorHandler",
    "ErrorReporter",
    "to_foundation_severity",
]
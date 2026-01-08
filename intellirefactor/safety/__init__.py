"""
Safety and validation systems for IntelliRefactor

This module provides comprehensive safety checks, backup mechanisms, and rollback
capabilities to prevent destructive operations and ensure safe refactoring.
"""

from .safety_manager import SafetyManager, SafetyCheck, SafetyResult, SafetyLevel
from .backup_manager import BackupManager, BackupResult
from .rollback_manager import RollbackManager, RollbackResult
from .destructive_operation_detector import DestructiveOperationDetector, OperationRisk
from .validation_tools import (
    RefactoringValidator,
    SemanticPreservationChecker,
    ValidationLevel,
    ValidationResult,
    ValidationCheck,
    ValidationReport,
)

__all__ = [
    "SafetyManager",
    "SafetyCheck",
    "SafetyResult",
    "SafetyLevel",
    "BackupManager",
    "BackupResult",
    "RollbackManager",
    "RollbackResult",
    "DestructiveOperationDetector",
    "OperationRisk",
    "RefactoringValidator",
    "SemanticPreservationChecker",
    "ValidationLevel",
    "ValidationResult",
    "ValidationCheck",
    "ValidationReport",
]

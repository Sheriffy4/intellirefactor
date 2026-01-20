"""
Diagnostic error handling for IntelliRefactor analysis issues (canonical module).

This module provides:
  - error detection and categorization
  - diagnostic information collection
  - suggested fix generation
  - formatting for text/json/markdown

Canonical import path:
  intellirefactor.analysis.diagnostics.error_handler
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import traceback
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from intellirefactor.analysis.foundation.models import Severity, parse_severity

logger = logging.getLogger(__name__)

__all__ = [
    "ErrorCategory",
    "ErrorSeverity",
    "AnalysisError",
    "AnalysisErrorHandler",
    "ErrorReporter",
    "to_foundation_severity",
]


class ErrorCategory(Enum):
    """Categories of analysis errors."""

    CONFIGURATION = auto()
    FILE_ACCESS = auto()
    SYNTAX_ERROR = auto()
    IMPORT_ERROR = auto()
    DEPENDENCY_MISSING = auto()
    PERMISSION_ERROR = auto()
    TIMEOUT_ERROR = auto()
    MEMORY_ERROR = auto()
    INTERNAL_ERROR = auto()
    VALIDATION_ERROR = auto()
    NETWORK_ERROR = auto()
    UNKNOWN = auto()


#
# Backward-compatible name. Canonical type is foundation.models.Severity.
#
ErrorSeverity = Severity


def to_foundation_severity(value: Any, default: Severity = Severity.MEDIUM) -> Severity:
    """Compatibility adapter: normalize old ErrorSeverity/strings/enums -> foundation Severity."""
    return parse_severity(value, default=default)


@dataclass
class AnalysisError:
    """Represents an analysis error with comprehensive information."""

    category: ErrorCategory
    severity: Severity
    message: str
    context: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    stack_trace: Optional[str] = None
    suggested_fixes: List[str] = field(default_factory=list)
    diagnostic_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            "category": self.category.name,
            "severity": self.severity.value,
            "message": self.message,
            "context": self.context,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "stack_trace": self.stack_trace,
            "suggested_fixes": self.suggested_fixes,
            "diagnostic_info": self.diagnostic_info,
            "timestamp": self.timestamp,
        }


class AnalysisErrorHandler:
    """
    Comprehensive error handler for analysis operations.

    Provides error detection, categorization, and diagnostic message generation
    for various types of analysis failures.
    """

    def __init__(self):
        self.error_patterns = self._initialize_error_patterns()
        self.fix_suggestions = self._initialize_fix_suggestions()
        self.diagnostic_collectors = self._initialize_diagnostic_collectors()

    def handle_analysis_error(
        self,
        exception: Exception,
        context: str,
        file_path: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None,
    ) -> AnalysisError:
        try:
            category = self._categorize_error(exception, context, file_path)
            severity = self._determine_severity(exception, category, context)

            message = self._extract_error_message(exception)
            stack_trace = self._extract_stack_trace(exception)
            line_number, column_number = self._extract_location_info(exception)

            suggested_fixes = self._generate_fix_suggestions(
                exception, category, context, file_path
            )

            diagnostic_info = self._collect_diagnostic_info(
                exception, category, context, file_path, additional_info
            )

            error = AnalysisError(
                category=category,
                severity=severity,
                message=message,
                context=context,
                file_path=file_path,
                line_number=line_number,
                column_number=column_number,
                stack_trace=stack_trace,
                suggested_fixes=suggested_fixes,
                diagnostic_info=diagnostic_info,
                timestamp=self._get_timestamp(),
            )

            logger.error(f"Analysis error handled: {category.name} - {message}")
            return error
        except Exception as handler_error:
            logger.error(f"Error handler itself failed: {handler_error}")
            return self._create_fallback_error(exception, context, file_path)

    def _categorize_error(
        self, exception: Exception, context: str, file_path: Optional[str]
    ) -> ErrorCategory:
        exception_type = type(exception).__name__
        exception_message = str(exception).lower()

        if isinstance(exception, PermissionError):
            return ErrorCategory.PERMISSION_ERROR
        elif isinstance(exception, FileNotFoundError):
            return ErrorCategory.FILE_ACCESS
        elif isinstance(exception, ImportError):
            return ErrorCategory.IMPORT_ERROR
        elif isinstance(exception, SyntaxError):
            return ErrorCategory.SYNTAX_ERROR
        elif isinstance(exception, TimeoutError):
            return ErrorCategory.TIMEOUT_ERROR
        elif isinstance(exception, MemoryError):
            return ErrorCategory.MEMORY_ERROR

        for pattern, category in self.error_patterns.items():
            if pattern in exception_type.lower() or pattern in exception_message:
                return category

        if "configuration" in context.lower() or "config" in context.lower():
            return ErrorCategory.CONFIGURATION
        elif "file" in context.lower() or "path" in context.lower():
            return ErrorCategory.FILE_ACCESS
        elif "import" in context.lower():
            return ErrorCategory.IMPORT_ERROR
        elif "syntax" in context.lower() or "parse" in context.lower():
            return ErrorCategory.SYNTAX_ERROR

        return ErrorCategory.UNKNOWN

    def _determine_severity(
        self, exception: Exception, category: ErrorCategory, context: str
    ) -> Severity:
        if category in [ErrorCategory.MEMORY_ERROR, ErrorCategory.INTERNAL_ERROR]:
            return Severity.CRITICAL
        if category in [
            ErrorCategory.CONFIGURATION,
            ErrorCategory.DEPENDENCY_MISSING,
            ErrorCategory.PERMISSION_ERROR,
        ]:
            return Severity.HIGH
        if category in [
            ErrorCategory.FILE_ACCESS,
            ErrorCategory.SYNTAX_ERROR,
            ErrorCategory.IMPORT_ERROR,
        ]:
            return Severity.MEDIUM
        if category in [ErrorCategory.VALIDATION_ERROR, ErrorCategory.TIMEOUT_ERROR]:
            return Severity.LOW
        return Severity.MEDIUM

    def _extract_error_message(self, exception: Exception) -> str:
        message = str(exception)
        if not message or message == exception.__class__.__name__:
            message = f"{exception.__class__.__name__} occurred"
        if len(message) > 500:
            message = message[:497] + "..."
        return message

    def _extract_stack_trace(self, exception: Exception) -> str:
        try:
            return "".join(
                traceback.format_exception(type(exception), exception, exception.__traceback__)
            )
        except Exception:
            return f"Stack trace unavailable for {type(exception).__name__}"

    def _extract_location_info(self, exception: Exception) -> tuple[Optional[int], Optional[int]]:
        line_number = getattr(exception, "lineno", None)
        column_number = getattr(exception, "offset", None)
        return line_number, column_number

    def _generate_fix_suggestions(
        self,
        exception: Exception,
        category: ErrorCategory,
        context: str,
        file_path: Optional[str],
    ) -> List[str]:
        suggestions: List[str] = []

        if category in self.fix_suggestions:
            suggestions.extend(self.fix_suggestions[category])

        exception_message = str(exception).lower()

        if category == ErrorCategory.FILE_ACCESS:
            if "not found" in exception_message:
                suggestions.append(f"Verify that the file path exists: {file_path}")
                suggestions.append("Check file permissions and accessibility")
            elif "permission denied" in exception_message:
                suggestions.append("Check file permissions - ensure read access")
                suggestions.append("Run with appropriate user privileges")

        elif category == ErrorCategory.IMPORT_ERROR:
            if "no module named" in exception_message:
                module_name = self._extract_module_name(exception_message)
                if module_name:
                    suggestions.append(f"Install missing module: pip install {module_name}")
                suggestions.append("Install required packages using pip")
                suggestions.append("Check Python package manager configuration")

        elif category == ErrorCategory.SYNTAX_ERROR:
            suggestions.append("Fix syntax errors in the target file")
            suggestions.append("Validate Python syntax using: python -m py_compile <file>")

        elif category == ErrorCategory.CONFIGURATION:
            suggestions.append("Check IntelliRefactor configuration file")
            suggestions.append("Validate configuration format and required fields")
            suggestions.append("Use default configuration if custom config is problematic")

        seen = set()
        unique: List[str] = []
        for s in suggestions:
            if s not in seen:
                seen.add(s)
                unique.append(s)
        return unique

    def _collect_diagnostic_info(
        self,
        exception: Exception,
        category: ErrorCategory,
        context: str,
        file_path: Optional[str],
        additional_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        diagnostic_info: Dict[str, Any] = {
            "exception_type": type(exception).__name__,
            "python_version": sys.version,
            "context": context,
        }

        if additional_info:
            diagnostic_info.update(additional_info)

        if category in self.diagnostic_collectors:
            try:
                collector = self.diagnostic_collectors[category]
                diagnostic_info.update(collector(exception, file_path))
            except Exception as e:
                diagnostic_info["diagnostic_collection_error"] = str(e)

        return diagnostic_info

    def _initialize_error_patterns(self) -> Dict[str, ErrorCategory]:
        return {
            "filenotfounderror": ErrorCategory.FILE_ACCESS,
            "permissionerror": ErrorCategory.PERMISSION_ERROR,
            "importerror": ErrorCategory.IMPORT_ERROR,
            "modulenotfounderror": ErrorCategory.IMPORT_ERROR,
            "syntaxerror": ErrorCategory.SYNTAX_ERROR,
            "indentationerror": ErrorCategory.SYNTAX_ERROR,
            "timeouterror": ErrorCategory.TIMEOUT_ERROR,
            "memoryerror": ErrorCategory.MEMORY_ERROR,
            "configuration": ErrorCategory.CONFIGURATION,
            "config": ErrorCategory.CONFIGURATION,
            "validation": ErrorCategory.VALIDATION_ERROR,
            "network": ErrorCategory.NETWORK_ERROR,
            "connection": ErrorCategory.NETWORK_ERROR,
            "no module named": ErrorCategory.DEPENDENCY_MISSING,
            "cannot import": ErrorCategory.IMPORT_ERROR,
            "invalid syntax": ErrorCategory.SYNTAX_ERROR,
            "unexpected eof": ErrorCategory.SYNTAX_ERROR,
            "access denied": ErrorCategory.PERMISSION_ERROR,
            "permission denied": ErrorCategory.PERMISSION_ERROR,
        }

    def _initialize_fix_suggestions(self) -> Dict[ErrorCategory, List[str]]:
        return {
            ErrorCategory.CONFIGURATION: [
                "Check IntelliRefactor configuration file format",
                "Validate all required configuration fields are present",
                "Try using default configuration",
                "Check configuration file permissions",
            ],
            ErrorCategory.FILE_ACCESS: [
                "Verify file path exists and is accessible",
                "Check file permissions",
                "Ensure file is not locked by another process",
                "Check disk space availability",
            ],
            ErrorCategory.SYNTAX_ERROR: [
                "Fix Python syntax errors in the target file",
                "Validate Python syntax using linter tools",
                "Check for proper Python indentation and syntax",
                "Use Python syntax validation tools to fix issues",
            ],
            ErrorCategory.IMPORT_ERROR: [
                "Install missing Python packages using pip",
                "Install required module dependencies",
                "Check package availability in package manager",
                "Use pip to install missing modules",
            ],
            ErrorCategory.DEPENDENCY_MISSING: [
                "Install required dependencies using pip",
                "Check requirements.txt file",
                "Verify package versions are compatible",
                "Update package manager and try again",
            ],
            ErrorCategory.PERMISSION_ERROR: [
                "Run with appropriate user privileges",
                "Check file and directory permissions",
                "Ensure write access to output directories",
                "Check if files are read-only",
            ],
            ErrorCategory.TIMEOUT_ERROR: [
                "Increase timeout settings",
                "Check system performance and resources",
                "Try analyzing smaller files or projects",
                "Check network connectivity if applicable",
            ],
            ErrorCategory.MEMORY_ERROR: [
                "Increase available memory",
                "Analyze smaller files or projects",
                "Close other memory-intensive applications",
                "Check for memory leaks in the analysis",
            ],
        }

    def _initialize_diagnostic_collectors(self) -> Dict[ErrorCategory, Callable]:
        return {
            ErrorCategory.FILE_ACCESS: self._collect_file_diagnostics,
            ErrorCategory.PERMISSION_ERROR: self._collect_file_diagnostics,
            ErrorCategory.IMPORT_ERROR: self._collect_import_diagnostics,
            ErrorCategory.SYNTAX_ERROR: self._collect_syntax_diagnostics,
            ErrorCategory.CONFIGURATION: self._collect_config_diagnostics,
            ErrorCategory.DEPENDENCY_MISSING: self._collect_dependency_diagnostics,
        }

    def _collect_file_diagnostics(
        self, exception: Exception, file_path: Optional[str]
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        if file_path:
            path_obj = Path(file_path)
            diagnostics["file_exists"] = path_obj.exists()
            diagnostics["file_path"] = str(path_obj.absolute())
            if path_obj.exists():
                try:
                    stat = path_obj.stat()
                    diagnostics["file_size"] = stat.st_size
                    diagnostics["file_permissions"] = oct(stat.st_mode)[-3:]
                    diagnostics["is_readable"] = path_obj.is_file() and os.access(path_obj, os.R_OK)
                except Exception as e:
                    diagnostics["stat_error"] = str(e)

            parent = path_obj.parent
            diagnostics["parent_exists"] = parent.exists()
            if parent.exists():
                diagnostics["parent_writable"] = os.access(parent, os.W_OK)
        return diagnostics

    def _collect_import_diagnostics(
        self, exception: Exception, file_path: Optional[str]
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {"python_path": sys.path, "installed_packages": []}
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                diagnostics["installed_packages"] = [pkg["name"] for pkg in packages]
        except Exception:
            diagnostics["package_list_error"] = "Could not retrieve installed packages"

        module_name = self._extract_module_name(str(exception))
        if module_name:
            diagnostics["missing_module"] = module_name
        return diagnostics

    def _collect_syntax_diagnostics(
        self, exception: Exception, file_path: Optional[str]
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        if file_path and Path(file_path).exists():
            try:
                content = Path(file_path).read_text(encoding="utf-8", errors="replace")
                try:
                    import ast
                    ast.parse(content)
                    diagnostics["ast_parse_success"] = True
                except SyntaxError as e:
                    diagnostics["ast_parse_success"] = False
                    diagnostics["ast_error_line"] = e.lineno
                    diagnostics["ast_error_offset"] = e.offset
                    diagnostics["ast_error_text"] = e.text

                lines = content.split("\n")
                diagnostics["total_lines"] = len(lines)
                diagnostics["non_empty_lines"] = len([line for line in lines if line.strip()])
            except Exception as e:
                diagnostics["file_read_error"] = str(e)
        return diagnostics

    def _collect_config_diagnostics(
        self, exception: Exception, file_path: Optional[str]
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        config_files = [
            ".intellirefactor/config.json",
            "intellirefactor.json",
            "pyproject.toml",
            "setup.cfg",
        ]
        for config_file in config_files:
            path = Path(config_file)
            diagnostics[f"{config_file}_exists"] = path.exists()
            if path.exists():
                diagnostics[f"{config_file}_readable"] = os.access(path, os.R_OK)
        return diagnostics

    def _collect_dependency_diagnostics(
        self, exception: Exception, file_path: Optional[str]
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        req_files = [
            "requirements.txt",
            "requirements-dev.txt",
            "Pipfile",
            "pyproject.toml",
        ]
        for req_file in req_files:
            path = Path(req_file)
            diagnostics[f"{req_file}_exists"] = path.exists()
        diagnostics["virtual_env"] = os.environ.get("VIRTUAL_ENV")
        diagnostics["conda_env"] = os.environ.get("CONDA_DEFAULT_ENV")
        return diagnostics

    def _extract_module_name(self, error_message: str) -> Optional[str]:
        import re
        patterns = [
            r"No module named '([^']+)'",
            r"No module named ([^\s]+)",
            r"cannot import name '([^']+)'",
            r"ModuleNotFoundError: No module named '([^']+)'",
        ]
        for pattern in patterns:
            match = re.search(pattern, error_message)
            if match:
                return match.group(1)
        return None

    def _get_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().isoformat()

    def _create_fallback_error(
        self, exception: Exception, context: str, file_path: Optional[str]
    ) -> AnalysisError:
        return AnalysisError(
            category=ErrorCategory.INTERNAL_ERROR,
            severity=Severity.CRITICAL,
            message=f"Error handler failed: {str(exception)}",
            context=context,
            file_path=file_path,
            suggested_fixes=[
                "Report this issue to IntelliRefactor developers",
                "Try with a simpler test case",
                "Check system resources and try again",
            ],
            diagnostic_info={"fallback_error": True},
            timestamp=self._get_timestamp(),
        )


class ErrorReporter:
    """Reports and formats analysis errors for different output formats."""

    def format_error(self, error: AnalysisError, format_type: str = "text") -> str:
        if format_type == "json":
            return json.dumps(error.to_dict(), indent=2, ensure_ascii=False)
        elif format_type == "markdown":
            return self._format_markdown(error)
        else:
            return self._format_text(error)

    def _format_text(self, error: AnalysisError) -> str:
        lines = [
            f"ERROR: {error.message}",
            f"Category: {error.category.name}",
            f"Severity: {error.severity.value.upper()}",
            f"Context: {error.context}",
        ]
        if error.file_path:
            location = error.file_path
            if error.line_number:
                location += f":{error.line_number}"
                if error.column_number:
                    location += f":{error.column_number}"
            lines.append(f"Location: {location}")

        if error.suggested_fixes:
            lines.append("\nSuggested fixes:")
            for i, fix in enumerate(error.suggested_fixes, 1):
                lines.append(f"  {i}. {fix}")

        if error.diagnostic_info:
            lines.append(f"\nDiagnostic info: {error.diagnostic_info}")

        return "\n".join(lines)

    def _format_markdown(self, error: AnalysisError) -> str:
        lines = [
            f"## Error: {error.message}",
            f"**Category:** {error.category.name}",
            f"**Severity:** {error.severity.value.upper()}",
            f"**Context:** {error.context}",
        ]

        if error.file_path:
            location = error.file_path
            if error.line_number:
                location += f":{error.line_number}"
                if error.column_number:
                    location += f":{error.column_number}"
            lines.append(f"**Location:** `{location}`")

        if error.suggested_fixes:
            lines.append("\n### Suggested Fixes")
            for i, fix in enumerate(error.suggested_fixes, 1):
                lines.append(f"{i}. {fix}")

        if error.diagnostic_info:
            lines.append("\n### Diagnostic Information")
            lines.append("```json")
            lines.append(json.dumps(error.diagnostic_info, indent=2, ensure_ascii=False))
            lines.append("```")

        return "\n".join(lines)

    def generate_error_summary(self, errors: List[AnalysisError]) -> Dict[str, Any]:
        if not errors:
            return {"total_errors": 0}

        summary: Dict[str, Any] = {
            "total_errors": len(errors),
            "by_category": {},
            "by_severity": {},
            "most_common_fixes": [],
            "files_with_errors": set(),
        }

        for error in errors:
            category = error.category.name
            severity = error.severity.value
            summary["by_category"][category] = summary["by_category"].get(category, 0) + 1
            summary["by_severity"][severity] = summary["by_severity"].get(severity, 0) + 1
            if error.file_path:
                summary["files_with_errors"].add(error.file_path)

        summary["files_with_errors"] = list(summary["files_with_errors"])

        fix_counts: Dict[str, int] = {}
        for error in errors:
            for fix in error.suggested_fixes:
                fix_counts[fix] = fix_counts.get(fix, 0) + 1

        summary["most_common_fixes"] = sorted(fix_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        return summary

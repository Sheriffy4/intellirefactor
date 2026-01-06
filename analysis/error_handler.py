"""
Comprehensive error handling system for IntelliRefactor analysis issues.

This module provides error detection, categorization, and diagnostic message generation
for various types of analysis failures and issues that can occur during IntelliRefactor operations.
"""

import logging
import traceback
import sys
import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
import subprocess
import json
import ast

logger = logging.getLogger(__name__)


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


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class AnalysisError:
    """Represents an analysis error with comprehensive information."""
    category: ErrorCategory
    severity: ErrorSeverity
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
            'category': self.category.name,
            'severity': self.severity.value,
            'message': self.message,
            'context': self.context,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'column_number': self.column_number,
            'stack_trace': self.stack_trace,
            'suggested_fixes': self.suggested_fixes,
            'diagnostic_info': self.diagnostic_info,
            'timestamp': self.timestamp
        }


class AnalysisErrorHandler:
    """
    Comprehensive error handler for analysis operations.
    
    Provides error detection, categorization, and diagnostic message generation
    for various types of analysis failures.
    """
    
    def __init__(self):
        """Initialize the error handler."""
        self.error_patterns = self._initialize_error_patterns()
        self.fix_suggestions = self._initialize_fix_suggestions()
        self.diagnostic_collectors = self._initialize_diagnostic_collectors()
    
    def handle_analysis_error(
        self,
        exception: Exception,
        context: str,
        file_path: Optional[str] = None,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> AnalysisError:
        """
        Handle and categorize an analysis error.
        
        Args:
            exception: The exception that occurred
            context: Context where the error occurred
            file_path: Optional file path where error occurred
            additional_info: Additional diagnostic information
            
        Returns:
            AnalysisError object with comprehensive error information
        """
        try:
            # Categorize the error
            category = self._categorize_error(exception, context, file_path)
            severity = self._determine_severity(exception, category, context)
            
            # Extract error details
            message = self._extract_error_message(exception)
            stack_trace = self._extract_stack_trace(exception)
            line_number, column_number = self._extract_location_info(exception)
            
            # Generate suggested fixes
            suggested_fixes = self._generate_fix_suggestions(
                exception, category, context, file_path
            )
            
            # Collect diagnostic information
            diagnostic_info = self._collect_diagnostic_info(
                exception, category, context, file_path, additional_info
            )
            
            # Create error object
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
                timestamp=self._get_timestamp()
            )
            
            logger.error(f"Analysis error handled: {category.name} - {message}")
            return error
            
        except Exception as handler_error:
            # Fallback error handling
            logger.error(f"Error handler itself failed: {handler_error}")
            return self._create_fallback_error(exception, context, file_path)
    
    def _categorize_error(
        self,
        exception: Exception,
        context: str,
        file_path: Optional[str]
    ) -> ErrorCategory:
        """Categorize the error based on exception type and context."""
        exception_type = type(exception).__name__
        exception_message = str(exception).lower()
        
        # Exception type-based categorization (prioritize specific types)
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
        
        # Check for specific error patterns
        for pattern, category in self.error_patterns.items():
            if pattern in exception_type.lower() or pattern in exception_message:
                return category
        
        # Context-based categorization
        if 'configuration' in context.lower() or 'config' in context.lower():
            return ErrorCategory.CONFIGURATION
        elif 'file' in context.lower() or 'path' in context.lower():
            return ErrorCategory.FILE_ACCESS
        elif 'import' in context.lower():
            return ErrorCategory.IMPORT_ERROR
        elif 'syntax' in context.lower() or 'parse' in context.lower():
            return ErrorCategory.SYNTAX_ERROR
        
        return ErrorCategory.UNKNOWN
    
    def _determine_severity(
        self,
        exception: Exception,
        category: ErrorCategory,
        context: str
    ) -> ErrorSeverity:
        """Determine the severity of the error."""
        # Critical errors that prevent system operation
        if category in [ErrorCategory.MEMORY_ERROR, ErrorCategory.INTERNAL_ERROR]:
            return ErrorSeverity.CRITICAL
        
        # High severity errors that prevent analysis
        if category in [
            ErrorCategory.CONFIGURATION,
            ErrorCategory.DEPENDENCY_MISSING,
            ErrorCategory.PERMISSION_ERROR
        ]:
            return ErrorSeverity.HIGH
        
        # Medium severity errors that affect specific files
        if category in [
            ErrorCategory.FILE_ACCESS,
            ErrorCategory.SYNTAX_ERROR,
            ErrorCategory.IMPORT_ERROR
        ]:
            return ErrorSeverity.MEDIUM
        
        # Low severity errors that are recoverable
        if category in [ErrorCategory.VALIDATION_ERROR, ErrorCategory.TIMEOUT_ERROR]:
            return ErrorSeverity.LOW
        
        return ErrorSeverity.MEDIUM
    
    def _extract_error_message(self, exception: Exception) -> str:
        """Extract a clear error message from the exception."""
        message = str(exception)
        
        # Clean up common error message patterns
        if not message or message == exception.__class__.__name__:
            message = f"{exception.__class__.__name__} occurred"
        
        # Truncate very long messages
        if len(message) > 500:
            message = message[:497] + "..."
        
        return message
    
    def _extract_stack_trace(self, exception: Exception) -> str:
        """Extract stack trace information."""
        try:
            return ''.join(traceback.format_exception(
                type(exception), exception, exception.__traceback__
            ))
        except Exception:
            return f"Stack trace unavailable for {type(exception).__name__}"
    
    def _extract_location_info(self, exception: Exception) -> tuple[Optional[int], Optional[int]]:
        """Extract line and column information if available."""
        line_number = None
        column_number = None
        
        if hasattr(exception, 'lineno'):
            line_number = exception.lineno
        if hasattr(exception, 'offset'):
            column_number = exception.offset
        
        return line_number, column_number
    
    def _generate_fix_suggestions(
        self,
        exception: Exception,
        category: ErrorCategory,
        context: str,
        file_path: Optional[str]
    ) -> List[str]:
        """Generate specific fix suggestions based on the error."""
        suggestions = []
        
        # Get category-specific suggestions
        if category in self.fix_suggestions:
            suggestions.extend(self.fix_suggestions[category])
        
        # Add context-specific suggestions
        exception_message = str(exception).lower()
        
        if category == ErrorCategory.FILE_ACCESS:
            if 'not found' in exception_message:
                suggestions.append(f"Verify that the file path exists: {file_path}")
                suggestions.append("Check file permissions and accessibility")
            elif 'permission denied' in exception_message:
                suggestions.append("Check file permissions - ensure read access")
                suggestions.append("Run with appropriate user privileges")
        
        elif category == ErrorCategory.IMPORT_ERROR:
            if 'no module named' in exception_message:
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
        
        # Remove duplicates while preserving order
        seen = set()
        unique_suggestions = []
        for suggestion in suggestions:
            if suggestion not in seen:
                seen.add(suggestion)
                unique_suggestions.append(suggestion)
        
        return unique_suggestions
    
    def _collect_diagnostic_info(
        self,
        exception: Exception,
        category: ErrorCategory,
        context: str,
        file_path: Optional[str],
        additional_info: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Collect comprehensive diagnostic information."""
        diagnostic_info = {
            'exception_type': type(exception).__name__,
            'python_version': sys.version,
            'context': context
        }
        
        # Add additional info if provided
        if additional_info:
            diagnostic_info.update(additional_info)
        
        # Collect category-specific diagnostics
        if category in self.diagnostic_collectors:
            try:
                collector = self.diagnostic_collectors[category]
                category_diagnostics = collector(exception, file_path)
                diagnostic_info.update(category_diagnostics)
            except Exception as e:
                diagnostic_info['diagnostic_collection_error'] = str(e)
        
        return diagnostic_info
    
    def _initialize_error_patterns(self) -> Dict[str, ErrorCategory]:
        """Initialize error pattern mappings."""
        return {
            'filenotfounderror': ErrorCategory.FILE_ACCESS,
            'permissionerror': ErrorCategory.PERMISSION_ERROR,
            'importerror': ErrorCategory.IMPORT_ERROR,
            'modulenotfounderror': ErrorCategory.IMPORT_ERROR,
            'syntaxerror': ErrorCategory.SYNTAX_ERROR,
            'indentationerror': ErrorCategory.SYNTAX_ERROR,
            'timeouterror': ErrorCategory.TIMEOUT_ERROR,
            'memoryerror': ErrorCategory.MEMORY_ERROR,
            'configuration': ErrorCategory.CONFIGURATION,
            'config': ErrorCategory.CONFIGURATION,
            'validation': ErrorCategory.VALIDATION_ERROR,
            'network': ErrorCategory.NETWORK_ERROR,
            'connection': ErrorCategory.NETWORK_ERROR,
            'no module named': ErrorCategory.DEPENDENCY_MISSING,
            'cannot import': ErrorCategory.IMPORT_ERROR,
            'invalid syntax': ErrorCategory.SYNTAX_ERROR,
            'unexpected eof': ErrorCategory.SYNTAX_ERROR,
            'access denied': ErrorCategory.PERMISSION_ERROR,
            'permission denied': ErrorCategory.PERMISSION_ERROR,
        }
    
    def _initialize_fix_suggestions(self) -> Dict[ErrorCategory, List[str]]:
        """Initialize fix suggestions for each error category."""
        return {
            ErrorCategory.CONFIGURATION: [
                "Check IntelliRefactor configuration file format",
                "Validate all required configuration fields are present",
                "Try using default configuration",
                "Check configuration file permissions"
            ],
            ErrorCategory.FILE_ACCESS: [
                "Verify file path exists and is accessible",
                "Check file permissions",
                "Ensure file is not locked by another process",
                "Check disk space availability"
            ],
            ErrorCategory.SYNTAX_ERROR: [
                "Fix Python syntax errors in the target file",
                "Validate Python syntax using linter tools",
                "Check for proper Python indentation and syntax",
                "Use Python syntax validation tools to fix issues"
            ],
            ErrorCategory.IMPORT_ERROR: [
                "Install missing Python packages using pip",
                "Install required module dependencies",
                "Check package availability in package manager",
                "Use pip to install missing modules"
            ],
            ErrorCategory.DEPENDENCY_MISSING: [
                "Install required dependencies using pip",
                "Check requirements.txt file",
                "Verify package versions are compatible",
                "Update package manager and try again"
            ],
            ErrorCategory.PERMISSION_ERROR: [
                "Run with appropriate user privileges",
                "Check file and directory permissions",
                "Ensure write access to output directories",
                "Check if files are read-only"
            ],
            ErrorCategory.TIMEOUT_ERROR: [
                "Increase timeout settings",
                "Check system performance and resources",
                "Try analyzing smaller files or projects",
                "Check network connectivity if applicable"
            ],
            ErrorCategory.MEMORY_ERROR: [
                "Increase available memory",
                "Analyze smaller files or projects",
                "Close other memory-intensive applications",
                "Check for memory leaks in the analysis"
            ]
        }
    
    def _initialize_diagnostic_collectors(self) -> Dict[ErrorCategory, Callable]:
        """Initialize diagnostic information collectors."""
        return {
            ErrorCategory.FILE_ACCESS: self._collect_file_diagnostics,
            ErrorCategory.PERMISSION_ERROR: self._collect_file_diagnostics,  # Also collect file diagnostics for permission errors
            ErrorCategory.IMPORT_ERROR: self._collect_import_diagnostics,
            ErrorCategory.SYNTAX_ERROR: self._collect_syntax_diagnostics,
            ErrorCategory.CONFIGURATION: self._collect_config_diagnostics,
            ErrorCategory.DEPENDENCY_MISSING: self._collect_dependency_diagnostics
        }
    
    def _collect_file_diagnostics(self, exception: Exception, file_path: Optional[str]) -> Dict[str, Any]:
        """Collect file-related diagnostic information."""
        diagnostics = {}
        
        if file_path:
            path_obj = Path(file_path)
            diagnostics['file_exists'] = path_obj.exists()
            diagnostics['file_path'] = str(path_obj.absolute())
            
            if path_obj.exists():
                try:
                    stat = path_obj.stat()
                    diagnostics['file_size'] = stat.st_size
                    diagnostics['file_permissions'] = oct(stat.st_mode)[-3:]
                    diagnostics['is_readable'] = path_obj.is_file() and os.access(path_obj, os.R_OK)
                except Exception as e:
                    diagnostics['stat_error'] = str(e)
            
            # Check parent directory
            parent = path_obj.parent
            diagnostics['parent_exists'] = parent.exists()
            if parent.exists():
                diagnostics['parent_writable'] = os.access(parent, os.W_OK)
        
        return diagnostics
    
    def _collect_import_diagnostics(self, exception: Exception, file_path: Optional[str]) -> Dict[str, Any]:
        """Collect import-related diagnostic information."""
        diagnostics = {
            'python_path': sys.path,
            'installed_packages': []
        }
        
        # Try to get installed packages
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'list', '--format=json'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                packages = json.loads(result.stdout)
                diagnostics['installed_packages'] = [pkg['name'] for pkg in packages]
        except Exception:
            diagnostics['package_list_error'] = "Could not retrieve installed packages"
        
        # Extract module name from error
        module_name = self._extract_module_name(str(exception))
        if module_name:
            diagnostics['missing_module'] = module_name
        
        return diagnostics
    
    def _collect_syntax_diagnostics(self, exception: Exception, file_path: Optional[str]) -> Dict[str, Any]:
        """Collect syntax-related diagnostic information."""
        diagnostics = {}
        
        if file_path and Path(file_path).exists():
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Try to parse with ast to get more details
                try:
                    ast.parse(content)
                    diagnostics['ast_parse_success'] = True
                except SyntaxError as e:
                    diagnostics['ast_parse_success'] = False
                    diagnostics['ast_error_line'] = e.lineno
                    diagnostics['ast_error_offset'] = e.offset
                    diagnostics['ast_error_text'] = e.text
                
                # Basic file statistics
                lines = content.split('\n')
                diagnostics['total_lines'] = len(lines)
                diagnostics['non_empty_lines'] = len([line for line in lines if line.strip()])
                
            except Exception as e:
                diagnostics['file_read_error'] = str(e)
        
        return diagnostics
    
    def _collect_config_diagnostics(self, exception: Exception, file_path: Optional[str]) -> Dict[str, Any]:
        """Collect configuration-related diagnostic information."""
        diagnostics = {}
        
        # Check for common config files
        config_files = [
            '.intellirefactor/config.json',
            'intellirefactor.json',
            'pyproject.toml',
            'setup.cfg'
        ]
        
        for config_file in config_files:
            path = Path(config_file)
            diagnostics[f'{config_file}_exists'] = path.exists()
            if path.exists():
                diagnostics[f'{config_file}_readable'] = os.access(path, os.R_OK)
        
        return diagnostics
    
    def _collect_dependency_diagnostics(self, exception: Exception, file_path: Optional[str]) -> Dict[str, Any]:
        """Collect dependency-related diagnostic information."""
        diagnostics = {}
        
        # Check for requirements files
        req_files = ['requirements.txt', 'requirements-dev.txt', 'Pipfile', 'pyproject.toml']
        for req_file in req_files:
            path = Path(req_file)
            diagnostics[f'{req_file}_exists'] = path.exists()
        
        # Check virtual environment
        diagnostics['virtual_env'] = os.environ.get('VIRTUAL_ENV')
        diagnostics['conda_env'] = os.environ.get('CONDA_DEFAULT_ENV')
        
        return diagnostics
    
    def _extract_module_name(self, error_message: str) -> Optional[str]:
        """Extract module name from import error message."""
        import re
        
        patterns = [
            r"No module named '([^']+)'",
            r"No module named ([^\s]+)",
            r"cannot import name '([^']+)'",
            r"ModuleNotFoundError: No module named '([^']+)'"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, error_message)
            if match:
                return match.group(1)
        
        return None
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _create_fallback_error(
        self,
        exception: Exception,
        context: str,
        file_path: Optional[str]
    ) -> AnalysisError:
        """Create a fallback error when error handling itself fails."""
        return AnalysisError(
            category=ErrorCategory.INTERNAL_ERROR,
            severity=ErrorSeverity.CRITICAL,
            message=f"Error handler failed: {str(exception)}",
            context=context,
            file_path=file_path,
            suggested_fixes=[
                "Report this issue to IntelliRefactor developers",
                "Try with a simpler test case",
                "Check system resources and try again"
            ],
            diagnostic_info={'fallback_error': True},
            timestamp=self._get_timestamp()
        )


class ErrorReporter:
    """Reports and formats analysis errors for different output formats."""
    
    def __init__(self):
        """Initialize the error reporter."""
        pass
    
    def format_error(self, error: AnalysisError, format_type: str = 'text') -> str:
        """
        Format an error for display.
        
        Args:
            error: The AnalysisError to format
            format_type: Output format ('text', 'json', 'markdown')
            
        Returns:
            Formatted error string
        """
        if format_type == 'json':
            return json.dumps(error.to_dict(), indent=2)
        elif format_type == 'markdown':
            return self._format_markdown(error)
        else:
            return self._format_text(error)
    
    def _format_text(self, error: AnalysisError) -> str:
        """Format error as plain text."""
        lines = [
            f"ERROR: {error.message}",
            f"Category: {error.category.name}",
            f"Severity: {error.severity.value.upper()}",
            f"Context: {error.context}"
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
        
        return '\n'.join(lines)
    
    def _format_markdown(self, error: AnalysisError) -> str:
        """Format error as markdown."""
        lines = [
            f"## Error: {error.message}",
            f"**Category:** {error.category.name}",
            f"**Severity:** {error.severity.value.upper()}",
            f"**Context:** {error.context}"
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
            lines.append(json.dumps(error.diagnostic_info, indent=2))
            lines.append("```")
        
        return '\n'.join(lines)
    
    def generate_error_summary(self, errors: List[AnalysisError]) -> Dict[str, Any]:
        """Generate a summary of multiple errors."""
        if not errors:
            return {'total_errors': 0}
        
        summary = {
            'total_errors': len(errors),
            'by_category': {},
            'by_severity': {},
            'most_common_fixes': [],
            'files_with_errors': set()
        }
        
        # Count by category and severity
        for error in errors:
            category = error.category.name
            severity = error.severity.value
            
            summary['by_category'][category] = summary['by_category'].get(category, 0) + 1
            summary['by_severity'][severity] = summary['by_severity'].get(severity, 0) + 1
            
            if error.file_path:
                summary['files_with_errors'].add(error.file_path)
        
        # Convert set to list for JSON serialization
        summary['files_with_errors'] = list(summary['files_with_errors'])
        
        # Find most common fixes
        fix_counts = {}
        for error in errors:
            for fix in error.suggested_fixes:
                fix_counts[fix] = fix_counts.get(fix, 0) + 1
        
        summary['most_common_fixes'] = sorted(
            fix_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        
        return summary
"""
Standardized error handling for IntelliRefactor analysis.

Provides utilities for safe execution of analysis stages and consistent
error reporting across all analyzers.

Key Features:
1. Safe execution wrapper with automatic error collection
2. Stage-based error classification
3. Consistent error formatting
4. Graceful degradation on failures
"""

import functools
import logging
import traceback
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, TypeVar

from .models import AnalysisError, AnalysisStage

logger = logging.getLogger(__name__)

T = TypeVar('T')


class StandardErrorHandler:
    """
    Central error handling coordinator.
    
    Collects and manages errors encountered during analysis.
    """
    
    def __init__(self):
        self.errors: List[AnalysisError] = []
    
    def add_error(self, error: AnalysisError) -> None:
        """Add an error to the collection."""
        self.errors.append(error)
        logger.debug(f"Added error: {error.file_path} - {error.message}")
    
    def get_errors(self) -> List[AnalysisError]:
        """Get all collected errors."""
        return self.errors.copy()
    
    def get_errors_by_stage(self, stage: AnalysisStage) -> List[AnalysisError]:
        """Get errors filtered by analysis stage."""
        return [e for e in self.errors if e.stage == stage]
    
    def get_errors_by_file(self, file_path: str) -> List[AnalysisError]:
        """Get errors for a specific file."""
        return [e for e in self.errors if e.file_path == file_path]
    
    def clear_errors(self) -> None:
        """Clear all collected errors."""
        self.errors.clear()
    
    def has_errors(self) -> bool:
        """Check if any errors have been collected."""
        return len(self.errors) > 0
    
    def error_count(self) -> int:
        """Get total number of errors."""
        return len(self.errors)
    
    def to_dict(self) -> dict:
        """Convert errors to dictionary format."""
        return {
            "total_errors": len(self.errors),
            "errors": [e.to_dict() for e in self.errors]
        }


def safe_run(
    stage: AnalysisStage,
    file_path: Path,
    func: Callable[..., T],
    *args,
    error_handler: Optional[StandardErrorHandler] = None,
    **kwargs
) -> Tuple[Optional[T], Optional[AnalysisError]]:
    """
    Safely execute a function and capture any exceptions.
    
    Args:
        stage: Analysis stage where execution occurs
        file_path: Path to file being analyzed
        func: Function to execute
        error_handler: Optional error handler to collect errors
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Tuple of (result, error) where one is None and the other contains data
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        error = _create_analysis_error(stage, file_path, e)
        
        if error_handler:
            error_handler.add_error(error)
        
        logger.warning(f"Analysis error in {file_path}: {error.message}")
        return None, error


def safe_run_with_default(
    stage: AnalysisStage,
    file_path: Path,
    func: Callable[..., T],
    default_value: T,
    error_handler: Optional[StandardErrorHandler] = None,
    *args,
    **kwargs
) -> T:
    """
    Safely execute a function, returning a default value on failure.
    
    Args:
        stage: Analysis stage where execution occurs
        file_path: Path to file being analyzed
        func: Function to execute
        default_value: Value to return if function fails
        error_handler: Optional error handler to collect errors
        *args: Positional arguments for func
        **kwargs: Keyword arguments for func
        
    Returns:
        Function result or default_value if exception occurs
    """
    result, error = safe_run(stage, file_path, func, *args, error_handler=error_handler, **kwargs)
    
    if error is not None:
        return default_value
    
    return result


def _create_analysis_error(
    stage: AnalysisStage,
    file_path: Path,
    exception: Exception
) -> AnalysisError:
    """Create standardized AnalysisError from exception."""
    return AnalysisError(
        file_path=str(file_path),
        stage=stage,
        error_type=type(exception).__name__,
        message=str(exception),
        traceback=traceback.format_exc()
    )


def safe_method(stage: AnalysisStage):
    """
    Decorator for making methods safe with automatic error handling.
    
    Usage:
        @safe_method(AnalysisStage.PARSE)
        def parse_file(self, file_path: Path) -> ParseResult:
            # Implementation here
            pass
            
    The decorated method will automatically catch exceptions and return
    (result, error) tuple, with errors added to self.error_handler if present.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., Tuple[Optional[T], Optional[AnalysisError]]]:
        @functools.wraps(func)
        def wrapper(self, file_path: Path, *args, **kwargs) -> Tuple[Optional[T], Optional[AnalysisError]]:
            # Check if object has error_handler attribute
            error_handler = getattr(self, 'error_handler', None)
            
            # IMPORTANT: positional args must go before keyword-only args
            return safe_run(
                stage,
                file_path,
                func,
                self,
                file_path,
                *args,
                error_handler=error_handler,
                **kwargs,
            )
        return wrapper
    return decorator


# Predefined safe functions for common operations

def safe_read_file(
    file_path: Path,
    encoding: str = 'utf-8',
    error_handler: Optional[StandardErrorHandler] = None
) -> Optional[str]:
    """
    Safely read a file's contents.
    
    Args:
        file_path: Path to file
        encoding: File encoding
        error_handler: Optional error handler
        
    Returns:
        File contents or None if error occurs
    """
    def _read():
        return file_path.read_text(encoding=encoding)
    
    return safe_run_with_default(
        stage=AnalysisStage.READ,
        file_path=file_path,
        func=_read,
        default_value=None,
        error_handler=error_handler
    )


def safe_parse_ast(
    source_code: str,
    file_path: Path,
    error_handler: Optional[StandardErrorHandler] = None
) -> Optional[Any]:
    """
    Safely parse Python source code into AST.
    
    Args:
        source_code: Python source code
        file_path: Path to source file (for error reporting)
        error_handler: Optional error handler
        
    Returns:
        AST object or None if parsing fails
    """
    import ast
    
    def _parse():
        return ast.parse(source_code, filename=str(file_path))
    
    return safe_run_with_default(
        stage=AnalysisStage.PARSE,
        file_path=file_path,
        func=_parse,
        default_value=None,
        error_handler=error_handler
    )

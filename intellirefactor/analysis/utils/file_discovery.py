"""
Unified file discovery utility for deterministic file scanning.

This module provides a single source of truth for discovering Python files
in a project, with proper exclusion patterns and deterministic sorting.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import List, Optional, Set


# Default exclusion patterns (directories and file patterns to ignore)
DEFAULT_EXCLUDE_DIRS = {
    ".git",
    ".hg", 
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache", 
    ".ruff_cache",
    ".tox",
    ".nox",
    "node_modules",
    ".intellirefactor",  # Our own output directory
}

DEFAULT_EXCLUDE_GLOBS = [
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "*.so",
    "*.dll",
    "*.egg",
    ".*",  # Hidden files
]


def discover_python_files(
    project_root: Path,
    include: Optional[List[str]] = None,
    exclude_dirs: Optional[Set[str]] = None,
    exclude_globs: Optional[List[str]] = None,
) -> List[Path]:
    """
    Discover Python files in a project with deterministic ordering.
    
    Args:
        project_root: Root directory to scan
        include: Glob patterns to include (default: ["**/*.py"])
        exclude_dirs: Directory names to exclude
        exclude_globs: File patterns to exclude
        
    Returns:
        Sorted list of Python file paths
    """
    project_root = project_root.resolve()
    
    if include is None:
        include = ["**/*.py"]
    
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS
        
    if exclude_globs is None:
        exclude_globs = DEFAULT_EXCLUDE_GLOBS
    
    found_files: Set[Path] = set()
    
    # Expand all include patterns
    for pattern in include:
        for file_path in project_root.glob(pattern):
            if file_path.is_file():
                found_files.add(file_path.resolve())
    
    # Filter out excluded directories and files
    filtered_files = []
    for file_path in found_files:
        # Check if file is in excluded directory
        try:
            relative_path = file_path.relative_to(project_root)
        except Exception:
            # If project_root is not a parent (shouldn't happen often), keep file but don't crash
            relative_path = file_path
        path_parts = set(relative_path.parts)
        
        if path_parts & exclude_dirs:
            continue
        
        # Exclude any *.egg-info directories
        if any(str(p).endswith(".egg-info") for p in relative_path.parts):
            continue
            
        # Check file patterns
        should_exclude = False
        for glob_pattern in exclude_globs:
            if file_path.match(glob_pattern):
                should_exclude = True
                break
                
        if not should_exclude:
            filtered_files.append(file_path)
    
    # Return deterministically sorted list
    return sorted(filtered_files)


def is_excluded_directory(dir_name: str, exclude_dirs: Optional[Set[str]] = None) -> bool:
    """Check if a directory name should be excluded."""
    if exclude_dirs is None:
        exclude_dirs = DEFAULT_EXCLUDE_DIRS
    return dir_name in exclude_dirs


def is_excluded_file(file_path: Path, exclude_globs: Optional[List[str]] = None) -> bool:
    """Check if a file should be excluded based on glob patterns."""
    if exclude_globs is None:
        exclude_globs = DEFAULT_EXCLUDE_GLOBS
    
    for pattern in exclude_globs:
        if fnmatch.fnmatch(file_path.name, pattern):
            return True
    return False


# Convenience function for common use case
def get_project_python_files(project_root: Path) -> List[Path]:
    """
    Get all Python files in a project using default settings.
    
    This is the recommended way to scan a project consistently.
    """
    return discover_python_files(project_root)
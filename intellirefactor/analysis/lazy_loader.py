"""
Lazy loading for project context in IntelliRefactor.

This module implements lazy loading functionality to efficiently handle large projects
where full indexing might take too long. It provides mechanisms to load project context
on-demand and cache frequently accessed data.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass
from functools import wraps
import threading
import time

from .index.store import IndexStore
from .index.query import IndexQuery


@dataclass
class LazyLoadConfig:
    """Configuration for lazy loading behavior."""

    enable_caching: bool = True
    cache_size_limit: int = 1000
    lazy_file_scanning: bool = True
    quick_dependency_check: bool = True
    min_file_size_for_full_analysis: int = 1024 * 10  # 10KB
    max_project_files_for_full_load: int = 1000


class LazyProjectContext:
    """
    Lazy loading wrapper for project context that provides on-demand access
    to project data without loading everything upfront.
    """

    def __init__(
        self,
        project_root: Path,
        index_store: IndexStore,
        config: Optional[LazyLoadConfig] = None,
    ):
        """
        Initialize lazy project context.

        Args:
            project_root: Root path of the project
            index_store: IndexStore instance for data access
            config: Lazy loading configuration
        """
        self.project_root = Path(project_root)
        self.index.store = index_store
        self.config = config or LazyLoadConfig()
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()

        # Cache for frequently accessed data
        self._file_cache: Dict[str, Any] = {}
        self._symbol_cache: Dict[str, Any] = {}
        self._dependency_cache: Dict[str, Any] = {}

        # Track loaded files to avoid repeated full analysis
        self._loaded_files: Set[str] = set()
        self._file_hashes: Optional[Dict[str, str]] = None

        # Initialize query interface
        self.query = IndexQuery(index_store)

        self.logger.info(f"LazyProjectContext initialized for {self.project_root}")

    def _with_cache(
        self,
        cache_name: str,
        key: str,
        loader: Callable[[], Any],
        max_size: Optional[int] = None,
    ):
        """
        Generic caching wrapper with size limit.

        Args:
            cache_name: Name of the cache attribute
            key: Cache key
            loader: Function to load data if not in cache
            max_size: Maximum cache size (None for unlimited)
        """
        cache = getattr(self, cache_name)

        if key in cache:
            return cache[key]

        if max_size and len(cache) >= max_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(cache))
            del cache[oldest_key]

        data = loader()
        cache[key] = data
        return data

    def _get_file_hash(self, file_path: str) -> Optional[str]:
        """Get hash for a file from the index."""
        if self._file_hashes is None:
            self._file_hashes = self.index.store.get_all_file_hashes()

        return self._file_hashes.get(file_path)

    def _is_file_modified(self, file_path: str) -> bool:
        """Check if file has been modified since last analysis."""

        file_path_obj = Path(file_path)
        if not file_path_obj.exists():
            return True

        current_hash = self._calculate_file_hash(file_path_obj)
        stored_hash = self._get_file_hash(file_path)

        return current_hash != stored_hash

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate hash for a file."""
        import hashlib

        with open(file_path, "rb") as f:
            content = f.read()
            return hashlib.blake2b(content, digest_size=16).hexdigest()

    def get_file_info(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get file information with lazy loading.

        Args:
            file_path: Path to the file

        Returns:
            File information or None if not found
        """

        def _load_file_info():
            return self.index.store.get_file(file_path)

        if self.config.enable_caching:
            return self._with_cache(
                "_file_cache", file_path, _load_file_info, self.config.cache_size_limit
            )
        else:
            return _load_file_info()

    def get_symbol_info(self, symbol_uid: str) -> Optional[Dict[str, Any]]:
        """
        Get symbol information with lazy loading.

        Args:
            symbol_uid: Unique identifier for the symbol

        Returns:
            Symbol information or None if not found
        """

        def _load_symbol_info():
            return self.index.store.get_symbol(symbol_uid)

        if self.config.enable_caching:
            return self._with_cache(
                "_symbol_cache",
                symbol_uid,
                _load_symbol_info,
                self.config.cache_size_limit,
            )
        else:
            return _load_symbol_info()

    def find_importers_of_module(self, module_path: str) -> List[Dict[str, Any]]:
        """
        Find which modules import the given module using lazy loading.

        Args:
            module_path: Path to the module to find importers for

        Returns:
            List of modules that import the given module
        """

        def _load_importers():
            return self.query.find_importers_of_module(module_path)

        cache_key = f"importers_{module_path}"
        if self.config.enable_caching:
            return self._with_cache(
                "_dependency_cache",
                cache_key,
                _load_importers,
                self.config.cache_size_limit,
            )
        else:
            return _load_importers()

    def find_usage_of_symbol(self, symbol_uid: str) -> List[Dict[str, Any]]:
        """
        Find where a symbol is used (reverse dependency lookup) with lazy loading.

        Args:
            symbol_uid: UID of the symbol to find usage for

        Returns:
            List of locations where the symbol is used
        """

        def _load_symbol_usage():
            return self.query.find_usage_of_symbol(symbol_uid)

        cache_key = f"usage_{symbol_uid}"
        if self.config.enable_caching:
            return self._with_cache(
                "_dependency_cache",
                cache_key,
                _load_symbol_usage,
                self.config.cache_size_limit,
            )
        else:
            return _load_symbol_usage()

    def is_symbol_used_in_project(self, symbol_uid: str) -> bool:
        """
        Check if a symbol is used anywhere in the project with lazy loading.

        Args:
            symbol_uid: UID of the symbol to check

        Returns:
            True if the symbol is used in the project, False otherwise
        """

        def _check_symbol_usage():
            return self.query.is_symbol_used_in_project(symbol_uid)

        cache_key = f"used_{symbol_uid}"
        if self.config.enable_caching:
            return self._with_cache(
                "_dependency_cache",
                cache_key,
                _check_symbol_usage,
                self.config.cache_size_limit,
            )
        else:
            return _check_symbol_usage()

    def quick_scan_file_for_imports(self, file_path: str) -> List[str]:
        """
        Quick scan a file for import statements without full AST parsing.

        Args:
            file_path: Path to the file to scan

        Returns:
            List of imported module names
        """
        import re

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Simple regex to find import statements
            # This is much faster than full AST parsing
            import_pattern = r"(?:^|\n)\s*(?:import|from)\s+([a-zA-Z0-9_.]+)"
            matches = re.findall(import_pattern, content)

            return list(set(matches))  # Remove duplicates
        except Exception:
            return []

    def lazy_load_project_structure(self) -> Dict[str, Any]:
        """
        Lazy load basic project structure without full analysis.

        Returns:
            Basic project structure information
        """
        structure = {
            "total_files": 0,
            "python_files": [],
            "module_imports": {},  # file_path -> [imported_modules]
            "last_updated": time.time(),
        }

        # Find Python files in the project
        python_files = list(self.project_root.rglob("*.py"))
        structure["total_files"] = len(python_files)
        structure["python_files"] = [str(f.relative_to(self.project_root)) for f in python_files]

        # For large projects, only scan for imports if enabled
        if (
            self.config.lazy_file_scanning
            and len(python_files) <= self.config.max_project_files_for_full_load
        ):
            for py_file in python_files:
                rel_path = str(py_file.relative_to(self.project_root))
                imports = self.quick_scan_file_for_imports(str(py_file))
                if imports:
                    structure["module_imports"][rel_path] = imports

        return structure

    def get_cached_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the lazy loading cache.

        Returns:
            Cache statistics
        """
        return {
            "file_cache_size": len(self._file_cache),
            "symbol_cache_size": len(self._symbol_cache),
            "dependency_cache_size": len(self._dependency_cache),
            "loaded_files_count": len(self._loaded_files),
            "total_cache_size": (
                len(self._file_cache) + len(self._symbol_cache) + len(self._dependency_cache)
            ),
        }

    def clear_cache(self) -> None:
        """Clear all caches."""
        with self._lock:
            self._file_cache.clear()
            self._symbol_cache.clear()
            self._dependency_cache.clear()
            self._loaded_files.clear()
            self._file_hashes = None

            self.logger.info("Lazy loading cache cleared")

    def preload_frequently_used_data(self, project_path: str) -> None:
        """
        Preload frequently used data for the project in a lazy manner.

        Args:
            project_path: Path to the project to preload data for
        """
        self.logger.info(f"Preloading frequently used data for {project_path}")

        # This is a placeholder for more sophisticated preloading logic
        # that could identify the most important files/modules to load first
        pass


def lazy_load_enabled(func):
    """
    Decorator to enable lazy loading for specific operations.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = time.time() - start_time

        if duration > 1.0:  # If operation took more than 1 second
            logging.getLogger(__name__).warning(
                f"Operation {func.__name__} took {duration:.2f}s - "
                f"consider optimizing for large projects"
            )

        return result

    return wrapper


class LazyContextManager:
    """
    Manager for lazy project contexts that handles multiple projects.
    """

    def __init__(self, config: Optional[LazyLoadConfig] = None):
        """
        Initialize lazy context manager.

        Args:
            config: Default lazy loading configuration
        """
        self.config = config or LazyLoadConfig()
        self._contexts: Dict[str, LazyProjectContext] = {}
        self.logger = logging.getLogger(__name__)

    def get_context(
        self,
        project_root: Path,
        index_store: IndexStore,
        config: Optional[LazyLoadConfig] = None,
    ) -> LazyProjectContext:
        """
        Get or create a lazy project context.

        Args:
            project_root: Root path of the project
            index_store: IndexStore instance for data access
            config: Lazy loading configuration (optional)

        Returns:
            LazyProjectContext instance
        """
        project_key = str(Path(project_root).resolve())

        if project_key not in self._contexts:
            context_config = config or self.config
            self._contexts[project_key] = LazyProjectContext(
                project_root, index_store, context_config
            )
            self.logger.info(f"Created new lazy context for {project_root}")

        return self._contexts[project_key]

    def clear_context(self, project_root: Path) -> bool:
        """
        Clear the context for a specific project.

        Args:
            project_root: Root path of the project to clear

        Returns:
            True if context was cleared, False if not found
        """
        project_key = str(Path(project_root).resolve())

        if project_key in self._contexts:
            del self._contexts[project_key]
            self.logger.info(f"Cleared lazy context for {project_root}")
            return True

        return False

    def clear_all_contexts(self) -> None:
        """Clear all project contexts."""
        self._contexts.clear()
        self.logger.info("Cleared all lazy contexts")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all managed contexts.

        Returns:
            Statistics about all contexts
        """
        stats = {"total_contexts": len(self._contexts), "contexts": {}}

        for project_key, context in self._contexts.items():
            stats["contexts"][project_key] = context.get_cached_stats()

        return stats

"""File operations and caching for decomposition analysis.

Extracted from DecompositionAnalyzer to reduce god class complexity.
Handles file I/O, AST parsing with caching, and path resolution.
"""
import ast
import difflib
import logging
import py_compile
import shutil
from pathlib import Path
from typing import Dict, Tuple, List, Optional, Any

from .models import FunctionalBlock


class FileOperations:
    """Manages file I/O operations with caching for decomposition analysis."""
    
    def __init__(self, config: Any, logger: logging.Logger):
        """Initialize file operations manager.
        
        Args:
            config: DecompositionConfig instance
            logger: Logger instance for operation logging
        """
        self.config = config
        self.logger = logger
        
        # Caches for performance
        self._file_text_cache: Dict[Tuple[Path, bool], str] = {}
        self._file_ast_cache: Dict[Path, ast.Module] = {}
    
    def read_text(self, path: Path, bom: bool = False) -> str:
        """Read text file with caching.
        
        Args:
            path: Path to file
            bom: If True, use utf-8-sig encoding to handle BOM
            
        Returns:
            File content as string (empty string if file doesn't exist)
        """
        path = path.resolve()
        key = (path, bool(bom))
        if key in self._file_text_cache:
            return self._file_text_cache[key]
        enc = "utf-8-sig" if bom else "utf-8"
        text = path.read_text(encoding=enc) if path.exists() else ""
        self._file_text_cache[key] = text
        return text
    
    def write_text(self, path: Path, content: str) -> None:
        """Write text to file and invalidate caches.
        
        Args:
            path: Path to file
            content: Content to write
        """
        path = path.resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        # Invalidate both bom variants
        self._file_text_cache.pop((path, False), None)
        self._file_text_cache.pop((path, True), None)
        self._file_ast_cache.pop(path, None)
    
    def parse_file(self, path: Path) -> ast.Module:
        """Parse Python file to AST with caching.
        
        Args:
            path: Path to Python file
            
        Returns:
            Parsed AST module
        """
        path = path.resolve()
        if path in self._file_ast_cache:
            return self._file_ast_cache[path]
        txt = self.read_text(path, bom=True)
        tree = ast.parse(txt, filename=str(path))
        self._file_ast_cache[path] = tree
        return tree
    
    def slice_lines(self, src: str, lineno: int, end_lineno: int) -> str:
        """Extract line range from source code.
        
        Args:
            src: Source code string
            lineno: Starting line number (1-indexed)
            end_lineno: Ending line number (1-indexed, inclusive)
            
        Returns:
            Extracted lines as string
        """
        lines = src.splitlines(True)
        a = max(1, int(lineno))
        b = max(a, int(end_lineno))
        return "".join(lines[a - 1 : b])
    
    def resolve_block_file_path(self, block: FunctionalBlock, package_root: Path) -> Path:
        """Resolve absolute path for a functional block's file.
        
        Args:
            block: FunctionalBlock instance
            package_root: Package root directory
            
        Returns:
            Absolute path to the block's file
        """
        p = Path(block.file_path)
        return p if p.is_absolute() else (package_root / p).resolve()
    
    def backup_file(self, file_path: Path, backup_root: Path, package_root: Path) -> Path:
        """Create backup copy of a file.
        
        Args:
            file_path: Path to file to backup
            backup_root: Root directory for backups
            package_root: Package root for relative path calculation
            
        Returns:
            Path to backup file
        """
        file_path = file_path.resolve()
        package_root = package_root.resolve()
        try:
            rel = file_path.relative_to(package_root)
            dest = backup_root / rel
        except Exception:
            dest = backup_root / file_path.name

        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, dest)
        return dest
    
    def write_patch(self, patch_path: Path, old: str, new: str, file_label: str) -> None:
        """Write unified diff patch file.
        
        Args:
            patch_path: Path where patch should be written
            old: Original content
            new: New content
            file_label: Label for the file in patch header
        """
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        diff = difflib.unified_diff(
            old.splitlines(True),
            new.splitlines(True),
            fromfile=f"{file_label}.orig",
            tofile=f"{file_label}.new",
            lineterm="",
        )
        patch_path.write_text("".join(diff), encoding="utf-8")
    
    def ensure_package_files(self, target_file: Path, *, package_root: Path) -> None:
        """Ensure all __init__.py files exist in package hierarchy.
        
        Args:
            target_file: Target file path
            package_root: Package root directory
        """
        target_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            rel = target_file.resolve().relative_to(package_root.resolve())
        except Exception:
            return

        cur = package_root.resolve()
        for part in rel.parts[:-1]:
            cur = cur / part
            init_py = cur / "__init__.py"
            if not init_py.exists():
                init_py.write_text('"""Auto-generated package."""\n', encoding="utf-8")
    
    def detect_package_root(self, project_root: Path) -> Tuple[Path, str]:
        """Detect package root directory and name.
        
        Args:
            project_root: Project root directory
            
        Returns:
            Tuple of (package_root_path, package_name)
        """
        if (project_root / "__init__.py").exists():
            return project_root, project_root.name

        for root_name in (self.config.project_package_roots or []):
            cand = project_root / root_name
            if (cand / "__init__.py").exists():
                return cand, cand.name

        return project_root, project_root.name
    
    def resolve_target_module_path(self, *, package_root: Path, target_module: str) -> Path:
        """Resolve target module string to absolute path.
        
        Args:
            package_root: Package root directory
            target_module: Target module path (relative or absolute)
            
        Returns:
            Absolute path to target module
        """
        rel = Path(target_module)
        if rel.is_absolute():
            return rel
        return (package_root / rel).resolve()
    
    def rollback(self, *, backups: List[Tuple[Path, Path]], created_files: List[Path]) -> None:
        """Rollback file changes by restoring backups and removing created files.
        
        Args:
            backups: List of (original_path, backup_path) tuples
            created_files: List of paths to files that were created
        """
        self.logger.warning("Rollback started: restoring backups and removing created files...")
        for orig, bkp in reversed(backups):
            try:
                if bkp.exists():
                    orig.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(bkp, orig)
            except Exception as e:
                self.logger.error("Rollback restore failed for %s: %s", orig, e)

        for f in reversed(created_files):
            try:
                if f.exists():
                    f.unlink()
            except Exception as e:
                self.logger.error("Rollback delete failed for %s: %s", f, e)
    
    def validate_files(self, files: List[Path], validate_unified_aliases_fn: Optional[callable] = None) -> None:
        """Validate Python files for syntax and compilation errors.
        
        Args:
            files: List of file paths to validate
            validate_unified_aliases_fn: Optional function to validate unified import aliases
            
        Raises:
            SyntaxError: If file has syntax errors
            py_compile.PyCompileError: If file fails to compile
        """
        for f in files:
            code = self.read_text(f, bom=True)
            ast.parse(code, filename=str(f))
            py_compile.compile(str(f), doraise=True)
            if validate_unified_aliases_fn and any(part == "unified" for part in f.parts):
                validate_unified_aliases_fn(file_path=f, code=code)

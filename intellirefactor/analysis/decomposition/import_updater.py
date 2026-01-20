"""
Import Updater Module

Handles the update of import statements during functional decomposition.
Rewrites imports to point to unified symbols in consolidated modules.

Extracted from DecompositionAnalyzer god class to improve modularity.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path
from typing import Any, Iterable, List, Optional, Set, Tuple

from .models import CanonicalizationPlan, FunctionalBlock, ProjectFunctionalMap


class ImportUpdater:
    """
    Manages import statement updates during functional decomposition.
    
    Rewrites from-import statements to point to unified symbols,
    handling aliases, comments, and module resolution.
    """

    def __init__(self, file_ops, logger):
        """
        Initialize the ImportUpdater.

        Args:
            file_ops: FileOperations instance for file I/O
            logger: Logger instance for diagnostics
        """
        self.file_ops = file_ops
        self.logger = logger

    def apply_update_imports_assisted(
        self,
        *,
        step: Any,
        plan: CanonicalizationPlan,
        functional_map: ProjectFunctionalMap,
        package_root: Path,
        package_name: str,
        backup_root: Path,
        patch_root: Path,
        backups: List[Tuple[Path, Path]],
        module_dotted_from_target_module_fn,
        qualify_module_fn,
    ) -> Tuple[List[Path], List[str]]:
        """
        Apply UPDATE_IMPORTS step: rewrite from-import statements.

        Args:
            step: Patch step containing source_blocks
            plan: Canonicalization plan with target module/symbol
            functional_map: Project functional map
            package_root: Root directory of the package
            package_name: Name of the package
            backup_root: Directory for backups
            patch_root: Directory for patches
            backups: List of (original, backup) file paths
            module_dotted_from_target_module_fn: Function to convert module path to dotted notation
            qualify_module_fn: Function to qualify module names with package

        Returns:
            Tuple of (changed_files, warnings)
        """
        changed: List[Path] = []
        warnings: List[str] = []

        unified_mod_dotted = f"{package_name}.{module_dotted_from_target_module_fn(plan.target_module)}"
        unified_symbol = plan.target_symbol

        for bid in (getattr(step, "source_blocks", None) or []):
            b = functional_map.blocks.get(bid)
            if not b:
                continue
            if b.is_method:
                warnings.append(f"{step.id}: UPDATE_IMPORTS skip method {b.qualname}")
                continue

            orig_mod = qualify_module_fn(b.module, package_name)
            orig_name = b.method_name

            for f in self._iter_python_files(package_root):
                src = self.file_ops.read_text(f, bom=True)
                try:
                    tree = ast.parse(src, filename=str(f))
                except SyntaxError:
                    continue

                lines = src.splitlines(True)
                file_mod = self._file_module_name(f, package_root, package_name)
                edits: List[Tuple[int, int, str]] = []

                for n in tree.body:
                    if not isinstance(n, ast.ImportFrom):
                        continue

                    abs_mod = self._resolve_importfrom_abs_module(n, file_mod)
                    abs_mod = qualify_module_fn(abs_mod, package_name) if abs_mod else abs_mod
                    if abs_mod != orig_mod:
                        continue

                    seg = self.file_ops.slice_lines(src, n.lineno, n.end_lineno or n.lineno)
                    if self._segment_has_comment(seg):
                        warnings.append(f"{f.name}:{n.lineno}: skipped import rewrite due to comments")
                        continue

                    moved: List[ast.alias] = []
                    remaining: List[ast.alias] = []
                    for a in n.names:
                        if a.name == orig_name:
                            moved.append(a)
                        else:
                            remaining.append(a)
                    if not moved:
                        continue

                    indent = re.match(r"\s*", lines[n.lineno - 1]).group(0) if lines else ""
                    repl: List[str] = []

                    if remaining:
                        repl.append(indent + self._format_importfrom(n.module, remaining, int(n.level or 0)) + "\n")

                    for a in moved:
                        local = a.asname or a.name
                        if local != unified_symbol:
                            repl.append(indent + f"from {unified_mod_dotted} import {unified_symbol} as {local}\n")
                        else:
                            repl.append(indent + f"from {unified_mod_dotted} import {unified_symbol}\n")

                    start = n.lineno - 1
                    end = (n.end_lineno or n.lineno)
                    edits.append((start, end, "".join(repl)))

                if not edits:
                    continue

                new_lines = lines[:]
                for start, end, text in sorted(edits, key=lambda x: x[0], reverse=True):
                    new_lines[start:end] = [text]
                new_src = "".join(new_lines)

                if new_src == src:
                    continue

                bkp = self.file_ops.backup_file(f, backup_root, package_root)
                backups.append((f, bkp))

                self.file_ops.write_text(f, new_src)

                patch_path = patch_root / f"{plan.cluster_id}_{step.id}_UPDATE_IMPORTS_{f.name}.patch"
                self.file_ops.write_patch(patch_path, src, new_src, str(f))
                changed.append(f)

        if not changed:
            warnings.append(f"{getattr(step, 'id', 'UPDATE_IMPORTS')}: no import sites updated")

        return changed, warnings

    def _segment_has_comment(self, segment: str) -> bool:
        """
        Check if a code segment contains comments.

        Args:
            segment: Code segment to check

        Returns:
            True if segment contains comments
        """
        try:
            tok = tokenize.generate_tokens(io.StringIO(segment).readline)
            return any(t.type == tokenize.COMMENT for t in tok)
        except Exception:
            return "#" in segment

    def _iter_python_files(self, package_root: Path) -> Iterable[Path]:
        """
        Iterate over Python files in package, skipping common directories.

        Args:
            package_root: Root directory of the package

        Yields:
            Path objects for Python files
        """
        skip_dirs = {"__pycache__", ".git", ".venv", "venv", "build", "dist"}
        for p in package_root.rglob("*.py"):
            if any(part in skip_dirs for part in p.parts):
                continue
            if "unified" in p.parts:
                continue
            yield p

    def _file_module_name(self, file_path: Path, package_root: Path, package_name: str) -> str:
        """
        Convert file path to module name.

        Args:
            file_path: Path to Python file
            package_root: Root directory of the package
            package_name: Name of the package

        Returns:
            Fully qualified module name
        """
        rel = file_path.resolve().relative_to(package_root.resolve()).as_posix()
        if rel.endswith(".py"):
            rel = rel[:-3]
        if rel.endswith("/__init__"):
            rel = rel[: -len("/__init__")]
        rel = rel.replace("/", ".")
        return f"{package_name}.{rel}" if rel else package_name

    def _resolve_importfrom_abs_module(self, node: ast.ImportFrom, file_module: str) -> str:
        """
        Resolve relative import to absolute module name.

        Args:
            node: ImportFrom AST node
            file_module: Module name of the file containing the import

        Returns:
            Absolute module name
        """
        mod = node.module or ""
        level = int(node.level or 0)
        if level <= 0:
            return mod
        parts = file_module.split(".")
        base = parts[:-level] if level <= len(parts) else []
        if mod:
            base += mod.split(".")
        return ".".join([p for p in base if p])

    def _format_importfrom(self, module: Optional[str], names: List[ast.alias], level: int) -> str:
        """
        Format an ImportFrom statement.

        Args:
            module: Module name (can be None for relative imports)
            names: List of imported names with optional aliases
            level: Relative import level (number of dots)

        Returns:
            Formatted import statement string
        """
        prefix = "." * max(0, level)
        mod = module or ""
        parts = []
        for a in names:
            parts.append(f"{a.name} as {a.asname}" if a.asname else a.name)
        return f"from {prefix}{mod} import " + ", ".join(parts)

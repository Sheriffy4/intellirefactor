"""
Validation Module

Handles validation of unified import aliases and other safety checks
during functional decomposition.

Extracted from DecompositionAnalyzer god class to improve modularity.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Dict


class UnifiedAliasValidator:
    """
    Validates unified import aliases to prevent runtime collisions.
    
    In unified modules we generate internal helper import aliases (__ir_*).
    This validator ensures that the same bound name is not assigned by
    multiple imports pointing to different sources.
    """

    def __init__(self):
        """Initialize the UnifiedAliasValidator."""
        pass

    def validate_unified_import_aliases(self, *, file_path: Path, code: str) -> None:
        """
        Validate unified import aliases to prevent collisions.
        
        In unified modules we generate internal helper import aliases (__ir_*).
        If the same bound name is assigned by multiple imports (pointing to different sources),
        it will break at runtime because name resolution happens at call time.

        Args:
            file_path: Path to the file being validated
            code: Source code to validate

        Raises:
            RuntimeError: If alias collision is detected
        """
        try:
            tree = ast.parse(code, filename=str(file_path))
        except SyntaxError:
            return

        seen: Dict[str, str] = {}

        for node in getattr(tree, "body", []):
            if isinstance(node, ast.Import):
                for a in node.names:
                    bound = a.asname or (a.name.split(".")[0] if a.name else "")
                    if not bound.startswith("__ir_"):
                        continue
                    src = a.name
                    prev = seen.get(bound)
                    if prev is not None and prev != src:
                        raise RuntimeError(
                            f"[unified alias collision] {file_path}: name {bound!r} imported from "
                            f"{prev!r} and {src!r}. Use unique aliases per canonical module."
                        )
                    seen[bound] = src

            elif isinstance(node, ast.ImportFrom):
                mod = "." * int(node.level or 0) + (node.module or "")
                for a in node.names:
                    bound = a.asname or a.name
                    if not bound.startswith("__ir_"):
                        continue
                    src = f"{mod}:{a.name}"
                    prev = seen.get(bound)
                    if prev is not None and prev != src:
                        raise RuntimeError(
                            f"[unified alias collision] {file_path}: name {bound!r} imported from "
                            f"{prev!r} and {src!r}. Use unique aliases per canonical module."
                        )
                    seen[bound] = src

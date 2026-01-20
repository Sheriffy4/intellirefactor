"""
Import Resolution Utilities

Handles extraction and resolution of imports from AST nodes.
"""

from __future__ import annotations

import ast
from typing import List, Optional

from .utils import iter_toplevel_import_nodes


class ImportResolver:
    """
    Resolves and extracts imports from AST nodes.

    Handles:
    - Absolute imports
    - Relative imports (from . import x)
    - Top-level imports within Try/If blocks
    """

    @staticmethod
    def extract_file_imports(tree: ast.Module, current_module: str = "") -> List[str]:
        """
        Extract top-level imports (including within top-level Try/If).
        Handles relative imports like `from . import x`.

        Args:
            tree: AST module to extract imports from
            current_module: Current module name for resolving relative imports

        Returns:
            List of import strings
        """
        imports: List[str] = []

        for node in iter_toplevel_import_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)

            elif isinstance(node, ast.ImportFrom):
                abs_mod = ImportResolver._resolve_absolute_module(node, current_module)
                if not abs_mod:
                    continue

                imports.append(abs_mod)
                for alias in node.names:
                    if alias.name != "*":
                        imports.append(f"{abs_mod}.{alias.name}")

        return imports

    @staticmethod
    def _resolve_absolute_module(
        node: ast.ImportFrom, current_module: str
    ) -> Optional[str]:
        """
        Resolve relative import to absolute module name.

        Examples:
            - from . import x -> current_module.x
            - from .. import x -> parent_module.x
            - from .submodule import x -> current_module.submodule.x

        Args:
            node: ImportFrom AST node
            current_module: Current module name

        Returns:
            Absolute module name or None if cannot be resolved
        """
        if node.level and node.level > 0:
            if not current_module:
                return None
            parts = current_module.split(".")
            if len(parts) < node.level:
                return None
            base = ".".join(parts[: -node.level])
            if node.module:
                return f"{base}.{node.module}" if base else node.module
            return base or None
        return node.module

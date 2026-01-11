"""
Common utilities for decomposition module.

Goals:
- Remove code duplication (imports scanning, regex detection, path normalization)
- Provide unified ID strategy and helpers for consistent data quality
"""

from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Iterator, List


# ============================================================================
# AST utilities
# ============================================================================

def iter_toplevel_import_nodes(tree: ast.Module) -> Iterator[ast.stmt]:
    """
    Yield nodes that may contain imports, including nested Try/If blocks.

    FIX: recursion (previous implementation only went 1 level deep).
    """
    stack: List[ast.stmt] = list(getattr(tree, "body", []))
    while stack:
        node = stack.pop(0)
        yield node

        if isinstance(node, ast.Try):
            stack[0:0] = list(node.body) + list(node.orelse) + list(node.finalbody)
            for h in node.handlers:
                stack[0:0] = list(h.body)

        elif isinstance(node, ast.If):
            stack[0:0] = list(node.body) + list(node.orelse)


# ============================================================================
# Regex detection (single source of truth)
# ============================================================================

_REGEX_STRONG_INDICATORS = (
    "[", "]", "()", r"\b", r"\d", r"\w", r"\s",
    ".*", ".+", ".?", "^", "$", "|"
)

_REGEX_PATTERNS = (
    re.compile(r"\\\w"),           # Escaped: \d, \w, \s
    re.compile(r"\[[^\]]+\]"),     # Character classes: [a-z]
    re.compile(r"\([^)]*\)"),      # Groups
    re.compile(r"\.\*"),           # .*
    re.compile(r"\.\+"),           # .+
    re.compile(r"\.\?"),           # .?
)

def is_likely_regex(literal: str, min_length: int = 3) -> bool:
    """Strict check if string is likely a regex pattern (avoid false positives)."""
    if not literal or len(literal) < min_length:
        return False

    if any(ind in literal for ind in _REGEX_STRONG_INDICATORS):
        return True

    return any(p.search(literal) for p in _REGEX_PATTERNS)


# ============================================================================
# Path / module normalization
# ============================================================================

def to_posix_path(p: str) -> str:
    try:
        return Path(p).as_posix()
    except Exception:
        return p.replace("\\", "/")

def normalize_module_path(path: str, project_roots: List[str]) -> str:
    """
    Remove known project roots from module paths.
    Example:
      normalize_module_path("intellirefactor.analysis.models", ["intellirefactor"])
      -> "analysis.models"
    """
    for root in project_roots or []:
        prefix = f"{root}."
        if path.startswith(prefix):
            return path[len(prefix):]
    return path


# ============================================================================
# ID generation
# ============================================================================

def make_block_id(
    project_root: str,
    file_path: str,
    module: str,
    qualname: str,
    lineno: int
) -> str:
    """
    Unified stable ID strategy:
      {module_or_relpath}:{qualname}:{lineno}

    - Prefer module when reliable
    - Otherwise use path relative to project_root (prevents collisions)
    """
    try:
        rel = Path(file_path).resolve().relative_to(Path(project_root).resolve()).as_posix()
    except Exception:
        rel = to_posix_path(file_path)

    module_part = module or rel
    return f"{module_part}:{qualname}:{lineno}"

def make_hash_id(content: str, prefix: str = "", length: int = 10) -> str:
    """Stable short hash id (sha256)."""
    h = hashlib.sha256(content.encode("utf-8")).hexdigest()[:length]
    return f"{prefix}_{h}" if prefix else h
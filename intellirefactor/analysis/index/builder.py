"""
IndexBuilder for IntelliRefactor persistent index.

This module implements the IndexBuilder class that creates and maintains
the SQLite index with incremental updates based on file content hashes.

Architecture principles:
1. Incremental processing - only analyze changed files
2. Bounded memory - batch processing for large projects
3. Facts-only storage - minimal data in SQLite
4. Stable symbol UIDs for consistent updates
"""

import ast
import copy
import hashlib
import json
import logging
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

from .schema import IndexSchema

_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_UID_RE = re.compile(r"^(?:[0-9a-f]{16}|[0-9a-f]{32})$")

def _relpath_posix(p: Path, root: Path) -> str:
    """Stable, cross-platform repo-relative path for DB/storage."""
    try:
        pr = p.resolve()
    except Exception:
        pr = p
    try:
        rr = root.resolve()
    except Exception:
        rr = root
    try:
        return pr.relative_to(rr).as_posix()
    except Exception:
        # Should be rare (symlinks/outside root). Keep stable string anyway.
        return pr.as_posix()

def _module_name_from_relative_path(relative_path: str) -> str:
    """
    Convert repo-relative path to dotted module name:
      a/b/c.py -> a.b.c
      a/b/__init__.py -> a.b
    """
    p = Path(relative_path.replace("\\", "/"))
    if p.name == "__init__.py":
        parts = list(p.parent.parts)
        # repo-root __init__.py: do not treat as "__init__" module
        if not parts:
            return ""
    else:
        parts = list(p.parent.parts) + [p.stem]
    parts = [x for x in parts if x and x != "."]
    return ".".join(parts) if parts else p.stem

# Map AST node types to canonical BlockType.value strings (foundation.models.BlockType).
# Canonical values: if/for/while/try/with (NOT if_block/for_loop/...).
_BLOCK_KIND_MAP = {
    ast.If: "if",
    ast.For: "for",
    ast.AsyncFor: "for",
    ast.While: "while",
    ast.Try: "try",
    ast.With: "with",
    ast.AsyncWith: "with",
}

_DEFAULT_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".intellirefactor",
    "intellirefactor_out",
}



@dataclass
class IndexBuildResult:
    """Result of index building operation."""

    success: bool
    files_processed: int
    files_skipped: int
    symbols_found: int
    blocks_found: int
    dependencies_found: int
    errors: List[str]
    build_time_seconds: float
    incremental: bool
    # Optional: totals from DB after build (useful for dashboards/specs).
    db_totals: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "files_processed": self.files_processed,
            "files_skipped": self.files_skipped,
            "symbols_found": self.symbols_found,
            "blocks_found": self.blocks_found,
            "dependencies_found": self.dependencies_found,
            "errors": list(self.errors),
            "build_time_seconds": self.build_time_seconds,
            "incremental": self.incremental,
            "db_totals": dict(self.db_totals) if isinstance(self.db_totals, dict) else None,
        }


@dataclass
class FileAnalysisResult:
    """Result of analyzing a single file."""

    file_path: str
    content_hash: str
    file_size: int
    lines_of_code: int
    is_test_file: bool
    last_modified: float
    symbols: List[Dict[str, Any]]
    blocks: List[Dict[str, Any]]
    dependencies: List[Dict[str, Any]]
    attribute_accesses: List[Dict[str, Any]]
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "content_hash": self.content_hash,
            "file_size": self.file_size,
            "lines_of_code": self.lines_of_code,
            "is_test_file": self.is_test_file,
            "last_modified": self.last_modified,
            "symbols": self.symbols,
            "blocks": self.blocks,
            "dependencies": self.dependencies,
            "attribute_accesses": self.attribute_accesses,
            "error": self.error,
        }


class _AstNormalizer(ast.NodeTransformer):
    """
    Normalizes an AST subtree to reduce noise from identifiers and constants.

    Goal: support Type-2/Type-3-ish clone detection (structural/normalized).
    This is not "true semantic" equivalence, but much better than reusing ast_fingerprint.
    """

    def __init__(self):
        super().__init__()
        self._name_map: Dict[str, str] = {}
        self._attr_map: Dict[str, str] = {}
        self._arg_map: Dict[str, str] = {}
        self._name_i = 0
        self._attr_i = 0
        self._arg_i = 0

    def _map_name(self, name: str) -> str:
        if name not in self._name_map:
            self._name_i += 1
            self._name_map[name] = f"VAR{self._name_i}"
        return self._name_map[name]

    def _map_attr(self, name: str) -> str:
        # attribute names can be very semantic; we normalize them but keep stable mapping within subtree
        if name not in self._attr_map:
            self._attr_i += 1
            self._attr_map[name] = f"ATTR{self._attr_i}"
        return self._attr_map[name]

    def _map_arg(self, name: str) -> str:
        if name not in self._arg_map:
            self._arg_i += 1
            self._arg_map[name] = f"ARG{self._arg_i}"
        return self._arg_map[name]

    # --- Definitions
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        node.name = "FUNC"
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        node = self.generic_visit(node)
        node.name = "ASYNC_FUNC"
        return node

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        node = self.generic_visit(node)
        node.name = "CLASS"
        return node

    # --- Identifiers
    def visit_Name(self, node: ast.Name) -> ast.AST:
        node.id = self._map_name(node.id)
        return node

    def visit_arg(self, node: ast.arg) -> ast.AST:
        node.arg = self._map_arg(node.arg)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = self.generic_visit(node)
        node.attr = self._map_attr(node.attr)
        return node

    def visit_keyword(self, node: ast.keyword) -> ast.AST:
        node = self.generic_visit(node)
        if node.arg is not None:
            node.arg = self._map_attr(node.arg)
        return node

    # --- Constants
    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        v = node.value
        if v is None:
            node.value = None
        elif isinstance(v, bool):
            node.value = True
        elif isinstance(v, int):
            node.value = 0
        elif isinstance(v, float):
            node.value = 0.0
        elif isinstance(v, str):
            node.value = ""
        elif isinstance(v, bytes):
            node.value = b""
        else:
            node.value = None
        return node


class ContextAwareVisitor(ast.NodeVisitor):
    """
    AST Visitor that tracks scope context to correctly identify
    qualified names, nesting levels, and avoid recursion issues.
    """

    def __init__(self, file_path: Path, project_root: Path, content: str):
        self.logger = logging.getLogger(__name__)  # FIX: was missing

        self.file_path = file_path
        self.project_root = project_root
        self.relative_path = _relpath_posix(file_path, project_root)
        self.module_name = _module_name_from_relative_path(self.relative_path)
        # Empty file => [] which makes module symbol line_end=0 (invalid range).
        # Treat empty file as 1 "logical" line for line range consistency.
        self.content_lines = content.splitlines() or [""]

        self.scope_stack: List[Tuple[str, str]] = []
        self.current_class_node: Optional[ast.ClassDef] = None

        self.scope_occurrence_map: Dict[Tuple[str, str], int] = defaultdict(int)

        self._dep_index: Dict[Tuple[str, str, str], int] = {}
        self._attribute_access_map: Dict[Tuple[str, str], Dict[str, Any]] = {}

        self.symbols: List[Dict[str, Any]] = []
        self.blocks: List[Dict[str, Any]] = []
        self.dependencies: List[Dict[str, Any]] = []

        self._create_module_symbol()

    @property
    def attribute_accesses(self) -> List[Dict[str, Any]]:
        """Returns the list of aggregated attribute accesses."""
        return list(self._attribute_access_map.values())

    def _create_module_symbol(self):
        """Creates a symbol representing the module itself."""
        module_qn = self.module_name or self.file_path.stem
        symbol_uid = IndexSchema.generate_symbol_uid(self.relative_path, module_qn, "module", 0, "")

        self.symbols.append(
            {
                "symbol_uid": symbol_uid,
                "name": module_qn.split(".")[-1] if module_qn else self.file_path.stem,
                "qualified_name": module_qn,
                "kind": "module",
                "line_start": 1,
                "line_end": len(self.content_lines),
                "signature": (f"module {module_qn}" if module_qn else "module"),
                "ast_fingerprint": "",
                "token_fingerprint": "",
                "semantic_category": "module",
                "responsibility_markers": json.dumps([]),
                "is_public": not module_qn.startswith("_"),
                "is_async": False,
                "is_property": False,
                "is_static": False,
                "is_classmethod": False,
                "complexity_score": 0,
            }
        )
        self.scope_stack.append((module_qn, symbol_uid))

    def _current_package_parts(self) -> List[str]:
        """
        For module "a.b.c" -> package parts ["a", "b"]
        For package "__init__.py" module "a.b" -> package parts ["a", "b"]
        """
        mn = self.module_name or ""
        if not mn:
            return []
        parts = mn.split(".")
        if self.file_path.name == "__init__.py":
            return parts
        return parts[:-1] if len(parts) > 1 else []

    def _resolve_importfrom_targets(self, node: ast.ImportFrom) -> List[str]:
        """
        Resolve ImportFrom into *module targets* (not symbol targets).
        This is critical to map imports to module symbols and fill target_symbol_id.
        """
        lvl = int(node.level or 0)
        pkg = self._current_package_parts()

        # Absolute import: from x.y import z  => depend on module x.y
        if lvl == 0:
            if not node.module:
                return []
            return [str(node.module)]

        # Relative import:
        # level=1 => same package
        # level=2 => parent package
        up = lvl - 1
        if up > len(pkg):
            return []
        base_parts = pkg[:-up] if up > 0 else pkg

        if node.module:
            base_parts = base_parts + str(node.module).split(".")
            base_mod = ".".join([p for p in base_parts if p])
            return [base_mod] if base_mod else []

        # from . import foo, bar  => depend on package.foo / package.bar
        base_mod = ".".join([p for p in base_parts if p])
        targets: List[str] = []
        for alias in node.names:
            nm = alias.name
            if nm == "*":
                if base_mod:
                    targets.append(base_mod)
                continue
            targets.append(f"{base_mod}.{nm}" if base_mod else nm)
        return targets

    def _get_current_qualified_name(self, name: str) -> str:
        # Include module prefix (scope_stack[0]) to avoid cross-file collisions and
        # to make qualified_name truly qualified: pkg.mod.Class.method, etc.
        parts: List[str] = []
        if self.scope_stack:
            mod = self.scope_stack[0][0]
            if mod:
                parts.append(mod)
        parts.extend([s[0] for s in self.scope_stack[1:]])
        parts.append(name)
        parts = [p for p in parts if p]
        return ".".join(parts)

    def _get_current_parent_uid(self) -> str:
        if not self.scope_stack:
            return ""
        return self.scope_stack[-1][1]

    def _get_occurrence_index(self, name: str) -> int:
        parent_uid = self._get_current_parent_uid()
        key = (parent_uid, name)
        idx = self.scope_occurrence_map[key]
        self.scope_occurrence_map[key] += 1
        return idx

    def visit_ClassDef(self, node: ast.ClassDef):
        qualified_name = self._get_current_qualified_name(node.name)
        occurrence_idx = self._get_occurrence_index(node.name)
        disambiguator = f":{occurrence_idx}" if occurrence_idx > 0 else ""

        symbol_uid = IndexSchema.generate_symbol_uid(
            self.relative_path,
            qualified_name,
            "class",
            0,
            f"class {node.name}{disambiguator}",
        )

        symbol_info = {
            "symbol_uid": symbol_uid,
            "name": node.name,
            "qualified_name": qualified_name,
            "kind": "class",
            "line_start": node.lineno,
            "line_end": getattr(node, "end_lineno", node.lineno),
            "signature": f"class {node.name}",
            "ast_fingerprint": self._calculate_ast_fingerprint(node),
            "token_fingerprint": self._calculate_token_fingerprint(node),
            "semantic_category": self._determine_semantic_category(node.name),
            "responsibility_markers": json.dumps(self._extract_responsibility_markers(node)),
            "is_public": not node.name.startswith("_"),
            "is_async": False,
            "is_property": False,
            "is_static": False,
            "is_classmethod": False,
            "complexity_score": 0,
        }
        self.symbols.append(symbol_info)

        self._extract_local_facts(node, symbol_uid)

        self.scope_stack.append((node.name, symbol_uid))
        prev_class_node = self.current_class_node
        self.current_class_node = node

        self.generic_visit(node)

        self.current_class_node = prev_class_node
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self._handle_function(node, is_async=False)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self._handle_function(node, is_async=True)

    def _handle_function(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool):
        qualified_name = self._get_current_qualified_name(node.name)
        occurrence_idx = self._get_occurrence_index(node.name)

        kind = "function"
        is_static = False
        is_classmethod = False

        if self.current_class_node:
            kind = "method"
            for dec in node.decorator_list:
                if isinstance(dec, ast.Name):
                    if dec.id == "staticmethod":
                        is_static = True
                    elif dec.id == "classmethod":
                        is_classmethod = True

        signature = self._get_function_signature(node, is_async)
        disambiguator = f":{occurrence_idx}" if occurrence_idx > 0 else ""

        symbol_uid = IndexSchema.generate_symbol_uid(
            self.relative_path, qualified_name, kind, 0, signature + disambiguator
        )

        symbol_info = {
            "symbol_uid": symbol_uid,
            "name": node.name,
            "qualified_name": qualified_name,
            "kind": kind,
            "line_start": node.lineno,
            "line_end": getattr(node, "end_lineno", node.lineno),
            "signature": signature,
            "ast_fingerprint": self._calculate_ast_fingerprint(node),
            "token_fingerprint": self._calculate_token_fingerprint(node),
            "semantic_category": self._determine_semantic_category(node.name),
            "responsibility_markers": json.dumps(self._extract_responsibility_markers(node)),
            "is_public": not node.name.startswith("_"),
            "is_async": is_async,
            "is_property": self._is_property(node),
            "is_static": is_static,
            "is_classmethod": is_classmethod,
            "complexity_score": self._calculate_complexity(node),
        }
        self.symbols.append(symbol_info)

        self._extract_local_facts(node, symbol_uid)

        self.scope_stack.append((node.name, symbol_uid))
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_Import(self, node: ast.Import):
        parent_uid = self._get_current_parent_uid()
        for alias in node.names:
            self._add_dependency(
                source_uid=parent_uid,
                target=alias.name,
                dependency_kind="imports",
                resolution="exact",
                confidence=1.0,
                evidence={
                    "line_number": node.lineno,
                    "ast_node_type": "Import",
                    "module_name": alias.name,
                },
            )

    def visit_ImportFrom(self, node: ast.ImportFrom):
        parent_uid = self._get_current_parent_uid()
        targets = self._resolve_importfrom_targets(node)
        if not targets:
            # keep legacy behavior as fallback (unresolved relative)
            base = "." * (node.level or 0)
            if node.module:
                base += str(node.module)
            targets = [base] if base else []

        imported_names = [a.name for a in node.names]
        for mod_target in targets:
            self._add_dependency(
                source_uid=parent_uid,
                target=str(mod_target),
                dependency_kind="imports",
                resolution="exact",
                confidence=1.0,
                evidence={
                    "line_number": node.lineno,
                    "ast_node_type": "ImportFrom",
                    "module": str(node.module) if node.module else None,
                    "level": int(node.level or 0),
                    "resolved_module_target": str(mod_target),
                    "imported_names": imported_names,
                },
            )

    def _add_dependency(
        self,
        source_uid: str,
        target: str,
        dependency_kind: str,
        resolution: str,
        confidence: float,
        evidence: Dict,
    ):
        key = (source_uid, target, dependency_kind)
        idx = self._dep_index.get(key)
        if idx is None:
            self._dep_index[key] = len(self.dependencies)
            self.dependencies.append(
                {
                    "source_symbol_uid": source_uid,
                    "target_external": target,
                    "dependency_kind": dependency_kind,
                    "resolution": resolution,
                    "confidence": confidence,
                    "evidence": evidence,
                    "usage_count": 1,
                }
            )
            return

        # Already seen: bump usage_count (keep first evidence as representative)
        dep = self.dependencies[idx]
        dep["usage_count"] = int(dep.get("usage_count", 1) or 1) + 1

    def _extract_local_facts(self, parent_node: ast.AST, symbol_uid: str):
        for child in ast.iter_child_nodes(parent_node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                continue
            self._extract_blocks_recursive(child, symbol_uid, nesting_level=1)
            self._extract_usage_recursive(child, symbol_uid)

    def _extract_blocks_recursive(self, node: ast.AST, symbol_uid: str, nesting_level: int):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return

        if isinstance(node, tuple(_BLOCK_KIND_MAP.keys())):
            block_kind = _BLOCK_KIND_MAP.get(type(node), type(node).__name__.lower())
            block_uid = IndexSchema.generate_block_uid(
                symbol_uid,
                block_kind,
                node.lineno,
                getattr(node, "end_lineno", node.lineno),
            )

            self.blocks.append(
                {
                    "block_uid": block_uid,
                    "symbol_uid": symbol_uid,
                    # canonical name (aligns with DB schema + other analyzers)
                    "block_type": block_kind,
                    "line_start": node.lineno,
                    "line_end": getattr(node, "end_lineno", node.lineno),
                    "lines_of_code": getattr(node, "end_lineno", node.lineno) - node.lineno + 1,
                    "nesting_level": nesting_level,
                    "ast_fingerprint": self._calculate_ast_fingerprint(node),
                    "token_fingerprint": self._calculate_token_fingerprint(node),
                    "normalized_fingerprint": self._calculate_normalized_fingerprint(node),
                }
            )
            next_level = nesting_level + 1
        else:
            next_level = nesting_level

        for child in ast.iter_child_nodes(node):
            self._extract_blocks_recursive(child, symbol_uid, next_level)

    def _extract_usage_recursive(self, node: ast.AST, symbol_uid: str):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return

        if isinstance(node, ast.Call):
            target = self._extract_call_target(node)
            if target:
                self._add_dependency(
                    source_uid=symbol_uid,
                    target=target,
                    dependency_kind="calls",
                    resolution="probable",
                    confidence=0.7,
                    evidence={
                        "line_number": node.lineno,
                        "ast_node_type": "Call",
                        "call_target": target,
                    },
                )

        elif isinstance(node, ast.Attribute):
            key = (symbol_uid, node.attr)
            if key in self._attribute_access_map:
                self._attribute_access_map[key]["count"] += 1
            else:
                self._attribute_access_map[key] = {
                    "symbol_uid": symbol_uid,
                    "attribute_name": node.attr,
                    "access_type": "read",
                    "line_number": node.lineno,
                    "confidence": 1.0,
                    "count": 1,
                    "evidence": {
                        "line_number": node.lineno,
                        "ast_node_type": "Attribute",
                        "attribute_name": node.attr,
                    },
                }

        for child in ast.iter_child_nodes(node):
            self._extract_usage_recursive(child, symbol_uid)

    # --- Helpers ---

    def _get_function_signature(
        self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef], is_async: bool
    ) -> str:
        args = [arg.arg for arg in node.args.args]
        async_prefix = "async " if is_async else ""
        return f"{async_prefix}def {node.name}({', '.join(args)})"

    def _calculate_ast_fingerprint(self, node: ast.AST) -> str:
        node_types = [type(child).__name__ for child in ast.walk(node)]
        fingerprint_str = "|".join(node_types)
        return hashlib.blake2b(fingerprint_str.encode("utf-8"), digest_size=16).hexdigest()

    def _calculate_token_fingerprint(self, node: ast.AST) -> str:
        try:
            if hasattr(ast, "unparse"):
                source = ast.unparse(node)
                source = " ".join(source.split())
                return hashlib.blake2b(source.encode("utf-8"), digest_size=16).hexdigest()
            return ""  # FIX: ensure str return always
        except Exception:
            self.logger.debug("Failed to build token fingerprint", exc_info=True)
            return ""

    def _calculate_normalized_fingerprint(self, node: ast.AST) -> str:
        """
        Normalized fingerprint: AST after identifier/constant normalization.
        Better for Type-2 clones than reusing ast_fingerprint.
        """
        try:
            cloned = copy.deepcopy(node)
            norm = _AstNormalizer().visit(cloned)
            ast.fix_missing_locations(norm)
            dumped = ast.dump(norm, include_attributes=False)
            dumped = " ".join(dumped.split())
            return hashlib.blake2b(dumped.encode("utf-8"), digest_size=16).hexdigest()
        except Exception:
            self.logger.debug("Failed to build normalized fingerprint", exc_info=True)
            return ""

    def _determine_semantic_category(self, name: str) -> str:
        name_lower = name.lower()
        if any(k in name_lower for k in ["test", "check", "verify", "validate"]):
            return "validation"
        if any(k in name_lower for k in ["transform", "convert", "parse"]):
            return "transformation"
        if any(k in name_lower for k in ["save", "load", "store", "db"]):
            return "persistence"
        if any(k in name_lower for k in ["get", "set", "is"]):
            return "access"
        return "business_logic"

    def _extract_responsibility_markers(self, node: ast.AST) -> List[str]:
        markers = []
        if hasattr(node, "name"):
            cat = self._determine_semantic_category(getattr(node, "name"))
            if cat != "business_logic":
                markers.append(cat)
        return markers

    def _is_property(self, node: ast.FunctionDef) -> bool:
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id == "property":
                return True
        return False

    def _calculate_complexity(self, node: ast.AST) -> int:
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

    def _extract_call_target(self, call_node: ast.Call) -> Optional[str]:
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        if isinstance(call_node.func, ast.Attribute):
            if isinstance(call_node.func.value, ast.Name):
                return f"{call_node.func.value.id}.{call_node.func.attr}"
            return call_node.func.attr
        return None


class IndexBuilder:
    """
    Builds and maintains the persistent SQLite index for IntelliRefactor.

    Supports incremental updates based on file content hashes and uses
    batch processing to handle large projects without memory exhaustion.
    """

    def __init__(self, db_path: Path, batch_size: int = 100):
        self.db_path = Path(db_path)
        self.batch_size = batch_size
        self.logger = logging.getLogger(__name__)

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.db_path.exists():
            self.logger.info("Creating new database at %s", self.db_path)
            self.conn = IndexSchema.create_database(str(self.db_path))
            self.conn.execute("PRAGMA foreign_keys = ON")
        else:
            self.conn = sqlite3.connect(str(self.db_path))
            self.conn.execute("PRAGMA foreign_keys = ON")

            if IndexSchema.needs_migration(self.conn):
                self.logger.warning("Database schema migration needed - recreating database")
                self.conn.close()
                self.db_path.unlink()
                self.conn = IndexSchema.create_database(str(self.db_path))

    @staticmethod
    def quote_ident(name: str) -> str:
        """sqlite identifiers can't be parametrized; validate + quote."""
        if not _IDENT_RE.match(name):
            raise ValueError(f"Invalid identifier: {name!r}")
        return f'"{name}"'

    def build_index(
        self,
        project_path: Path,
        incremental: bool = True,
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
    ) -> IndexBuildResult:
        start_time = datetime.now()

        try:
            self.logger.info("Starting index build for %s (incremental=%s)", project_path, incremental)

            python_files = sorted(self._iter_python_files(project_path), key=lambda p: p.as_posix())
            self.logger.info("Found %d Python files", len(python_files))

            if incremental:
                files_to_process = self._filter_changed_files(python_files, project_path)
                files_skipped = len(python_files) - len(files_to_process)

                orphaned_count = self._cleanup_orphaned_records(project_path, python_files)
                if orphaned_count > 0:
                    self.logger.info("Cleaned up %d orphaned file records", orphaned_count)
            else:
                files_to_process = python_files
                files_skipped = 0

            self.logger.info("Processing %d files (skipped %d)", len(files_to_process), files_skipped)

            # delta counters (only for processed files)
            delta_symbols = 0
            delta_blocks = 0
            delta_dependencies = 0
            errors: List[str] = []
            processed_files = 0

            for batch_idx, batch in enumerate(self._batch_files(files_to_process)):
                batch_result = self._process_file_batch(batch, project_path)
                delta_symbols += batch_result["symbols"]
                delta_blocks += batch_result["blocks"]
                delta_dependencies += batch_result["dependencies"]
                errors.extend(batch_result["errors"])
                processed_files += batch_result["files"]

                if progress_callback:
                    progress = processed_files / len(files_to_process) if files_to_process else 1.0
                    progress_callback(progress, processed_files, len(files_to_process))

                self.conn.commit()

                self.logger.info(
                    "Processed batch %d: %d files, %d symbols, %d blocks",
                    batch_idx + 1,
                    batch_result["files"],
                    batch_result["symbols"],
                    batch_result["blocks"],
                )

            self.conn.commit()

            build_time = (datetime.now() - start_time).total_seconds()

            # IMPORTANT:
            # IndexBuildResult should reflect the INDEX STATE after the build,
            # not only the delta from changed files.
            # Otherwise incremental builds that process 0 files will show 0 deps/symbols/blocks,
            # which breaks specs/dashboards.
            db_totals = self._get_db_totals()

            result = IndexBuildResult(
                success=len(errors) == 0,
                files_processed=len(files_to_process),
                files_skipped=files_skipped,
                symbols_found=int(db_totals.get("symbols_count", 0)),
                blocks_found=int(db_totals.get("blocks_count", 0)),
                dependencies_found=int(db_totals.get("dependencies_count", 0)),
                errors=errors,
                build_time_seconds=build_time,
                incremental=incremental,
                db_totals=db_totals,
            )

            self.logger.info(
                "Index build completed in %.2fs: %d files, %d symbols",
                build_time,
                result.files_processed,
                result.symbols_found,
            )

            return result

        except Exception as e:
            self.logger.error("Index build failed: %s", e, exc_info=True)
            return IndexBuildResult(
                success=False,
                files_processed=0,
                files_skipped=0,
                symbols_found=0,
                blocks_found=0,
                dependencies_found=0,
                errors=[str(e)],
                build_time_seconds=(datetime.now() - start_time).total_seconds(),
                incremental=incremental,
            )

    def rebuild_index(
        self,
        project_path: Path,
        progress_callback: Optional[Callable[[float, int, int], None]] = None,
    ) -> IndexBuildResult:
        self.logger.info("Rebuilding index from scratch")
        self._clear_index()
        return self.build_index(project_path, incremental=False, progress_callback=progress_callback)

    def _get_db_totals(self) -> Dict[str, int]:
        """
        Totals for the whole DB after the build.
        Cheap aggregates (COUNT(*)) to make reports consistent in incremental mode.
        """
        totals: Dict[str, int] = {}
        try:
            cur = self.conn.execute("SELECT COUNT(*) FROM files")
            totals["files_count"] = int(cur.fetchone()[0] or 0)
        except Exception:
            totals["files_count"] = 0
        try:
            cur = self.conn.execute("SELECT COUNT(*) FROM symbols")
            totals["symbols_count"] = int(cur.fetchone()[0] or 0)
        except Exception:
            totals["symbols_count"] = 0
        try:
            cur = self.conn.execute("SELECT COUNT(*) FROM blocks")
            totals["blocks_count"] = int(cur.fetchone()[0] or 0)
        except Exception:
            totals["blocks_count"] = 0
        try:
            cur = self.conn.execute("SELECT COUNT(*) FROM dependencies")
            totals["dependencies_count"] = int(cur.fetchone()[0] or 0)
        except Exception:
            totals["dependencies_count"] = 0
        try:
            cur = self.conn.execute("SELECT COUNT(*) FROM attribute_access")
            totals["attribute_access_count"] = int(cur.fetchone()[0] or 0)
        except Exception:
            totals["attribute_access_count"] = 0
        return totals

    def _filter_changed_files(self, python_files: List[Path], project_root: Path) -> List[Path]:
        changed_files: List[Path] = []
        chunk_size = 500

        for i in range(0, len(python_files), chunk_size):
            chunk = python_files[i : i + chunk_size]

            rel_paths: List[str] = []
            path_map: Dict[str, Path] = {}

            for f in chunk:
                try:
                    rel = _relpath_posix(f, project_root)
                    rel_paths.append(rel)
                    path_map[rel] = f
                except ValueError:
                    continue

            if not rel_paths:
                continue

            placeholders = ",".join(["?"] * len(rel_paths))
            sql = (
                "SELECT file_path, content_hash FROM files "
                "WHERE file_path IN (" + placeholders + ")"
            )
            cursor = self.conn.execute(sql, rel_paths)
            db_hashes = {row[0]: row[1] for row in cursor.fetchall()}

            for rel_path in rel_paths:
                file_path = path_map[rel_path]
                try:
                    file_size = file_path.stat().st_size
                    if file_size > 10 * 1024 * 1024:
                        self.logger.warning("Skipping large file %s (%d bytes)", file_path, file_size)
                        continue

                    with open(file_path, "r", encoding="utf-8-sig") as f:
                        content = f.read()
                    current_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

                    if rel_path not in db_hashes or db_hashes[rel_path] != current_hash:
                        changed_files.append(file_path)
                        self.logger.debug("File changed: %s", rel_path)
                    else:
                        self.logger.debug("File unchanged: %s", rel_path)

                except (UnicodeDecodeError, PermissionError) as e:
                    self.logger.warning("Cannot read file %s: %s", file_path, e)
                    changed_files.append(file_path)
                except Exception as e:
                    self.logger.error("Unexpected error processing %s: %s", file_path, e, exc_info=True)
                    changed_files.append(file_path)

        self.logger.info(
            "Incremental update: %d changed files out of %d total",
            len(changed_files),
            len(python_files),
        )
        return changed_files

    def _cleanup_orphaned_records(self, project_root: Path, current_files: List[Path]) -> int:
        cursor = self.conn.execute("SELECT file_path FROM files")
        db_files = {row[0] for row in cursor.fetchall()}
        current_relative_paths = {_relpath_posix(f, project_root) for f in current_files}
        orphaned_files = db_files - current_relative_paths

        if orphaned_files:
            self.logger.info("Cleaning up %d orphaned file records", len(orphaned_files))
            for orphaned_file in orphaned_files:
                self.conn.execute("DELETE FROM files WHERE file_path = ?", (orphaned_file,))

        return len(orphaned_files)

    def _iter_python_files(self, project_path: Path) -> Iterator[Path]:
        """Iterate project *.py files excluding common non-source directories."""
        for p in project_path.rglob("*.py"):
            try:
                rel_parts = set(p.relative_to(project_path).parts)
            except Exception:
                rel_parts = set(p.parts)
            if rel_parts.intersection(_DEFAULT_SKIP_DIRS):
                continue
            yield p

    def _batch_files(self, files: List[Path]) -> Iterator[List[Path]]:
        for i in range(0, len(files), self.batch_size):
            yield files[i : i + self.batch_size]

    def _process_file_batch(self, batch: List[Path], project_root: Path) -> Dict[str, Any]:
        batch_stats: Dict[str, Any] = {
            "files": 0,
            "symbols": 0,
            "blocks": 0,
            "dependencies": 0,
            "errors": [],
        }

        for file_path in batch:
            try:
                analysis = self._analyze_file(file_path, project_root)
                if analysis.error:
                    batch_stats["errors"].append(f"{file_path}: {analysis.error}")
                    continue

                file_id = self._store_file_record(analysis)
                symbol_ids = self._store_symbols(file_id, analysis.symbols)
                batch_stats["symbols"] += len(symbol_ids)

                self._store_blocks(symbol_ids, analysis.blocks)
                batch_stats["blocks"] += len(analysis.blocks)

                self._store_dependencies(symbol_ids, analysis.dependencies)
                batch_stats["dependencies"] += len(analysis.dependencies)

                self._store_attribute_accesses(symbol_ids, analysis.attribute_accesses)

                batch_stats["files"] += 1

            except Exception as e:
                batch_stats["errors"].append(f"{file_path}: {str(e)}")
                self.logger.error("Error processing %s: %s", file_path, e, exc_info=True)

        return batch_stats

    def _analyze_file(self, file_path: Path, project_root: Path) -> FileAnalysisResult:
        try:
            file_size = file_path.stat().st_size
            if file_size > 10 * 1024 * 1024:
                return FileAnalysisResult(
                    file_path=_relpath_posix(file_path, project_root),
                    content_hash="",
                    file_size=file_size,
                    lines_of_code=0,
                    is_test_file=self._is_test_file(file_path),
                    last_modified=file_path.stat().st_mtime,
                    symbols=[],
                    blocks=[],
                    dependencies=[],
                    attribute_accesses=[],
                    error="File too large (>10MB)",
                )

            with open(file_path, "r", encoding="utf-8-sig") as f:
                content = f.read()

            content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            last_modified = file_path.stat().st_mtime
            lines_of_code = len([line for line in content.splitlines() if line.strip()])

            try:
                visitor = ContextAwareVisitor(file_path, project_root, content)
                tree = ast.parse(content)
                visitor.visit(tree)

                return FileAnalysisResult(
                    file_path=_relpath_posix(file_path, project_root),
                    content_hash=content_hash,
                    file_size=len(content.encode("utf-8")),
                    lines_of_code=lines_of_code,
                    is_test_file=self._is_test_file(file_path),
                    last_modified=last_modified,
                    symbols=visitor.symbols,
                    blocks=visitor.blocks,
                    dependencies=visitor.dependencies,
                    attribute_accesses=visitor.attribute_accesses,
                )

            except SyntaxError as e:
                try:
                    rel = _relpath_posix(file_path, project_root)
                except Exception:
                    rel = file_path.as_posix()
                return FileAnalysisResult(
                    file_path=rel,
                    content_hash=content_hash,
                    file_size=len(content.encode("utf-8")),
                    lines_of_code=lines_of_code,
                    is_test_file=self._is_test_file(file_path),
                    last_modified=last_modified,
                    symbols=[],
                    blocks=[],
                    dependencies=[],
                    attribute_accesses=[],
                    error=f"Syntax error: {e}",
                )

        except Exception as e:
            try:
                rel = _relpath_posix(file_path, project_root)
            except Exception:
                rel = file_path.as_posix()
            return FileAnalysisResult(
                file_path=rel,
                content_hash="",
                file_size=0,
                lines_of_code=0,
                is_test_file=False,
                last_modified=0.0,
                symbols=[],
                blocks=[],
                dependencies=[],
                attribute_accesses=[],
                error=str(e),
            )

    def _is_test_file(self, file_path: Path) -> bool:
        return (
            file_path.name.startswith("test_")
            or file_path.name.endswith("_test.py")
            or "test" in file_path.parts
        )

    def _store_file_record(self, analysis: FileAnalysisResult) -> int:
        self.conn.execute("DELETE FROM files WHERE file_path = ?", (analysis.file_path,))
        now = datetime.now()

        cursor = self.conn.execute(
            """
            INSERT INTO files (file_path, content_hash, last_modified, last_analyzed, file_size, lines_of_code, is_test_file)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                analysis.file_path,
                analysis.content_hash,
                datetime.fromtimestamp(analysis.last_modified),
                now,
                analysis.file_size,
                analysis.lines_of_code,
                analysis.is_test_file,
            ),
        )
        return cursor.lastrowid

    def _store_symbols(self, file_id: int, symbols: List[Dict[str, Any]]) -> Dict[str, int]:
        symbol_ids: Dict[str, int] = {}

        for symbol in symbols:
            self.conn.execute("DELETE FROM symbols WHERE symbol_uid = ?", (symbol["symbol_uid"],))

            cursor = self.conn.execute(
                """
                INSERT INTO symbols (
                    symbol_uid, file_id, name, qualified_name, kind, line_start, line_end,
                    signature, ast_fingerprint, token_fingerprint, semantic_category,
                    responsibility_markers, is_public, is_async, is_property,
                    is_static, is_classmethod, complexity_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol["symbol_uid"],
                    file_id,
                    symbol["name"],
                    symbol["qualified_name"],
                    symbol["kind"],
                    symbol["line_start"],
                    symbol["line_end"],
                    symbol["signature"],
                    symbol.get("ast_fingerprint", "") or "",
                    symbol.get("token_fingerprint", "") or "",
                    symbol.get("semantic_category", "") or "",
                    symbol.get("responsibility_markers", "[]") or "[]",
                    bool(symbol.get("is_public", True)),
                    bool(symbol.get("is_async", False)),
                    bool(symbol.get("is_property", False)),
                    bool(symbol.get("is_static", False)),
                    bool(symbol.get("is_classmethod", False)),
                    int(symbol.get("complexity_score", 0) or 0),
                ),
            )
            symbol_ids[symbol["symbol_uid"]] = cursor.lastrowid

        return symbol_ids

    def _store_blocks(self, symbol_ids: Dict[str, int], blocks: List[Dict[str, Any]]) -> None:
        for block in blocks:
            symbol_id = symbol_ids.get(block["symbol_uid"])
            if not symbol_id:
                continue

            self.conn.execute("DELETE FROM blocks WHERE block_uid = ?", (block["block_uid"],))
            block_type = block.get("block_type") or "other"
            statement_count = int(block.get("statement_count", 0) or 0)
            parent_block_id = block.get("parent_block_id", None)
            min_clone_size = int(block.get("min_clone_size", 3) or 3)
            is_extractable = bool(block.get("is_extractable", True))
            confidence = float(block.get("confidence", 1.0) or 1.0)
            metadata = block.get("metadata", {}) or {}

            self.conn.execute(
                """
                INSERT INTO blocks (
                    block_uid, symbol_id, block_type, line_start, line_end, lines_of_code,
                    statement_count, nesting_level, parent_block_id,
                    ast_fingerprint, token_fingerprint, normalized_fingerprint,
                    min_clone_size, is_extractable, confidence, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    block["block_uid"],
                    symbol_id,
                    block_type,
                    int(block.get("line_start", 1) or 1),
                    int(block.get("line_end", 1) or 1),
                    int(block.get("lines_of_code", 0) or 0),
                    statement_count,
                    int(block.get("nesting_level", 0) or 0),
                    parent_block_id,
                    block.get("ast_fingerprint", "") or "",
                    block.get("token_fingerprint", "") or "",
                    block.get("normalized_fingerprint", "") or "",
                    min_clone_size,
                    is_extractable,
                    confidence,
                    json.dumps(metadata),
                ),
            )

    def _store_dependencies(self, symbol_ids: Dict[str, int], dependencies: List[Dict[str, Any]]) -> None:
        for dep in dependencies:
            source_symbol_id = symbol_ids.get(dep["source_symbol_uid"])
            if not source_symbol_id:
                continue

            target_external = str(dep.get("target_external") or "")
            dep_kind = str(dep.get("dependency_kind") or "")

            target_symbol_id = self._resolve_target_symbol_id(
                source_symbol_id=source_symbol_id,
                target_external=target_external,
                kind=dep_kind,
            )

            self.conn.execute(
                """
                INSERT INTO dependencies (
                    source_symbol_id, target_symbol_id, target_external, dependency_kind,
                    resolution, confidence, evidence_json, usage_count
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    source_symbol_id,
                    target_symbol_id,
                    target_external,
                    dep_kind,
                    dep.get("resolution", "probable"),
                    float(dep.get("confidence", 0.7) or 0.7),
                    json.dumps(dep.get("evidence", {})),
                    int(dep.get("usage_count", 1) or 1),
                ),
            )

    def _resolve_target_symbol_id(self, *, source_symbol_id: int, target_external: str, kind: str) -> Optional[int]:
        """
        Best-effort resolver for target_symbol_id.
        Resolve only when match is unambiguous.
        """
        if not target_external:
            return None

        # 1) direct symbol_uid reference
        if _UID_RE.match(target_external):
            cur = self.conn.execute(
                "SELECT symbol_id FROM symbols WHERE symbol_uid = ? LIMIT 2",
                (target_external,),
            )
            rows = cur.fetchall()
            if len(rows) == 1:
                return int(rows[0][0])

        # 2) imports -> resolve to module symbol by qualified_name
        if kind == "imports":
            candidate = target_external.strip()
            if candidate.startswith("."):
                return None
            for _ in range(6):
                cur = self.conn.execute(
                    "SELECT symbol_id FROM symbols WHERE kind = 'module' AND qualified_name = ? LIMIT 2",
                    (candidate,),
                )
                rows = cur.fetchall()
                if len(rows) == 1:
                    return int(rows[0][0])
                if "." not in candidate:
                    break
                candidate = candidate.rsplit(".", 1)[0]
            return None

        # 3) calls -> resolve simple foo() within same file when unique
        if kind == "calls" and "." not in target_external:
            cur = self.conn.execute("SELECT file_id FROM symbols WHERE symbol_id = ? LIMIT 1", (source_symbol_id,))
            row = cur.fetchone()
            if not row:
                return None
            file_id = int(row[0])
            cur = self.conn.execute(
                """
                SELECT symbol_id
                FROM symbols
                WHERE file_id = ?
                  AND name = ?
                  AND kind IN ('function', 'method')
                LIMIT 2
                """,
                (file_id, target_external),
            )
            rows = cur.fetchall()
            if len(rows) == 1:
                return int(rows[0][0])
            return None

        return None

    def _store_attribute_accesses(self, symbol_ids: Dict[str, int], accesses: List[Dict[str, Any]]) -> None:
        for access in accesses:
            symbol_id = symbol_ids.get(access["symbol_uid"])
            if not symbol_id:
                continue

            self.conn.execute(
                """
                INSERT INTO attribute_access (
                    symbol_id, attribute_name, access_type, line_number, confidence, evidence_json, count
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol_id,
                    access["attribute_name"],
                    access["access_type"],
                    access["line_number"],
                    access["confidence"],
                    json.dumps(access["evidence"]),
                    access.get("count", 1),
                ),
            )

    def _clear_index(self):
        self.logger.info("Clearing existing index data")

        tables_to_clear = [
            "attribute_access",
            "dependencies",
            "blocks",
            "symbols",
            "files",
            "duplicate_members",
            "duplicate_groups",
            "refactoring_decisions",
            "problems",
            "analysis_runs",
        ]

        for table in tables_to_clear:
            try:
                self.conn.execute("DELETE FROM " + self.quote_ident(table))  # nosec B608
            except sqlite3.OperationalError as e:
                self.logger.debug("Could not clear table %s: %s", table, e)

        self.conn.commit()
        self.logger.info("Index data cleared")

    def get_index_status(self) -> Dict[str, Any]:
        cursor = self.conn.execute("SELECT COUNT(*) FROM files")
        files_count = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM symbols")
        symbols_count = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM blocks")
        blocks_count = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT COUNT(*) FROM dependencies")
        dependencies_count = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT MAX(last_analyzed) FROM files")
        last_analysis = cursor.fetchone()[0]

        return {
            "files_indexed": files_count,
            "symbols_indexed": symbols_count,
            "blocks_indexed": blocks_count,
            "dependencies_indexed": dependencies_count,
            "last_analysis": last_analysis,
            "schema_version": IndexSchema.get_schema_version(self.conn),
            "database_size_mb": self._get_database_size_mb(),
        }

    def get_detailed_statistics(self) -> Dict[str, Any]:
        stats = self.get_index_status()

        cursor = self.conn.execute(
            """
            SELECT is_test_file, COUNT(*) as count, SUM(lines_of_code) as total_loc
            FROM files
            GROUP BY is_test_file
            """
        )
        file_breakdown: Dict[str, Any] = {}
        for row in cursor.fetchall():
            file_type = "test_files" if row[0] else "source_files"
            file_breakdown[file_type] = {"count": row[1], "total_lines_of_code": row[2] or 0}

        cursor = self.conn.execute(
            """
            SELECT kind, COUNT(*) as count, AVG(complexity_score) as avg_complexity
            FROM symbols
            GROUP BY kind
            """
        )
        symbol_breakdown: Dict[str, Any] = {}
        for row in cursor.fetchall():
            symbol_breakdown[row[0]] = {"count": row[1], "avg_complexity": round(row[2] or 0, 2)}

        cursor = self.conn.execute(
            """
            SELECT semantic_category, COUNT(*) as count
            FROM symbols
            WHERE semantic_category IS NOT NULL
            GROUP BY semantic_category
            """
        )
        category_breakdown: Dict[str, Any] = {}
        for row in cursor.fetchall():
            category_breakdown[row[0]] = row[1]

        stats.update(
            {
                "file_breakdown": file_breakdown,
                "symbol_breakdown": symbol_breakdown,
                "category_breakdown": category_breakdown,
            }
        )
        return stats

    def _get_database_size_mb(self) -> float:
        try:
            if self.db_path.exists():
                size_bytes = self.db_path.stat().st_size
                return round(size_bytes / (1024 * 1024), 2)
            return 0.0
        except Exception:
            return 0.0

    def close(self):
        if hasattr(self, "conn") and self.conn:
            try:
                self.conn.close()
                self.conn = None
            except Exception as e:
                self.logger.warning("Error closing database connection: %s", e)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()

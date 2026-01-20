"""Unified symbol generation for functional decomposition.

Extracted from DecompositionAnalyzer to reduce god class complexity.
Handles generation of unified symbols via two strategies:
1. moved_impl: Copy implementation to unified module (for top-level functions)
2. delegating: Create wrapper that delegates to canonical implementation
"""
import ast
import builtins
import hashlib
import re
import textwrap
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from .models import FunctionalBlock


def stable_suffix(s: str) -> str:
    """Generate stable 8-character hash suffix for a string."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:8]


def has_top_level_line(src: str, line: str) -> bool:
    """Check if line already exists as standalone top-level line in source.
    
    Matches loosely on leading/trailing whitespace.
    """
    return re.search(rf"^\s*{re.escape(line)}\s*$", src, flags=re.M) is not None


def build_call_arguments(args: ast.arguments) -> str:
    """Build call argument string from AST arguments.
    
    Handles positional-only, regular, *args, keyword-only, and **kwargs.
    """
    parts: List[str] = []
    for a in getattr(args, "posonlyargs", []):
        parts.append(a.arg)
    for a in args.args:
        parts.append(a.arg)
    if args.vararg:
        parts.append(f"*{args.vararg.arg}")
    for a in args.kwonlyargs:
        parts.append(f"{a.arg}={a.arg}")
    if args.kwarg:
        parts.append(f"**{args.kwarg.arg}")
    return ", ".join(parts)


def canonical_method_call_expr(cls_expr: str, meth: str, dec_kind: str) -> str:
    """Generate method call expression for canonical method.
    
    Handles different decorator kinds (classmethod, staticmethod, property, etc.)
    """
    if dec_kind == "classmethod":
        return f"getattr({cls_expr}, {meth!r}).__func__(args[0], *args[1:], **kwargs)"
    if dec_kind == "staticmethod":
        return f"getattr({cls_expr}, {meth!r})(*args, **kwargs)"
    if dec_kind in ("property", "cached_property", "functools.cached_property"):
        return f"(getattr({cls_expr}, {meth!r}).fget)(args[0], *args[1:], **kwargs)"
    return f"getattr({cls_expr}, {meth!r})(args[0], *args[1:], **kwargs)"


def find_existing_unified_symbol_meta(src: str, symbol: str) -> Optional[Dict[str, str]]:
    """Find metadata for existing unified symbol in source.
    
    Looks for [IR_CANONICAL] and [IR_CLUSTER] markers in comments above function def.
    """
    lines = src.splitlines()
    pat = re.compile(rf"^\s*(async\s+def|def)\s+{re.escape(symbol)}\s*\(")
    for i, line in enumerate(lines):
        if not pat.match(line):
            continue
        meta: Dict[str, str] = {}
        for j in range(max(0, i - 12), i):
            m = re.search(r"\[IR_CANONICAL\]\s*(.+)$", lines[j])
            if m:
                meta["canonical_block_id"] = m.group(1).strip()
            m = re.search(r"\[IR_CLUSTER\]\s*(.+)$", lines[j])
            if m:
                meta["cluster_id"] = m.group(1).strip()
        return meta  # may be empty -> still treat as "exists"
    return None


class UnifiedSymbolGenerator:
    """Generates unified symbols for functional decomposition."""
    
    def __init__(self, file_ops, ast_utils_module, wrapper_marker: str):
        """Initialize generator.
        
        Args:
            file_ops: FileOperations instance
            ast_utils_module: ast_utils module for utility functions
            wrapper_marker: Marker string for wrapped functions
        """
        self.file_ops = file_ops
        self.ast_utils = ast_utils_module
        self.wrapper_marker = wrapper_marker
    
    def ensure_unified_symbol_moved_impl(
        self,
        *,
        target_file: Path,
        target_symbol: str,
        canonical_block: FunctionalBlock,
        package_root: Path,
        package_name: str,
    ) -> Tuple[str, bool, List[str]]:
        """Ensure unified symbol exists by moving implementation.
        
        Copies the canonical implementation to the unified module.
        Only works for top-level functions (not methods).
        
        Args:
            target_file: Target unified module file
            target_symbol: Symbol name in unified module
            canonical_block: Canonical functional block
            package_root: Package root path
            package_name: Package name
            
        Returns:
            Tuple of (new_source, success, warnings)
        """
        warnings: List[str] = []
        existing = self.file_ops.read_text(target_file)

        if re.search(rf"^\s*(async\s+def|def)\s+{re.escape(target_symbol)}\s*\(", existing, flags=re.M):
            return existing, True, ["symbol already exists"]

        if canonical_block.is_method:
            return existing, False, ["moved_impl only for top-level functions"]

        src_file = self.file_ops.resolve_block_file_path(canonical_block, package_root)
        src_text = self.file_ops.read_text(src_file, bom=True)

        try:
            mod_tree = ast.parse(src_text, filename=str(src_file))
            fn_node = self.ast_utils.find_def_node(mod_tree, canonical_block.qualname, lineno=canonical_block.lineno)
        except Exception as e:
            return existing, False, [f"cannot locate canonical def: {e}"]

        if not isinstance(fn_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return existing, False, ["canonical node is not a function def"]

        if fn_node.decorator_list:
            return existing, False, ["decorators present on canonical callable"]

        snippet = self.file_ops.slice_lines(src_text, canonical_block.lineno, canonical_block.end_lineno)
        snippet = textwrap.dedent(snippet)

        try:
            sn_tree = ast.parse(snippet)
        except SyntaxError as e:
            return existing, False, [f"snippet parse failed: {e}"]

        sn_fn = next((n for n in sn_tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
        if not sn_fn:
            return existing, False, ["no def in snippet"]

        sn_fn.name = target_symbol
        sn_fn.decorator_list = []
        ast.fix_missing_locations(sn_fn)

        fn_src = ast.unparse(sn_fn)
        if "__ir_unified_" in fn_src or self.wrapper_marker in fn_src:
            return existing, False, ["canonical already looks like wrapper/delegation; skip moved_impl"]

        # Conservative free-name check
        free = self.ast_utils.collect_free_names(sn_fn)
        free = {n for n in free if not hasattr(builtins, n)}
        if free:
            return existing, False, [f"unknown free names in moved_impl: {sorted(free)[:5]}"]

        header = existing
        if not header.strip():
            header = (
                '"""Auto-generated unified implementations.\n'
                "Generated by functional decomposition analyzer.\n"
                '"""\n\n'
                "from __future__ import annotations\n\n"
            )

        block = "\n".join(
            [
                "",
                f"# --- Auto-generated: {target_symbol} (moved from {canonical_block.qualname}) ---",
                ast.unparse(sn_fn),
                "",
            ]
        )
        return header + block, True, warnings
    
    def ensure_unified_symbol_delegating(
        self,
        *,
        target_file: Path,
        target_symbol: str,
        canonical_block: FunctionalBlock,
        canonical_block_id: str,
        cluster_id: str,
        package_root: Path,
        package_name: str,
        module_dotted_from_filepath_fn,
        qualify_module_fn,
    ) -> str:
        """Ensure unified symbol exists by creating delegating wrapper.
        
        Creates a wrapper function that delegates to the canonical implementation.
        Works for both functions and methods.
        
        Args:
            target_file: Target unified module file
            target_symbol: Symbol name in unified module
            canonical_block: Canonical functional block
            canonical_block_id: Canonical block ID
            cluster_id: Cluster ID
            package_root: Package root path
            package_name: Package name
            module_dotted_from_filepath_fn: Function to convert filepath to dotted module
            qualify_module_fn: Function to qualify module name
            
        Returns:
            New source code with unified symbol
        """
        existing = self.file_ops.read_text(target_file)
        meta = find_existing_unified_symbol_meta(existing, target_symbol)
        if meta is not None:
            # symbol exists
            if meta.get("canonical_block_id") != canonical_block_id:
                raise RuntimeError(
                    f"unified symbol collision: {target_file}::{target_symbol} already bound to "
                    f"{meta.get('canonical_block_id')!r}, current {canonical_block_id!r}"
                )
            return existing

        src_file = self.file_ops.resolve_block_file_path(canonical_block, package_root)
        orig_mod = qualify_module_fn(canonical_block.module, package_name) or module_dotted_from_filepath_fn(src_file, package_root, package_name)
        
        # Detect if async
        try:
            tree = self.file_ops.parse_file(src_file)
            node = self.ast_utils.find_def_node(tree, canonical_block.qualname, lineno=canonical_block.lineno)
            is_async = isinstance(node, ast.AsyncFunctionDef)
        except Exception:
            is_async = False

        header = existing
        if not header.strip():
            header = (
                '"""Auto-generated unified implementations.\n'
                "Safe mode delegates to canonical originals.\n"
                '"""\n\n'
                "from __future__ import annotations\n\n"
            )

        lines: List[str] = []
        lines.append("")
        lines.append(f"# --- Auto-generated: {target_symbol} (delegate to canonical) ---")
        lines.append(f"# [IR_CLUSTER] {cluster_id}")
        lines.append(f"# [IR_CANONICAL] {canonical_block_id}")

        if canonical_block.is_method:
            mod_alias = f"__ir_canonical_mod_{stable_suffix(orig_mod)}"
            import_line = f"import {orig_mod} as {mod_alias}"
            if not has_top_level_line(existing, import_line):
                lines.append(import_line)

            class_chain = canonical_block.qualname.split(".")[:-1]
            meth = canonical_block.method_name
            cls_expr = mod_alias + "".join(f".{c}" for c in class_chain)

            dec_kind = ""
            try:
                tree = self.file_ops.parse_file(src_file)
                node = self.ast_utils.find_def_node(tree, canonical_block.qualname, lineno=canonical_block.lineno)
                dec_kind = self.ast_utils.get_decorator_kind(node)
            except Exception:
                dec_kind = ""

            lines.append("")
            if is_async:
                lines.append(f"async def {target_symbol}(*args, **kwargs):")
                call = canonical_method_call_expr(cls_expr, meth, dec_kind)
                lines.append(f"    return await {call}")
            else:
                lines.append(f"def {target_symbol}(*args, **kwargs):")
                call = canonical_method_call_expr(cls_expr, meth, dec_kind)
                lines.append(f"    return {call}")
        else:
            fn = canonical_block.method_name
            alias = f"__ir_canonical_{fn}_{stable_suffix(orig_mod)}"
            from_line = f"from {orig_mod} import {fn} as {alias}"
            if not has_top_level_line(existing, from_line):
                lines.append(from_line)
            lines.append("")
            if is_async:
                lines.append(f"async def {target_symbol}(*args, **kwargs):")
                lines.append(f"    return await {alias}(*args, **kwargs)")
            else:
                lines.append(f"def {target_symbol}(*args, **kwargs):")
                lines.append(f"    return {alias}(*args, **kwargs)")

        lines.append("")
        return header + "\n".join(lines)

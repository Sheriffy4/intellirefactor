from __future__ import annotations

"""
Static type-consistency audit for IntelliRefactor (AST-based).

Primary purpose:
  - Find duplicated/competing enums for severity (AuditSeverity/ErrorSeverity/etc.)
  - Find dataclass fields named "severity" that are typed as str or non-canonical enum
  - Find suspicious imports of *Severity from non-foundation modules
  - Find comparisons of severity to string literals (common post-migration bug)
  - Find severity rendered directly into text reports (should typically use `.value`)

This helps you safely remove compatibility aliases over time.

Usage:
  python -m intellirefactor.tools.type_consistency_audit <repo_root> --format json
  python intellirefactor/tools/type_consistency_audit.py <repo_root> --output report.json
"""

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


CANONICAL_SEVERITY_TOKENS = {"critical", "high", "medium", "low", "info"}

DEFAULT_SKIP_DIRS = {
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
    "node_modules",
    "dist",
    "build",
}


def _is_skipped(path: Path) -> bool:
    parts = set(path.parts)
    return bool(parts.intersection(DEFAULT_SKIP_DIRS))


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)  # py>=3.9
    except Exception:
        return node.__class__.__name__


def _base_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def _is_enum_base(base: ast.AST) -> bool:
    # Accept "Enum" or "enum.Enum"
    if isinstance(base, ast.Name) and base.id == "Enum":
        return True
    if isinstance(base, ast.Attribute) and base.attr == "Enum":
        return True
    return False


def _has_dataclass_decorator(node: ast.ClassDef) -> bool:
    for dec in node.decorator_list or []:
        if isinstance(dec, ast.Name) and dec.id == "dataclass":
            return True
        if isinstance(dec, ast.Attribute) and dec.attr == "dataclass":
            return True
        if isinstance(dec, ast.Call):
            # @dataclass(...)
            f = dec.func
            if isinstance(f, ast.Name) and f.id == "dataclass":
                return True
            if isinstance(f, ast.Attribute) and f.attr == "dataclass":
                return True
    return False


def _literal_str(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _collect_str_constants(node: ast.AST) -> List[str]:
    """Collect all string literals inside a node."""
    out: List[str] = []
    for n in ast.walk(node):
        s = _literal_str(n)
        if s is not None:
            out.append(s)
    return out


def _scope_uses_severity_enum(scope: ast.AST) -> bool:
    """
    Heuristic: does THIS scope (module/class/function body) show evidence of enum-based severity usage?
    Important: should not "see" nested scopes.

    Signals:
      A) Explicit enum names: Severity / ErrorSeverity / AuditSeverity
      B) Enum-ish value rendering without naming enum: <severity_like>.value
         where severity_like is:
           - severity / obj.severity
           - getattr(obj, "severity")
           - mapping["severity"]
           - mapping.get("severity")
    """
    SEV_NAMES = {"Severity", "ErrorSeverity", "AuditSeverity"}

    def _slice_literal_str(slc: ast.AST) -> Optional[str]:
        # py<3.9: ast.Index(value=...)
        if hasattr(ast, "Index") and isinstance(slc, ast.Index):  # type: ignore[attr-defined]
            return _literal_str(slc.value)
        return _literal_str(slc)

    def _looks_like_severity_like_expr(n: ast.AST) -> bool:
        # severity / obj.severity
        if isinstance(n, ast.Name) and n.id == "severity":
            return True
        if isinstance(n, ast.Attribute) and n.attr == "severity":
            return True

        # getattr(obj, "severity")
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Name)
            and n.func.id == "getattr"
            and len(n.args) >= 2
            and _literal_str(n.args[1]) == "severity"
        ):
            return True

        # mapping["severity"]
        if isinstance(n, ast.Subscript):
            if _slice_literal_str(n.slice) == "severity":
                return True

        # mapping.get("severity")
        if (
            isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "get"
            and n.args
            and _literal_str(n.args[0]) == "severity"
        ):
            return True

        return False

    for n in _iter_nodes_no_nested_scopes(scope):
        if isinstance(n, ast.Name) and n.id in SEV_NAMES:
            return True
        if isinstance(n, ast.Attribute):
            # Severity.HIGH / Severity.high etc.
            if isinstance(n.value, ast.Name) and n.value.id in SEV_NAMES:
                return True
            # enum-ish render: <severity_like>.value
            if n.attr == "value" and _looks_like_severity_like_expr(n.value):
                return True
        if isinstance(n, ast.Call):
            # Severity("high") / ErrorSeverity("medium")
            if isinstance(n.func, ast.Name) and n.func.id in SEV_NAMES:
                return True
            # isinstance(x, Severity)
            if isinstance(n.func, ast.Name) and n.func.id == "isinstance" and len(n.args) >= 2:
                if isinstance(n.args[1], ast.Name) and n.args[1].id in SEV_NAMES:
                    return True
        if isinstance(n, ast.ImportFrom) and getattr(n, "module", None):
            for a in n.names or []:
                if getattr(a, "name", None) in SEV_NAMES:
                    return True
    return False


def _subscript_key_literal(sub: ast.Subscript) -> Optional[str]:
    """
    Return literal string key for dict-like subscription: x["severity"].
    Handles Python 3.8 ast.Index and 3.9+ direct slice nodes.
    """
    slc = sub.slice
    # py<3.9: ast.Index(value=...)
    if hasattr(ast, "Index") and isinstance(slc, ast.Index):  # type: ignore[attr-defined]
        return _literal_str(slc.value)
    return _literal_str(slc)


def _is_getattr_severity_call(node: ast.AST) -> bool:
    """
    getattr(obj, "severity"[,...])
    """
    if not isinstance(node, ast.Call):
        return False
    if not (isinstance(node.func, ast.Name) and node.func.id == "getattr"):
        return False
    if len(node.args) < 2:
        return False
    k = _literal_str(node.args[1])
    return (k == "severity")


def _is_dict_get_severity_call(node: ast.AST) -> bool:
    """
    some_dict.get("severity"[,...])
    """
    if not isinstance(node, ast.Call):
        return False
    if not (isinstance(node.func, ast.Attribute) and node.func.attr == "get"):
        return False
    if not node.args:
        return False
    k = _literal_str(node.args[0])
    return (k == "severity")


def _is_severity_subscript(node: ast.AST) -> bool:
    """
    some_dict["severity"]
    """
    if not isinstance(node, ast.Subscript):
        return False
    k = _subscript_key_literal(node)
    return (k == "severity")


def _is_logger_method_call(node: ast.AST) -> bool:
    """
    logger.info(...) or self.logger.info(...)
    """
    if not isinstance(node, ast.Call):
        return False
    f = node.func
    if not isinstance(f, ast.Attribute):
        return False
    if f.attr not in {"debug", "info", "warning", "error", "exception", "critical"}:
        return False

    base = f.value
    if isinstance(base, ast.Name) and base.id == "logger":
        return True
    if isinstance(base, ast.Attribute) and base.attr == "logger" and isinstance(base.value, ast.Name) and base.value.id == "self":
        return True
    return False


def _classify_severity_expr_str(expr: str, *, strict_aliases: set[str], uncertain_aliases: set[str]) -> str:
    """
    Best-effort classification based on unparsed expression string and alias sets.
    """
    if expr in strict_aliases:
        return "alias:strict"
    if expr in uncertain_aliases:
        return "alias:uncertain"

    e = expr.replace('"', "'")
    if e == "severity" or e.endswith(".severity"):
        return "direct"
    if e.startswith("getattr(") and "'severity'" in e:
        return "accessor:getattr"
    if ".get('severity'" in e:
        return "accessor:dict_get"
    if "['severity']" in e:
        return "accessor:subscript"
    return "unknown"


def _confidence_for_finding(
    *,
    kinds_map: Dict[str, str],
    uses_enum_in_scope: bool,
    alias_value_hints: set[str],
) -> str:
    """
    More accurate confidence:
      - high: direct severity OR strict alias from `.severity`
      - medium: scope is enum-aware OR we have evidence that an alias is used with `.value`
      - low: accessor-only with no enum evidence (likely JSON/dict string severity)

    alias_value_hints: names of alias variables that appear as `<alias>.value` somewhere in the same function scope.
    """
    if any(k in {"direct", "alias:strict"} for k in kinds_map.values()):
        return "high"

    # If any referenced alias is known to be used as enum via `.value`, treat as enum-aware locally.
    for expr, kind in kinds_map.items():
        if kind.startswith("alias:") and expr in alias_value_hints:
            return "medium"

    return "medium" if uses_enum_in_scope else "low"


def _iter_nodes_no_nested_scopes(node: ast.AST) -> List[ast.AST]:
    """
    Walk node children but do NOT descend into nested scopes:
      - FunctionDef / AsyncFunctionDef / Lambda / ClassDef

    This prevents outer-function alias tracking from "seeing" inner-function bodies.
    """
    out: List[ast.AST] = []
    stack: List[ast.AST] = [node]
    while stack:
        n = stack.pop()
        out.append(n)
        for ch in ast.iter_child_nodes(n):
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            stack.append(ch)
    return out


def _extract_assigned_names(target: ast.AST) -> List[str]:
    """
    Extract simple assigned variable names from assignment targets:
      - Name
      - Tuple/List destructuring
    """
    names: List[str] = []

    def rec(t: ast.AST) -> None:
        if isinstance(t, ast.Name):
            names.append(t.id)
            return
        if isinstance(t, (ast.Tuple, ast.List)):
            for el in t.elts:
                rec(el)
            return
        # ignore attributes/subscripts/etc (we don't model them)

    rec(target)
    return names


def _is_severity_ref(node: ast.AST) -> bool:
    """
    True if node is a direct reference to a `severity` symbol:
      - severity
      - something.severity
    """
    if isinstance(node, ast.Name) and node.id == "severity":
        return True
    if isinstance(node, ast.Attribute) and node.attr == "severity":
        return True
    return False


def _is_alias_ref(node: ast.AST, aliases: set[str]) -> bool:
    """True if node is Name(...) and that name is in alias set."""
    return isinstance(node, ast.Name) and node.id in aliases


def _is_severity_accessor_expr(node: ast.AST) -> bool:
    """
    "severity-like" access patterns that often replace direct `.severity`:
      - getattr(obj, "severity")
      - mapping["severity"]
      - mapping.get("severity")
    """
    return _is_getattr_severity_call(node) or _is_severity_subscript(node) or _is_dict_get_severity_call(node)


def _is_severity_like_ref(node: ast.AST, aliases: set[str], *, include_direct: bool) -> bool:
    if include_direct and _is_severity_ref(node):
        return True
    if include_direct and _is_severity_accessor_expr(node):
        return True
    if _is_alias_ref(node, aliases):
        return True
    return False


def _is_severity_value_access(node: ast.AST) -> bool:
    """
    True if node is: <severity_ref>.value  (e.g. finding.severity.value, severity.value)
    """
    if isinstance(node, ast.Attribute) and node.attr == "value":
        return _is_severity_ref(node.value)
    return False


def _is_severity_like_value_access(node: ast.AST, aliases: set[str], *, include_direct: bool) -> bool:
    if isinstance(node, ast.Attribute) and node.attr == "value":
        return _is_severity_like_ref(node.value, aliases, include_direct=include_direct)
    return False


def _find_unvalued_severity_exprs(node: ast.AST) -> List[str]:
    """
    Return stringified subexpressions that reference `severity` but are NOT already `.value`.

    We use a parent-aware walk to avoid flagging `something.severity` when it is part of
    `something.severity.value`.
    """
    found: List[str] = []

    def walk(n: ast.AST, parent: Optional[ast.AST]) -> None:
        # if `n` is severity ref and parent is ".value" attribute access -> do not count
        if _is_severity_ref(n):
            if isinstance(parent, ast.Attribute) and parent.attr == "value":
                return
            found.append(_safe_unparse(n))
            return

        # if the whole node is already severity.value -> skip exploring inside
        if _is_severity_value_access(n):
            return

        for ch in ast.iter_child_nodes(n):
            walk(ch, n)

    walk(node, None)
    # de-dup while keeping order
    uniq: List[str] = []
    seen = set()
    for x in found:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _find_unvalued_severity_like_exprs(
    node: ast.AST,
    aliases: set[str],
    *,
    include_direct: bool,
) -> List[str]:
    """
    Like `_find_unvalued_severity_exprs`, but also treats alias variables as severity-like refs.

    include_direct:
      - True  => count direct `severity` / `x.severity` too
      - False => count only aliases (e.g. `sev`) and ignore direct refs (to avoid duplicates)
    """
    found: List[str] = []

    def walk(n: ast.AST, parent: Optional[ast.AST]) -> None:
        if _is_severity_like_ref(n, aliases, include_direct=include_direct):
            if isinstance(parent, ast.Attribute) and parent.attr == "value":
                return
            found.append(_safe_unparse(n))
            return

        if _is_severity_like_value_access(n, aliases, include_direct=include_direct):
            return

        for ch in ast.iter_child_nodes(n):
            # Do not descend into nested scopes
            if isinstance(ch, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)):
                continue
            walk(ch, n)

    walk(node, None)

    uniq: List[str] = []
    seen = set()
    for x in found:
        if x not in seen:
            uniq.append(x)
            seen.add(x)
    return uniq


def _looks_like_text_template(node: ast.AST) -> bool:
    """
    Heuristic: is node likely a text template / report string?
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return True
    if isinstance(node, ast.JoinedStr):
        return True
    return False


@dataclass
class EnumInfo:
    file: str
    line: int
    name: str
    values: List[str]
    looks_like_severity: bool
    matches_canonical: bool


@dataclass
class SeverityFieldInfo:
    file: str
    line: int
    class_name: str
    is_dataclass: bool
    annotation: str
    default: str


@dataclass
class ImportInfo:
    file: str
    line: int
    module: str
    names: List[str]


@dataclass
class SeverityStringComparisonInfo:
    file: str
    line: int
    expr: str
    severity_exprs: List[str]
    string_literals: List[str]
    severity_expr_kinds: Dict[str, str]
    confidence: str


@dataclass
class SeverityStringRenderInfo:
    file: str
    line: int
    kind: str  # f-string / format / percent / concat
    expr: str
    severity_exprs: List[str]
    severity_expr_kinds: Dict[str, str]
    confidence: str


def _scan_expr_for_string_cmp_and_render(
    expr: ast.AST,
    *,
    py_file: Path,
    strict_aliases: set[str],
    uncertain_aliases: set[str],
    include_direct: bool,
    uses_enum_in_scope: bool,
    alias_value_hints: set[str],
    out_comparisons: List[SeverityStringComparisonInfo],
    out_renderings: List[SeverityStringRenderInfo],
) -> None:
    """
    Scan expression-like AST (or a statement subtree) for:
      - severity/alias comparisons to canonical string tokens
      - severity/alias rendering into strings

    `include_direct=False` is used for alias-based pass to avoid duplicating direct findings.
    """
    aliases_all = set(strict_aliases) | set(uncertain_aliases)
    for node in _iter_nodes_no_nested_scopes(expr):
        # (1) Compare severity-like to string literal(s) without `.value`
        if isinstance(node, ast.Compare):
            sev_exprs = _find_unvalued_severity_like_exprs(node, aliases_all, include_direct=include_direct)
            if sev_exprs:
                strs = [s for s in _collect_str_constants(node) if s.lower() in CANONICAL_SEVERITY_TOKENS]
                if strs:
                    kinds_map = {
                        s: _classify_severity_expr_str(s, strict_aliases=strict_aliases, uncertain_aliases=uncertain_aliases)
                        for s in sev_exprs
                    }
                    conf = _confidence_for_finding(
                        kinds_map=kinds_map,
                        uses_enum_in_scope=uses_enum_in_scope,
                        alias_value_hints=alias_value_hints,
                    )
                    out_comparisons.append(
                        SeverityStringComparisonInfo(
                            file=str(py_file),
                            line=getattr(node, "lineno", 1),
                            expr=_safe_unparse(node),
                            severity_exprs=sev_exprs,
                            string_literals=strs,
                            severity_expr_kinds=kinds_map,
                            confidence=conf,
                        )
                    )

        # (2) f"...{sev}..."
        if isinstance(node, ast.JoinedStr):
            sev_exprs: List[str] = []
            for part in node.values:
                if isinstance(part, ast.FormattedValue):
                    sev_exprs.extend(
                        _find_unvalued_severity_like_exprs(
                            part.value, aliases_all, include_direct=include_direct
                        )
                    )
            if sev_exprs:
                kinds_map = {
                    s: _classify_severity_expr_str(s, strict_aliases=strict_aliases, uncertain_aliases=uncertain_aliases)
                    for s in sorted(set(sev_exprs))
                }
                conf = _confidence_for_finding(
                    kinds_map=kinds_map,
                    uses_enum_in_scope=uses_enum_in_scope,
                    alias_value_hints=alias_value_hints,
                )
                out_renderings.append(
                    SeverityStringRenderInfo(
                        file=str(py_file),
                        line=getattr(node, "lineno", 1),
                        kind="f-string",
                        expr=_safe_unparse(node),
                        severity_exprs=sorted(set(sev_exprs)),
                        severity_expr_kinds=kinds_map,
                        confidence=conf,
                    )
                )

        # (3) "...%s..." % sev
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mod) and _looks_like_text_template(node.left):
            sev_exprs = _find_unvalued_severity_like_exprs(node.right, aliases_all, include_direct=include_direct)
            if sev_exprs:
                kinds_map = {
                    s: _classify_severity_expr_str(s, strict_aliases=strict_aliases, uncertain_aliases=uncertain_aliases)
                    for s in sev_exprs
                }
                conf = _confidence_for_finding(
                    kinds_map=kinds_map,
                    uses_enum_in_scope=uses_enum_in_scope,
                    alias_value_hints=alias_value_hints,
                )
                out_renderings.append(
                    SeverityStringRenderInfo(
                        file=str(py_file),
                        line=getattr(node, "lineno", 1),
                        kind="percent-format",
                        expr=_safe_unparse(node),
                        severity_exprs=sev_exprs,
                        severity_expr_kinds=kinds_map,
                        confidence=conf,
                    )
                )

        # (4) "...{}".format(sev) / template.format(severity=sev)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "format":
            sev_exprs: List[str] = []
            for a in node.args:
                sev_exprs.extend(_find_unvalued_severity_like_exprs(a, aliases_all, include_direct=include_direct))
            for kw in node.keywords or []:
                if kw.value is not None:
                    sev_exprs.extend(
                        _find_unvalued_severity_like_exprs(kw.value, aliases_all, include_direct=include_direct)
                    )
            if sev_exprs:
                sev_exprs_u = sorted(set(sev_exprs))
                kinds_map = {
                    s: _classify_severity_expr_str(s, strict_aliases=strict_aliases, uncertain_aliases=uncertain_aliases)
                    for s in sev_exprs_u
                }
                conf = _confidence_for_finding(
                    kinds_map=kinds_map,
                    uses_enum_in_scope=uses_enum_in_scope,
                    alias_value_hints=alias_value_hints,
                )
                out_renderings.append(
                    SeverityStringRenderInfo(
                        file=str(py_file),
                        line=getattr(node, "lineno", 1),
                        kind="str.format",
                        expr=_safe_unparse(node),
                        severity_exprs=sev_exprs_u,
                        severity_expr_kinds=kinds_map,
                        confidence=conf,
                    )
                )

        # (5) "Severity: " + sev
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            if _looks_like_text_template(node.left) or _looks_like_text_template(node.right):
                sev_exprs = _find_unvalued_severity_like_exprs(node, aliases_all, include_direct=include_direct)
                if sev_exprs:
                    kinds_map = {
                        s: _classify_severity_expr_str(s, strict_aliases=strict_aliases, uncertain_aliases=uncertain_aliases)
                        for s in sev_exprs
                    }
                    conf = _confidence_for_finding(
                        kinds_map=kinds_map,
                        uses_enum_in_scope=uses_enum_in_scope,
                        alias_value_hints=alias_value_hints,
                    )
                    out_renderings.append(
                        SeverityStringRenderInfo(
                            file=str(py_file),
                            line=getattr(node, "lineno", 1),
                            kind="concat",
                            expr=_safe_unparse(node),
                            severity_exprs=sev_exprs,
                            severity_expr_kinds=kinds_map,
                            confidence=conf,
                        )
                    )

        # (6) lazy logging: logger.info("severity=%s", sev)
        # We keep it conservative: only when the first arg is a literal string containing "severity".
        if isinstance(node, ast.Call) and _is_logger_method_call(node):
            pos = [a for a in node.args]  # keywords are separate in ast.Call.keywords
            if pos and isinstance(pos[0], ast.Constant) and isinstance(pos[0].value, str):
                msg = pos[0].value
                if "severity" in msg.lower():
                    sev_exprs: List[str] = []
                    # check subsequent positional args only (logger lazy formatting)
                    for a in pos[1:]:
                        sev_exprs.extend(
                            _find_unvalued_severity_like_exprs(a, aliases_all, include_direct=include_direct)
                        )
                    if sev_exprs:
                        sev_exprs_u = sorted(set(sev_exprs))
                        kinds_map = {
                            s: _classify_severity_expr_str(s, strict_aliases=strict_aliases, uncertain_aliases=uncertain_aliases)
                            for s in sev_exprs_u
                        }
                        conf = _confidence_for_finding(
                            kinds_map=kinds_map,
                            uses_enum_in_scope=uses_enum_in_scope,
                            alias_value_hints=alias_value_hints,
                        )
                        out_renderings.append(
                            SeverityStringRenderInfo(
                                file=str(py_file),
                                line=getattr(node, "lineno", 1),
                                kind="logger-lazy-format",
                                expr=_safe_unparse(node),
                                severity_exprs=sev_exprs_u,
                                severity_expr_kinds=kinds_map,
                                confidence=conf,
                            )
                        )


def _scan_function_for_severity_alias_issues(
    fn: ast.AST,
    *,
    py_file: Path,
    out_comparisons: List[SeverityStringComparisonInfo],
    out_renderings: List[SeverityStringRenderInfo],
    uses_enum_in_scope: bool,
) -> None:
    """
    Very light intra-function alias tracking:
      sev = finding.severity
      sev2 = sev
    and then detect string comparisons/rendering involving sev/sev2.
    """
    strict_aliases: set[str] = set()
    uncertain_aliases: set[str] = set()

    def _collect_alias_value_hints() -> set[str]:
        """
        Pre-pass: collect alias variables that are used as `<alias>.value` somewhere in this function,
        under the alias-tracking model (strict/uncertain) and respecting control-flow/scope boundaries.
        """
        hints: set[str] = set()

        def _local_rhs_kind(rhs: Optional[ast.AST], cur_s: set[str], cur_u: set[str]) -> Optional[str]:
            if rhs is None:
                return None
            if _is_severity_like_value_access(rhs, cur_s | cur_u, include_direct=True):
                return None
            if _is_severity_ref(rhs):
                return "strict"
            if _is_getattr_severity_call(rhs):
                return "uncertain"
            if _is_severity_subscript(rhs) or _is_dict_get_severity_call(rhs):
                return "uncertain"
            if isinstance(rhs, ast.Name):
                if rhs.id in cur_s:
                    return "strict"
                if rhs.id in cur_u:
                    return "uncertain"
            return None

        def _local_update_aliases_from_assign(
            targets: List[ast.AST],
            rhs: Optional[ast.AST],
            cur_s: set[str],
            cur_u: set[str],
        ) -> Tuple[set[str], set[str]]:
            cur_s = set(cur_s)
            cur_u = set(cur_u)
            assigned: List[str] = []
            for t in targets:
                assigned.extend(_extract_assigned_names(t))

            k = _local_rhs_kind(rhs, cur_s, cur_u)
            if k == "strict":
                for nm in assigned:
                    cur_s.add(nm)
                    cur_u.discard(nm)
            elif k == "uncertain":
                for nm in assigned:
                    cur_u.add(nm)
                    cur_s.discard(nm)
            else:
                for nm in assigned:
                    cur_s.discard(nm)
                    cur_u.discard(nm)
            return cur_s, cur_u

        def has_alias_value_access(stmt_or_expr: ast.AST, cur_s: set[str], cur_u: set[str]) -> None:
            aliases = cur_s | cur_u
            for n in _iter_nodes_no_nested_scopes(stmt_or_expr):
                if isinstance(n, ast.Attribute) and n.attr == "value" and isinstance(n.value, ast.Name):
                    if n.value.id in aliases:
                        hints.add(n.value.id)

        def hint_stmt(stmt: ast.stmt, cur_s: set[str], cur_u: set[str]) -> Tuple[set[str], set[str]]:
            # Check `.value` usage in the statement itself (including header expressions)
            has_alias_value_access(stmt, cur_s, cur_u)

            # Control flow (mirror the alias-tracker structure, but without reporting)
            if isinstance(stmt, ast.If):
                has_alias_value_access(stmt.test, cur_s, cur_u)
                bs, bu = hint_block(stmt.body, set(cur_s), set(cur_u))
                es, eu = hint_block(stmt.orelse, set(cur_s), set(cur_u))
                return (cur_s | bs | es), (cur_u | bu | eu)

            if isinstance(stmt, (ast.For, ast.AsyncFor)):
                has_alias_value_access(stmt.iter, cur_s, cur_u)
                # kill loop targets
                ks = set(cur_s)
                ku = set(cur_u)
                for nm in _extract_assigned_names(stmt.target):
                    ks.discard(nm)
                    ku.discard(nm)
                bs, bu = hint_block(stmt.body, ks, ku)
                es, eu = hint_block(stmt.orelse, set(cur_s), set(cur_u))
                return (cur_s | bs | es), (cur_u | bu | eu)

            if isinstance(stmt, ast.While):
                has_alias_value_access(stmt.test, cur_s, cur_u)
                bs, bu = hint_block(stmt.body, set(cur_s), set(cur_u))
                es, eu = hint_block(stmt.orelse, set(cur_s), set(cur_u))
                return (cur_s | bs | es), (cur_u | bu | eu)

            if isinstance(stmt, (ast.With, ast.AsyncWith)):
                for item in stmt.items:
                    has_alias_value_access(item.context_expr, cur_s, cur_u)
                # kill assigned optional vars
                ks = set(cur_s)
                ku = set(cur_u)
                for item in stmt.items:
                    if item.optional_vars is not None:
                        for nm in _extract_assigned_names(item.optional_vars):
                            ks.discard(nm)
                            ku.discard(nm)
                bs, bu = hint_block(stmt.body, ks, ku)
                return (cur_s | bs), (cur_u | bu)

            if isinstance(stmt, ast.Try):
                bs, bu = hint_block(stmt.body, set(cur_s), set(cur_u))
                os, ou = hint_block(stmt.orelse, set(cur_s), set(cur_u))
                fs, fu = hint_block(stmt.finalbody, set(cur_s), set(cur_u))
                hs: set[str] = set()
                hu: set[str] = set()
                for h in stmt.handlers or []:
                    hcs = set(cur_s)
                    hcu = set(cur_u)
                    if getattr(h, "name", None) and isinstance(h.name, str):
                        hcs.discard(h.name)
                        hcu.discard(h.name)
                    x_s, x_u = hint_block(h.body, hcs, hcu)
                    hs |= x_s
                    hu |= x_u
                return (cur_s | bs | os | fs | hs), (cur_u | bu | ou | fu | hu)

            # Transfer function for aliases (same as in main pass)
            if isinstance(stmt, ast.Assign):
                return _local_update_aliases_from_assign(stmt.targets, stmt.value, cur_s, cur_u)
            if isinstance(stmt, ast.AnnAssign):
                return _local_update_aliases_from_assign([stmt.target], stmt.value, cur_s, cur_u)
            if isinstance(stmt, ast.AugAssign):
                ns = set(cur_s)
                nu = set(cur_u)
                for nm in _extract_assigned_names(stmt.target):
                    ns.discard(nm)
                    nu.discard(nm)
                return ns, nu

            return cur_s, cur_u

        def hint_block(stmts: List[ast.stmt], cur_s: set[str], cur_u: set[str]) -> Tuple[set[str], set[str]]:
            cur_s = set(cur_s)
            cur_u = set(cur_u)
            for st in stmts or []:
                cur_s, cur_u = hint_stmt(st, cur_s, cur_u)
            return cur_s, cur_u

        body = getattr(fn, "body", None)
        if isinstance(body, list):
            hint_block(body, set(), set())
        return hints

    # Precompute alias `.value` hints for stable confidence (no order-dependent confidence)
    alias_value_hints = _collect_alias_value_hints()
    # If we saw `<alias>.value`, treat this scope as enum-aware even if Severity isn't named explicitly
    uses_enum_in_scope = bool(uses_enum_in_scope) or bool(alias_value_hints)

    def rhs_kind(rhs: Optional[ast.AST], cur_strict: set[str], cur_uncertain: set[str]) -> Optional[str]:
        if rhs is None:
            return None
        # If it's already `.value`, it's (likely) a string; not an enum alias
        if _is_severity_like_value_access(rhs, cur_strict | cur_uncertain, include_direct=True):
            return None
        if _is_severity_ref(rhs):
            return "strict"
        # getattr(obj, "severity")
        if _is_getattr_severity_call(rhs):
            return "uncertain"
        # mapping["severity"] or mapping.get("severity")
        if _is_severity_subscript(rhs) or _is_dict_get_severity_call(rhs):
            return "uncertain"
        if isinstance(rhs, ast.Name):
            if rhs.id in cur_strict:
                return "strict"
            if rhs.id in cur_uncertain:
                return "uncertain"
        return None

    def update_aliases_from_assign(
        targets: List[ast.AST],
        rhs: Optional[ast.AST],
        cur_strict: set[str],
        cur_uncertain: set[str],
    ) -> Tuple[set[str], set[str]]:
        cur_strict = set(cur_strict)
        cur_uncertain = set(cur_uncertain)
        assigned_names: List[str] = []
        for t in targets:
            assigned_names.extend(_extract_assigned_names(t))

        k = rhs_kind(rhs, cur_strict, cur_uncertain)
        if k == "strict":
            for nm in assigned_names:
                cur_strict.add(nm)
                cur_uncertain.discard(nm)
        elif k == "uncertain":
            for nm in assigned_names:
                cur_uncertain.add(nm)
                cur_strict.discard(nm)
        else:
            for nm in assigned_names:
                cur_strict.discard(nm)
                cur_uncertain.discard(nm)
        return cur_strict, cur_uncertain

    def process_stmt(stmt: ast.stmt, cur_strict: set[str], cur_uncertain: set[str]) -> Tuple[set[str], set[str]]:
        # Compound statements: scan header expressions only, recurse into blocks separately
        if isinstance(stmt, ast.If):
            _scan_expr_for_string_cmp_and_render(
                stmt.test,
                py_file=py_file,
                strict_aliases=cur_strict,
                uncertain_aliases=cur_uncertain,
                include_direct=False,
                uses_enum_in_scope=uses_enum_in_scope,
                alias_value_hints=alias_value_hints,
                out_comparisons=out_comparisons,
                out_renderings=out_renderings,
            )
            b_strict, b_uncertain = process_block(stmt.body, set(cur_strict), set(cur_uncertain))
            e_strict, e_uncertain = process_block(stmt.orelse, set(cur_strict), set(cur_uncertain))
            return (cur_strict | b_strict | e_strict), (cur_uncertain | b_uncertain | e_uncertain)

        if isinstance(stmt, (ast.For, ast.AsyncFor)):
            _scan_expr_for_string_cmp_and_render(
                stmt.iter,
                py_file=py_file,
                strict_aliases=cur_strict,
                uncertain_aliases=cur_uncertain,
                include_direct=False,
                uses_enum_in_scope=uses_enum_in_scope,
                alias_value_hints=alias_value_hints,
                out_comparisons=out_comparisons,
                out_renderings=out_renderings,
            )
            # Target assigns names each iteration -> kill those aliases in loop body entry
            k_strict = set(cur_strict)
            k_uncertain = set(cur_uncertain)
            for nm in _extract_assigned_names(stmt.target):
                k_strict.discard(nm)
                k_uncertain.discard(nm)
            b_strict, b_uncertain = process_block(stmt.body, k_strict, k_uncertain)
            e_strict, e_uncertain = process_block(stmt.orelse, set(cur_strict), set(cur_uncertain))
            return (cur_strict | b_strict | e_strict), (cur_uncertain | b_uncertain | e_uncertain)

        if isinstance(stmt, ast.While):
            _scan_expr_for_string_cmp_and_render(
                stmt.test,
                py_file=py_file,
                strict_aliases=cur_strict,
                uncertain_aliases=cur_uncertain,
                include_direct=False,
                uses_enum_in_scope=uses_enum_in_scope,
                alias_value_hints=alias_value_hints,
                out_comparisons=out_comparisons,
                out_renderings=out_renderings,
            )
            b_strict, b_uncertain = process_block(stmt.body, set(cur_strict), set(cur_uncertain))
            e_strict, e_uncertain = process_block(stmt.orelse, set(cur_strict), set(cur_uncertain))
            return (cur_strict | b_strict | e_strict), (cur_uncertain | b_uncertain | e_uncertain)

        if isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                _scan_expr_for_string_cmp_and_render(
                    item.context_expr,
                    py_file=py_file,
                    strict_aliases=cur_strict,
                    uncertain_aliases=cur_uncertain,
                    include_direct=False,
                    uses_enum_in_scope=uses_enum_in_scope,
                    alias_value_hints=alias_value_hints,
                    out_comparisons=out_comparisons,
                    out_renderings=out_renderings,
                )
            # optional_vars assigns -> kill those aliases on entry
            k_strict = set(cur_strict)
            k_uncertain = set(cur_uncertain)
            for item in stmt.items:
                if item.optional_vars is not None:
                    for nm in _extract_assigned_names(item.optional_vars):
                        k_strict.discard(nm)
                        k_uncertain.discard(nm)
            b_strict, b_uncertain = process_block(stmt.body, k_strict, k_uncertain)
            return (cur_strict | b_strict), (cur_uncertain | b_uncertain)

        if isinstance(stmt, ast.Try):
            b_strict, b_uncertain = process_block(stmt.body, set(cur_strict), set(cur_uncertain))
            o_strict, o_uncertain = process_block(stmt.orelse, set(cur_strict), set(cur_uncertain))
            f_strict, f_uncertain = process_block(stmt.finalbody, set(cur_strict), set(cur_uncertain))
            h_strict: set[str] = set()
            h_uncertain: set[str] = set()
            for h in stmt.handlers or []:
                hs = set(cur_strict)
                hu = set(cur_uncertain)
                # except ... as name  assigns name -> kill alias
                if getattr(h, "name", None):
                    if isinstance(h.name, str):
                        hs.discard(h.name)
                        hu.discard(h.name)
                bs, bu = process_block(h.body, hs, hu)
                h_strict |= bs
                h_uncertain |= bu
            return (cur_strict | b_strict | o_strict | f_strict | h_strict), (cur_uncertain | b_uncertain | o_uncertain | f_uncertain | h_uncertain)

        # Simple statement: scan whole stmt subtree (no nested scopes)
        _scan_expr_for_string_cmp_and_render(
            stmt,
            py_file=py_file,
            strict_aliases=cur_strict,
            uncertain_aliases=cur_uncertain,
            include_direct=False,
            uses_enum_in_scope=uses_enum_in_scope,
            alias_value_hints=alias_value_hints,
            out_comparisons=out_comparisons,
            out_renderings=out_renderings,
        )

        # Transfer function for aliases:
        if isinstance(stmt, ast.Assign):
            return update_aliases_from_assign(stmt.targets, stmt.value, cur_strict, cur_uncertain)
        if isinstance(stmt, ast.AnnAssign):
            return update_aliases_from_assign([stmt.target], stmt.value, cur_strict, cur_uncertain)
        if isinstance(stmt, ast.AugAssign):
            # sev += ...  => no longer a clean alias
            ns = set(cur_strict)
            nu = set(cur_uncertain)
            for nm in _extract_assigned_names(stmt.target):
                ns.discard(nm)
                nu.discard(nm)
            return ns, nu

        # Other statements don't affect alias set in our light model
        return cur_strict, cur_uncertain

    def process_block(stmts: List[ast.stmt], cur_strict: set[str], cur_uncertain: set[str]) -> Tuple[set[str], set[str]]:
        cur_strict = set(cur_strict)
        cur_uncertain = set(cur_uncertain)
        for st in stmts or []:
            cur_strict, cur_uncertain = process_stmt(st, cur_strict, cur_uncertain)
        return cur_strict, cur_uncertain

    # Function bodies are different fields depending on node type
    body = getattr(fn, "body", None)
    if isinstance(body, list):
        process_block(body, strict_aliases, uncertain_aliases)


def _scan_file(py_file: Path) -> Dict[str, Any]:
    text = py_file.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(text, filename=str(py_file))
    except SyntaxError as e:
        return {"parse_error": f"SyntaxError: {e}"}

    enums: List[EnumInfo] = []
    severity_fields: List[SeverityFieldInfo] = []
    suspicious_imports: List[ImportInfo] = []
    severity_string_comparisons: List[SeverityStringComparisonInfo] = []
    severity_string_renderings: List[SeverityStringRenderInfo] = []
    uses_enum_in_module_scope = _scope_uses_severity_enum(tree)

    # --- imports
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported = [a.name for a in node.names if a.name]
            if any(name.endswith("Severity") or name in {"AuditSeverity", "ErrorSeverity"} for name in imported):
                # allow canonical import
                if node.module.endswith("analysis.foundation.models"):
                    continue
                suspicious_imports.append(
                    ImportInfo(
                        file=str(py_file),
                        line=getattr(node, "lineno", 1),
                        module=node.module,
                        names=imported,
                    )
                )

    # --- direct scan by scopes (prevents "file-level Severity leaks" into unrelated functions)
    # Module top-level (won't descend into classes/functions)
    _scan_expr_for_string_cmp_and_render(
        tree,
        py_file=py_file,
        strict_aliases=set(),
        uncertain_aliases=set(),
        include_direct=True,
        uses_enum_in_scope=uses_enum_in_module_scope,
        alias_value_hints=set(),
        out_comparisons=severity_string_comparisons,
        out_renderings=severity_string_renderings,
    )

    # Class scopes (without descending into nested defs/classes)
    for top in tree.body:
        if isinstance(top, ast.ClassDef):
            uses_enum_in_class_scope = _scope_uses_severity_enum(top)
            _scan_expr_for_string_cmp_and_render(
                top,
                py_file=py_file,
                strict_aliases=set(),
                uncertain_aliases=set(),
                include_direct=True,
                uses_enum_in_scope=uses_enum_in_class_scope,
                alias_value_hints=set(),
                out_comparisons=severity_string_comparisons,
                out_renderings=severity_string_renderings,
            )

    # Function scopes (without descending into nested defs/classes)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            uses_enum_in_fn_scope = _scope_uses_severity_enum(node)
            _scan_expr_for_string_cmp_and_render(
                node,
                py_file=py_file,
                strict_aliases=set(),
                uncertain_aliases=set(),
                include_direct=True,
                uses_enum_in_scope=uses_enum_in_fn_scope,
                alias_value_hints=set(),
                out_comparisons=severity_string_comparisons,
                out_renderings=severity_string_renderings,
            )

    # --- comparison + rendering checks (function-local alias tracking)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            uses_enum_in_fn_scope = _scope_uses_severity_enum(node)
            _scan_function_for_severity_alias_issues(
                node,
                py_file=py_file,
                out_comparisons=severity_string_comparisons,
                out_renderings=severity_string_renderings,
                uses_enum_in_scope=uses_enum_in_fn_scope,
            )

    # --- classes
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        # Enums
        is_enum = any(_is_enum_base(b) for b in (node.bases or []))
        if is_enum:
            values: List[str] = []
            for stmt in node.body:
                if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                    v = _literal_str(stmt.value)
                    if v is not None:
                        values.append(v)
                    else:
                        # Could be auto() or something else
                        values.append(_safe_unparse(stmt.value))

            looks_like_severity = "severity" in node.name.lower() or node.name.lower().endswith("severity")
            # Check if enum values are exactly the canonical tokens (subset/equals)
            lowered = {v.lower() for v in values if isinstance(v, str)}
            matches_canonical = bool(lowered) and lowered.issubset(CANONICAL_SEVERITY_TOKENS)

            enums.append(
                EnumInfo(
                    file=str(py_file),
                    line=getattr(node, "lineno", 1),
                    name=node.name,
                    values=values,
                    looks_like_severity=looks_like_severity,
                    matches_canonical=matches_canonical,
                )
            )

        # severity fields in classes (dataclass or not)
        is_dc = _has_dataclass_decorator(node)
        for stmt in node.body:
            # severity: Type = default
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.target.id == "severity":
                ann = _safe_unparse(stmt.annotation) if stmt.annotation else ""
                default = _safe_unparse(stmt.value) if stmt.value else ""
                severity_fields.append(
                    SeverityFieldInfo(
                        file=str(py_file),
                        line=getattr(stmt, "lineno", getattr(node, "lineno", 1)),
                        class_name=node.name,
                        is_dataclass=is_dc,
                        annotation=ann,
                        default=default,
                    )
                )
            # severity = "high" (untyped)
            elif isinstance(stmt, ast.Assign):
                for tgt in stmt.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "severity":
                        severity_fields.append(
                            SeverityFieldInfo(
                                file=str(py_file),
                                line=getattr(stmt, "lineno", getattr(node, "lineno", 1)),
                                class_name=node.name,
                                is_dataclass=is_dc,
                                annotation="(untyped)",
                                default=_safe_unparse(stmt.value),
                            )
                        )

    # De-dup (line+expr+kind) to avoid noise when multiple passes detect same site
    def _dedup(items: List[Dict[str, Any]], key_fields: List[str]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        seen = set()
        for it in items:
            key = tuple(it.get(k) for k in key_fields)
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    comparisons_dicts = [c.__dict__ for c in severity_string_comparisons]
    renderings_dicts = [r.__dict__ for r in severity_string_renderings]
    comparisons_dicts = _dedup(comparisons_dicts, ["file", "line", "expr"])
    renderings_dicts = _dedup(renderings_dicts, ["file", "line", "kind", "expr"])

    # NOTE: filtering is done in run_audit() where we have the flag (noise-reduction mode).

    return {
        "enums": [e.__dict__ for e in enums],
        "severity_fields": [f.__dict__ for f in severity_fields],
        "suspicious_imports": [i.__dict__ for i in suspicious_imports],
        "severity_string_comparisons": comparisons_dicts,
        "severity_string_renderings": renderings_dicts,
        "_uses_severity_enum_in_module_scope": uses_enum_in_module_scope,
    }


def run_audit(repo_root: Path) -> Dict[str, Any]:
    repo_root = repo_root.resolve()
    files: List[Path] = []
    for p in repo_root.rglob("*.py"):
        if _is_skipped(p):
            continue
        files.append(p)

    enums_all: List[Dict[str, Any]] = []
    fields_all: List[Dict[str, Any]] = []
    imports_all: List[Dict[str, Any]] = []
    comparisons_all: List[Dict[str, Any]] = []
    renderings_all: List[Dict[str, Any]] = []
    comparisons_suppressed: List[Dict[str, Any]] = []
    renderings_suppressed: List[Dict[str, Any]] = []
    parse_errors: List[Dict[str, Any]] = []

    # Backward-compatible signature: noise reduction is controlled in main() and passed via closure.
    # We set a function attribute from main(); default False.
    noise_reduction = bool(getattr(run_audit, "_noise_reduction", False))

    for f in sorted(files, key=lambda x: x.as_posix()):
        data = _scan_file(f)
        if "parse_error" in data:
            parse_errors.append({"file": str(f), "error": data["parse_error"]})
            continue
        enums_all.extend(data.get("enums", []))
        fields_all.extend(data.get("severity_fields", []))
        imports_all.extend(data.get("suspicious_imports", []))

        comps = list(data.get("severity_string_comparisons", []))
        rends = list(data.get("severity_string_renderings", []))

        # Noise-reduction: suppress low-confidence accessor-only findings (likely JSON/dict strings)
        if noise_reduction:
            for c in comps:
                if (c.get("confidence") == "low"):
                    c = dict(c)
                    c["suppression_reason"] = "noise-reduction: low-confidence (accessor-only, no Severity usage evidence in file)"
                    comparisons_suppressed.append(c)
                else:
                    comparisons_all.append(c)
            for r in rends:
                if (r.get("confidence") == "low"):
                    r = dict(r)
                    r["suppression_reason"] = "noise-reduction: low-confidence (accessor-only, no Severity usage evidence in file)"
                    renderings_suppressed.append(r)
                else:
                    renderings_all.append(r)
        else:
            comparisons_all.extend(comps)
            renderings_all.extend(rends)

    # Derive some actionable notes
    duplicated_severity_enums = [
        e for e in enums_all
        if e.get("looks_like_severity") and not e.get("matches_canonical")
    ]

    string_severity_fields = [
        f for f in fields_all
        if "str" in (f.get("annotation") or "").lower()
    ]

    return {
        "repo_root": str(repo_root),
        "summary": {
            "python_files_scanned": len(files),
            "parse_errors": len(parse_errors),
            "enums_found": len(enums_all),
            "severity_fields_found": len(fields_all),
            "suspicious_imports_found": len(imports_all),
            "duplicated_severity_enums": len(duplicated_severity_enums),
            "string_severity_fields": len(string_severity_fields),
            "severity_string_comparisons_found": len(comparisons_all),
            "severity_string_renderings_found": len(renderings_all),
            "severity_string_comparisons_suppressed": len(comparisons_suppressed),
            "severity_string_renderings_suppressed": len(renderings_suppressed),
            "noise_reduction_enabled": bool(noise_reduction),
        },
        "parse_errors": parse_errors,
        "enums": enums_all,
        "severity_fields": fields_all,
        "suspicious_imports": imports_all,
        "severity_string_comparisons": comparisons_all,
        "severity_string_renderings": renderings_all,
        "severity_string_comparisons_suppressed": comparisons_suppressed,
        "severity_string_renderings_suppressed": renderings_suppressed,
        "recommendations": [
            "Prefer intellirefactor.analysis.foundation.models.Severity everywhere.",
            "Replace string-typed severity fields with Severity enum.",
            "Replace comparisons like `x.severity == 'high'` with `x.severity is Severity.high` (or `== Severity.high`).",
            "When rendering severity into text reports/logs, prefer `severity.value` (or a dedicated formatter) to avoid `Severity.high` output.",
            "Eliminate extra *Severity enums and keep only compatibility aliases temporarily.",
            "After codebase migration, delete aliases AuditSeverity/ErrorSeverity.",
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser(prog="type_consistency_audit")
    ap.add_argument("repo_root", help="Path to repo/project root")
    ap.add_argument("--output", "-o", help="Write JSON report to file")
    ap.add_argument("--format", choices=["json"], default="json")
    ap.add_argument(
        "--noise-reduction",
        action="store_true",
        help="Suppress low-confidence findings from accessor-only patterns (e.g. JSON/dict severity strings). "
             "Suppressed items are still included in *_suppressed lists.",
    )
    args = ap.parse_args()

    # pass flag into run_audit without breaking external API
    setattr(run_audit, "_noise_reduction", bool(args.noise_reduction))
    payload = run_audit(Path(args.repo_root))

    s = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    if args.output:
        Path(args.output).write_text(s, encoding="utf-8")
    else:
        print(s)


if __name__ == "__main__":
    main()





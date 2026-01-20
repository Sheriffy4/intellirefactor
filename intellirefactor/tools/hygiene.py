from __future__ import annotations

import argparse
import fnmatch
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from typing import Any


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
}


@dataclass
class Issue:
    code: str
    message: str
    file_path: str
    line: int
    column: int = 1
    snippet: str = ""
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "file_path": self.file_path,
            "line": self.line,
            "column": self.column,
            "snippet": self.snippet,
            "suggestion": self.suggestion,
        }


_RE_TYPE_ANY = re.compile(r"(?P<prefix>->\s*|:\s*|,\s*)\bany\b")
_RE_TYPE_ANY_GENERIC = re.compile(r"\b(any)\b")
_RE_READ_TEXT_UTF8 = re.compile(r"\bread_text\s*\(\s*[^)]*encoding\s*=\s*['\"]utf-8['\"][^)]*\)")
_RE_OPEN_UTF8 = re.compile(r"\bopen\s*\(\s*[^)]*encoding\s*=\s*['\"]utf-8['\"][^)]*\)")
_RE_HAS_ERRORS_ARG = re.compile(r"\berrors\s*=")
_RE_STR_RELATIVE_TO = re.compile(r"str\s*\(\s*[^)]*\.relative_to\s*\(")
_RE_MD5_USEDFORSECURITY = re.compile(r"hashlib\.md5\s*\([^)]*usedforsecurity\s*=")
_RE_LOGGER_FSTRING = re.compile(r"\blogger\.(debug|info|warning|error|exception)\(\s*f[\"']")
_RE_UNIFIED_PLACEHOLDER = re.compile(r"\bintellirefactor\.unified\.")


def _iter_py_files(root: Path, exclude_globs: List[str]) -> Iterable[Path]:
    root = root.resolve()
    exc = [e.replace("\\", "/") for e in (exclude_globs or [])]

    for p in root.rglob("*.py"):
        try:
            rel_parts = set(p.relative_to(root).parts)
        except Exception:
            rel_parts = set(p.parts)
        if rel_parts.intersection(DEFAULT_SKIP_DIRS):
            continue

        rel = None
        try:
            rel = p.relative_to(root).as_posix()
        except Exception:
            rel = p.as_posix()

        if exc and any(fnmatch.fnmatch(rel, pat) for pat in exc):
            continue

        yield p


def scan_file(project_root: Path, file_path: Path) -> List[Issue]:
    issues: List[Issue] = []
    try:
        text = file_path.read_text(encoding="utf-8-sig", errors="replace")
    except Exception as e:
        issues.append(
            Issue(
                code="READ_FAILED",
                message=f"Cannot read file: {e}",
                file_path=str(file_path),
                line=1,
            )
        )
        return issues

    try:
        rel = file_path.resolve().relative_to(project_root.resolve()).as_posix()
    except Exception:
        rel = file_path.as_posix()

    for i, line in enumerate(text.splitlines(), start=1):
        s = line.strip()

        # any -> Any (typing)
        if "any" in s:
            if _RE_TYPE_ANY.search(line) or ("Dict[" in line and "any" in line):
                issues.append(
                    Issue(
                        code="TYPE_ANY",
                        message="Found 'any' in type annotations (should be 'Any').",
                        file_path=rel,
                        line=i,
                        snippet=line[:240],
                        suggestion="Replace 'any' with 'Any' in annotations (preferably via safe autocorrect).",
                    )
                )

        # read_text encoding utf-8 without errors=...
        if "read_text" in s and "encoding" in s and "utf-8" in s:
            if _RE_READ_TEXT_UTF8.search(line) and not _RE_HAS_ERRORS_ARG.search(line):
                issues.append(
                    Issue(
                        code="READ_UTF8_NO_ERRORS",
                        message="read_text(..., encoding='utf-8') without errors=... (can crash on bad bytes).",
                        file_path=rel,
                        line=i,
                        snippet=line[:240],
                        suggestion="Consider encoding='utf-8-sig', errors='replace' (or decide project policy).",
                    )
                )

        # open encoding utf-8 without errors=...
        if "open" in s and "encoding" in s and "utf-8" in s:
            if _RE_OPEN_UTF8.search(line) and not _RE_HAS_ERRORS_ARG.search(line):
                issues.append(
                    Issue(
                        code="OPEN_UTF8_NO_ERRORS",
                        message="open(..., encoding='utf-8') without errors=... (can crash on bad bytes).",
                        file_path=rel,
                        line=i,
                        snippet=line[:240],
                        suggestion="Consider encoding='utf-8-sig', errors='replace' (or decide project policy).",
                    )
                )

        # str(relative_to(...)) path normalization
        if ".relative_to(" in line and "str(" in line:
            if _RE_STR_RELATIVE_TO.search(line):
                issues.append(
                    Issue(
                        code="PATH_STR_RELATIVE",
                        message="str(Path.relative_to(...)) is OS-dependent; prefer .as_posix() for stable reports/DB.",
                        file_path=rel,
                        line=i,
                        snippet=line[:240],
                        suggestion="Use path.relative_to(root).as_posix() (stable cross-platform).",
                    )
                )

        # md5 usedforsecurity
        if "hashlib.md5" in line and "usedforsecurity" in line:
            if _RE_MD5_USEDFORSECURITY.search(line):
                issues.append(
                    Issue(
                        code="HASHLIB_MD5_USEDFORSECURITY",
                        message="hashlib.md5(..., usedforsecurity=...) is not supported on all Python builds (may TypeError).",
                        file_path=rel,
                        line=i,
                        snippet=line[:240],
                        suggestion="Wrap in try/except TypeError fallback or drop usedforsecurity kwarg.",
                    )
                )

        # logger f-strings (style/perf)
        if "logger." in line and "f\"" in line:
            if _RE_LOGGER_FSTRING.search(line):
                issues.append(
                    Issue(
                        code="LOGGER_FSTRING",
                        message="logger.<level>(f'...') found; consider logger.<level>('...%s...', arg) style.",
                        file_path=rel,
                        line=i,
                        snippet=line[:240],
                        suggestion="Use lazy formatting: logger.info('x=%s', x).",
                    )
                )

        # unified placeholder
        if "intellirefactor.unified." in line:
            if _RE_UNIFIED_PLACEHOLDER.search(line):
                issues.append(
                    Issue(
                        code="UNIFIED_PLACEHOLDER",
                        message="Found reference to intellirefactor.unified.* (likely autogenerated placeholder / missing module).",
                        file_path=rel,
                        line=i,
                        snippet=line[:240],
                        suggestion="Remove placeholder or guard with try/except, or implement the referenced module.",
                    )
                )

    return issues


def scan_project(root: Path, *, exclude: List[str]) -> List[Issue]:
    root = root.resolve()
    all_issues: List[Issue] = []
    for f in _iter_py_files(root, exclude):
        all_issues.extend(scan_file(root, f))
    return all_issues


def _has_future_annotations(src: str) -> bool:
    # cheap and sufficient for safety gate
    return "from __future__ import annotations" in src


def fix_any_in_annotations(root: Path, *, exclude: List[str], write: bool) -> Dict[str, Any]:
    """
    Safe auto-fix: replaces Name('any') -> Name('Any') in annotation contexts.
    Guard: only if file contains 'from __future__ import annotations' to avoid runtime NameError.
    """
    try:
        pass
    except Exception:
        return {
            "success": False,
            "error": "libcst is not installed; cannot auto-fix. Install libcst or run scan-only.",
            "files_changed": 0,
        }

def fix_missing_any_import(root: Path, *, exclude: List[str], write: bool) -> Dict[str, Any]:
    """
    Auto-fix (safe): add typing.Any import when Any is used in annotations but not imported.

    Safety rules:
    - Only if Any is referenced inside Annotation nodes (type annotations).
    - Only if Any is not already imported from typing/typing_extensions (or via star import).
    - Prefer extending existing `from typing import ...`; otherwise insert a new import.
    """
    try:
        import libcst as cst
    except Exception:
        return {
            "success": False,
            "error": "libcst is not installed; cannot auto-fix. Install libcst or run scan-only.",
            "files_changed": 0,
        }

    def _module_name(mod_expr: Optional["cst.BaseExpression"]) -> Optional[str]:
        if mod_expr is None:
            return None
        if isinstance(mod_expr, cst.Name):
            return mod_expr.value
        if isinstance(mod_expr, cst.Attribute):
            # take leftmost name
            x = mod_expr
            while isinstance(x, cst.Attribute):
                x = x.value
            if isinstance(x, cst.Name):
                return x.value
        return None

    def _is_docstring_stmt(stmt: "cst.CSTNode") -> bool:
        if not isinstance(stmt, cst.SimpleStatementLine):
            return False
        if len(stmt.body) != 1:
            return False
        b0 = stmt.body[0]
        return isinstance(b0, cst.Expr) and isinstance(b0.value, cst.SimpleString)

    def _is_future_import(stmt: "cst.CSTNode") -> bool:
        if not isinstance(stmt, cst.SimpleStatementLine):
            return False
        if len(stmt.body) != 1:
            return False
        b0 = stmt.body[0]
        return isinstance(b0, cst.ImportFrom) and _module_name(b0.module) == "__future__"

    def _is_import_stmt(stmt: "cst.CSTNode") -> bool:
        if not isinstance(stmt, cst.SimpleStatementLine):
            return False
        if len(stmt.body) != 1:
            return False
        return isinstance(stmt.body[0], (cst.Import, cst.ImportFrom))

    class AnyUsageVisitor(cst.CSTVisitor):
        def __init__(self) -> None:
            self._in_annotation = 0
            self.uses_any_in_annotations = False

        def visit_Annotation(self, node: cst.Annotation) -> Optional[bool]:
            self._in_annotation += 1
            return True

        def leave_Annotation(self, original_node: cst.Annotation) -> None:
            self._in_annotation -= 1

        def visit_Name(self, node: cst.Name) -> Optional[bool]:
            if self._in_annotation and node.value == "Any":
                self.uses_any_in_annotations = True
            return True

    def _has_any_import(mod: "cst.Module") -> bool:
        for stmt in mod.body:
            if not isinstance(stmt, cst.SimpleStatementLine) or len(stmt.body) != 1:
                continue
            b0 = stmt.body[0]
            if not isinstance(b0, cst.ImportFrom):
                continue
            m = _module_name(b0.module)
            if m not in {"typing", "typing_extensions"}:
                continue
            if isinstance(b0.names, cst.ImportStar):
                return True
            if isinstance(b0.names, list):
                for a in b0.names:
                    if isinstance(a, cst.ImportAlias) and isinstance(a.name, cst.Name):
                        if a.name.value == "Any":
                            return True
        return False

    def _add_any_import(mod: "cst.Module") -> "cst.Module":
        body = list(mod.body)

        # 1) extend existing "from typing import ..."
        for idx, stmt in enumerate(body):
            if not isinstance(stmt, cst.SimpleStatementLine) or len(stmt.body) != 1:
                continue
            b0 = stmt.body[0]
            if not isinstance(b0, cst.ImportFrom):
                continue
            if _module_name(b0.module) != "typing":
                continue
            if isinstance(b0.names, cst.ImportStar):
                return mod
            if isinstance(b0.names, list):
                if any(
                    isinstance(a, cst.ImportAlias)
                    and isinstance(a.name, cst.Name)
                    and a.name.value == "Any"
                    for a in b0.names
                ):
                    return mod
                new_names = list(b0.names) + [cst.ImportAlias(name=cst.Name("Any"))]
                body[idx] = stmt.with_changes(body=[b0.with_changes(names=new_names)])
                return mod.with_changes(body=body)

        # 2) insert new "from typing import Any" after docstring/future/import block
        insert_at = 0
        if body and _is_docstring_stmt(body[0]):
            insert_at = 1
        while insert_at < len(body) and _is_future_import(body[insert_at]):
            insert_at += 1
        while insert_at < len(body) and _is_import_stmt(body[insert_at]):
            insert_at += 1

        new_stmt = cst.SimpleStatementLine(
            body=[
                cst.ImportFrom(
                    module=cst.Name("typing"),
                    names=[cst.ImportAlias(name=cst.Name("Any"))],
                )
            ]
        )
        body.insert(insert_at, new_stmt)
        return mod.with_changes(body=body)

    root = root.resolve()
    files_changed = 0
    for f in _iter_py_files(root, exclude):
        src = f.read_text(encoding="utf-8-sig", errors="replace")
        if "Any" not in src:
            continue
        try:
            mod = cst.parse_module(src)
        except Exception:
            continue

        v = AnyUsageVisitor()
        mod.visit(v)
        if not v.uses_any_in_annotations:
            continue
        if _has_any_import(mod):
            continue

        new_mod = _add_any_import(mod)
        new_src = new_mod.code
        if new_src != src:
            files_changed += 1
            if write:
                f.write_text(new_src, encoding="utf-8-sig")

    return {"success": True, "files_changed": files_changed, "write": write}


def fix_logger_fstring(root: Path, *, exclude: List[str], write: bool) -> Dict[str, Any]:
    """
    Auto-fix (safe subset):
      logger.info(f"...{x}...") -> logger.info("...%s...", x)

    Safety constraints:
      - only logger.<level>(<fstring>) with NO extra positional args
      - f-string contains only literal text + {expr} without conversion/format_spec
      - keyword args are preserved (fmt args are inserted before keyword args)
    """
    try:
        import libcst as cst
    except Exception:
        return {
            "success": False,
            "error": "libcst is not installed; cannot auto-fix. Install libcst or run scan-only.",
            "files_changed": 0,
        }

    LEVELS = {"debug", "info", "warning", "error", "exception", "critical"}

    def _is_logger_call_func(expr: "cst.BaseExpression") -> bool:
        # logger.info / self.logger.info
        if not isinstance(expr, cst.Attribute):
            return False
        if not (isinstance(expr.attr, cst.Name) and expr.attr.value in LEVELS):
            return False
        v = expr.value
        if isinstance(v, cst.Name) and v.value == "logger":
            return True
        if isinstance(v, cst.Attribute) and isinstance(v.attr, cst.Name) and v.attr.value == "logger":
            if isinstance(v.value, cst.Name) and v.value.value == "self":
                return True
        return False

    def _escape_double_quoted(s: str) -> str:
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'

    class Fixer(cst.CSTTransformer):
        def __init__(self) -> None:
            self.calls_changed = 0

        def leave_Call(self, original_node: "cst.Call", updated_node: "cst.Call") -> "cst.Call":
            if not _is_logger_call_func(updated_node.func):
                return updated_node

            # split args: positional vs keyword
            pos = [a for a in updated_node.args if a.keyword is None and a.star != "**" and a.star != "*"]
            kw = [a for a in updated_node.args if a.keyword is not None or a.star in ("*", "**")]

            # only one positional arg (the message)
            if len(pos) != 1:
                return updated_node

            msg = pos[0].value
            if not isinstance(msg, cst.FormattedString):
                return updated_node

            fmt_parts: List[str] = []
            fmt_args: List[cst.Arg] = []

            for part in msg.parts:
                if isinstance(part, cst.FormattedStringText):
                    fmt_parts.append(part.value)
                    continue
                if isinstance(part, cst.FormattedStringExpression):
                    if part.conversion is not None:
                        return updated_node
                    if part.format_spec is not None:
                        return updated_node
                    fmt_parts.append("%s")
                    fmt_args.append(cst.Arg(value=part.expression))
                    continue
                # any other part => skip
                return updated_node

            new_msg = cst.SimpleString(_escape_double_quoted("".join(fmt_parts)))

            new_args: List[cst.Arg] = []
            # replace the only positional message arg
            new_args.append(pos[0].with_changes(value=new_msg))
            # insert formatting args BEFORE keyword args to keep valid call syntax
            new_args.extend(fmt_args)
            new_args.extend(kw)

            self.calls_changed += 1
            return updated_node.with_changes(args=new_args)

    root = root.resolve()
    files_changed = 0
    calls_changed_total = 0
    for f in _iter_py_files(root, exclude):
        src = f.read_text(encoding="utf-8-sig", errors="replace")
        if "logger." not in src or ("f\"" not in src and "f'" not in src):
            continue
        try:
            import libcst as cst
            mod = cst.parse_module(src)
        except Exception:
            continue
        fixer = Fixer()
        new_mod = mod.visit(fixer)
        if new_mod.code != src:
            files_changed += 1
            calls_changed_total += fixer.calls_changed
            if write:
                f.write_text(new_mod.code, encoding="utf-8-sig")

    return {
        "success": True,
        "files_changed": files_changed,
        "calls_changed": calls_changed_total,
        "write": write,
        "note": "Only safe subset transformed (no format specs/conversions, no extra positional args).",
    }


def fix_encoding_errors(root: Path, *, exclude: List[str], write: bool) -> Dict[str, Any]:
    """
    Auto-fix (policy, safe subset):
      - add errors="replace" to Path.read_text(..., encoding="utf-8|utf-8-sig") if missing
      - add errors="replace" to open(..., encoding="utf-8|utf-8-sig") if missing and not binary mode
    """
    try:
        import libcst as cst
    except Exception:
        return {
            "success": False,
            "error": "libcst is not installed; cannot auto-fix. Install libcst or run scan-only.",
            "files_changed": 0,
        }

    def _has_kw(call: "cst.Call", kw: str) -> bool:
        return any(a.keyword and isinstance(a.keyword, cst.Name) and a.keyword.value == kw for a in call.args)

    def _get_kw(call: "cst.Call", kw: str) -> Optional["cst.Arg"]:
        for a in call.args:
            if a.keyword and isinstance(a.keyword, cst.Name) and a.keyword.value == kw:
                return a
        return None

    def _is_utf8_encoding(call: "cst.Call") -> bool:
        a = _get_kw(call, "encoding")
        if not a or not isinstance(a.value, cst.SimpleString):
            return False
        v = a.value.value.strip()
        return v in {'"utf-8"', "'utf-8'", '"utf-8-sig"', "'utf-8-sig'"}

    def _is_binary_open(call: "cst.Call") -> bool:
        # open(file, mode="rb") or open(file, "rb") -> contains 'b'
        m = _get_kw(call, "mode")
        if m and isinstance(m.value, cst.SimpleString):
            return "b" in m.value.value
        pos = [a for a in call.args if a.keyword is None and a.star is None]
        if len(pos) >= 2 and isinstance(pos[1].value, cst.SimpleString):
            return "b" in pos[1].value.value
        return False

    class Fixer(cst.CSTTransformer):
        def __init__(self) -> None:
            self.calls_changed = 0

        def leave_Call(self, original_node: "cst.Call", updated_node: "cst.Call") -> "cst.Call":
            # Path.read_text(...)
            if isinstance(updated_node.func, cst.Attribute) and isinstance(updated_node.func.attr, cst.Name):
                if updated_node.func.attr.value == "read_text":
                    if _has_kw(updated_node, "errors"):
                        return updated_node
                    if not _is_utf8_encoding(updated_node):
                        return updated_node
                    self.calls_changed += 1
                    return updated_node.with_changes(
                        args=list(updated_node.args)
                        + [cst.Arg(keyword=cst.Name("errors"), value=cst.SimpleString('"replace"'))]
                    )

            # open(...)
            if isinstance(updated_node.func, cst.Name) and updated_node.func.value == "open":
                if _has_kw(updated_node, "errors"):
                    return updated_node
                if not _is_utf8_encoding(updated_node):
                    return updated_node
                if _is_binary_open(updated_node):
                    return updated_node
                self.calls_changed += 1
                return updated_node.with_changes(
                    args=list(updated_node.args)
                    + [cst.Arg(keyword=cst.Name("errors"), value=cst.SimpleString('"replace"'))]
                )

            return updated_node

    root = root.resolve()
    files_changed = 0
    calls_changed_total = 0
    for f in _iter_py_files(root, exclude):
        src = f.read_text(encoding="utf-8-sig", errors="replace")
        if "read_text" not in src and "open(" not in src:
            continue
        try:
            import libcst as cst
            mod = cst.parse_module(src)
        except Exception:
            continue
        fixer = Fixer()
        new_mod = mod.visit(fixer)
        if new_mod.code != src:
            files_changed += 1
            calls_changed_total += fixer.calls_changed
            if write:
                f.write_text(new_mod.code, encoding="utf-8-sig")

    return {"success": True, "files_changed": files_changed, "calls_changed": calls_changed_total, "write": write}

    class AnyFixer(cst.CSTTransformer):
        def __init__(self) -> None:
            self._in_annotation = 0

        def visit_Annotation(self, node: cst.Annotation) -> Optional[bool]:
            self._in_annotation += 1
            return True

        def leave_Annotation(self, original_node: cst.Annotation, updated_node: cst.Annotation) -> cst.Annotation:
            self._in_annotation -= 1
            return updated_node

        def leave_Name(self, original_node: cst.Name, updated_node: cst.Name) -> cst.BaseExpression:
            if self._in_annotation and original_node.value == "any":
                return updated_node.with_changes(value="Any")
            return updated_node

    root = root.resolve()
    changed = 0
    skipped = 0

    for f in _iter_py_files(root, exclude):
        src = f.read_text(encoding="utf-8-sig", errors="replace")
        if "any" not in src:
            continue
        if not _has_future_annotations(src):
            skipped += 1
            continue

        try:
            m = cst.parse_module(src)
            new_m = m.visit(AnyFixer())
            new_src = new_m.code
        except Exception:
            continue

        if new_src != src:
            changed += 1
            if write:
                f.write_text(new_src, encoding="utf-8-sig")

    return {
        "success": True,
        "files_changed": changed,
        "files_skipped_no_future_annotations": skipped,
        "write": write,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="intellirefactor-hygiene")
    sub = p.add_subparsers(dest="cmd", required=True)

    scan_p = sub.add_parser("scan", help="Scan project for hygiene issues.")
    scan_p.add_argument("path", nargs="?", default=".", help="Project root path")
    scan_p.add_argument("--format", choices=["text", "json"], default="text")
    scan_p.add_argument("--exclude", action="append", default=[], help="Exclude glob (relative posix). Can repeat.")

    fix_p = sub.add_parser("fix-any", help="Auto-fix: replace 'any' -> 'Any' in annotations (safe mode).")
    fix_p.add_argument("path", nargs="?", default=".", help="Project root path")
    fix_p.add_argument("--exclude", action="append", default=[], help="Exclude glob (relative posix). Can repeat.")
    fix_p.add_argument("--write", action="store_true", help="Write changes to disk (otherwise dry-run).")
    fix_p.add_argument("--format", choices=["text", "json"], default="text")
    
    fix_anyimp = sub.add_parser(
        "fix-missing-any-import",
        help="Auto-fix: add `from typing import Any` (or extend existing import) when Any is used but not imported.",
    )
    fix_anyimp.add_argument("path", nargs="?", default=".", help="Project root path")
    fix_anyimp.add_argument("--exclude", action="append", default=[], help="Exclude glob (relative posix). Can repeat.")
    fix_anyimp.add_argument("--write", action="store_true", help="Write changes to disk (otherwise dry-run).")
    fix_anyimp.add_argument("--format", choices=["text", "json"], default="text")

    fix_log = sub.add_parser(
        "fix-logger-fstring",
        help="Auto-fix (safe subset): logger.info(f'..{x}..') -> logger.info('..%s..', x).",
    )
    fix_log.add_argument("path", nargs="?", default=".", help="Project root path")
    fix_log.add_argument("--exclude", action="append", default=[], help="Exclude glob (relative posix). Can repeat.")
    fix_log.add_argument("--write", action="store_true", help="Write changes to disk (otherwise dry-run).")
    fix_log.add_argument("--format", choices=["text", "json"], default="text")

    fix_enc = sub.add_parser(
        "fix-encoding-errors",
        help="Auto-fix (safe subset): add errors='replace' to utf-8/utf-8-sig open()/read_text() when missing.",
    )
    fix_enc.add_argument("path", nargs="?", default=".", help="Project root path")
    fix_enc.add_argument("--exclude", action="append", default=[], help="Exclude glob (relative posix). Can repeat.")
    fix_enc.add_argument("--write", action="store_true", help="Write changes to disk (otherwise dry-run).")
    fix_enc.add_argument("--format", choices=["text", "json"], default="text")

    args = p.parse_args(argv)
    root = Path(args.path).resolve()

    if args.cmd == "scan":
        issues = scan_project(root, exclude=args.exclude)
        if args.format == "json":
            print(json.dumps([i.to_dict() for i in issues], ensure_ascii=True, indent=2))
        else:
            by_file: Dict[str, List[Issue]] = {}
            for it in issues:
                by_file.setdefault(it.file_path, []).append(it)

            for fp in sorted(by_file.keys()):
                print(fp)
                for it in by_file[fp]:
                    print(f"  L{it.line:>4}  {it.code:<24}  {it.message}")
                    if it.snippet:
                        print(f"        {it.snippet.strip()}")
                    if it.suggestion:
                        print(f"        Suggestion: {it.suggestion}")
                print()

            print(f"Total issues: {len(issues)}")
        return 1 if issues else 0

    if args.cmd == "fix-any":
        res = fix_any_in_annotations(root, exclude=args.exclude, write=bool(args.write))
        if args.format == "json":
            print(json.dumps(res, ensure_ascii=True, indent=2))
        else:
            if not res.get("success"):
                print("ERROR:", res.get("error"))
                return 2
            print("files_changed:", res.get("files_changed", 0))
            print("files_skipped_no_future_annotations:", res.get("files_skipped_no_future_annotations", 0))
            print("write:", res.get("write"))
        return 0 if res.get("success") else 2
        
    if args.cmd == "fix-missing-any-import":
        res = fix_missing_any_import(root, exclude=args.exclude, write=bool(args.write))
        if args.format == "json":
            print(json.dumps(res, ensure_ascii=True, indent=2))
        else:
            if not res.get("success"):
                print("ERROR:", res.get("error"))
                return 2
            print("files_changed:", res.get("files_changed", 0))
            print("write:", res.get("write"))
        return 0 if res.get("success") else 2

    if args.cmd == "fix-logger-fstring":
        res = fix_logger_fstring(root, exclude=args.exclude, write=bool(args.write))
        if args.format == "json":
            print(json.dumps(res, ensure_ascii=True, indent=2))
        else:
            if not res.get("success"):
                print("ERROR:", res.get("error"))
                return 2
            print("files_changed:", res.get("files_changed", 0))
            print("calls_changed:", res.get("calls_changed", 0))
            print("write:", res.get("write"))
            if res.get("note"):
                print("note:", res["note"])
        return 0 if res.get("success") else 2

    if args.cmd == "fix-encoding-errors":
        res = fix_encoding_errors(root, exclude=args.exclude, write=bool(args.write))
        if args.format == "json":
            print(json.dumps(res, ensure_ascii=True, indent=2))
        else:
            if not res.get("success"):
                print("ERROR:", res.get("error"))
                return 2
            print("files_changed:", res.get("files_changed", 0))
            print("calls_changed:", res.get("calls_changed", 0))
            print("write:", res.get("write"))
        return 0 if res.get("success") else 2

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import ast
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    ".intellirefactor", "intellirefactor_out",
}

SQL_HINT_RE = re.compile(
    r"\b(SELECT|INSERT|UPDATE|DELETE|CREATE|DROP|ALTER|WITH|PRAGMA)\b",
    re.IGNORECASE,
)

DDL_HINT_RE = re.compile(r"\b(CREATE|DROP|ALTER)\b", re.IGNORECASE)

@dataclass
class Hit:
    file: str
    line: int
    col: int
    rule: str
    match: str
    suggestion: str
    snippet: str
    level: str  # "error" | "advisory"

# “Жёсткие” несовпадения схемы (то, что реально ломает runtime)
BAD_SQL_RULES: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bfiles\s*\.\s*path\b", re.IGNORECASE), "files.path", "Use files.file_path"),
    (re.compile(r"\bf\s*\.\s*path\b", re.IGNORECASE), "f.path", "Use f.file_path"),

    (re.compile(r"\bfiles\s*\.\s*loc\b", re.IGNORECASE), "files.loc", "Use files.lines_of_code"),
    (re.compile(r"\bf\s*\.\s*loc\b", re.IGNORECASE), "f.loc", "Use f.lines_of_code"),
    (re.compile(r"\bblocks\s*\.\s*loc\b", re.IGNORECASE), "blocks.loc", "Use blocks.lines_of_code"),
    (re.compile(r"\bb\s*\.\s*loc\b", re.IGNORECASE), "b.loc", "Use b.lines_of_code"),

    (re.compile(r"\bdependencies\s*\.\s*kind\b", re.IGNORECASE), "dependencies.kind", "Use dependencies.dependency_kind"),
    (re.compile(r"\bd\s*\.\s*kind\b", re.IGNORECASE), "d.kind", "Use d.dependency_kind"),
    (re.compile(r"\bdependencies\s*\.\s*count\b", re.IGNORECASE), "dependencies.count", "Use dependencies.usage_count"),
    (re.compile(r"\bd\s*\.\s*count\b", re.IGNORECASE), "d.count", "Use COALESCE(d.usage_count, 1)"),

    (re.compile(r"\bblocks\s*\.\s*kind\b", re.IGNORECASE), "blocks.kind", "Use blocks.block_type"),

    (re.compile(r"\bblock_fingerprint\b", re.IGNORECASE), "block_fingerprint", "Use normalized_fingerprint (or token_fingerprint/ast_fingerprint)"),
    (re.compile(r"\bis_private\b", re.IGNORECASE), "is_private", "Schema v3 uses is_public; replace with (is_public = 0)"),
]

# “Советы” (не schema drift, но потенциально слабое место)
ADVISORY_SQL_RULES: List[Tuple[re.Pattern, str, str]] = [
    (re.compile(r"\bcyclomatic_complexity\b", re.IGNORECASE),
     "cyclomatic_complexity",
     "OK in schema; in queries prefer complexity_score or COALESCE(cyclomatic_complexity, complexity_score) if you actually populate it"),
]

def iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        try:
            rel_parts = set(p.relative_to(root).parts)
        except Exception:
            rel_parts = set(p.parts)
        if rel_parts.intersection(SKIP_DIRS):
            continue
        yield p

def looks_like_sql(text: str) -> bool:
    if not text:
        return False
    if not SQL_HINT_RE.search(text):
        return False
    return True

def stringify_node(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value

    if isinstance(node, ast.JoinedStr):
        parts: List[str] = []
        for v in node.values:
            if isinstance(v, ast.Constant) and isinstance(v.value, str):
                parts.append(v.value)
            else:
                parts.append("{...}")
        return "".join(parts)

    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = stringify_node(node.left)
        right = stringify_node(node.right)
        if left is not None and right is not None:
            return left + right

    return None

def _shorten_sql(txt: str, max_len: int = 200) -> str:
    snippet = " ".join(txt.strip().split())
    return snippet if len(snippet) <= max_len else snippet[:max_len] + "..."

def scan_file(fp: Path, *, include_ddl: bool, include_advisory: bool) -> List[Hit]:
    hits: List[Hit] = []
    try:
        src = fp.read_text(encoding="utf-8-sig", errors="replace")
        tree = ast.parse(src)
    except Exception:
        return hits

    for node in ast.walk(tree):
        txt = stringify_node(node)
        if not isinstance(txt, str):
            continue
        if not looks_like_sql(txt):
            continue

        is_ddl = bool(DDL_HINT_RE.search(txt))
        if is_ddl and not include_ddl:
            continue

        # strict mismatches
        for rx, rule_name, suggestion in BAD_SQL_RULES:
            m = rx.search(txt)
            if not m:
                continue
            hits.append(
                Hit(
                    file=fp.as_posix(),
                    line=getattr(node, "lineno", 1) or 1,
                    col=getattr(node, "col_offset", 0) or 0,
                    rule=rule_name,
                    match=m.group(0),
                    suggestion=suggestion,
                    snippet=_shorten_sql(txt),
                    level="error",
                )
            )

        # advisory
        if include_advisory:
            for rx, rule_name, suggestion in ADVISORY_SQL_RULES:
                m = rx.search(txt)
                if not m:
                    continue
                hits.append(
                    Hit(
                        file=fp.as_posix(),
                        line=getattr(node, "lineno", 1) or 1,
                        col=getattr(node, "col_offset", 0) or 0,
                        rule=rule_name,
                        match=m.group(0),
                        suggestion=suggestion,
                        snippet=_shorten_sql(txt),
                        level="advisory",
                    )
                )

    return hits

def main(argv: List[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else ".").resolve()

    out_json = "--json" in argv
    include_advisory = "--advisory" in argv
    include_ddl = "--include-ddl" in argv

    all_hits: List[Hit] = []
    for fp in iter_py_files(root):
        all_hits.extend(scan_file(fp, include_ddl=include_ddl, include_advisory=include_advisory))

    all_hits.sort(key=lambda h: (h.level, h.file, h.line, h.col, h.rule))

    if out_json:
        print(json.dumps([h.__dict__ for h in all_hits], indent=2, ensure_ascii=False))
    else:
        for h in all_hits:
            print(f"{h.file}:{h.line}:{h.col} [{h.level}] [{h.rule}] match={h.match!r}")
            print(f"  suggestion: {h.suggestion}")
            print(f"  sql: {h.snippet}")
            print()

        by_rule: Dict[str, int] = {}
        for h in all_hits:
            by_rule[f"{h.level}:{h.rule}"] = by_rule.get(f"{h.level}:{h.rule}", 0) + 1
        if by_rule:
            print("Summary:")
            for k in sorted(by_rule.keys()):
                print(f"  {k}: {by_rule[k]}")
            print()

    # Возвращаем 1 только если есть именно error (mismatch)
    has_errors = any(h.level == "error" for h in all_hits)
    print(f"TOTAL sql schema mismatch hits: {sum(1 for h in all_hits if h.level=='error')}")
    if include_advisory:
        print(f"TOTAL advisory hits: {sum(1 for h in all_hits if h.level=='advisory')}")
    return 1 if has_errors else 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
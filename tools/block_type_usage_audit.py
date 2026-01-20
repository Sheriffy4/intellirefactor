from __future__ import annotations

import argparse
import ast
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", ".mypy_cache", ".pytest_cache",
    ".tox", "venv", ".venv", "env", ".env", "build", "dist",
    ".intellirefactor", "intellirefactor_out",
}

# Strings we want to catch (both current DB/index-builder style and "canonical short" style)
BLOCK_TYPE_STRINGS = {
    # index-builder style
    "if_block", "for_loop", "while_loop", "try_block", "with_block",
    # canonical short style (foundation BlockType values)
    "if", "for", "while", "try", "with",
    # other common block types
    "statement_group", "function_body", "method_body", "class_body",
}

KEY_STRINGS = {"block_type", "block_kind", "kind"}


def iter_py_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def safe_read_text(p: Path) -> str:
    data = p.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


@dataclass
class Hit:
    kind: str                  # e.g. "call", "enum", "string", "dict_key"
    file_path: str
    line: int
    col: int
    symbol: str                # e.g. "parse_block_type", "BlockType.IF_BLOCK", "if_block"
    context: str               # one-line context/snippet


class UsageVisitor(ast.NodeVisitor):
    def __init__(self, file_path: Path, source_lines: List[str]):
        self.file_path = file_path
        self.lines = source_lines
        self.hits: List[Hit] = []

    def _ctx(self, node: ast.AST) -> str:
        ln = getattr(node, "lineno", None)
        if not isinstance(ln, int) or ln <= 0 or ln > len(self.lines):
            return ""
        return self.lines[ln - 1].rstrip()

    def _add(self, node: ast.AST, kind: str, symbol: str):
        ln = int(getattr(node, "lineno", 0) or 0)
        col = int(getattr(node, "col_offset", 0) or 0)
        self.hits.append(
            Hit(
                kind=kind,
                file_path=self.file_path.as_posix(),
                line=ln,
                col=col,
                symbol=symbol,
                context=self._ctx(node),
            )
        )

    # ---- 1) calls to parse_block_type(...)
    def visit_Call(self, node: ast.Call):
        func = node.func
        name = None
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        if name == "parse_block_type":
            self._add(node, "call", "parse_block_type(...)")
        self.generic_visit(node)

    # ---- 2) enum usage BlockType.X
    def visit_Attribute(self, node: ast.Attribute):
        if isinstance(node.value, ast.Name) and node.value.id == "BlockType":
            self._add(node, "enum", f"BlockType.{node.attr}")
        self.generic_visit(node)

    # ---- 3) string usage: "if_block" / "for" etc
    def visit_Constant(self, node: ast.Constant):
        if isinstance(node.value, str):
            s = node.value.strip()
            if s in BLOCK_TYPE_STRINGS:
                self._add(node, "string", repr(s))
            if s in KEY_STRINGS:
                self._add(node, "dict_key", repr(s))
        self.generic_visit(node)


def scan(root: Path) -> Dict[str, Any]:
    hits: List[Hit] = []
    files_scanned = 0
    parse_errors: List[str] = []

    for p in iter_py_files(root):
        files_scanned += 1
        try:
            text = safe_read_text(p)
            lines = text.splitlines()
            tree = ast.parse(text, filename=str(p))
            v = UsageVisitor(p, lines)
            v.visit(tree)
            hits.extend(v.hits)
        except SyntaxError as e:
            parse_errors.append(f"{p}: SyntaxError: {e}")
        except Exception as e:
            parse_errors.append(f"{p}: {type(e).__name__}: {e}")

    # aggregate
    by_kind: Dict[str, int] = {}
    by_symbol: Dict[str, int] = {}
    for h in hits:
        by_kind[h.kind] = by_kind.get(h.kind, 0) + 1
        by_symbol[h.symbol] = by_symbol.get(h.symbol, 0) + 1

    return {
        "root": str(root),
        "files_scanned": files_scanned,
        "hits_total": len(hits),
        "hits_by_kind": dict(sorted(by_kind.items(), key=lambda kv: kv[0])),
        "hits_by_symbol_top": sorted(
            [{"symbol": k, "count": v} for k, v in by_symbol.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:50],
        "parse_errors_total": len(parse_errors),
        "parse_errors_preview": parse_errors[:50],
        "hits": [asdict(h) for h in hits],
    }


def to_markdown(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# BlockType / parse_block_type usage audit")
    lines.append("")
    lines.append(f"- **Root:** `{payload.get('root')}`")
    lines.append(f"- **Files scanned:** `{payload.get('files_scanned')}`")
    lines.append(f"- **Hits total:** `{payload.get('hits_total')}`")
    lines.append(f"- **Parse errors:** `{payload.get('parse_errors_total')}`")
    lines.append("")

    lines.append("## Hits by kind")
    lines.append("")
    lines.append("| kind | count |")
    lines.append("|---|---:|")
    for k, v in (payload.get("hits_by_kind") or {}).items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Top symbols")
    lines.append("")
    lines.append("| symbol | count |")
    lines.append("|---|---:|")
    for row in payload.get("hits_by_symbol_top") or []:
        lines.append(f"| `{row['symbol']}` | {row['count']} |")
    lines.append("")

    lines.append("## Matches")
    lines.append("")
    lines.append("| file | line | kind | symbol | context |")
    lines.append("|---|---:|---|---|---|")
    for h in payload.get("hits") or []:
        fp = h.get("file_path", "")
        ln = h.get("line", 0)
        kind = h.get("kind", "")
        sym = h.get("symbol", "")
        ctx = (h.get("context", "") or "").replace("|", "\\|")
        lines.append(f"| `{fp}` | {ln} | {kind} | `{sym}` | `{ctx}` |")
    lines.append("")

    if payload.get("parse_errors_preview"):
        lines.append("## Parse errors (preview)")
        lines.append("")
        for e in payload["parse_errors_preview"]:
            lines.append(f"- `{e}`")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit BlockType/parse_block_type usages across project.")
    ap.add_argument("root", nargs="?", default=".", help="Project root to scan")
    ap.add_argument("--out-json", default=None, help="Write JSON report to file")
    ap.add_argument("--out-md", default=None, help="Write Markdown report to file")
    ap.add_argument("--stdout", action="store_true", help="Print Markdown to stdout")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    payload = scan(root)

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.out_md:
        Path(args.out_md).write_text(to_markdown(payload), encoding="utf-8")
    if args.stdout or (not args.out_json and not args.out_md):
        print(to_markdown(payload))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

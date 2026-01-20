from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Set
import ast

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    ".intellirefactor", "intellirefactor_out",
}

SQL_PATTERNS = [
    re.compile(r"\bdependencies\s*\.\s*kind\b"),
    re.compile(r"\bblocks\s*\.\s*kind\b"),
    re.compile(r"\bd\s*\.\s*kind\b"),
    re.compile(r"\bd\s*\.\s*count\b"),
    re.compile(r"\busage_count\b.*\bcount\b|\bcount\b.*\busage_count\b"),
]

@dataclass
class Hit:
    file: str
    line: int
    col: int
    kind: str
    detail: str

def iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        try:
            parts = set(p.relative_to(root).parts)
        except Exception:
            parts = set(p.parts)
        if parts.intersection(SKIP_DIRS):
            continue
        yield p

class DriftVisitor(ast.NodeVisitor):
    """
    Detect only index-related 'kind' drift:
    - dicts that look like dependency facts and contain key 'kind'
    - dicts that look like block facts and contain key 'kind'
    - calls .get("kind") on variables named dep/block (heuristic)
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.hits: List[Hit] = []

    def _add(self, node: ast.AST, kind: str, detail: str) -> None:
        self.hits.append(
            Hit(
                self.file_path,
                getattr(node, "lineno", 1) or 1,
                getattr(node, "col_offset", 0) or 0,
                kind,
                detail,
            )
        )

    def visit_Dict(self, node: ast.Dict) -> None:
        keys: Set[str] = set()
        for k in node.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.add(k.value)

        if "kind" in keys:
            # dependency-like dict?
            if {"source_symbol_uid", "target_external"}.issubset(keys):
                self._add(node, "dep_dict_kind", 'dependency dict contains key "kind"')
            # block-like dict?
            if {"block_uid", "symbol_uid"}.issubset(keys):
                self._add(node, "block_dict_kind", 'block dict contains key "kind"')

        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # dep.get("kind") / block.get("kind") (heuristic by var name)
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "kind":
                recv = node.func.value
                if isinstance(recv, ast.Name) and recv.id.lower() in {"dep", "dependency", "d", "block", "b"}:
                    self._add(node, "get_kind", f'{recv.id}.get("kind")')
        self.generic_visit(node)

def scan_file_text_for_sql(fp: Path) -> List[Hit]:
    out: List[Hit] = []
    try:
        lines = fp.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except Exception:
        return out

    for i, line in enumerate(lines, start=1):
        for pat in SQL_PATTERNS:
            if pat.search(line):
                out.append(Hit(fp.as_posix(), i, 0, "sql_drift", line.strip()))
                break
    return out

def main(argv: List[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else ".").resolve()
    hits: List[Hit] = []

    for fp in iter_py_files(root):
        hits.extend(scan_file_text_for_sql(fp))

        try:
            text = fp.read_text(encoding="utf-8-sig", errors="replace")
            tree = ast.parse(text)
        except Exception:
            continue
        v = DriftVisitor(fp.as_posix())
        v.visit(tree)
        hits.extend(v.hits)

    hits_sorted = sorted(hits, key=lambda h: (h.file, h.line, h.col, h.kind))
    for h in hits_sorted:
        print(f"{h.file}:{h.line}:{h.col} [{h.kind}] {h.detail}")

    print(f"\nTOTAL drift hits: {len(hits_sorted)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
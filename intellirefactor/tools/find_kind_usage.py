from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

SKIP_DIRS = {
    ".git", ".hg", ".svn", ".venv", "venv",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    ".intellirefactor", "intellirefactor_out",
}

@dataclass
class Hit:
    file: str
    line: int
    col: int
    kind: str
    detail: str

class KindVisitor(ast.NodeVisitor):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.hits: List[Hit] = []

    def _add(self, node: ast.AST, kind: str, detail: str) -> None:
        line = getattr(node, "lineno", 1) or 1
        col = getattr(node, "col_offset", 0) or 0
        self.hits.append(Hit(self.file_path, line, col, kind, detail))

    def visit_Dict(self, node: ast.Dict) -> None:
        # {"kind": ...}
        for k in node.keys:
            if isinstance(k, ast.Constant) and k.value == "kind":
                self._add(k, "dict_key", 'dict key "kind"')
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        # obj.get("kind")
        if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
            if node.args and isinstance(node.args[0], ast.Constant) and node.args[0].value == "kind":
                self._add(node, "get_call", '.get("kind")')
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        # obj["kind"]
        sl = node.slice
        if isinstance(sl, ast.Constant) and sl.value == "kind":
            self._add(node, "subscript", '["kind"]')
        self.generic_visit(node)

def iter_py_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*.py"):
        try:
            rel_parts = set(p.relative_to(root).parts)
        except Exception:
            rel_parts = set(p.parts)
        if rel_parts.intersection(SKIP_DIRS):
            continue
        yield p

def scan(root: Path) -> List[Hit]:
    hits: List[Hit] = []
    for fp in iter_py_files(root):
        try:
            text = fp.read_text(encoding="utf-8-sig", errors="replace")
            tree = ast.parse(text)
        except Exception:
            continue
        v = KindVisitor(fp.as_posix())
        v.visit(tree)
        hits.extend(v.hits)
    return hits

def main(argv: List[str]) -> int:
    root = Path(argv[1] if len(argv) > 1 else ".").resolve()
    hits = scan(root)
    for h in sorted(hits, key=lambda x: (x.file, x.line, x.col)):
        print(f"{h.file}:{h.line}:{h.col} [{h.kind}] {h.detail}")
    print(f"\nTOTAL hits: {len(hits)}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
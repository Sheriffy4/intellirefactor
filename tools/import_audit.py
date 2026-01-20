from __future__ import annotations

import importlib
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "intellirefactor"
PKG_NAME = "intellirefactor"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


@dataclass
class ImportFailure:
    module: str
    error: str
    traceback: str
    seconds: float


def find_name_collisions(pkg_dir: Path) -> list[str]:
    collisions = []
    for d in pkg_dir.rglob("*"):
        if not d.is_dir():
            continue
        # проверяем только package-подобные папки
        items = {p.stem for p in d.glob("*.py")}
        dirs = {p.name for p in d.iterdir() if p.is_dir() and not p.name.startswith("__")}
        for name in sorted(items & dirs):
            collisions.append(f"Collision in {d}: both '{name}.py' and '{name}/' exist")
    return collisions


def python_file_to_module(py_file: Path) -> str | None:
    if py_file.name == "__init__.py":
        rel = py_file.parent.relative_to(REPO_ROOT)
    else:
        rel = py_file.relative_to(REPO_ROOT).with_suffix("")

    parts = rel.parts
    if not parts or parts[0] != PKG_NAME:
        return None

    return ".".join(parts)


def iter_package_modules(pkg_dir: Path) -> list[str]:
    modules = []
    for py in pkg_dir.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        if py.name.endswith(".bak.py") or py.name.endswith(".bak"):
            continue
        m = python_file_to_module(py)
        if m:
            modules.append(m)
    return sorted(set(modules))


def try_import(module: str) -> ImportFailure | None:
    start = time.perf_counter()
    try:
        importlib.import_module(module)
        return None
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return ImportFailure(
            module=module,
            error=f"{type(e).__name__}: {e}",
            traceback=tb,
            seconds=time.perf_counter() - start,
        )


def main():
    collisions = find_name_collisions(PKG_DIR)
    if collisions:
        print("NAME COLLISIONS FOUND:")
        for c in collisions:
            print(" -", c)
        print()

    modules = iter_package_modules(PKG_DIR)
    failures: list[ImportFailure] = []
    slow: list[tuple[str, float]] = []

    for m in modules:
        t0 = time.perf_counter()
        f = try_import(m)
        dt = time.perf_counter() - t0
        if dt > 0.5:
            slow.append((m, dt))
        if f:
            failures.append(f)

    print(f"TOTAL MODULES: {len(modules)}")
    print(f"BAD: {len(failures)}")
    if slow:
        print(f"SLOW IMPORTS (>0.5s): {len(slow)}")
        for m, dt in sorted(slow, key=lambda x: x[1], reverse=True)[:20]:
            print(f"  {dt:0.3f}s  {m}")

    for f in failures[:50]:
        print("\n===", f.module, f.error, f"{f.seconds:0.3f}s", "===\n", f.traceback)

    # JSON-отчет (по желанию)
    report = {
        "total_modules": len(modules),
        "bad": len(failures),
        "collisions": collisions,
        "failures": [f.__dict__ for f in failures],
        "slow_imports": [{"module": m, "seconds": dt} for m, dt in slow],
    }
    out = REPO_ROOT / "import_audit_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
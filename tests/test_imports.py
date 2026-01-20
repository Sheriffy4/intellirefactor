import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "intellirefactor"

# Ensure the repo root is in the Python path
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def iter_modules():
    """Iterate through all modules in the intellirefactor package."""
    modules = []
    for py in PKG_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        rel = py.relative_to(REPO_ROOT)
        if py.name == "__init__.py":
            rel = py.parent.relative_to(REPO_ROOT)
        else:
            rel = rel.with_suffix("")
        modules.append(".".join(rel.parts))
    return sorted(set(modules))


def test_all_modules_import():
    """
    Test that all modules in the intellirefactor package can be imported.
    
    This is a critical CI check to catch import regressions early.
    """
    bad = []
    for m in iter_modules():
        try:
            importlib.import_module(m)
        except Exception as e:
            bad.append((m, type(e).__name__, str(e)))
    
    assert not bad, "Import failures:\n" + "\n".join(
        f"{m}: {t}: {msg}" for m, t, msg in bad
    )
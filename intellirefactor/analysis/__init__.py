"""
Analysis module for IntelliRefactor

Provides project and file analysis capabilities including:
- Project structure analysis
- Code metrics calculation
- Duplicate code detection
- Refactoring opportunity identification
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, List

# NOTE:
# We intentionally avoid importing heavy analyzers at package import time.
# This keeps CLI/GUI snappy and prevents cyclic import issues.

__all__ = ["ProjectAnalyzer", "FileAnalyzer", "MetricsAnalyzer"]

_LAZY_EXPORTS: Dict[str, str] = {
    # canonical locations
    "ProjectAnalyzer": "intellirefactor.analysis.refactor.project_analyzer",
    "FileAnalyzer": "intellirefactor.analysis.refactor.file_analyzer",
    "MetricsAnalyzer": "intellirefactor.analysis.refactor.metrics_analyzer",
}

if TYPE_CHECKING:
    from intellirefactor.analysis.refactor.project_analyzer import ProjectAnalyzer as ProjectAnalyzer
    from intellirefactor.analysis.refactor.file_analyzer import FileAnalyzer as FileAnalyzer
    from intellirefactor.analysis.refactor.metrics_analyzer import MetricsAnalyzer as MetricsAnalyzer


def __getattr__(name: str) -> Any:
    """
    Lazy attribute loading (PEP 562).
    Allows `from intellirefactor.analysis import ProjectAnalyzer` without eager imports.
    """
    mod_path = _LAZY_EXPORTS.get(name)
    if not mod_path:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    import importlib

    mod = importlib.import_module(mod_path)
    try:
        return getattr(mod, name)
    except AttributeError as e:
        raise AttributeError(
            f"module {mod_path!r} does not define {name!r} (lazy export from {__name__!r})"
        ) from e


def __dir__() -> List[str]:
    return sorted(set(globals().keys()) | set(_LAZY_EXPORTS.keys()))

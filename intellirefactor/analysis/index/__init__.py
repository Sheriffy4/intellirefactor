"""
Index package for IntelliRefactor.

This package contains modules for indexing and querying code analysis data.

Modules:
- schema: Database schema definitions
- query: Query interface for indexed data
- store: Storage backend implementation
- builder: Index building utilities
- neighbor_extractor: Direct neighbor extraction for LLM navigation
"""

from __future__ import annotations

import importlib.util

__all__ = ["IndexBuilder", "IndexBuildResult", "NeighborExtractor"]

# Optional submodules (export only if present in this repo layout)
if importlib.util.find_spec(__name__ + ".schema"):
    __all__.append("IndexSchema")
if importlib.util.find_spec(__name__ + ".store"):
    __all__.append("IndexStore")
if importlib.util.find_spec(__name__ + ".query"):
    __all__.append("IndexQuery")
if importlib.util.find_spec(__name__ + ".neighbor_extractor"):
    __all__.append("NeighborExtractor")


def __getattr__(name: str):
    if name == "IndexSchema":
        from .schema import IndexSchema
        return IndexSchema
    if name == "IndexStore":
        from .store import IndexStore
        return IndexStore
    if name == "IndexQuery":
        from .query import IndexQuery
        return IndexQuery
    if name == "NeighborExtractor":
        from .neighbor_extractor import NeighborExtractor
        return NeighborExtractor
    if name in {"IndexBuilder", "IndexBuildResult"}:
        from .builder import IndexBuilder, IndexBuildResult
        return {"IndexBuilder": IndexBuilder, "IndexBuildResult": IndexBuildResult}[name]
    raise AttributeError(name)
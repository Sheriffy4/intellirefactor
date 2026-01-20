"""
Compatibility models for analysis.index.

This module re-exports canonical models from intellirefactor.analysis.foundation.models
to maintain backward compatibility with code that imports from analysis.index.models.
"""

from __future__ import annotations

from intellirefactor.analysis.foundation.models import (
    FileReference,
    BlockInfo,
    BlockType,
    Evidence,
    DependencyInfo,
    DependencyResolution,
    DeepMethodInfo,
    DeepClassInfo,
    SemanticCategory,
    ResponsibilityMarker,
    parse_semantic_category,
    parse_responsibility_markers,
    parse_dependency_resolution,
    parse_block_type,
)

__all__ = [
    "FileReference",
    "BlockInfo",
    "BlockType",
    "Evidence",
    "DependencyInfo",
    "DependencyResolution",
    "DeepMethodInfo",
    "DeepClassInfo",
    "SemanticCategory",
    "ResponsibilityMarker",
    "parse_semantic_category",
    "parse_responsibility_markers",
    "parse_dependency_resolution",
    "parse_block_type",
]

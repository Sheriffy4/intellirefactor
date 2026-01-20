"""
Foundation layer for IntelliRefactor.

This package provides the core contracts and utilities that all analysis
components depend on. It should have zero dependencies on other analysis modules.

Components:
- Standardized data models (Finding, AnalysisError, etc.)
- Error handling utilities
- Configuration contracts (coming soon)
"""

from .models import (
    Severity,
    AnalysisStage,
    Location,
    FileReference,
    Evidence,
    Finding,
    AnalysisError,
    AnalysisStats,
    AnalysisReport,
    BlockType,
    BlockInfo,
    # migrated from legacy analysis.models
    SemanticCategory,
    ResponsibilityMarker,
    DependencyResolution,
    DeepMethodInfo,
    DeepClassInfo,
    DependencyInfo,
    RiskLevel,
    DependencyInterface,
    InterfaceUsage,
    MethodGroup,
    CohesionMatrix,
    parse_semantic_category,
    parse_responsibility_markers,
    parse_dependency_resolution,
    parse_block_type,
)

from .error_handler import (
    StandardErrorHandler,
    safe_run,
    safe_run_with_default,
    safe_method,
    safe_read_file,
    safe_parse_ast
)

from .stubs import (
    not_implemented,
    stub_audit_entry
)

__all__ = [
    # Models
    'Severity',
    'AnalysisStage', 
    'Location',
    'FileReference',
    'Evidence',
    'Finding',
    'AnalysisError',
    'AnalysisStats',
    'AnalysisReport',
    'BlockType',
    'BlockInfo',
    
    # Deep semantic models
    'SemanticCategory',
    'ResponsibilityMarker',
    'DependencyResolution',
    'DeepMethodInfo',
    'DeepClassInfo',
    'DependencyInfo',
    'RiskLevel',
    'DependencyInterface',
    'InterfaceUsage',
    'MethodGroup',
    'CohesionMatrix',
    'parse_semantic_category',
    'parse_responsibility_markers',
    'parse_dependency_resolution',
    'parse_block_type',
    
    # Error Handling
    'StandardErrorHandler',
    'safe_run',
    'safe_run_with_default',
    'safe_method',
    'safe_read_file',
    'safe_parse_ast',
    
    # Stubs
    'not_implemented',
    'stub_audit_entry',
]

"""
Functional Decomposition & Consolidation Pipeline for IntelliRefactor

This module implements the functional decomposition system described in ref.md,
providing safe and incremental refactoring through:

1. Functional mapping (atomic blocks extraction)
2. Similarity clustering (finding duplicate functionality)
3. Consolidation planning (canonicalization + migration)
4. Patch-based application (small, validated steps)

The system integrates with existing IntelliRefactor components and extends
them with functional decomposition capabilities.
"""

from .models import (
    FunctionalBlock,
    Capability,
    SimilarityCluster,
    CanonicalizationPlan,
    PatchStep,
    ProjectFunctionalMap,
    DecompositionConfig,
    ApplicationMode,
    RecommendationType,
    RiskLevel,
    EffortClass,
    PatchStepKind,
)

from .block_extractor import FunctionalBlockExtractor
from .categorizer import FunctionCategorizer
from .fingerprints import FingerprintGenerator
from .similarity import SimilarityCalculator
from .clustering import FunctionalClusterer
from .functional_map import FunctionalMapBuilder
from .consolidation_planner import ConsolidationPlanner
from .report_generator import DecompositionReportGenerator
from .decomposition_analyzer import DecompositionAnalyzer

# Extracted modules (refactored from DecompositionAnalyzer)
from .file_operations import FileOperations
from .safe_exact_evaluator import SafeExactEvaluator
from .unified_symbol_generator import UnifiedSymbolGenerator
from .wrapper_patcher import WrapperPatcher
from .import_updater import ImportUpdater
from .validation import UnifiedAliasValidator
from .statistics_generator import StatisticsGenerator

# Helper modules (for advanced usage)
from .ast_helpers import (
    looks_like_class_ref,
    call_key_from_ast,
    attr_path_from_ast,
    calculate_cyclomatic_complexity,
)
from .signature_builder import SignatureBuilder
from .dependency_extractor import DependencyExtractor
from .type_collectors import ModuleTypeHintCollector, LocalTypeHintVisitor
from .visitors import FunctionVisitor, AssignedNameVisitor
from .import_resolver import ImportResolver

__all__ = [
    # Models
    "FunctionalBlock",
    "Capability",
    "SimilarityCluster",
    "CanonicalizationPlan",
    "PatchStep",
    "ProjectFunctionalMap",
    "DecompositionConfig",
    "ApplicationMode",
    "RecommendationType",
    "RiskLevel",
    "EffortClass",
    "PatchStepKind",
    # Components
    "FunctionalBlockExtractor",
    "FunctionCategorizer",
    "FingerprintGenerator",
    "SimilarityCalculator",
    "FunctionalClusterer",
    "FunctionalMapBuilder",
    "ConsolidationPlanner",
    "DecompositionReportGenerator",
    "DecompositionAnalyzer",
    # Extracted modules (refactored components)
    "FileOperations",
    "SafeExactEvaluator",
    "UnifiedSymbolGenerator",
    "WrapperPatcher",
    "ImportUpdater",
    "UnifiedAliasValidator",
    "StatisticsGenerator",
    # Helper utilities (advanced usage)
    "looks_like_class_ref",
    "call_key_from_ast",
    "attr_path_from_ast",
    "calculate_cyclomatic_complexity",
    "SignatureBuilder",
    "DependencyExtractor",
    "ModuleTypeHintCollector",
    "LocalTypeHintVisitor",
    "FunctionVisitor",
    "AssignedNameVisitor",
    "ImportResolver",
]

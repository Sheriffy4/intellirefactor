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
]
"""
Expert Refactoring Analysis Module

Provides deep analysis capabilities for expert-level refactoring,
extracting critical data needed for safe and precise code restructuring.

This module implements the expert refactoring analyzer as specified in the requirements,
focusing on Priority 0 (critical) analyses:
- Internal call graph analysis with cycle detection
- External caller analysis for impact assessment  
- Behavioral contract extraction from docstrings and code
- Dependency interface analysis for understanding module boundaries

Priority 1 (important) analyses:
- Cohesion matrix for class decomposition suggestions
- Test discovery and characterization test generation
- Concrete duplicate code detection with extraction suggestions

Priority 2 (enhancement) analyses:
- Git history analysis for co-change patterns
- Compatibility analysis for breaking change assessment
"""

from .expert_analyzer import ExpertRefactoringAnalyzer
from .models import (
    ExpertAnalysisResult,
    CallGraph,
    CallNode,
    CallEdge,
    CohesionMatrix,
    MethodGroup,
    BehavioralContract,
    CharacterizationTest,
    DependencyInterface,
    ExternalCaller,
    UsageAnalysis,
    ImpactAssessment,
    TestDiscoveryResult,
    DuplicateFragment,
    GitChangePattern,
    RiskLevel,
    CallType,
    TestCategory,
)

__all__ = [
    # Main analyzer
    'ExpertRefactoringAnalyzer',
    
    # Core result types
    'ExpertAnalysisResult',
    'CallGraph',
    'CallNode', 
    'CallEdge',
    'CohesionMatrix',
    'MethodGroup',
    'BehavioralContract',
    'CharacterizationTest',
    'DependencyInterface',
    'ExternalCaller',
    'UsageAnalysis',
    'ImpactAssessment',
    'TestDiscoveryResult',
    'DuplicateFragment',
    'GitChangePattern',
    
    # Enums
    'RiskLevel',
    'CallType',
    'TestCategory',
]
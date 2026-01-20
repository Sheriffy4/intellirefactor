"""
Specification Generation Modules

This package contains specialized generators for creating specification documents
from audit results. Each module focuses on a specific aspect of spec generation.
"""

from .sections import (
    RequirementsHeaderGenerator,
    DesignHeaderGenerator,
    ImplementationHeaderGenerator,
)
from .statistics import (
    ExecutiveSummaryGenerator,
    StatisticsGenerator,
)
from .findings import (
    CriticalFindingsGenerator,
    HighPriorityGenerator,
    DuplicateCodeGenerator,
    UnusedCodeGenerator,
    QualityPerformanceGenerator,
)
from .design import (
    ArchitectureOverviewGenerator,
    ComponentAnalysisGenerator,
    RefactoringStrategyGenerator,
    DependencyAnalysisGenerator,
    RiskAssessmentGenerator,
    DesignDecisionsGenerator,
)
from .implementation import (
    TaskBreakdownGenerator,
    PriorityPhasesGenerator,
    RefactoringTasksGenerator,
    TestingStrategyGenerator,
    RollbackPlanGenerator,
    SuccessCriteriaGenerator,
)
from .extractors import (
    RefactoringPriorityExtractor,
    CleanupTaskExtractor,
)
from .recommendations import (
    ImplementationRecommendationsGenerator,
)
from .appendix import (
    RequirementsAppendixGenerator,
    DesignAppendixGenerator,
    ImplementationAppendixGenerator,
)

__all__ = [
    "RequirementsHeaderGenerator",
    "DesignHeaderGenerator",
    "ImplementationHeaderGenerator",
    "ExecutiveSummaryGenerator",
    "StatisticsGenerator",
    "CriticalFindingsGenerator",
    "HighPriorityGenerator",
    "DuplicateCodeGenerator",
    "UnusedCodeGenerator",
    "QualityPerformanceGenerator",
    "ArchitectureOverviewGenerator",
    "ComponentAnalysisGenerator",
    "RefactoringStrategyGenerator",
    "DependencyAnalysisGenerator",
    "RiskAssessmentGenerator",
    "DesignDecisionsGenerator",
    "TaskBreakdownGenerator",
    "PriorityPhasesGenerator",
    "RefactoringTasksGenerator",
    "TestingStrategyGenerator",
    "RollbackPlanGenerator",
    "SuccessCriteriaGenerator",
    "RefactoringPriorityExtractor",
    "CleanupTaskExtractor",
    "ImplementationRecommendationsGenerator",
    "RequirementsAppendixGenerator",
    "DesignAppendixGenerator",
    "ImplementationAppendixGenerator",
]

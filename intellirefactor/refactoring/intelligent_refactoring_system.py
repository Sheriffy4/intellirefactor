"""
Intelligent Refactoring System Foundation.

This module provides the foundation for an intelligent refactoring system that can:
- analyze Python code and extract metrics
- detect refactoring opportunities
- assess refactoring quality
- export a machine-readable knowledge base
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Robust import: package-relative in real project, fallback for direct execution/tests.
try:
    from ..interfaces import BaseRefactoringSystem, GenericRefactoringOpportunity
except Exception:  # pragma: no cover
    try:
        from interfaces import BaseRefactoringSystem, GenericRefactoringOpportunity  # type: ignore
    except Exception:  # pragma: no cover

        class BaseRefactoringSystem:  # minimal stub
            pass

        @dataclass
        class GenericRefactoringOpportunity:  # minimal stub for local runs
            id: str
            type: str
            priority: int
            description: str
            target_files: List[str]
            estimated_impact: Dict[str, float]
            prerequisites: List[str]
            automation_confidence: float
            risk_level: str


class RefactoringOpportunityType(Enum):
    """Types of refactoring opportunities that can be detected."""

    GOD_CLASS = "god_class"
    LONG_METHOD = "long_method"
    DUPLICATE_CODE = "duplicate_code"
    LARGE_CONFIGURATION = "large_configuration"
    TIGHT_COUPLING = "tight_coupling"
    LOW_COHESION = "low_cohesion"
    MISSING_ABSTRACTION = "missing_abstraction"
    COMPLEX_CONDITIONAL = "complex_conditional"
    FEATURE_ENVY = "feature_envy"
    DATA_CLUMPS = "data_clumps"


class RefactoringQualityMetric(Enum):
    """Quality metrics for measuring refactoring success."""

    COMPLEXITY_REDUCTION = "complexity_reduction"
    COUPLING_REDUCTION = "coupling_reduction"
    COHESION_IMPROVEMENT = "cohesion_improvement"
    TESTABILITY_IMPROVEMENT = "testability_improvement"
    MAINTAINABILITY_IMPROVEMENT = "maintainability_improvement"
    PERFORMANCE_IMPACT = "performance_impact"
    BACKWARD_COMPATIBILITY = "backward_compatibility"
    CODE_DUPLICATION_REDUCTION = "code_duplication_reduction"
    INTERFACE_SEGREGATION = "interface_segregation"
    DEPENDENCY_INVERSION = "dependency_inversion"


@dataclass
class CodeMetrics:
    """Code metrics used in decision making."""

    file_size: int
    cyclomatic_complexity: int
    number_of_methods: int
    number_of_responsibilities: int
    coupling_level: float  # 0.0 to 1.0
    cohesion_level: float  # 0.0 to 1.0
    test_coverage: float  # 0.0 to 1.0
    number_of_dependencies: int
    number_of_clients: int


@dataclass
class RefactoringContext:
    """Context information for refactoring decisions."""

    has_existing_clients: bool
    backward_compatibility_required: bool
    performance_critical: bool
    team_experience_level: str  # "junior", "intermediate", "senior"
    project_timeline: str  # "tight", "moderate", "flexible"
    testing_infrastructure: str  # "minimal", "good", "excellent"


@dataclass
class RefactoringPattern:
    """Represents a refactoring pattern."""

    pattern_id: str
    name: str
    description: str
    applicability_conditions: List[str]
    transformation_steps: List[str]
    risk_level: str
    automation_potential: float


@dataclass
class CodeTransformationRule:
    """Represents a code transformation rule."""

    rule_id: str
    name: str
    description: str
    pattern_type: str
    source_pattern: str
    target_pattern: str
    conditions: List[str]
    transformations: List[str]


@dataclass
class RefactoringOpportunity:
    """Represents a detected refactoring opportunity."""

    opportunity_id: str
    type: RefactoringOpportunityType
    severity: str  # "low", "medium", "high", "critical"
    confidence: float  # 0.0 to 1.0

    file_path: str
    line_start: int
    line_end: int
    affected_elements: List[str]

    description: str
    current_metrics: Dict[str, float]
    impact_assessment: str

    recommended_patterns: List[str]
    estimated_effort_hours: float
    risk_level: str

    can_auto_refactor: bool
    automation_confidence: float
    manual_steps_required: List[str]

    detected_date: str
    detection_method: str
    related_opportunities: List[str] = field(default_factory=list)


@dataclass
class RefactoringQualityAssessment:
    """Assessment of refactoring quality and success."""

    assessment_id: str
    refactoring_id: str
    before_metrics: Dict[str, float]
    after_metrics: Dict[str, float]
    improvement_scores: Dict[RefactoringQualityMetric, float]
    overall_quality_score: float  # 0.0 to 1.0
    success_indicators: List[str]
    quality_issues: List[str]
    tests_pass: bool
    performance_acceptable: bool
    backward_compatible: bool
    improvement_suggestions: List[str]
    follow_up_refactorings: List[str]
    assessed_date: str
    assessor: str


@dataclass
class MachineReadableKnowledgeBase:
    """Machine-readable knowledge base for intelligent refactoring."""

    knowledge_base_id: str
    version: str
    created_date: str

    transformation_rules: List[CodeTransformationRule]
    refactoring_patterns: List[RefactoringPattern]
    decision_criteria: Dict[str, Any]

    detection_rules: Dict[RefactoringOpportunityType, Dict[str, Any]]
    quality_thresholds: Dict[str, float]

    quality_benchmarks: Dict[RefactoringQualityMetric, Dict[str, float]]
    success_patterns: List[Dict[str, Any]]

    automation_rules: List[Dict[str, Any]]
    reusable_components: List[Dict[str, Any]]

    historical_refactorings: List[Dict[str, Any]]
    success_correlations: Dict[str, float]


def _safe_div(numer: float, denom: float, default: float = 0.0) -> float:
    """Safe division helper to avoid ZeroDivisionError."""
    if denom == 0:
        return default
    return numer / denom


class CodeAnalyzer:
    """Analyzes code to extract metrics and detect refactoring opportunities."""

    def __init__(self) -> None:
        self.ast_cache: Dict[str, ast.AST] = {}

    def analyze_file(self, file_path: str) -> CodeMetrics:
        """
        Analyze a Python file and extract code metrics.

        Never raises: on error returns default metrics and logs the failure.
        """
        try:
            with open(file_path, "r", encoding="utf-8-sig") as f:
                content = f.read()

            tree = ast.parse(content)
            self.ast_cache[file_path] = tree
            return self._extract_metrics(tree, content)

        except (OSError, UnicodeError, SyntaxError) as e:
            logger.error("Error analyzing file %s: %s", file_path, e)
            return self._default_metrics()
        except Exception as e:
            logger.exception("Unexpected error analyzing file %s: %s", file_path, e)
            return self._default_metrics()

    def _extract_metrics(self, tree: ast.AST, content: str) -> CodeMetrics:
        """Extract metrics from AST and raw content."""
        lines = content.splitlines()
        file_size = len(lines)

        methods = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        complexity = self._calculate_complexity(tree)
        responsibilities = self._estimate_responsibilities(tree, content)
        coupling = self._estimate_coupling(tree)
        cohesion = self._estimate_cohesion(tree)
        dependencies = self._count_dependencies(tree)

        return CodeMetrics(
            file_size=file_size,
            cyclomatic_complexity=complexity,
            number_of_methods=len(methods),
            number_of_responsibilities=responsibilities,
            coupling_level=coupling,
            cohesion_level=cohesion,
            test_coverage=0.0,
            number_of_dependencies=dependencies,
            number_of_clients=0,
        )

    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate simplified cyclomatic complexity."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.Try, ast.With, ast.Match)):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                complexity += max(0, len(node.values) - 1)
        return complexity

    def _estimate_responsibilities(self, tree: ast.AST, content: str) -> int:
        """Estimate number of responsibilities (heuristic)."""
        methods = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        prefixes = {m.split("_", 1)[0] for m in methods if "_" in m}

        keywords = {
            "generate",
            "analyze",
            "test",
            "cache",
            "config",
            "metric",
            "monitor",
            "validate",
            "process",
            "handle",
            "manage",
        }
        lowered = content.lower()
        found = {k for k in keywords if k in lowered}

        return max(len(prefixes), len(found), 1)

    def _estimate_coupling(self, tree: ast.AST) -> float:
        """Estimate coupling level (0.0 to 1.0)."""
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        attrs = [n for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
        return min(1.0, (len(imports) + len(attrs)) / 50.0)

    def _estimate_cohesion(self, tree: ast.AST) -> float:
        """Estimate cohesion level (0.0 to 1.0)."""
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        if not classes:
            return 0.5

        largest = max(classes, key=lambda c: len(c.body))
        methods = [n for n in largest.body if isinstance(n, ast.FunctionDef)]

        if len(methods) <= 3:
            return 0.8
        if len(methods) <= 10:
            return 0.6
        return 0.3

    def _count_dependencies(self, tree: ast.AST) -> int:
        """Count import statements."""
        imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
        return len(imports)

    def _default_metrics(self) -> CodeMetrics:
        """Default metrics when analysis fails."""
        return CodeMetrics(
            file_size=0,
            cyclomatic_complexity=1,
            number_of_methods=0,
            number_of_responsibilities=1,
            coupling_level=0.5,
            cohesion_level=0.5,
            test_coverage=0.0,
            number_of_dependencies=0,
            number_of_clients=0,
        )


class OpportunityDetector:
    """Detects refactoring opportunities in code."""

    def __init__(self, knowledge_base: MachineReadableKnowledgeBase) -> None:
        self.knowledge_base = knowledge_base
        self.code_analyzer = CodeAnalyzer()

    def detect_opportunities(
        self,
        file_path: str,
        context: RefactoringContext,
    ) -> List[RefactoringOpportunity]:
        """
        Detect refactoring opportunities in a file.

        `context` is reserved for future rule tuning (e.g., performance-critical code).
        """
        _ = context  # not used yet
        metrics = self.code_analyzer.analyze_file(file_path)
        opportunities: List[RefactoringOpportunity] = []

        if self._is_god_class(metrics):
            opportunities.append(self._create_god_class_opportunity(file_path, metrics))
        if self._is_large_configuration(file_path, metrics):
            opportunities.append(self._create_config_split_opportunity(file_path, metrics))
        if self._has_tight_coupling(metrics):
            opportunities.append(self._create_coupling_opportunity(file_path, metrics))
        if self._has_low_cohesion(metrics):
            opportunities.append(self._create_cohesion_opportunity(file_path, metrics))

        return opportunities

    def _is_god_class(self, metrics: CodeMetrics) -> bool:
        thresholds = self.knowledge_base.detection_rules.get(
            RefactoringOpportunityType.GOD_CLASS,
            {"file_size_threshold": 1000, "responsibility_threshold": 3, "cohesion_threshold": 0.5},
        )
        return (
            metrics.file_size > thresholds["file_size_threshold"]
            and metrics.number_of_responsibilities > thresholds["responsibility_threshold"]
            and metrics.cohesion_level < thresholds["cohesion_threshold"]
        )

    def _is_large_configuration(self, file_path: str, metrics: CodeMetrics) -> bool:
        thresholds = self.knowledge_base.detection_rules.get(
            RefactoringOpportunityType.LARGE_CONFIGURATION,
            {
                "file_size_threshold": 200,
                "responsibility_threshold": 4,
                "filename_patterns": ["config", "settings"],
            },
        )
        filename_matches = any(p in file_path.lower() for p in thresholds["filename_patterns"])
        return (
            filename_matches
            and metrics.file_size > thresholds["file_size_threshold"]
            and metrics.number_of_responsibilities > thresholds["responsibility_threshold"]
        )

    def _has_tight_coupling(self, metrics: CodeMetrics) -> bool:
        threshold = self.knowledge_base.detection_rules.get(
            RefactoringOpportunityType.TIGHT_COUPLING,
            {"coupling_threshold": 0.7},
        )["coupling_threshold"]
        return metrics.coupling_level > threshold

    def _has_low_cohesion(self, metrics: CodeMetrics) -> bool:
        threshold = self.knowledge_base.detection_rules.get(
            RefactoringOpportunityType.LOW_COHESION,
            {"cohesion_threshold": 0.4},
        )["cohesion_threshold"]
        return metrics.cohesion_level < threshold

    def _create_god_class_opportunity(self, file_path: str, metrics: CodeMetrics) -> RefactoringOpportunity:
        now = datetime.now()
        return RefactoringOpportunity(
            opportunity_id=f"god_class_{Path(file_path).stem}_{now.strftime('%Y%m%d_%H%M%S')}",
            type=RefactoringOpportunityType.GOD_CLASS,
            severity="high",
            confidence=0.9,
            file_path=file_path,
            line_start=1,
            line_end=metrics.file_size,
            affected_elements=["entire_class"],
            description=(
                f"Large class with {metrics.number_of_responsibilities} responsibilities "
                f"and {metrics.file_size} lines"
            ),
            current_metrics={
                "file_size": float(metrics.file_size),
                "responsibilities": float(metrics.number_of_responsibilities),
                "cohesion": float(metrics.cohesion_level),
                "complexity": float(metrics.cyclomatic_complexity),
            },
            impact_assessment="High impact - difficult to maintain, test, and extend",
            recommended_patterns=["extract_component", "dependency_injection", "facade_pattern"],
            estimated_effort_hours=40.0,
            risk_level="medium",
            can_auto_refactor=True,
            automation_confidence=0.8,
            manual_steps_required=[
                "Review extracted component boundaries",
                "Validate interface contracts",
                "Update integration tests",
            ],
            detected_date=now.isoformat(),
            detection_method="automated_analysis",
        )

    def _create_config_split_opportunity(self, file_path: str, metrics: CodeMetrics) -> RefactoringOpportunity:
        now = datetime.now()
        return RefactoringOpportunity(
            opportunity_id=f"config_split_{Path(file_path).stem}_{now.strftime('%Y%m%d_%H%M%S')}",
            type=RefactoringOpportunityType.LARGE_CONFIGURATION,
            severity="medium",
            confidence=0.85,
            file_path=file_path,
            line_start=1,
            line_end=metrics.file_size,
            affected_elements=["configuration_class"],
            description=f"Large configuration with {metrics.number_of_responsibilities} distinct domains",
            current_metrics={
                "file_size": float(metrics.file_size),
                "responsibilities": float(metrics.number_of_responsibilities),
            },
            impact_assessment="Medium impact - configuration is hard to understand and maintain",
            recommended_patterns=["split_configuration"],
            estimated_effort_hours=8.0,
            risk_level="low",
            can_auto_refactor=True,
            automation_confidence=0.9,
            manual_steps_required=["Validate domain boundaries", "Test configuration loading"],
            detected_date=now.isoformat(),
            detection_method="automated_analysis",
        )

    def _create_coupling_opportunity(self, file_path: str, metrics: CodeMetrics) -> RefactoringOpportunity:
        now = datetime.now()
        return RefactoringOpportunity(
            opportunity_id=f"coupling_{Path(file_path).stem}_{now.strftime('%Y%m%d_%H%M%S')}",
            type=RefactoringOpportunityType.TIGHT_COUPLING,
            severity="medium",
            confidence=0.75,
            file_path=file_path,
            line_start=1,
            line_end=metrics.file_size,
            affected_elements=["class_dependencies"],
            description=f"High coupling level ({metrics.coupling_level:.2f}) detected",
            current_metrics={
                "coupling_level": float(metrics.coupling_level),
                "dependencies": float(metrics.number_of_dependencies),
            },
            impact_assessment="Medium impact - difficult to test and modify independently",
            recommended_patterns=["dependency_injection", "extract_interface"],
            estimated_effort_hours=16.0,
            risk_level="medium",
            can_auto_refactor=True,
            automation_confidence=0.7,
            manual_steps_required=[
                "Design interface contracts",
                "Update dependency injection configuration",
                "Create test doubles",
            ],
            detected_date=now.isoformat(),
            detection_method="automated_analysis",
        )

    def _create_cohesion_opportunity(self, file_path: str, metrics: CodeMetrics) -> RefactoringOpportunity:
        now = datetime.now()
        return RefactoringOpportunity(
            opportunity_id=f"cohesion_{Path(file_path).stem}_{now.strftime('%Y%m%d_%H%M%S')}",
            type=RefactoringOpportunityType.LOW_COHESION,
            severity="medium",
            confidence=0.7,
            file_path=file_path,
            line_start=1,
            line_end=metrics.file_size,
            affected_elements=["class_methods"],
            description=f"Low cohesion level ({metrics.cohesion_level:.2f}) detected",
            current_metrics={
                "cohesion_level": float(metrics.cohesion_level),
                "methods": float(metrics.number_of_methods),
            },
            impact_assessment="Medium impact - class has unclear purpose and mixed responsibilities",
            recommended_patterns=["extract_component", "single_responsibility"],
            estimated_effort_hours=24.0,
            risk_level="medium",
            can_auto_refactor=False,
            automation_confidence=0.5,
            manual_steps_required=[
                "Analyze method relationships",
                "Identify cohesive groups",
                "Design component boundaries",
                "Extract components manually",
            ],
            detected_date=now.isoformat(),
            detection_method="automated_analysis",
        )


class QualityAssessor:
    """Assesses the quality and success of refactoring efforts."""

    def __init__(self, knowledge_base: MachineReadableKnowledgeBase) -> None:
        self.knowledge_base = knowledge_base
        self.code_analyzer = CodeAnalyzer()

    def assess_refactoring_quality(
        self,
        refactoring_id: str,
        before_files: List[str],
        after_files: List[str],
    ) -> RefactoringQualityAssessment:
        """Assess the quality of a completed refactoring."""
        before_metrics = self._analyze_files(before_files)
        after_metrics = self._analyze_files(after_files)
        improvements = self._calculate_improvements(before_metrics, after_metrics)
        quality_score = self._calculate_quality_score(improvements)

        return RefactoringQualityAssessment(
            assessment_id=f"assessment_{refactoring_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            refactoring_id=refactoring_id,
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            improvement_scores=improvements,
            overall_quality_score=quality_score,
            success_indicators=self._identify_success_indicators(improvements),
            quality_issues=self._identify_quality_issues(improvements),
            tests_pass=True,
            performance_acceptable=True,
            backward_compatible=True,
            improvement_suggestions=self._generate_improvement_suggestions(improvements),
            follow_up_refactorings=self._suggest_follow_up_refactorings(after_metrics),
            assessed_date=datetime.now().isoformat(),
            assessor="intelligent_refactoring_system",
        )

    def _analyze_files(self, file_paths: List[str]) -> Dict[str, float]:
        """Analyze multiple files and aggregate metrics."""
        total: Dict[str, float] = {
            "file_size": 0.0,
            "complexity": 0.0,
            "coupling": 0.0,
            "cohesion": 0.0,
            "responsibilities": 0.0,
            "dependencies": 0.0,
            "file_count": float(len(file_paths)),
        }

        analyzed = 0
        for file_path in file_paths:
            if Path(file_path).exists():
                m = self.code_analyzer.analyze_file(file_path)
                total["file_size"] += float(m.file_size)
                total["complexity"] += float(m.cyclomatic_complexity)
                total["coupling"] += float(m.coupling_level)
                total["cohesion"] += float(m.cohesion_level)
                total["responsibilities"] += float(m.number_of_responsibilities)
                total["dependencies"] += float(m.number_of_dependencies)
                analyzed += 1

        denom = float(analyzed) if analyzed > 0 else 1.0
        total["avg_coupling"] = total["coupling"] / denom
        total["avg_cohesion"] = total["cohesion"] / denom
        total["avg_file_size"] = total["file_size"] / denom
        total["analyzed_count"] = float(analyzed)

        return total

    def _calculate_improvements(
        self,
        before: Dict[str, float],
        after: Dict[str, float],
    ) -> Dict[RefactoringQualityMetric, float]:
        """Calculate improvement scores (can be negative => regressions)."""
        improvements: Dict[RefactoringQualityMetric, float] = {}

        before_complexity = before.get("complexity", 0.0)
        after_complexity = after.get("complexity", 0.0)
        if before_complexity > 0:
            improvements[RefactoringQualityMetric.COMPLEXITY_REDUCTION] = _safe_div(
                before_complexity - after_complexity,
                before_complexity,
                default=0.0,
            )

        before_c = before.get("avg_coupling", 0.0)
        after_c = after.get("avg_coupling", 0.0)
        if before_c > 0:
            improvements[RefactoringQualityMetric.COUPLING_REDUCTION] = _safe_div(
                before_c - after_c,
                before_c,
                default=0.0,
            )

        before_h = before.get("avg_cohesion", 0.0)
        after_h = after.get("avg_cohesion", 0.0)
        # Normalize to remaining room to improve; guard against 1.0
        denom = 1.0 - before_h
        if denom > 0:
            improvements[RefactoringQualityMetric.COHESION_IMPROVEMENT] = _safe_div(
                after_h - before_h,
                denom,
                default=0.0,
            )
        else:
            improvements[RefactoringQualityMetric.COHESION_IMPROVEMENT] = 0.0

        before_m = 1.0 / (
            1.0 + before.get("avg_file_size", 1000.0) / 1000.0 + before.get("complexity", 10.0) / 10.0
        )
        after_m = 1.0 / (
            1.0 + after.get("avg_file_size", 1000.0) / 1000.0 + after.get("complexity", 10.0) / 10.0
        )
        denom = 1.0 - before_m
        improvements[RefactoringQualityMetric.MAINTAINABILITY_IMPROVEMENT] = (
            _safe_div(after_m - before_m, denom, default=0.0) if denom > 0 else 0.0
        )

        return improvements

    def _calculate_quality_score(self, improvements: Dict[RefactoringQualityMetric, float]) -> float:
        """Calculate overall score clamped to [0, 1]."""
        if not improvements:
            return 0.0

        weights = {
            RefactoringQualityMetric.COMPLEXITY_REDUCTION: 0.25,
            RefactoringQualityMetric.COUPLING_REDUCTION: 0.25,
            RefactoringQualityMetric.COHESION_IMPROVEMENT: 0.25,
            RefactoringQualityMetric.MAINTAINABILITY_IMPROVEMENT: 0.25,
        }

        weighted = 0.0
        total_w = 0.0
        for metric, value in improvements.items():
            w = weights.get(metric, 0.1)
            weighted += value * w
            total_w += w

        raw = weighted / total_w if total_w else 0.0
        return max(0.0, min(1.0, raw))

    def _identify_success_indicators(self, improvements: Dict[RefactoringQualityMetric, float]) -> List[str]:
        indicators: List[str] = []
        for metric, improvement in improvements.items():
            if improvement > 0.5:
                indicators.append(f"Significant {metric.value}: {improvement:.2%}")
            elif improvement > 0.2:
                indicators.append(f"Moderate {metric.value}: {improvement:.2%}")
        return indicators

    def _identify_quality_issues(self, improvements: Dict[RefactoringQualityMetric, float]) -> List[str]:
        issues: List[str] = []
        for metric, improvement in improvements.items():
            if improvement < 0:
                issues.append(f"Regression in {metric.value}: {improvement:.2%}")
            elif improvement < 0.1:
                issues.append(f"Limited {metric.value}: {improvement:.2%}")
        return issues

    def _generate_improvement_suggestions(self, improvements: Dict[RefactoringQualityMetric, float]) -> List[str]:
        suggestions: List[str] = []
        for metric, improvement in improvements.items():
            if improvement < 0.3:
                if metric == RefactoringQualityMetric.COMPLEXITY_REDUCTION:
                    suggestions.append("Consider further method extraction to reduce complexity")
                elif metric == RefactoringQualityMetric.COUPLING_REDUCTION:
                    suggestions.append("Consider introducing more interfaces to reduce coupling")
                elif metric == RefactoringQualityMetric.COHESION_IMPROVEMENT:
                    suggestions.append("Consider grouping related methods into separate components")
        return suggestions

    def _suggest_follow_up_refactorings(self, after_metrics: Dict[str, float]) -> List[str]:
        suggestions: List[str] = []
        if after_metrics.get("avg_file_size", 0.0) > 500:
            suggestions.append("Consider further component extraction for large files")
        if after_metrics.get("avg_coupling", 0.0) > 0.5:
            suggestions.append("Consider implementing more dependency injection")
        if after_metrics.get("avg_cohesion", 1.0) < 0.7:
            suggestions.append("Consider improving component cohesion")
        return suggestions


class IntelligentRefactoringSystem(BaseRefactoringSystem):
    """Main intelligent refactoring system that orchestrates all components."""

    def __init__(self, config: Optional[Any] = None) -> None:
        if config is not None and hasattr(config, "__dict__") and not isinstance(config, dict):
            self.config = {
                "safety_level": getattr(config, "safety_level", "moderate"),
                "auto_apply": getattr(config, "auto_apply", False),
                "backup_enabled": getattr(config, "backup_enabled", True),
                "validation_required": getattr(config, "validation_required", True),
                "max_operations_per_session": getattr(config, "max_operations_per_session", 50),
                "stop_on_failure": getattr(config, "stop_on_failure", True),
            }
        else:
            self.config = config or {}

        self.knowledge_base = self._create_knowledge_base()
        self.opportunity_detector = OpportunityDetector(self.knowledge_base)
        self.quality_assessor = QualityAssessor(self.knowledge_base)

    def _create_knowledge_base(self) -> MachineReadableKnowledgeBase:
        detection_rules = self.config.get(
            "detection_rules",
            {
                RefactoringOpportunityType.GOD_CLASS: {
                    "file_size_threshold": 1000,
                    "responsibility_threshold": 3,
                    "cohesion_threshold": 0.5,
                    "confidence_base": 0.9,
                },
                RefactoringOpportunityType.LARGE_CONFIGURATION: {
                    "file_size_threshold": 200,
                    "responsibility_threshold": 4,
                    "filename_patterns": ["config", "settings"],
                    "confidence_base": 0.85,
                },
                RefactoringOpportunityType.TIGHT_COUPLING: {
                    "coupling_threshold": 0.7,
                    "dependency_threshold": 10,
                    "confidence_base": 0.75,
                },
                RefactoringOpportunityType.LOW_COHESION: {
                    "cohesion_threshold": 0.4,
                    "method_threshold": 5,
                    "confidence_base": 0.7,
                },
            },
        )

        quality_benchmarks = self.config.get(
            "quality_benchmarks",
            {
                RefactoringQualityMetric.COMPLEXITY_REDUCTION: {
                    "excellent": 0.8,
                    "good": 0.5,
                    "acceptable": 0.2,
                    "poor": 0.0,
                },
                RefactoringQualityMetric.COUPLING_REDUCTION: {
                    "excellent": 0.7,
                    "good": 0.4,
                    "acceptable": 0.2,
                    "poor": 0.0,
                },
                RefactoringQualityMetric.COHESION_IMPROVEMENT: {
                    "excellent": 0.8,
                    "good": 0.5,
                    "acceptable": 0.3,
                    "poor": 0.0,
                },
            },
        )

        return MachineReadableKnowledgeBase(
            knowledge_base_id=self.config.get("knowledge_base_id", "generic_refactoring_kb"),
            version="1.0.0",
            created_date=datetime.now().isoformat(),
            transformation_rules=[],
            refactoring_patterns=[],
            decision_criteria={},
            detection_rules=detection_rules,
            quality_thresholds=self.config.get(
                "quality_thresholds",
                {
                    "complexity_threshold": 15,
                    "coupling_threshold": 0.7,
                    "cohesion_threshold": 0.4,
                    "file_size_threshold": 1000,
                },
            ),
            quality_benchmarks=quality_benchmarks,
            success_patterns=[],
            automation_rules=[],
            reusable_components=[],
            historical_refactorings=[],
            success_correlations={},
        )

    def analyze_project(self, project_path: str) -> Dict[str, Any]:
        """Analyze an entire project for refactoring opportunities."""
        root = Path(project_path)
        python_files = list(root.rglob("*.py"))

        context = RefactoringContext(
            has_existing_clients=True,
            backward_compatibility_required=True,
            performance_critical=False,
            team_experience_level="senior",
            project_timeline="flexible",
            testing_infrastructure="excellent",
        )

        all_opps: List[RefactoringOpportunity] = []
        for file_path in python_files:
            all_opps.extend(self.opportunity_detector.detect_opportunities(str(file_path), context))

        prioritized = self._prioritize_opportunities(all_opps)
        plan = self._generate_refactoring_plan(prioritized)

        return {
            "total_files_analyzed": len(python_files),
            "opportunities_detected": len(all_opps),
            "high_priority_opportunities": len([o for o in all_opps if o.severity == "high"]),
            "automation_ready": len([o for o in all_opps if o.can_auto_refactor]),
            "estimated_total_effort_hours": sum(o.estimated_effort_hours for o in all_opps),
            "prioritized_opportunities": prioritized[:10],
            "refactoring_plan": plan,
        }

    def _prioritize_opportunities(self, opportunities: List[RefactoringOpportunity]) -> List[RefactoringOpportunity]:
        """Prioritize refactoring opportunities."""
        def priority_score(opp: RefactoringOpportunity) -> float:
            severity_scores = {"critical": 4, "high": 3, "medium": 2, "low": 1}
            s = severity_scores.get(opp.severity, 1)
            return (
                s * opp.confidence * (1.0 + opp.automation_confidence)
                / (1.0 + opp.estimated_effort_hours / 10.0)
            )

        return sorted(opportunities, key=priority_score, reverse=True)

    def _generate_refactoring_plan(self, opportunities: List[Any]) -> List[Dict[str, Any]]:
        """
        Generate a refactoring execution plan.

        Accepts either internal RefactoringOpportunity objects or objects with compatible attributes.
        """
        plan: List[Dict[str, Any]] = []

        for i, opp in enumerate(opportunities[:5], 1):
            opp_type = getattr(opp, "type", getattr(opp, "opportunity_type", "unknown"))
            if hasattr(opp_type, "value"):
                opp_type_value = opp_type.value
            else:
                opp_type_value = str(opp_type)

            plan.append(
                {
                    "step": i,
                    "opportunity_id": getattr(opp, "opportunity_id", getattr(opp, "id", "unknown")),
                    "description": getattr(opp, "description", ""),
                    "type": opp_type_value,
                    "estimated_effort_hours": float(getattr(opp, "estimated_effort_hours", 0.0)),
                    "automation_ready": bool(getattr(opp, "can_auto_refactor", False)),
                    "recommended_patterns": list(getattr(opp, "recommended_patterns", [])),
                    "manual_steps": list(getattr(opp, "manual_steps_required", [])),
                    "risk_level": getattr(opp, "risk_level", getattr(opp, "severity", "unknown")),
                }
            )

        return plan

    def identify_opportunities(self, analysis_data: Dict[str, Any]) -> List[GenericRefactoringOpportunity]:
        """Identify refactoring opportunities from analysis data."""
        opportunities: List[GenericRefactoringOpportunity] = []
        project_path = analysis_data.get("project_path", ".")

        if "legacy_analysis" in analysis_data:
            legacy_data = analysis_data["legacy_analysis"]
            candidates = legacy_data.get("refactoring_candidates", [])

            for c in candidates:
                opportunities.append(
                    GenericRefactoringOpportunity(
                        id=f"candidate_{c.get('filepath', '').replace('/', '_')}",
                        type="file_refactoring",
                        priority=int(c.get("refactoring_priority", 5)),
                        description=(
                            f"Refactor {c.get('filepath', 'file')} with "
                            f"{len(c.get('issues', []))} issues"
                        ),
                        target_files=[c.get("filepath", "")],
                        estimated_impact={
                            "complexity_reduction": 0.3,
                            "maintainability_improvement": 0.4,
                            "automation_potential": float(c.get("automation_potential", 0.5)),
                        },
                        prerequisites=[],
                        automation_confidence=float(c.get("automation_potential", 0.5)),
                        risk_level="medium",
                    )
                )
            return opportunities

        try:
            analysis_result = self.analyze_project(project_path)
            prioritized = analysis_result.get("prioritized_opportunities", [])

            for opp in prioritized:
                opportunities.append(
                    GenericRefactoringOpportunity(
                        id=opp.opportunity_id,
                        type=opp.type.value,
                        priority=int(max(1, min(10, opp.confidence * 10))),
                        description=opp.description,
                        target_files=[opp.file_path],
                        estimated_impact={
                            "complexity_reduction": 0.4,
                            "maintainability_improvement": 0.5,
                            "automation_potential": float(opp.automation_confidence),
                        },
                        prerequisites=list(getattr(opp, "manual_steps_required", [])),
                        automation_confidence=float(opp.automation_confidence),
                        risk_level=str(getattr(opp, "risk_level", opp.severity)),
                    )
                )
        except Exception as e:
            logger.exception("Failed to analyze project for opportunities: %s", e)

        return opportunities

    def generate_refactoring_plan(self, opportunities: List[GenericRefactoringOpportunity]) -> Dict[str, Any]:
        """
        Generate a refactoring plan from generic opportunities.

        Fix: previously this created incompatible pseudo-objects and crashed in _generate_refactoring_plan().
        Now we directly build a plan in a compatible format.
        """
        # Sort by priority desc; take top 5
        top = sorted(opportunities, key=lambda o: o.priority, reverse=True)[:5]

        steps: List[Dict[str, Any]] = []
        for i, opp in enumerate(top, 1):
            # Very rough effort estimate: higher priority => lower effort is not always true,
            # but keeps the demo predictable.
            effort = float(max(1, 12 - int(opp.priority)))

            steps.append(
                {
                    "step": i,
                    "opportunity_id": opp.id,
                    "description": opp.description,
                    "type": opp.type,
                    "estimated_effort_hours": effort,
                    "automation_ready": bool(opp.automation_confidence > 0.7),
                    "recommended_patterns": [],
                    "manual_steps": list(opp.prerequisites or []),
                    "risk_level": opp.risk_level,
                    "target_files": list(opp.target_files or []),
                }
            )

        return {
            "id": f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "opportunities_count": len(opportunities),
            "estimated_total_effort": sum(float(max(1, 12 - int(o.priority))) for o in opportunities),
            "steps": steps,
            "created_at": datetime.now().isoformat(),
            "automation_ready_count": len([o for o in opportunities if o.automation_confidence > 0.7]),
        }

    def export_knowledge_base(self, filepath: str) -> None:
        """
        Export the knowledge base to a JSON file.

        Fix: Enum keys in dicts are converted to strings recursively, otherwise json.dump fails.
        """
        export_data = asdict(self.knowledge_base)

        def to_jsonable(obj: Any) -> Any:
            if isinstance(obj, dict):
                new: Dict[Any, Any] = {}
                for k, v in obj.items():
                    kk = to_jsonable(k)
                    # JSON requires string keys
                    if not isinstance(kk, str):
                        kk = str(kk)
                    new[kk] = to_jsonable(v)
                return new
            if isinstance(obj, list):
                return [to_jsonable(x) for x in obj]
            if isinstance(obj, Enum):
                return obj.value
            return obj

        export_data = to_jsonable(export_data)
        export_data["export_timestamp"] = datetime.now().isoformat()

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

        logger.info("Exported knowledge base to %s", filepath)


def create_intelligent_refactoring_system(config: Optional[Dict[str, Any]] = None) -> IntelligentRefactoringSystem:
    """Create and initialize the intelligent refactoring system."""
    return IntelligentRefactoringSystem(config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    system = create_intelligent_refactoring_system()
    system.export_knowledge_base("intelligent_refactoring_knowledge_base.json")
    logger.info("Intelligent refactoring system foundation created successfully")
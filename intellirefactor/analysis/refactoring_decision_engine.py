"""
Refactoring Decision Engine with Multi-Criteria Analysis

This module implements a comprehensive decision engine that analyzes all available
data sources and makes intelligent refactoring recommendations with priority scoring,
confidence assessment, and step-by-step implementation plans.

Features:
- Multi-criteria analysis integrating all existing analysis components
- Rule-based decision making with configurable criteria weights
- Priority calculation by impact, confidence, effort, and architecture
- Step-by-step implementation plan generation
- Decision tracking with analysis_run_id and versioning
- Support for custom refactoring rules and decision criteria
- Machine-readable exports (JSON, YAML)
"""

import uuid
import json
from datetime import datetime
from typing import List, Dict, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import logging

# Optional YAML support
try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)

# Robust import handling for internal models
_MODELS_AVAILABLE = False
try:
    from .models import Evidence, FileReference
    from .audit_models import AuditResult, AuditFinding, AuditSeverity
    from .block_clone_detector import CloneGroup, CloneType, ExtractionStrategy
    from .architectural_smell_detector import (
        ArchitecturalSmell,
        SmellType,
        SmellSeverity,
    )
    from .unused_code_detector import UnusedCodeFinding, UnusedCodeType, ConfidenceLevel
    from .responsibility_clusterer import ClusteringResult, ResponsibilityCluster

    _MODELS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Internal models not available, running in standalone mode: {e}")
    # Define placeholders to prevent NameErrors during runtime checks
    Evidence = None
    FileReference = None
    AuditResult = None
    AuditFinding = None
    CloneGroup = None


# ============================================================================
# Helpers
# ============================================================================


def _norm_token(val: Any) -> str:
    """
    Normalize an enum or string value to a simple lowercase token.
    Handles cases like 'AuditSeverity.CRITICAL' -> 'critical'.
    """
    if val is None:
        return ""
    s = val.value if hasattr(val, "value") else str(val)
    s = s.lower().strip()
    # If string is like 'enumclass.member', take the last part
    if "." in s:
        s = s.split(".")[-1]
    return s


# ============================================================================
# Enums
# ============================================================================


class RefactoringType(Enum):
    """Types of refactoring decisions."""

    EXTRACT_METHOD = "extract_method"
    EXTRACT_CLASS = "extract_class"
    REMOVE_UNUSED_CODE = "remove_unused_code"
    DECOMPOSE_GOD_CLASS = "decompose_god_class"
    REDUCE_METHOD_COMPLEXITY = "reduce_method_complexity"
    ELIMINATE_DUPLICATES = "eliminate_duplicates"
    IMPROVE_COHESION = "improve_cohesion"
    REDUCE_COUPLING = "reduce_coupling"
    PARAMETERIZE_DUPLICATES = "parameterize_duplicates"
    TEMPLATE_METHOD_PATTERN = "template_method_pattern"


class RefactoringPriority(Enum):
    """Priority levels for refactoring decisions."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEFERRED = "deferred"

    def __lt__(self, other: "RefactoringPriority") -> bool:
        """Enable comparison for sorting."""
        if not isinstance(other, RefactoringPriority):
            return NotImplemented
        # Order from lowest to highest priority
        order = [
            RefactoringPriority.DEFERRED,
            RefactoringPriority.LOW,
            RefactoringPriority.MEDIUM,
            RefactoringPriority.HIGH,
            RefactoringPriority.CRITICAL,
        ]
        return order.index(self) < order.index(other)


class ImpactCategory(Enum):
    """Categories of refactoring impact."""

    MAINTAINABILITY = "maintainability"
    PERFORMANCE = "performance"
    READABILITY = "readability"
    TESTABILITY = "testability"
    REUSABILITY = "reusability"
    SECURITY = "security"
    ARCHITECTURE = "architecture"


# ============================================================================
# Data Classes
# ============================================================================


@dataclass
class DecisionCriteria:
    """Configurable criteria for decision making."""

    # Criteria weights (must sum to 1.0)
    code_quality_weight: float = 0.30
    performance_weight: float = 0.20
    architecture_weight: float = 0.25
    risk_weight: float = 0.15
    effort_weight: float = 0.10

    # Thresholds
    min_confidence_threshold: float = 0.7
    min_impact_threshold: float = 0.5
    max_effort_threshold: float = 0.8

    # Priority calculation parameters
    critical_priority_threshold: float = 0.9
    high_priority_threshold: float = 0.7
    medium_priority_threshold: float = 0.5

    def __post_init__(self):
        """Validate and normalize criteria weights."""
        total_weight = (
            self.code_quality_weight
            + self.performance_weight
            + self.architecture_weight
            + self.risk_weight
            + self.effort_weight
        )

        if abs(total_weight - 1.0) > 0.01:
            if total_weight > 0:
                logger.warning(f"Criteria weights sum to {total_weight:.3f}, normalizing to 1.0.")
                self.code_quality_weight /= total_weight
                self.performance_weight /= total_weight
                self.architecture_weight /= total_weight
                self.risk_weight /= total_weight
                self.effort_weight /= total_weight
            else:
                raise ValueError("Total weight cannot be zero.")

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class ImpactAssessment:
    """Assessment of refactoring impact."""

    category: ImpactCategory
    score: float  # 0.0 to 1.0
    description: str
    quantified_benefits: List[str] = field(default_factory=list)
    evidence: Optional[Any] = None

    def __post_init__(self):
        # Clamp score to valid range
        if self.score < 0.0:
            self.score = 0.0
        elif self.score > 1.0:
            self.score = 1.0

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "category": self.category.value,
            "score": self.score,
            "description": self.description,
            "quantified_benefits": self.quantified_benefits,
        }
        if self.evidence:
            result["evidence"] = (
                self.evidence.to_dict() if hasattr(self.evidence, "to_dict") else str(self.evidence)
            )
        return result


@dataclass
class FeasibilityAnalysis:
    """Analysis of refactoring feasibility."""

    effort_score: float = 0.5  # 0.0 to 1.0 (higher = more effort)
    risk_score: float = 0.5  # 0.0 to 1.0 (higher = more risk)
    complexity_score: float = 0.5  # 0.0 to 1.0 (higher = more complex)

    # Risk factors
    breaking_changes_risk: bool = False
    test_coverage_risk: bool = False
    dependency_risk: bool = False

    # Effort estimation
    estimated_hours: Optional[float] = None
    prerequisite_count: int = 0

    def __post_init__(self):
        for score_name in ["effort_score", "risk_score", "complexity_score"]:
            val = getattr(self, score_name)
            if val < 0.0:
                setattr(self, score_name, 0.0)
            elif val > 1.0:
                setattr(self, score_name, 1.0)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class ImplementationStep:
    """A single step in the implementation plan."""

    step_number: int
    title: str
    description: str
    estimated_time: str = "30 minutes"
    prerequisites: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    automation_possible: bool = False
    risk_level: str = "low"  # low, medium, high

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def _create_default_feasibility() -> FeasibilityAnalysis:
    return FeasibilityAnalysis()


@dataclass
class RefactoringDecision:
    """A refactoring decision with complete analysis."""

    # Basic identification
    refactoring_type: RefactoringType
    priority: RefactoringPriority = RefactoringPriority.MEDIUM
    confidence: float = 0.5  # 0.0 to 1.0

    # Target information
    target_files: List[str] = field(default_factory=list)
    target_symbols: List[str] = field(default_factory=list)

    # Analysis results
    impact_assessments: List[ImpactAssessment] = field(default_factory=list)
    feasibility: Optional[FeasibilityAnalysis] = field(default_factory=_create_default_feasibility)

    # Decision rationale
    title: str = ""
    description: str = ""
    rationale: str = ""
    evidence: List[Any] = field(default_factory=list)

    # Implementation guidance
    implementation_plan: List[ImplementationStep] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)

    # Tracking information
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    analysis_run_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.confidence < 0.0:
            self.confidence = 0.0
        elif self.confidence > 1.0:
            self.confidence = 1.0

        if not self.analysis_run_id:
            self.analysis_run_id = str(uuid.uuid4())

        if self.feasibility is None:
            self.feasibility = FeasibilityAnalysis()

        # Deduplicate target_files while preserving order
        seen = set()
        unique_files = []
        for f in self.target_files:
            if f and f not in seen:
                seen.add(f)
                unique_files.append(f)
        self.target_files = unique_files

    def get_overall_impact_score(self) -> float:
        """Calculate overall impact score from all assessments."""
        if not self.impact_assessments:
            return 0.0
        return sum(assessment.score for assessment in self.impact_assessments) / len(
            self.impact_assessments
        )

    def get_impact_score_by_category(self, category: ImpactCategory) -> float:
        """Get average impact score for a specific category."""
        assessments = [a for a in self.impact_assessments if a.category == category]
        if not assessments:
            return 0.0
        return sum(a.score for a in assessments) / len(assessments)

    def get_risk_adjusted_priority_score(self) -> float:
        """Calculate risk-adjusted priority score."""
        impact_score = self.get_overall_impact_score()
        # Feasibility is guaranteed not None by __post_init__
        risk_adjustment = 1.0 - (self.feasibility.risk_score * 0.3)
        return impact_score * self.confidence * risk_adjustment

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "decision_id": self.decision_id,
            "refactoring_type": self.refactoring_type.value,
            "priority": self.priority.value,
            "confidence": self.confidence,
            "target_files": self.target_files,
            "target_symbols": self.target_symbols,
            "impact_assessments": [a.to_dict() for a in self.impact_assessments],
            "feasibility": self.feasibility.to_dict() if self.feasibility else None,
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "evidence": [(e.to_dict() if hasattr(e, "to_dict") else str(e)) for e in self.evidence],
            "implementation_plan": [step.to_dict() for step in self.implementation_plan],
            "prerequisites": self.prerequisites,
            "analysis_run_id": self.analysis_run_id,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "metadata": self.metadata,
        }


@dataclass
class DecisionAnalysisResult:
    """Result of comprehensive decision analysis."""

    project_path: str
    analysis_run_id: str
    decisions: List[RefactoringDecision]
    criteria_used: DecisionCriteria
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def get_decisions_by_priority(self, priority: RefactoringPriority) -> List[RefactoringDecision]:
        return [d for d in self.decisions if d.priority == priority]

    def get_high_confidence_decisions(
        self, min_confidence: float = 0.8
    ) -> List[RefactoringDecision]:
        return [d for d in self.decisions if d.confidence >= min_confidence]

    def get_decisions_by_type(self, refactoring_type: RefactoringType) -> List[RefactoringDecision]:
        return [d for d in self.decisions if d.refactoring_type == refactoring_type]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_path": self.project_path,
            "analysis_run_id": self.analysis_run_id,
            "decisions": [d.to_dict() for d in self.decisions],
            "criteria_used": self.criteria_used.to_dict(),
            "analysis_metadata": self.analysis_metadata,
            "created_at": self.created_at.isoformat(),
        }


# ============================================================================
# Decision Engine
# ============================================================================

# Type alias for custom rule functions
CustomRuleFunc = Callable[[Any, str], List[RefactoringDecision]]


class RefactoringDecisionEngine:
    """
    Multi-criteria decision engine for refactoring recommendations.
    """

    # Mapping for clone extraction strategies to refactoring types
    CLONE_STRATEGY_MAP: Dict[str, RefactoringType] = {
        "extract_method": RefactoringType.EXTRACT_METHOD,
        "extract_class": RefactoringType.EXTRACT_CLASS,
        "parameterize": RefactoringType.PARAMETERIZE_DUPLICATES,
        "template_method": RefactoringType.TEMPLATE_METHOD_PATTERN,
    }

    # Mapping for smell types to refactoring types
    SMELL_REFACTORING_MAP: Dict[str, RefactoringType] = {
        "god_class": RefactoringType.DECOMPOSE_GOD_CLASS,
        "long_method": RefactoringType.EXTRACT_METHOD,
        "high_complexity": RefactoringType.REDUCE_METHOD_COMPLEXITY,
        "srp_violation": RefactoringType.EXTRACT_CLASS,
        "feature_envy": RefactoringType.REDUCE_COUPLING,
        "inappropriate_intimacy": RefactoringType.REDUCE_COUPLING,
    }

    def __init__(self, criteria: Optional[DecisionCriteria] = None):
        """Initialize decision engine with criteria."""
        self.criteria = criteria or DecisionCriteria()
        self.custom_rules: List[CustomRuleFunc] = []
        self.decision_history: List[DecisionAnalysisResult] = []

        logger.info("RefactoringDecisionEngine initialized")

    def add_custom_rule(self, rule_function: CustomRuleFunc) -> None:
        """Add a custom decision rule function."""
        self.custom_rules.append(rule_function)
        logger.info(f"Added custom rule: {getattr(rule_function, '__name__', 'anonymous')}")

    def analyze_project(self, audit_result: Any) -> DecisionAnalysisResult:
        """
        Analyze project audit results and generate refactoring decisions.
        """
        analysis_run_id = str(uuid.uuid4())
        project_path = getattr(audit_result, "project_path", "unknown")

        logger.info(f"Starting decision analysis for project: {project_path}")

        decisions: List[RefactoringDecision] = []

        # 1. Analyze clone groups
        try:
            clone_groups = getattr(audit_result, "clone_groups", [])
            if clone_groups:
                decisions.extend(self._analyze_clone_groups(clone_groups, analysis_run_id))
        except Exception as e:
            logger.error(f"Error analyzing clone groups: {e}", exc_info=True)

        # 2. Analyze architectural smells
        smell_findings = []
        try:
            findings = getattr(audit_result, "findings", [])
            smell_types = {
                "god_class",
                "long_method",
                "high_complexity",
                "srp_violation",
                "feature_envy",
                "inappropriate_intimacy",
            }

            for f in findings:
                ftype = getattr(f, "finding_type", None)
                val = _norm_token(ftype)
                if val in smell_types:
                    smell_findings.append(f)

            if smell_findings:
                decisions.extend(
                    self._analyze_architectural_smells(smell_findings, analysis_run_id)
                )
        except Exception as e:
            logger.error(f"Error analyzing architectural smells: {e}", exc_info=True)

        # 3. Analyze unused code
        try:
            unused_result = getattr(audit_result, "unused_result", None)
            if unused_result:
                decisions.extend(self._analyze_unused_code(unused_result, analysis_run_id))
        except Exception as e:
            logger.error(f"Error analyzing unused code: {e}", exc_info=True)

        # 4. Apply custom rules
        for rule in self.custom_rules:
            try:
                custom_decisions = rule(audit_result, analysis_run_id)
                if custom_decisions:
                    decisions.extend(custom_decisions)
            except Exception as e:
                logger.warning(f"Custom rule failed: {e}", exc_info=True)

        # 5. Calculate priorities and filter
        decisions = self._calculate_priorities(decisions)
        decisions = self._filter_by_criteria(decisions)

        # Sort by priority score
        decisions.sort(key=lambda d: d.get_risk_adjusted_priority_score(), reverse=True)

        result = DecisionAnalysisResult(
            project_path=project_path,
            analysis_run_id=analysis_run_id,
            decisions=decisions,
            criteria_used=self.criteria,
            analysis_metadata={
                "total_decisions": len(decisions),
                "analysis_sources": {
                    "clone_groups": len(getattr(audit_result, "clone_groups", [])),
                    "smell_findings": len(smell_findings),
                    "unused_findings": len(
                        getattr(getattr(audit_result, "unused_result", None), "findings", [])
                    ),
                },
            },
        )

        self.decision_history.append(result)
        logger.info(f"Decision analysis completed: {len(decisions)} decisions generated")

        return result

    def _analyze_clone_groups(
        self, clone_groups: List[Any], analysis_run_id: str
    ) -> List[RefactoringDecision]:
        """Analyze clone groups and generate duplicate elimination decisions."""
        decisions = []

        for clone_group in clone_groups:
            instances = getattr(clone_group, "instances", [])
            if len(instances) < 2:
                continue

            # Determine refactoring type
            refactoring_type = self._determine_clone_refactoring_type(clone_group)

            # Collect target files safely
            target_files = []
            for inst in instances:
                path = getattr(inst, "file_path", None)
                if path:
                    target_files.append(path)

            # Calculate metrics
            total_lines = getattr(clone_group, "total_lines", len(instances) * 5)
            confidence = getattr(clone_group, "confidence", 0.8)
            clone_type_val = _norm_token(getattr(clone_group, "clone_type", None))

            # Impact Assessment
            impact_assessments = [
                ImpactAssessment(
                    category=ImpactCategory.MAINTAINABILITY,
                    score=min(0.9, 0.3 + len(instances) * 0.1),
                    description=f"Eliminating {len(instances)} duplicate code instances",
                    quantified_benefits=[
                        f"Reduces code duplication by ~{total_lines} lines",
                        f"Improves maintainability across {len(set(target_files))} files",
                    ],
                ),
                ImpactAssessment(
                    category=ImpactCategory.READABILITY,
                    score=0.6,
                    description="Improves code readability by reducing repetition",
                ),
            ]

            # Feasibility
            feasibility = FeasibilityAnalysis(
                effort_score=min(0.8, 0.2 + len(instances) * 0.1),
                risk_score=0.3 if clone_type_val == "exact" else 0.5,
                complexity_score=0.4,
                test_coverage_risk=True,
                estimated_hours=max(0.5, len(instances) * 0.5),
            )

            # Implementation Plan
            implementation_plan = self._create_clone_elimination_plan(clone_group)

            # Evidence Generation
            evidence_list = []
            try:
                if _MODELS_AVAILABLE and Evidence is not None:
                    # Try to construct proper evidence if possible
                    file_refs = []
                    for inst in instances:
                        if hasattr(inst, "block_info") and hasattr(
                            inst.block_info, "file_reference"
                        ):
                            file_refs.append(inst.block_info.file_reference)
                        elif hasattr(inst, "file_path") and hasattr(inst, "line_start"):
                            # Fallback to creating a FileReference if available
                            try:
                                file_refs.append(
                                    FileReference(
                                        file_path=inst.file_path,
                                        line_start=inst.line_start,
                                        line_end=getattr(inst, "line_end", inst.line_start),
                                    )
                                )
                            except:
                                pass

                    evidence_list.append(
                        Evidence(
                            description=f"Clone group with {len(instances)} instances",
                            confidence=confidence,
                            file_references=file_refs,
                            metadata={
                                "clone_type": clone_type_val,
                                "total_lines": total_lines,
                                "similarity_score": getattr(clone_group, "similarity_score", 0.0),
                            },
                        )
                    )
                else:
                    # Synthetic evidence if models are missing
                    raise ImportError("Models unavailable")
            except Exception as e:
                logger.warning(
                    f"Could not generate full evidence for clone group: {e}",
                    exc_info=True,
                )
                # Fallback synthetic evidence
                evidence_list.append(
                    {
                        "description": f"Clone group with {len(instances)} instances",
                        "confidence": confidence,
                        "metadata": {
                            "clone_type": clone_type_val,
                            "total_lines": total_lines,
                        },
                    }
                )

            decision = RefactoringDecision(
                refactoring_type=refactoring_type,
                priority=RefactoringPriority.MEDIUM,
                confidence=confidence,
                target_files=target_files,
                impact_assessments=impact_assessments,
                feasibility=feasibility,
                title=f"Eliminate {clone_type_val} code clones",
                description=f"Extract common functionality from {len(instances)} duplicate code blocks",
                rationale=f"Found {len(instances)} similar code blocks that can be consolidated",
                evidence=evidence_list,
                implementation_plan=implementation_plan,
                analysis_run_id=analysis_run_id,
                metadata={"clone_group_id": getattr(clone_group, "group_id", str(uuid.uuid4()))},
            )

            decisions.append(decision)

        return decisions

    def _analyze_architectural_smells(
        self, smell_findings: List[Any], analysis_run_id: str
    ) -> List[RefactoringDecision]:
        """Analyze architectural smells and generate refactoring decisions."""
        decisions = []

        for finding in smell_findings:
            try:
                ftype = getattr(finding, "finding_type", None)
                finding_type_val = _norm_token(ftype)

                severity = getattr(finding, "severity", None)
                severity_val = _norm_token(severity)

                refactoring_type = self.SMELL_REFACTORING_MAP.get(
                    finding_type_val, RefactoringType.IMPROVE_COHESION
                )

                # Impact calculation
                impact_score = 0.5
                if severity_val == "critical":
                    impact_score = 0.9
                elif severity_val == "high":
                    impact_score = 0.7
                elif severity_val == "low":
                    impact_score = 0.3

                impact_assessments = [
                    ImpactAssessment(
                        category=ImpactCategory.MAINTAINABILITY,
                        score=impact_score,
                        description=f"Addressing {finding_type_val} smell",
                        quantified_benefits=getattr(finding, "recommendations", []),
                    ),
                    ImpactAssessment(
                        category=ImpactCategory.ARCHITECTURE,
                        score=impact_score * 0.8,
                        description="Improves overall architecture quality",
                    ),
                ]

                # Feasibility
                feasibility = self._calculate_smell_feasibility(finding)

                # Plan
                implementation_plan = self._create_smell_remediation_plan(finding)

                # Evidence
                evidence = []
                finding_evidence = getattr(finding, "evidence", None)
                if finding_evidence:
                    evidence = (
                        [finding_evidence]
                        if not isinstance(finding_evidence, list)
                        else finding_evidence
                    )
                else:
                    # Synthetic evidence fallback
                    evidence = [
                        {
                            "description": f"Detected {finding_type_val} smell",
                            "severity": severity_val,
                            "file_path": getattr(finding, "file_path", ""),
                            "confidence": getattr(finding, "confidence", 0.7),
                        }
                    ]

                file_path = getattr(finding, "file_path", "")

                decision = RefactoringDecision(
                    refactoring_type=refactoring_type,
                    priority=RefactoringPriority.MEDIUM,
                    confidence=getattr(finding, "confidence", 0.7),
                    target_files=[file_path] if file_path else [],
                    impact_assessments=impact_assessments,
                    feasibility=feasibility,
                    title=f"Address {finding_type_val}",
                    description=getattr(finding, "description", ""),
                    rationale=f"Detected {finding_type_val} with {severity_val} severity",
                    evidence=evidence,
                    implementation_plan=implementation_plan,
                    analysis_run_id=analysis_run_id,
                    metadata={"finding_id": getattr(finding, "finding_id", "")},
                )

                decisions.append(decision)
            except Exception as e:
                logger.warning(f"Error processing smell finding: {e}", exc_info=True)
                continue

        return decisions

    def _analyze_unused_code(
        self, unused_result: Any, analysis_run_id: str
    ) -> List[RefactoringDecision]:
        """Analyze unused code findings and generate removal decisions."""
        decisions = []
        findings = getattr(unused_result, "findings", [])

        for finding in findings:
            try:
                # Check confidence
                conf_level = getattr(finding, "confidence_level", None)
                conf_val = _norm_token(conf_level)

                if conf_val != "high":
                    continue

                file_path = getattr(finding, "file_path", "")
                symbol_name = getattr(finding, "symbol_name", "unknown")

                impact_assessments = [
                    ImpactAssessment(
                        category=ImpactCategory.MAINTAINABILITY,
                        score=0.6,
                        description="Removes unused code",
                        quantified_benefits=[
                            "Reduces cognitive load",
                            "Decreases codebase size",
                        ],
                    )
                ]

                feasibility = FeasibilityAnalysis(
                    effort_score=0.2,
                    risk_score=0.2,
                    complexity_score=0.2,
                    test_coverage_risk=False,
                    estimated_hours=0.5,
                )

                implementation_plan = [
                    ImplementationStep(
                        1,
                        "Verify unused code",
                        "Double-check for dynamic usage",
                        "15 minutes",
                    ),
                    ImplementationStep(
                        2,
                        "Remove code",
                        f"Remove {symbol_name}",
                        "15 minutes",
                        automation_possible=True,
                    ),
                ]

                evidence = []
                finding_evidence = getattr(finding, "evidence", None)
                if finding_evidence:
                    evidence = [finding_evidence]

                decision = RefactoringDecision(
                    refactoring_type=RefactoringType.REMOVE_UNUSED_CODE,
                    priority=RefactoringPriority.LOW,
                    confidence=getattr(finding, "confidence", 0.9),
                    target_files=[file_path] if file_path else [],
                    target_symbols=[symbol_name],
                    impact_assessments=impact_assessments,
                    feasibility=feasibility,
                    title=f"Remove unused code: {symbol_name}",
                    description=f"Remove unused symbol {symbol_name}",
                    rationale="Code analysis indicates this symbol is not used.",
                    evidence=evidence,
                    implementation_plan=implementation_plan,
                    analysis_run_id=analysis_run_id,
                )

                decisions.append(decision)
            except Exception as e:
                logger.warning(f"Error processing unused finding: {e}", exc_info=True)
                continue

        return decisions

    def _determine_clone_refactoring_type(self, clone_group: Any) -> RefactoringType:
        strategy = getattr(clone_group, "extraction_strategy", None)
        val = _norm_token(strategy)

        for key, rtype in self.CLONE_STRATEGY_MAP.items():
            if key in val:
                return rtype
        return RefactoringType.ELIMINATE_DUPLICATES

    def _calculate_smell_feasibility(self, finding: Any) -> FeasibilityAnalysis:
        severity = getattr(finding, "severity", None)
        sev_val = _norm_token(severity)

        effort = 0.5
        if sev_val == "critical":
            effort = 0.8
        elif sev_val == "low":
            effort = 0.2

        return FeasibilityAnalysis(
            effort_score=effort,
            risk_score=0.5 if sev_val in ("critical", "high") else 0.3,
            complexity_score=effort,
            estimated_hours=effort * 8,
        )

    def _create_clone_elimination_plan(self, clone_group: Any) -> List[ImplementationStep]:
        """Create an implementation plan for clone elimination."""
        # Use clone type to customize the plan slightly
        clone_type = _norm_token(getattr(clone_group, "clone_type", ""))

        return [
            ImplementationStep(
                1, "Analyze clones", f"Review {clone_type} variations", "30 minutes"
            ),
            ImplementationStep(
                2,
                "Design extraction",
                "Define new method/class signature",
                "45 minutes",
            ),
            ImplementationStep(
                3,
                "Extract",
                "Create shared component",
                "1 hour",
                automation_possible=True,
            ),
            ImplementationStep(
                4,
                "Replace",
                "Update call sites",
                "45 minutes",
                automation_possible=True,
            ),
        ]

    def _create_smell_remediation_plan(self, finding: Any) -> List[ImplementationStep]:
        """Create an implementation plan for smell remediation."""
        finding_type = _norm_token(getattr(finding, "finding_type", ""))

        if finding_type == "god_class":
            return [
                ImplementationStep(
                    1, "Identify responsibilities", "Group related methods", "1 hour"
                ),
                ImplementationStep(2, "Extract classes", "Create focused components", "2 hours"),
                ImplementationStep(3, "Update dependencies", "Refactor call sites", "1 hour"),
            ]
        elif finding_type == "long_method":
            return [
                ImplementationStep(
                    1, "Find logical blocks", "Identify cohesive sections", "30 minutes"
                ),
                ImplementationStep(2, "Extract methods", "Create helper methods", "45 minutes"),
            ]

        return [
            ImplementationStep(
                1, "Analyze smell", f"Understand impact of {finding_type}", "30 minutes"
            ),
            ImplementationStep(2, "Refactor", "Apply pattern", "1 hour"),
        ]

    def _calculate_priorities(
        self, decisions: List[RefactoringDecision]
    ) -> List[RefactoringDecision]:
        for decision in decisions:
            score = self._calculate_priority_score(decision)

            # Assign priority based on score and confidence
            if decision.confidence < 0.5 and decision.get_overall_impact_score() > 0.6:
                # High impact but low confidence -> Deferred
                decision.priority = RefactoringPriority.DEFERRED
            elif score >= self.criteria.critical_priority_threshold:
                decision.priority = RefactoringPriority.CRITICAL
            elif score >= self.criteria.high_priority_threshold:
                decision.priority = RefactoringPriority.HIGH
            elif score >= self.criteria.medium_priority_threshold:
                decision.priority = RefactoringPriority.MEDIUM
            else:
                decision.priority = RefactoringPriority.LOW
        return decisions

    def _calculate_priority_score(self, decision: RefactoringDecision) -> float:
        # Weighted sum of impacts
        impact = decision.get_overall_impact_score()

        # Specific category boosts
        perf_score = decision.get_impact_score_by_category(ImpactCategory.PERFORMANCE)
        arch_score = decision.get_impact_score_by_category(ImpactCategory.ARCHITECTURE)

        # Risk/Effort penalty (inverse)
        risk_factor = 1.0 - decision.feasibility.risk_score
        effort_factor = 1.0 - decision.feasibility.effort_score

        # Combine using criteria weights
        raw_score = (
            impact * self.criteria.code_quality_weight
            + perf_score * self.criteria.performance_weight
            + arch_score * self.criteria.architecture_weight
            + risk_factor * self.criteria.risk_weight
            + effort_factor * self.criteria.effort_weight
        )

        return raw_score * decision.confidence

    def _filter_by_criteria(
        self, decisions: List[RefactoringDecision]
    ) -> List[RefactoringDecision]:
        filtered = []
        for d in decisions:
            if d.confidence < self.criteria.min_confidence_threshold:
                continue
            if d.get_overall_impact_score() < self.criteria.min_impact_threshold:
                continue
            if d.feasibility.effort_score > self.criteria.max_effort_threshold:
                continue
            filtered.append(d)
        return filtered

    def export_decisions(self, result: DecisionAnalysisResult, format: str = "json") -> str:
        data = result.to_dict()
        if format.lower() == "json":
            return json.dumps(data, indent=2, default=str)
        elif format.lower() == "yaml":
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML not installed")
            return yaml.safe_dump(data, default_flow_style=False, sort_keys=False)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def save_decisions_to_file(
        self, result: DecisionAnalysisResult, path: str, format: str = "json"
    ) -> None:
        content = self.export_decisions(result, format)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def clear_history(self) -> None:
        self.decision_history.clear()

    def get_decision_history(self) -> List[DecisionAnalysisResult]:
        """Get history of all decision analyses."""
        return self.decision_history.copy()

    def get_decision_by_id(self, decision_id: str) -> Optional[RefactoringDecision]:
        """Retrieve a specific decision by its ID from history."""
        for result in self.decision_history:
            for decision in result.decisions:
                if decision.decision_id == decision_id:
                    return decision
        return None

    def track_decision_success(self, decision_id: str, success_metrics: Dict[str, Any]) -> bool:
        """Track success metrics for a decision implementation."""
        for result in self.decision_history:
            for decision in result.decisions:
                if decision.decision_id == decision_id:
                    decision.metadata["success_metrics"] = success_metrics
                    decision.metadata["tracked_at"] = datetime.now().isoformat()
                    return True
        return False

    def get_learning_data(self) -> List[Dict[str, Any]]:
        """Extract data for future ML training."""
        data = []
        for res in self.decision_history:
            for dec in res.decisions:
                data.append(
                    {
                        "type": dec.refactoring_type.value,
                        "confidence": dec.confidence,
                        "impact": dec.get_overall_impact_score(),
                        "feasibility": dec.feasibility.to_dict(),
                    }
                )
        return data

"""
Refactoring Decision Engine with Multi-Criteria Analysis.

This module analyzes aggregated results (e.g. AuditResult) and produces a list of
refactoring decisions with priorities, confidence, and implementation steps.

NOTE: This file replaces a previously broken/non-valid Python version.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

try:  # optional dependency
    import yaml  # type: ignore
    YAML_AVAILABLE = True
except Exception:  # pragma: no cover
    yaml = None
    YAML_AVAILABLE = False

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def _norm_token(val: Any) -> str:
    """Normalize enum/string to lowercase token (e.g. AuditSeverity.CRITICAL -> critical)."""
    if val is None:
        return ""
    s = val.value if hasattr(val, "value") else str(val)
    s = s.lower().strip()
    if "." in s:
        s = s.split(".")[-1]
    return s


def _safe_get(obj: Any, attr: str, default: Any = None) -> Any:
    try:
        return getattr(obj, attr, default)
    except Exception:
        return default


def _location_dict(file_path: str, line_start: Optional[int] = None, line_end: Optional[int] = None) -> Dict[str, Any]:
    return {"file_path": file_path, "line_start": line_start, "line_end": line_end}


def _evidence_dict(
    description: str,
    confidence: float,
    locations: Optional[List[Dict[str, Any]]] = None,
    code_snippets: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "description": description,
        "confidence": _clamp01(confidence),
        "locations": locations or [],
        "code_snippets": code_snippets or [],
        "metadata": metadata or {},
    }


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------

class RefactoringType(Enum):
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
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEFERRED = "deferred"

    def __lt__(self, other: "RefactoringPriority") -> bool:
        if not isinstance(other, RefactoringPriority):
            return NotImplemented
        order = [
            RefactoringPriority.DEFERRED,
            RefactoringPriority.LOW,
            RefactoringPriority.MEDIUM,
            RefactoringPriority.HIGH,
            RefactoringPriority.CRITICAL,
        ]
        return order.index(self) < order.index(other)


class ImpactCategory(Enum):
    MAINTAINABILITY = "maintainability"
    PERFORMANCE = "performance"
    READABILITY = "readability"
    TESTABILITY = "testability"
    REUSABILITY = "reusability"
    SECURITY = "security"
    ARCHITECTURE = "architecture"


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------

@dataclass
class DecisionCriteria:
    # weights (auto-normalized)
    code_quality_weight: float = 0.30
    performance_weight: float = 0.20
    architecture_weight: float = 0.25
    risk_weight: float = 0.15
    effort_weight: float = 0.10

    # thresholds
    min_confidence_threshold: float = 0.7
    min_impact_threshold: float = 0.5
    max_effort_threshold: float = 0.8

    # priority thresholds
    critical_priority_threshold: float = 0.9
    high_priority_threshold: float = 0.7
    medium_priority_threshold: float = 0.5

    def __post_init__(self) -> None:
        total = (
            self.code_quality_weight
            + self.performance_weight
            + self.architecture_weight
            + self.risk_weight
            + self.effort_weight
        )
        if abs(total - 1.0) > 0.01:
            if total <= 0:
                raise ValueError("Total weight cannot be zero.")
            logger.warning("Criteria weights sum to %.3f, normalizing.", total)
            self.code_quality_weight /= total
            self.performance_weight /= total
            self.architecture_weight /= total
            self.risk_weight /= total
            self.effort_weight /= total

    def to_dict(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class ImpactAssessment:
    category: ImpactCategory
    score: float
    description: str
    quantified_benefits: List[str] = field(default_factory=list)
    evidence: Optional[Any] = None

    def __post_init__(self) -> None:
        self.score = _clamp01(self.score)

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "category": self.category.value,
            "score": self.score,
            "description": self.description,
            "quantified_benefits": self.quantified_benefits,
        }
        if self.evidence is not None:
            out["evidence"] = self.evidence.to_dict() if hasattr(self.evidence, "to_dict") else self.evidence
        return out


@dataclass
class FeasibilityAnalysis:
    effort_score: float = 0.5
    risk_score: float = 0.5
    complexity_score: float = 0.5

    breaking_changes_risk: bool = False
    test_coverage_risk: bool = False
    dependency_risk: bool = False

    estimated_hours: Optional[float] = None
    prerequisite_count: int = 0

    def __post_init__(self) -> None:
        self.effort_score = _clamp01(self.effort_score)
        self.risk_score = _clamp01(self.risk_score)
        self.complexity_score = _clamp01(self.complexity_score)

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


@dataclass
class ImplementationStep:
    step_number: int
    title: str
    description: str
    estimated_time: str = "30 minutes"
    prerequisites: List[str] = field(default_factory=list)
    validation_criteria: List[str] = field(default_factory=list)
    automation_possible: bool = False
    risk_level: str = "low"

    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


def _create_default_feasibility() -> FeasibilityAnalysis:
    return FeasibilityAnalysis()


@dataclass
class RefactoringDecision:
    refactoring_type: RefactoringType
    priority: RefactoringPriority = RefactoringPriority.MEDIUM
    confidence: float = 0.5

    target_files: List[str] = field(default_factory=list)
    target_symbols: List[str] = field(default_factory=list)

    impact_assessments: List[ImpactAssessment] = field(default_factory=list)
    feasibility: FeasibilityAnalysis = field(default_factory=_create_default_feasibility)

    title: str = ""
    description: str = ""
    rationale: str = ""
    evidence: List[Any] = field(default_factory=list)

    implementation_plan: List[ImplementationStep] = field(default_factory=list)
    prerequisites: List[str] = field(default_factory=list)

    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    analysis_run_id: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.confidence = _clamp01(self.confidence)
        if not self.analysis_run_id:
            self.analysis_run_id = str(uuid.uuid4())

        # dedup target_files
        seen: set[str] = set()
        uniq: List[str] = []
        for f in self.target_files:
            if f and f not in seen:
                seen.add(f)
                uniq.append(f)
        self.target_files = uniq

    def get_overall_impact_score(self) -> float:
        if not self.impact_assessments:
            return 0.0
        return sum(a.score for a in self.impact_assessments) / len(self.impact_assessments)

    def get_impact_score_by_category(self, category: ImpactCategory) -> float:
        xs = [a.score for a in self.impact_assessments if a.category == category]
        return (sum(xs) / len(xs)) if xs else 0.0

    def get_risk_adjusted_priority_score(self) -> float:
        impact = self.get_overall_impact_score()
        risk_adj = 1.0 - (self.feasibility.risk_score * 0.3)
        return impact * self.confidence * risk_adj

    def to_dict(self) -> Dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "refactoring_type": self.refactoring_type.value,
            "priority": self.priority.value,
            "confidence": self.confidence,
            "target_files": self.target_files,
            "target_symbols": self.target_symbols,
            "impact_assessments": [a.to_dict() for a in self.impact_assessments],
            "feasibility": self.feasibility.to_dict(),
            "title": self.title,
            "description": self.description,
            "rationale": self.rationale,
            "evidence": [(e.to_dict() if hasattr(e, "to_dict") else e) for e in self.evidence],
            "implementation_plan": [s.to_dict() for s in self.implementation_plan],
            "prerequisites": self.prerequisites,
            "analysis_run_id": self.analysis_run_id,
            "created_at": self.created_at.isoformat(),
            "version": self.version,
            "metadata": self.metadata,
        }


@dataclass
class DecisionAnalysisResult:
    project_path: str
    analysis_run_id: str
    decisions: List[RefactoringDecision]
    criteria_used: DecisionCriteria
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def get_decisions_by_priority(self, priority: RefactoringPriority) -> List[RefactoringDecision]:
        return [d for d in self.decisions if d.priority == priority]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "project_path": self.project_path,
            "analysis_run_id": self.analysis_run_id,
            "decisions": [d.to_dict() for d in self.decisions],
            "criteria_used": self.criteria_used.to_dict(),
            "analysis_metadata": self.analysis_metadata,
            "created_at": self.created_at.isoformat(),
        }


CustomRuleFunc = Callable[[Any, str], List[RefactoringDecision]]


class RefactoringDecisionEngine:
    """
    Multi-criteria decision engine for refactoring recommendations.
    """

    CLONE_STRATEGY_MAP: Dict[str, RefactoringType] = {
        "extract_method": RefactoringType.EXTRACT_METHOD,
        "extract_function": RefactoringType.EXTRACT_METHOD,
        "extract_class": RefactoringType.EXTRACT_CLASS,
        "parameterize": RefactoringType.PARAMETERIZE_DUPLICATES,
        "template_method": RefactoringType.TEMPLATE_METHOD_PATTERN,
    }

    SMELL_REFACTORING_MAP: Dict[str, RefactoringType] = {
        "god_class": RefactoringType.DECOMPOSE_GOD_CLASS,
        "long_method": RefactoringType.EXTRACT_METHOD,
        "high_complexity": RefactoringType.REDUCE_METHOD_COMPLEXITY,
        "srp_violation": RefactoringType.EXTRACT_CLASS,
        "feature_envy": RefactoringType.REDUCE_COUPLING,
        "inappropriate_intimacy": RefactoringType.REDUCE_COUPLING,
    }

    def __init__(self, criteria: Optional[DecisionCriteria] = None) -> None:
        self.criteria = criteria or DecisionCriteria()
        self.custom_rules: List[CustomRuleFunc] = []
        self.decision_history: List[DecisionAnalysisResult] = []
        logger.info("RefactoringDecisionEngine initialized")

    def add_custom_rule(self, rule_function: CustomRuleFunc) -> None:
        self.custom_rules.append(rule_function)
        logger.info("Added custom rule: %s", getattr(rule_function, "__name__", "anonymous"))

    def analyze_project(self, audit_result: Any) -> DecisionAnalysisResult:
        analysis_run_id = str(uuid.uuid4())
        project_path = _safe_get(audit_result, "project_path", "unknown")

        decisions: List[RefactoringDecision] = []

        # 1) Clone groups
        try:
            clone_groups = _safe_get(audit_result, "clone_groups", []) or []
            if clone_groups:
                decisions.extend(self._analyze_clone_groups(clone_groups, analysis_run_id))
        except Exception as e:
            logger.error("Error analyzing clone groups: %s", e, exc_info=True)

        # 2) Smells (by finding_type/type token)
        smell_findings: List[Any] = []
        try:
            findings = _safe_get(audit_result, "findings", []) or []
            smell_types = {
                "god_class",
                "long_method",
                "high_complexity",
                "srp_violation",
                "feature_envy",
                "inappropriate_intimacy",
            }
            for f in findings:
                ftype = _safe_get(f, "finding_type", None) or _safe_get(f, "type", None)
                if _norm_token(ftype) in smell_types:
                    smell_findings.append(f)
            if smell_findings:
                decisions.extend(self._analyze_architectural_smells(smell_findings, analysis_run_id))
        except Exception as e:
            logger.error("Error analyzing architectural smells: %s", e, exc_info=True)

        # 3) Unused code
        try:
            unused_result = _safe_get(audit_result, "unused_result", None)
            if unused_result:
                decisions.extend(self._analyze_unused_code(unused_result, analysis_run_id))
        except Exception as e:
            logger.error("Error analyzing unused code: %s", e, exc_info=True)

        # 4) Custom rules
        for rule in self.custom_rules:
            try:
                extra = rule(audit_result, analysis_run_id)
                if extra:
                    decisions.extend(extra)
            except Exception as e:
                logger.warning("Custom rule failed: %s", e, exc_info=True)

        # 5) Score + filter
        decisions = self._calculate_priorities(decisions)
        decisions = self._filter_by_criteria(decisions)
        decisions.sort(key=lambda d: d.get_risk_adjusted_priority_score(), reverse=True)

        res = DecisionAnalysisResult(
            project_path=str(project_path),
            analysis_run_id=analysis_run_id,
            decisions=decisions,
            criteria_used=self.criteria,
            analysis_metadata={
                "total_decisions": len(decisions),
                "analysis_sources": {
                    "clone_groups": len(_safe_get(audit_result, "clone_groups", []) or []),
                    "smell_findings": len(smell_findings),
                    "unused_findings": len(_safe_get(_safe_get(audit_result, "unused_result", None), "findings", []) or []),
                },
            },
        )
        self.decision_history.append(res)
        return res

    # -----------------------------------------------------------------
    # Clone groups
    # -----------------------------------------------------------------
    def _analyze_clone_groups(self, clone_groups: List[Any], analysis_run_id: str) -> List[RefactoringDecision]:
        decisions: List[RefactoringDecision] = []
        for g in clone_groups:
            instances = _safe_get(g, "instances", []) or []
            if len(instances) < 2:
                continue

            extraction_strategy = _norm_token(_safe_get(g, "extraction_strategy", None))
            extraction_confidence = float(_safe_get(g, "extraction_confidence", 0.0) or 0.0)
            if extraction_strategy in ("no_extraction", "") and extraction_confidence < 0.5:
                continue

            similarity = float(_safe_get(g, "similarity_score", 0.0) or 0.0)
            confidence = _clamp01(max(similarity, extraction_confidence, 0.5))

            target_files: List[str] = []
            locs: List[Dict[str, Any]] = []
            total_loc = 0
            for inst in instances:
                fp = _safe_get(inst, "file_path", None)
                ls = _safe_get(inst, "line_start", None)
                le = _safe_get(inst, "line_end", None)
                if fp:
                    target_files.append(fp)
                    locs.append(_location_dict(fp, ls, le))
                bi = _safe_get(inst, "block_info", None)
                if bi is not None:
                    total_loc += int(_safe_get(bi, "lines_of_code", 0) or 0)

            unique_files = len(set(target_files)) if target_files else 0
            clone_type_val = _norm_token(_safe_get(g, "clone_type", None))
            ref_type = self._determine_clone_refactoring_type(g)

            impact = [
                ImpactAssessment(
                    category=ImpactCategory.MAINTAINABILITY,
                    score=_clamp01(0.3 + len(instances) * 0.1),
                    description=f"Eliminating {len(instances)} duplicate code instances",
                    quantified_benefits=[
                        f"Files affected: {unique_files}",
                        f"Estimated duplicated LOC (sum instances): {total_loc}",
                    ],
                )
            ]
            feasibility = FeasibilityAnalysis(
                effort_score=_clamp01(0.2 + len(instances) * 0.1),
                risk_score=0.3 if clone_type_val == "exact" else 0.5,
                complexity_score=0.4,
                test_coverage_risk=True,
                estimated_hours=max(0.5, len(instances) * 0.5),
            )

            evidence_list = [
                _evidence_dict(
                    description=f"Clone group with {len(instances)} instances",
                    confidence=confidence,
                    locations=locs[:10],
                    metadata={
                        "clone_type": clone_type_val,
                        "similarity_score": similarity,
                        "extraction_strategy": extraction_strategy,
                        "extraction_confidence": extraction_confidence,
                        "group_id": _safe_get(g, "group_id", None),
                    },
                )
            ]

            decisions.append(
                RefactoringDecision(
                    refactoring_type=ref_type,
                    priority=RefactoringPriority.MEDIUM,
                    confidence=confidence,
                    target_files=target_files,
                    impact_assessments=impact,
                    feasibility=feasibility,
                    title=f"Eliminate {clone_type_val or 'duplicate'} clones",
                    description=f"Consolidate {len(instances)} similar code blocks",
                    rationale="Detected clone group suitable for consolidation",
                    evidence=evidence_list,
                    implementation_plan=self._create_clone_elimination_plan(g),
                    analysis_run_id=analysis_run_id,
                )
            )

        return decisions

    def _determine_clone_refactoring_type(self, clone_group: Any) -> RefactoringType:
        strategy = _norm_token(_safe_get(clone_group, "extraction_strategy", None))
        for key, rtype in self.CLONE_STRATEGY_MAP.items():
            if key in strategy:
                return rtype
        return RefactoringType.ELIMINATE_DUPLICATES if False else RefactoringType.ELIMINATE_DUPLICATES

    # -----------------------------------------------------------------
    # Smells
    # -----------------------------------------------------------------
    def _analyze_architectural_smells(self, smell_findings: List[Any], analysis_run_id: str) -> List[RefactoringDecision]:
        decisions: List[RefactoringDecision] = []
        for f in smell_findings:
            try:
                ftype = _safe_get(f, "finding_type", None) or _safe_get(f, "type", None)
                ftok = _norm_token(ftype)
                sev = _norm_token(_safe_get(f, "severity", None))

                rtype = self.SMELL_REFACTORING_MAP.get(ftok, RefactoringType.IMPROVE_COHESION)
                conf = float(_safe_get(f, "confidence", 0.7) or 0.7)

                impact_score = 0.5
                if sev == "critical":
                    impact_score = 0.9
                elif sev == "high":
                    impact_score = 0.7
                elif sev == "low":
                    impact_score = 0.3

                impact = [
                    ImpactAssessment(
                        category=ImpactCategory.MAINTAINABILITY,
                        score=impact_score,
                        description=f"Addressing {ftok} smell",
                        quantified_benefits=_safe_get(f, "recommendations", []) or [],
                    )
                ]

                file_path = _safe_get(f, "file_path", "") or ""
                line = _safe_get(f, "line_start", None) or _safe_get(f, "line", None)

                evidence_list: List[Any] = []
                existing_evidence = _safe_get(f, "evidence", None)
                if existing_evidence:
                    evidence_list = existing_evidence if isinstance(existing_evidence, list) else [existing_evidence]
                else:
                    evidence_list = [
                        _evidence_dict(
                            description=f"Detected smell: {ftok}",
                            confidence=conf,
                            locations=[_location_dict(file_path, line, line)] if file_path else [],
                            metadata={"severity": sev},
                        )
                    ]

                decisions.append(
                    RefactoringDecision(
                        refactoring_type=rtype,
                        priority=RefactoringPriority.MEDIUM,
                        confidence=_clamp01(conf),
                        target_files=[file_path] if file_path else [],
                        impact_assessments=impact,
                        feasibility=self._calculate_smell_feasibility(f),
                        title=f"Address {ftok}",
                        description=str(_safe_get(f, "description", "") or ""),
                        rationale=f"Detected {ftok} ({sev})",
                        evidence=evidence_list,
                        implementation_plan=self._create_smell_remediation_plan(f),
                        analysis_run_id=analysis_run_id,
                    )
                )
            except Exception as e:
                logger.warning("Error processing smell finding: %s", e, exc_info=True)
        return decisions

    def _calculate_smell_feasibility(self, finding: Any) -> FeasibilityAnalysis:
        sev = _norm_token(_safe_get(finding, "severity", None))
        effort = 0.5
        if sev == "critical":
            effort = 0.8
        elif sev == "low":
            effort = 0.2
        return FeasibilityAnalysis(
            effort_score=effort,
            risk_score=0.5 if sev in ("critical", "high") else 0.3,
            complexity_score=effort,
            estimated_hours=effort * 8,
        )

    # -----------------------------------------------------------------
    # Unused code
    # -----------------------------------------------------------------
    def _analyze_unused_code(self, unused_result: Any, analysis_run_id: str) -> List[RefactoringDecision]:
        decisions: List[RefactoringDecision] = []
        findings = _safe_get(unused_result, "findings", []) or []
        for f in findings:
            try:
                conf_level = _safe_get(f, "confidence_level", None)
                if _norm_token(conf_level) and _norm_token(conf_level) != "high":
                    continue

                file_path = _safe_get(f, "file_path", "") or ""
                symbol_name = _safe_get(f, "symbol_name", "unknown") or "unknown"
                conf = float(_safe_get(f, "confidence", 0.9) or 0.9)

                impact = [
                    ImpactAssessment(
                        category=ImpactCategory.MAINTAINABILITY,
                        score=0.6,
                        description="Removes unused code",
                        quantified_benefits=["Reduces cognitive load", "Decreases codebase size"],
                    )
                ]
                feasibility = FeasibilityAnalysis(effort_score=0.2, risk_score=0.2, complexity_score=0.2, estimated_hours=0.5)
                plan = [
                    ImplementationStep(1, "Verify unused code", "Double-check for dynamic usage", "15 minutes"),
                    ImplementationStep(2, "Remove code", f"Remove {symbol_name}", "15 minutes", automation_possible=True),
                ]

                existing_evidence = _safe_get(f, "evidence", None)
                evidence = [existing_evidence] if existing_evidence else [
                    _evidence_dict(
                        description=f"Unused symbol: {symbol_name}",
                        confidence=conf,
                        locations=[_location_dict(file_path)] if file_path else [],
                    )
                ]

                decisions.append(
                    RefactoringDecision(
                        refactoring_type=RefactoringType.REMOVE_UNUSED_CODE,
                        priority=RefactoringPriority.LOW,
                        confidence=_clamp01(conf),
                        target_files=[file_path] if file_path else [],
                        target_symbols=[symbol_name],
                        impact_assessments=impact,
                        feasibility=feasibility,
                        title=f"Remove unused code: {symbol_name}",
                        description=f"Remove unused symbol {symbol_name}",
                        rationale="High confidence unused code finding",
                        evidence=evidence,
                        implementation_plan=plan,
                        analysis_run_id=analysis_run_id,
                    )
                )
            except Exception as e:
                logger.warning("Error processing unused finding: %s", e, exc_info=True)
        return decisions

    # -----------------------------------------------------------------
    # Plans
    # -----------------------------------------------------------------
    def _create_clone_elimination_plan(self, clone_group: Any) -> List[ImplementationStep]:
        clone_type = _norm_token(_safe_get(clone_group, "clone_type", ""))
        return [
            ImplementationStep(1, "Analyze clones", f"Review {clone_type} variations", "30 minutes"),
            ImplementationStep(2, "Design extraction", "Define new method/class signature", "45 minutes"),
            ImplementationStep(3, "Extract", "Create shared component", "1 hour", automation_possible=True),
            ImplementationStep(4, "Replace", "Update call sites", "45 minutes", automation_possible=True),
        ]

    def _create_smell_remediation_plan(self, finding: Any) -> List[ImplementationStep]:
        ft = _norm_token(_safe_get(finding, "finding_type", "") or _safe_get(finding, "type", ""))
        if ft == "god_class":
            return [
                ImplementationStep(1, "Identify responsibilities", "Group related methods", "1 hour"),
                ImplementationStep(2, "Extract classes", "Create focused components", "2 hours"),
                ImplementationStep(3, "Update dependencies", "Refactor call sites", "1 hour"),
            ]
        if ft == "long_method":
            return [
                ImplementationStep(1, "Find logical blocks", "Identify cohesive sections", "30 minutes"),
                ImplementationStep(2, "Extract methods", "Create helper methods", "45 minutes"),
            ]
        return [
            ImplementationStep(1, "Analyze smell", f"Understand impact of {ft}", "30 minutes"),
            ImplementationStep(2, "Refactor", "Apply appropriate refactoring/pattern", "1 hour"),
        ]

    # -----------------------------------------------------------------
    # Scoring / filtering / export
    # -----------------------------------------------------------------
    def _calculate_priorities(self, decisions: List[RefactoringDecision]) -> List[RefactoringDecision]:
        for d in decisions:
            score = self._calculate_priority_score(d)
            if d.confidence < 0.5 and d.get_overall_impact_score() > 0.6:
                d.priority = RefactoringPriority.DEFERRED
            elif score >= self.criteria.critical_priority_threshold:
                d.priority = RefactoringPriority.CRITICAL
            elif score >= self.criteria.high_priority_threshold:
                d.priority = RefactoringPriority.HIGH
            elif score >= self.criteria.medium_priority_threshold:
                d.priority = RefactoringPriority.MEDIUM
            else:
                d.priority = RefactoringPriority.LOW
        return decisions

    def _calculate_priority_score(self, d: RefactoringDecision) -> float:
        impact = d.get_overall_impact_score()
        perf = d.get_impact_score_by_category(ImpactCategory.PERFORMANCE)
        arch = d.get_impact_score_by_category(ImpactCategory.ARCHITECTURE)
        risk = 1.0 - d.feasibility.risk_score
        effort = 1.0 - d.feasibility.effort_score
        raw = (
            impact * self.criteria.code_quality_weight
            + perf * self.criteria.performance_weight
            + arch * self.criteria.architecture_weight
            + risk * self.criteria.risk_weight
            + effort * self.criteria.effort_weight
        )
        return _clamp01(raw * d.confidence)

    def _filter_by_criteria(self, decisions: List[RefactoringDecision]) -> List[RefactoringDecision]:
        out: List[RefactoringDecision] = []
        for d in decisions:
            if d.confidence < self.criteria.min_confidence_threshold:
                continue
            if d.get_overall_impact_score() < self.criteria.min_impact_threshold:
                continue
            if d.feasibility.effort_score > self.criteria.max_effort_threshold:
                continue
            out.append(d)
        return out

    def export_decisions(self, result: DecisionAnalysisResult, format: str = "json") -> str:
        data = result.to_dict()
        fmt = format.lower().strip()
        if fmt == "json":
            return json.dumps(data, indent=2, default=str, ensure_ascii=False)
        if fmt == "yaml":
            if not YAML_AVAILABLE:
                raise ImportError("PyYAML not installed")
            assert yaml is not None
            return yaml.safe_dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
        raise ValueError(f"Unsupported format: {format}")

    def save_decisions_to_file(self, result: DecisionAnalysisResult, path: str, format: str = "json") -> None:
        content = self.export_decisions(result, format=format)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def clear_history(self) -> None:
        self.decision_history.clear()

    def get_decision_history(self) -> List[DecisionAnalysisResult]:
        return self.decision_history.copy()

    def get_decision_by_id(self, decision_id: str) -> Optional[RefactoringDecision]:
        for res in self.decision_history:
            for dec in res.decisions:
                if dec.decision_id == decision_id:
                    return dec
        return None

    def track_decision_success(self, decision_id: str, success_metrics: Dict[str, Any]) -> bool:
        dec = self.get_decision_by_id(decision_id)
        if not dec:
            return False
        dec.metadata["success_metrics"] = success_metrics
        dec.metadata["tracked_at"] = datetime.now().isoformat()
        return True

    def get_learning_data(self) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for res in self.decision_history:
            for d in res.decisions:
                out.append(
                    {
                        "type": d.refactoring_type.value,
                        "confidence": d.confidence,
                        "impact": d.get_overall_impact_score(),
                        "feasibility": d.feasibility.to_dict(),
                    }
                )
        return out
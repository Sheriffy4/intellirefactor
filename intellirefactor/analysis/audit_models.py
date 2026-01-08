"""
Audit models for IntelliRefactor

Shared data models for audit engine and specification generator.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum

from .models import Evidence


class AuditSeverity(Enum):
    """Severity levels for audit findings."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class AuditFindingType(Enum):
    """Types of audit findings."""

    DUPLICATE_BLOCK = "duplicate_block"
    DUPLICATE_METHOD = "duplicate_method"
    UNUSED_CODE = "unused_code"
    INDEX_ISSUE = "index_issue"
    PERFORMANCE_ISSUE = "performance_issue"
    QUALITY_ISSUE = "quality_issue"


@dataclass
class AuditFinding:
    """Represents a single audit finding with evidence."""

    finding_id: str
    finding_type: AuditFindingType
    severity: AuditSeverity
    title: str
    description: str
    file_path: str
    line_start: int
    line_end: int
    confidence: float
    evidence: Evidence
    recommendations: List[str] = field(default_factory=list)
    related_findings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "finding_id": self.finding_id,
            "finding_type": self.finding_type.value,
            "severity": self.severity.value,
            "title": self.title,
            "description": self.description,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "confidence": self.confidence,
            "evidence": self.evidence.to_dict(),
            "recommendations": self.recommendations,
            "related_findings": self.related_findings,
            "metadata": self.metadata,
        }


@dataclass
class AuditStatistics:
    """Statistics for audit results."""

    total_findings: int
    findings_by_severity: Dict[str, int]
    findings_by_type: Dict[str, int]
    files_analyzed: int
    analysis_time_seconds: float
    confidence_distribution: Dict[str, int]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "total_findings": self.total_findings,
            "findings_by_severity": self.findings_by_severity,
            "findings_by_type": self.findings_by_type,
            "files_analyzed": self.files_analyzed,
            "analysis_time_seconds": self.analysis_time_seconds,
            "confidence_distribution": self.confidence_distribution,
        }


@dataclass
class AuditResult:
    """Complete audit results with findings and metadata."""

    project_path: str
    findings: List[AuditFinding]
    statistics: AuditStatistics
    index_result: Optional[Any]  # IndexBuildResult
    unused_result: Optional[Any]  # UnusedCodeAnalysisResult
    clone_groups: List[Any]  # CloneGroup objects
    analysis_metadata: Dict[str, Any]

    def get_findings_by_severity(self, severity: AuditSeverity) -> List[AuditFinding]:
        """Get findings filtered by severity."""
        return [f for f in self.findings if f.severity == severity]

    def get_findings_by_type(self, finding_type: AuditFindingType) -> List[AuditFinding]:
        """Get findings filtered by type."""
        return [f for f in self.findings if f.finding_type == finding_type]

    def get_critical_findings(self) -> List[AuditFinding]:
        """Get critical severity findings."""
        return self.get_findings_by_severity(AuditSeverity.CRITICAL)

    def get_high_confidence_findings(self, min_confidence: float = 0.8) -> List[AuditFinding]:
        """Get high confidence findings."""
        return [f for f in self.findings if f.confidence >= min_confidence]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "project_path": self.project_path,
            "findings": [f.to_dict() for f in self.findings],
            "statistics": self.statistics.to_dict(),
            "index_result": (
                self.index_result.to_dict()
                if self.index_result and hasattr(self.index_result, "to_dict")
                else None
            ),
            "unused_result": (
                self.unused_result.to_dict()
                if self.unused_result and hasattr(self.unused_result, "to_dict")
                else None
            ),
            "clone_groups": [
                g.to_dict() if hasattr(g, "to_dict") else str(g) for g in self.clone_groups
            ],
            "analysis_metadata": self.analysis_metadata,
        }

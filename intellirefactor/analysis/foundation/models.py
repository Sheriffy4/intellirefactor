"""
Unified foundation models for IntelliRefactor.

This module provides core data models used as a contract between analyzers.

Fixes / additions:
- Evidence is now backward compatible:
  - supports both `locations: List[Location]` (new style)
  - and `file_references: List[FileReference]` (older style used by clone detector / decision engine)
- Added FileReference, BlockType, BlockInfo for clone detection workflows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


def parse_severity(value: Any, default: Severity = Severity.MEDIUM) -> Severity:
    """
    Normalize severity coming from different subsystems:
    - foundation.models.Severity
    - workflows.audit_models.AuditSeverity (alias or separate Enum)
    - debug_cycle_manager.ErrorSeverity (alias or separate Enum)
    - analysis.error_handler.ErrorSeverity (string values)
    - raw strings like "high"/"HIGH"/"Critical"
    """
    if value is None:
        return default

    if isinstance(value, Severity):
        return value

    # Other Enum types: try .value or .name
    if hasattr(value, "value"):
        try:
            return Severity(str(getattr(value, "value")).lower())
        except Exception:
            pass
    if hasattr(value, "name"):
        try:
            return Severity(str(getattr(value, "name")).lower())
        except Exception:
            pass

    # Strings
    try:
        s = str(value).strip().lower()
        return Severity(s)
    except Exception:
        return default


class AnalysisStage(Enum):
    READ = "read"
    PARSE = "parse"
    ANALYZE = "analyze"
    INDEX = "index"
    REPORT = "report"


@dataclass(frozen=True)
class Location:
    file_path: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None

    def __str__(self) -> str:
        if self.line_start and self.line_end:
            return f"{self.file_path}:{self.line_start}-{self.line_end}"
        if self.line_start:
            return f"{self.file_path}:{self.line_start}"
        return self.file_path


@dataclass(frozen=True)
class FileReference:
    """
    Backward-compatible file reference used in some analyzers.

    Equivalent to Location but kept for compatibility with earlier modules.
    """
    file_path: str
    line_start: int
    line_end: int

    def to_location(self) -> Location:
        return Location(file_path=self.file_path, line_start=self.line_start, line_end=self.line_end)

    def to_dict(self) -> Dict[str, Any]:
        return {"file_path": self.file_path, "line_start": self.line_start, "line_end": self.line_end}


@dataclass
class Evidence:
    """
    Standard evidence structure with full backward compatibility.

    Supports both:
    - locations: List[Location] (new style)
    - file_references: List[FileReference] (legacy style)
    Both are normalized and available in to_dict()
    """
    description: str
    confidence: float  # 0.0 to 1.0
    # legacy-friendly: keep both representations available
    file_references: List[FileReference] = field(default_factory=list)
    locations: List[Location] = field(default_factory=list)
    code_snippets: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

        # normalize file_references -> locations
        if self.file_references:
            existing = {(l.file_path, l.line_start, l.line_end) for l in self.locations}
            for fr in self.file_references:
                loc = fr.to_location()
                key = (loc.file_path, loc.line_start, loc.line_end)
                if key not in existing:
                    self.locations.append(loc)
                    existing.add(key)

        # normalize locations -> file_references (when possible)
        if self.locations and not self.file_references:
            refs: List[FileReference] = []
            for l in self.locations:
                if l.line_start is not None and l.line_end is not None:
                    refs.append(FileReference(l.file_path, int(l.line_start), int(l.line_end)))
            self.file_references = refs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "description": self.description,
            "confidence": self.confidence,
            # Keep both keys for compatibility across old/new modules
            "file_references": [fr.to_dict() for fr in self.file_references],
            "locations": [
                {"file_path": l.file_path, "line_start": l.line_start, "line_end": l.line_end}
                for l in self.locations
            ],
            "code_snippets": self.code_snippets,
            "metadata": self.metadata,
        }


@dataclass
class Finding:
    id: str
    type: str
    severity: Severity
    confidence: float
    title: str
    description: str
    location: Location
    evidence: Evidence
    recommendations: List[str] = field(default_factory=list)
    related_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "title": self.title,
            "description": self.description,
            "location": {
                "file_path": self.location.file_path,
                "line_start": self.location.line_start,
                "line_end": self.location.line_end,
            },
            "evidence": self.evidence.to_dict(),
            "recommendations": self.recommendations,
            "related_ids": self.related_ids,
            "metadata": self.metadata,
        }


@dataclass
class AnalysisError:
    file_path: str
    stage: AnalysisStage
    error_type: str
    message: str
    line_number: Optional[int] = None
    traceback: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "file_path": self.file_path,
            "stage": self.stage.value,
            "error_type": self.error_type,
            "message": self.message,
            "line_number": self.line_number,
            "traceback": self.traceback,
            "metadata": self.metadata,
        }


@dataclass
class AnalysisStats:
    files_scanned: int = 0
    files_analyzed: int = 0
    errors_encountered: int = 0
    findings_count: int = 0
    analysis_duration: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "files_scanned": self.files_scanned,
            "files_analyzed": self.files_analyzed,
            "errors_encountered": self.errors_encountered,
            "findings_count": self.findings_count,
            "analysis_duration": self.analysis_duration,
            "metadata": self.metadata,
        }


@dataclass
class AnalysisReport:
    tool_name: str
    tool_version: str
    analysis_type: str
    project_root: str
    started_at: str
    completed_at: str
    stats: AnalysisStats
    findings: List[Finding] = field(default_factory=list)
    errors: List[AnalysisError] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "tool_version": self.tool_version,
            "analysis_type": self.analysis_type,
            "project_root": self.project_root,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "stats": self.stats.to_dict(),
            "findings": [f.to_dict() for f in self.findings],
            "errors": [e.to_dict() for e in self.errors],
            "metadata": self.metadata,
        }


# ----------------------------------------------------------------------
# Clone / block models (canonical contract used by dedup extractor/detector)
# ----------------------------------------------------------------------

class BlockType(Enum):
    """Types of code blocks for clone detection."""
    IF_BLOCK = "if"
    FOR_LOOP = "for"
    WHILE_LOOP = "while"
    TRY_BLOCK = "try"
    WITH_BLOCK = "with"
    STATEMENT_GROUP = "statement_group"
    FUNCTION_BODY = "function_body"
    METHOD_BODY = "method_body"
    CLASS_BODY = "class_body"


@dataclass
class BlockInfo:
    """
    Information about a code block for clone detection.
    Designed to match dedup.block_extractor / dedup.block_clone_detector usage.
    """
    block_type: BlockType
    file_reference: FileReference
    parent_method: Optional[str] = None

    ast_fingerprint: str = ""
    token_fingerprint: str = ""
    normalized_fingerprint: str = ""

    nesting_level: int = 0
    lines_of_code: int = 0
    statement_count: int = 0

    min_clone_size: int = 3
    is_extractable: bool = True

    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "block_type": self.block_type.value,
            "file_reference": self.file_reference.to_dict(),
            "parent_method": self.parent_method,
            "ast_fingerprint": self.ast_fingerprint,
            "token_fingerprint": self.token_fingerprint,
            "normalized_fingerprint": self.normalized_fingerprint,
            "nesting_level": self.nesting_level,
            "lines_of_code": self.lines_of_code,
            "statement_count": self.statement_count,
            "min_clone_size": self.min_clone_size,
            "is_extractable": self.is_extractable,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


# -----------------------------------------------------------------------
# Deep semantic / dependency models (migrated from legacy analysis.models)
# -----------------------------------------------------------------------

class SemanticCategory(Enum):
    VALIDATION = "validation"
    TRANSFORMATION = "transformation"
    CACHING = "caching"
    PERSISTENCE = "persistence"
    COMPUTATION = "computation"
    COORDINATION = "coordination"
    FACTORY = "factory"
    UTILITY = "utility"
    CONFIGURATION = "configuration"
    LOGGING = "logging"
    ERROR_HANDLING = "error_handling"
    UNKNOWN = "unknown"


class ResponsibilityMarker(Enum):
    DATA_ACCESS = "data_access"
    BUSINESS_LOGIC = "business_logic"
    VALIDATION = "validation"
    FORMATTING = "formatting"
    CACHING = "caching"
    LOGGING = "logging"
    ERROR_HANDLING = "error_handling"
    COORDINATION = "coordination"
    CONFIGURATION = "configuration"
    PERSISTENCE = "persistence"


class DependencyResolution(Enum):
    EXACT = "exact"
    PROBABLE = "probable"
    UNKNOWN = "unknown"


@dataclass
class DeepMethodInfo:
    name: str
    qualified_name: str
    file_reference: FileReference
    signature: str = ""

    ast_fingerprint: str = ""
    token_fingerprint: str = ""
    operation_signature: str = ""

    semantic_category: SemanticCategory = SemanticCategory.UNKNOWN
    responsibility_markers: set = field(default_factory=set)
    side_effects: set = field(default_factory=set)

    complexity_score: int = 0
    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0

    is_public: bool = True
    is_async: bool = False
    is_property: bool = False
    is_static: bool = False
    is_classmethod: bool = False

    calls_external: List[str] = field(default_factory=list)
    uses_attributes: List[str] = field(default_factory=list)
    imports_used: List[str] = field(default_factory=list)

    confidence: float = 1.0
    analysis_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "file_reference": self.file_reference.to_dict(),
            "signature": self.signature,
            "ast_fingerprint": self.ast_fingerprint,
            "token_fingerprint": self.token_fingerprint,
            "operation_signature": self.operation_signature,
            "semantic_category": self.semantic_category.value,
            "responsibility_markers": sorted([m.value for m in self.responsibility_markers]),
            "side_effects": sorted(list(self.side_effects)),
            "complexity_score": self.complexity_score,
            "lines_of_code": self.lines_of_code,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "is_public": self.is_public,
            "is_async": self.is_async,
            "is_property": self.is_property,
            "is_static": self.is_static,
            "is_classmethod": self.is_classmethod,
            "calls_external": list(self.calls_external),
            "uses_attributes": list(self.uses_attributes),
            "imports_used": list(self.imports_used),
            "confidence": self.confidence,
            "analysis_version": self.analysis_version,
            "metadata": self.metadata,
        }


@dataclass
class DeepClassInfo:
    name: str
    qualified_name: str
    file_reference: FileReference

    methods: List[str] = field(default_factory=list)
    attributes: List[str] = field(default_factory=list)
    class_attributes: List[str] = field(default_factory=list)

    base_classes: List[str] = field(default_factory=list)
    derived_classes: List[str] = field(default_factory=list)

    responsibility_markers: set = field(default_factory=set)
    cohesion_score: float = 0.0

    lines_of_code: int = 0
    method_count: int = 0
    attribute_count: int = 0
    complexity_score: int = 0

    is_abstract: bool = False
    is_interface: bool = False
    is_data_class: bool = False
    is_singleton: bool = False

    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not 0.0 <= self.cohesion_score <= 1.0:
            raise ValueError(f"Cohesion score must be between 0.0 and 1.0, got {self.cohesion_score}")


@dataclass
class DependencyInfo:
    source_symbol: str
    target_symbol: Optional[str]
    target_external: Optional[str]
    dependency_kind: str

    resolution: DependencyResolution = DependencyResolution.UNKNOWN
    confidence: float = 1.0
    evidence: Optional[Evidence] = None

    usage_count: int = 1
    usage_contexts: List[str] = field(default_factory=list)

    is_critical: bool = False
    is_circular: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.target_symbol is None and self.target_external is None:
            raise ValueError("Either target_symbol or target_external must be specified")


# -----------------------------------------------------------------------
# Extra models (required by some expert analyzers in legacy code)
# -----------------------------------------------------------------------

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class DependencyInterface:
    module_name: str
    used_methods: List[str] = field(default_factory=list)
    used_attributes: List[str] = field(default_factory=list)
    import_style: str = ""
    criticality: RiskLevel = RiskLevel.MEDIUM
    version_constraints: Optional[str] = None


@dataclass
class InterfaceUsage:
    total_dependencies: int
    critical_interfaces: List[DependencyInterface] = field(default_factory=list)
    unused_imports: List[str] = field(default_factory=list)
    potential_violations: List[str] = field(default_factory=list)


@dataclass
class MethodGroup:
    methods: List[str]
    shared_attributes: List[str]
    cohesion_score: float
    extraction_recommendation: str = ""


@dataclass
class CohesionMatrix:
    methods: List[str]
    attributes: List[str]
    matrix: List[List[float]]
    cohesion_scores: Dict[str, float] = field(default_factory=dict)
    suggested_groups: List[MethodGroup] = field(default_factory=list)


# -----------------------------------------------------------------------
# Parsers (compat helpers used by index/store/importers)
# -----------------------------------------------------------------------

def parse_semantic_category(category_str: Optional[str]) -> SemanticCategory:
    try:
        return SemanticCategory(str(category_str or "").lower())
    except Exception:
        return SemanticCategory.UNKNOWN


def parse_responsibility_markers(markers: Optional[List[str]]) -> set:
    out: set = set()
    for marker_str in markers or []:
        try:
            out.add(ResponsibilityMarker(str(marker_str).lower()))
        except Exception:
            continue
    return out


def parse_dependency_resolution(resolution_str: Optional[str]) -> DependencyResolution:
    try:
        return DependencyResolution(str(resolution_str or "").lower())
    except Exception:
        return DependencyResolution.UNKNOWN


def parse_block_type(block_type_str: Optional[str]) -> BlockType:
    """
    Parse block_type coming from multiple subsystems / historical versions.

    Canonical values (BlockType.value):
      if / for / while / try / with / statement_group / function_body / method_body / class_body

    Legacy values (previous index builder):
      if_block / for_loop / while_loop / try_block / with_block

    Also tolerated legacy shorthands used in some heuristics:
      function / method
    """
    s = str(block_type_str or "").strip().lower()
    if not s:
        return BlockType.STATEMENT_GROUP

    normalize_map = {
        # legacy index builder
        "if_block": "if",
        "for_loop": "for",
        "while_loop": "while",
        "try_block": "try",
        "with_block": "with",
        # older shorthands
        "function": "function_body",
        "method": "method_body",
    }
    s = normalize_map.get(s, s)

    try:
        return BlockType(s)
    except Exception:
        return BlockType.STATEMENT_GROUP

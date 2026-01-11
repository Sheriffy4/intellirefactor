"""
Core data models for IntelliRefactor analysis.

This module defines minimal dataclasses that represent the core entities
in the analysis system. These models focus on fingerprints, semantic categories,
and responsibility markers while avoiding storing heavy data like full source code.

Architecture principles:
1. Minimal models - store references to heavy data, not the data itself
2. Fingerprint-focused - enable fast duplicate detection and similarity matching
3. Evidence-based - all findings include concrete evidence and confidence scores
4. Semantic-aware - capture semantic categories and responsibility patterns
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set
from enum import Enum


class SemanticCategory(Enum):
    """Semantic categories for methods and functions."""

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


class BlockType(Enum):
    """Types of code blocks for clone detection."""

    IF_BLOCK = "if"
    FOR_LOOP = "for"
    WHILE_LOOP = "while"
    TRY_BLOCK = "try"
    WITH_BLOCK = "with"
    STATEMENT_GROUP = "statement_group"
    FUNCTION_BODY = "function_body"
    CLASS_BODY = "class_body"


class DependencyResolution(Enum):
    """Resolution levels for dependencies."""

    EXACT = "exact"  # Fully resolved to specific symbol
    PROBABLE = "probable"  # Likely resolved but uncertain
    UNKNOWN = "unknown"  # Cannot resolve


class ResponsibilityMarker(Enum):
    """Responsibility markers for SRP analysis."""

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


@dataclass(frozen=True)
class FileReference:
    """Reference to a file with line information."""

    file_path: str
    line_start: int
    line_end: int

    def __str__(self) -> str:
        return f"{self.file_path}:{self.line_start}-{self.line_end}"


@dataclass(frozen=True)
class Evidence:
    """Evidence supporting an analysis finding."""

    description: str
    confidence: float  # 0.0 to 1.0
    file_references: List[FileReference] = field(default_factory=list)
    code_snippets: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "description": self.description,
            "confidence": self.confidence,
            "file_references": [
                {
                    "file_path": ref.file_path,
                    "line_start": ref.line_start,
                    "line_end": ref.line_end,
                }
                for ref in self.file_references
            ],
            "code_snippets": self.code_snippets,
            "metadata": self.metadata,
        }


@dataclass
class DeepMethodInfo:
    """
    Comprehensive method information for deep analysis.

    Focuses on fingerprints, semantic categories, and responsibility markers
    rather than storing full source code.
    """

    # Basic identification
    name: str
    qualified_name: str
    file_reference: FileReference
    signature: str

    # Fingerprints for duplicate detection
    ast_fingerprint: str  # Structural fingerprint from AST
    token_fingerprint: str  # Token-based fingerprint
    operation_signature: str  # Sequence of operations (validate->transform->return)

    # Semantic analysis
    semantic_category: SemanticCategory
    responsibility_markers: Set[ResponsibilityMarker]
    side_effects: Set[str] = field(default_factory=set)  # file_io, network, state_modification

    # Metrics and complexity
    complexity_score: int = 0
    lines_of_code: int = 0
    cyclomatic_complexity: int = 0
    cognitive_complexity: int = 0

    # Method characteristics
    is_public: bool = True
    is_async: bool = False
    is_property: bool = False
    is_static: bool = False
    is_classmethod: bool = False

    # Dependencies (references, not full objects)
    calls_external: List[str] = field(default_factory=list)  # External method calls
    uses_attributes: List[str] = field(default_factory=list)  # self.* attributes used
    imports_used: List[str] = field(default_factory=list)  # Import statements used

    # Analysis metadata
    confidence: float = 1.0
    analysis_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def get_responsibility_score(self) -> float:
        """Calculate responsibility diversity score (higher = more responsibilities)."""
        # [IR_DELEGATED] Auto-generated wrapper (functional decomposition)
        from intellirefactor.unified.analysis import get_responsibility_score as __ir_unified_get_responsibility_score
        return __ir_unified_get_responsibility_score(self)

    def has_side_effects(self) -> bool:
        """Check if method has any side effects."""
        return len(self.side_effects) > 0

    def is_complex(self, threshold: int = 10) -> bool:
        """Check if method exceeds complexity threshold."""
        return self.cyclomatic_complexity > threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "file_reference": {
                "file_path": self.file_reference.file_path,
                "line_start": self.file_reference.line_start,
                "line_end": self.file_reference.line_end,
            },
            "signature": self.signature,
            "ast_fingerprint": self.ast_fingerprint,
            "token_fingerprint": self.token_fingerprint,
            "operation_signature": self.operation_signature,
            "semantic_category": self.semantic_category.value,
            "responsibility_markers": [marker.value for marker in self.responsibility_markers],
            "side_effects": list(self.side_effects),
            "complexity_score": self.complexity_score,
            "lines_of_code": self.lines_of_code,
            "cyclomatic_complexity": self.cyclomatic_complexity,
            "cognitive_complexity": self.cognitive_complexity,
            "is_public": self.is_public,
            "is_async": self.is_async,
            "is_property": self.is_property,
            "is_static": self.is_static,
            "is_classmethod": self.is_classmethod,
            "calls_external": self.calls_external,
            "uses_attributes": self.uses_attributes,
            "imports_used": self.imports_used,
            "confidence": self.confidence,
            "analysis_version": self.analysis_version,
            "metadata": self.metadata,
        }


@dataclass
class BlockInfo:
    """
    Information about a code block for clone detection.

    Stores block type, normalized fingerprint, and line references
    without storing full source code.
    """

    # Basic identification
    block_type: BlockType
    file_reference: FileReference
    parent_method: str  # Qualified name of containing method

    # Fingerprints for clone detection
    ast_fingerprint: str  # Structural fingerprint
    token_fingerprint: str  # Token-based fingerprint
    normalized_fingerprint: str  # Variable/literal normalized fingerprint

    # Block characteristics
    nesting_level: int = 0
    lines_of_code: int = 0
    statement_count: int = 0

    # Clone detection metadata
    min_clone_size: int = 3  # Minimum lines for clone detection
    is_extractable: bool = True  # Can this block be extracted as a method?

    # Analysis metadata
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def is_clone_candidate(self) -> bool:
        """Check if block is large enough for clone detection."""
        return self.lines_of_code >= self.min_clone_size

    def get_complexity_score(self) -> float:
        """Calculate block complexity based on nesting and statements."""
        return self.nesting_level * 2 + self.statement_count

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "block_type": self.block_type.value,
            "file_reference": {
                "file_path": self.file_reference.file_path,
                "line_start": self.file_reference.line_start,
                "line_end": self.file_reference.line_end,
            },
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


@dataclass
class DependencyInfo:
    """
    Information about dependencies between symbols.

    Includes resolution level, confidence, and evidence for Python's
    dynamic nature uncertainty.
    """

    # Basic dependency information
    source_symbol: str  # Qualified name of source symbol
    target_symbol: Optional[str]  # Qualified name of target symbol (if resolved)
    target_external: Optional[str]  # External target (module.function)
    dependency_kind: str  # calls, imports, inherits, uses_attr, instantiates

    # Resolution and confidence
    resolution: DependencyResolution
    confidence: float  # 0.0 to 1.0
    evidence: Evidence

    # Usage information
    usage_count: int = 1  # How many times this dependency occurs
    usage_contexts: List[str] = field(default_factory=list)  # Where it's used

    # Analysis metadata
    is_critical: bool = False  # Is this dependency critical for functionality?
    is_circular: bool = False  # Part of circular dependency?
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if self.target_symbol is None and self.target_external is None:
            raise ValueError("Either target_symbol or target_external must be specified")

    def is_resolved(self) -> bool:
        """Check if dependency is fully resolved."""
        return self.resolution == DependencyResolution.EXACT

    def is_external(self) -> bool:
        """Check if dependency is external to the project."""
        return self.target_external is not None

    def get_target_name(self) -> str:
        """Get the target name (internal or external)."""
        return self.target_symbol or self.target_external or "unknown"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_symbol": self.source_symbol,
            "target_symbol": self.target_symbol,
            "target_external": self.target_external,
            "dependency_kind": self.dependency_kind,
            "resolution": self.resolution.value,
            "confidence": self.confidence,
            "evidence": {
                "description": self.evidence.description,
                "confidence": self.evidence.confidence,
                "file_references": [
                    {
                        "file_path": ref.file_path,
                        "line_start": ref.line_start,
                        "line_end": ref.line_end,
                    }
                    for ref in self.evidence.file_references
                ],
                "code_snippets": self.evidence.code_snippets,
                "metadata": self.evidence.metadata,
            },
            "usage_count": self.usage_count,
            "usage_contexts": self.usage_contexts,
            "is_critical": self.is_critical,
            "is_circular": self.is_circular,
            "metadata": self.metadata,
        }


@dataclass
class DeepClassInfo:
    """
    Comprehensive class information for architectural analysis.

    Focuses on responsibility analysis and architectural smell detection.
    """

    # Basic identification
    name: str
    qualified_name: str
    file_reference: FileReference

    # Methods and attributes
    methods: List[str] = field(default_factory=list)  # Qualified names of methods
    attributes: List[str] = field(default_factory=list)  # Instance attributes
    class_attributes: List[str] = field(default_factory=list)  # Class attributes

    # Inheritance information
    base_classes: List[str] = field(default_factory=list)
    derived_classes: List[str] = field(default_factory=list)

    # Responsibility analysis
    responsibility_markers: Set[ResponsibilityMarker] = field(default_factory=set)
    cohesion_score: float = 0.0  # 0.0 to 1.0, higher is more cohesive

    # Metrics
    lines_of_code: int = 0
    method_count: int = 0
    attribute_count: int = 0
    complexity_score: int = 0

    # Architectural characteristics
    is_abstract: bool = False
    is_interface: bool = False
    is_data_class: bool = False
    is_singleton: bool = False

    # Analysis metadata
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not 0.0 <= self.cohesion_score <= 1.0:
            raise ValueError(
                f"Cohesion score must be between 0.0 and 1.0, got {self.cohesion_score}"
            )

    def get_responsibility_score(self) -> float:
        """Calculate responsibility diversity score (higher = more responsibilities)."""
        return len(self.responsibility_markers) / len(ResponsibilityMarker)

    def is_god_class(
        self,
        method_threshold: int = 15,
        responsibility_threshold: int = 3,
        cohesion_threshold: float = 0.5,
    ) -> bool:
        """Check if class exhibits God Class pattern."""
        return (
            self.method_count > method_threshold
            and len(self.responsibility_markers) > responsibility_threshold
            and self.cohesion_score < cohesion_threshold
        )

    def is_large_class(self, loc_threshold: int = 500, method_threshold: int = 20) -> bool:
        """Check if class is considered large."""
        return self.lines_of_code > loc_threshold or self.method_count > method_threshold

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "qualified_name": self.qualified_name,
            "file_reference": {
                "file_path": self.file_reference.file_path,
                "line_start": self.file_reference.line_start,
                "line_end": self.file_reference.line_end,
            },
            "methods": self.methods,
            "attributes": self.attributes,
            "class_attributes": self.class_attributes,
            "base_classes": self.base_classes,
            "derived_classes": self.derived_classes,
            "responsibility_markers": [marker.value for marker in self.responsibility_markers],
            "cohesion_score": self.cohesion_score,
            "lines_of_code": self.lines_of_code,
            "method_count": self.method_count,
            "attribute_count": self.attribute_count,
            "complexity_score": self.complexity_score,
            "is_abstract": self.is_abstract,
            "is_interface": self.is_interface,
            "is_data_class": self.is_data_class,
            "is_singleton": self.is_singleton,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


# Utility functions for working with models


def create_file_reference(file_path: str, line_start: int, line_end: int) -> FileReference:
    """Create a file reference with validation."""
    if line_start < 1:
        raise ValueError(f"Line start must be >= 1, got {line_start}")
    if line_end < line_start:
        raise ValueError(f"Line end ({line_end}) must be >= line start ({line_start})")

    return FileReference(file_path=file_path, line_start=line_start, line_end=line_end)


def create_evidence(
    description: str,
    confidence: float,
    file_references: Optional[List[FileReference]] = None,
    code_snippets: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Evidence:
    """Create evidence with validation."""
    return Evidence(
        description=description,
        confidence=confidence,
        file_references=file_references or [],
        code_snippets=code_snippets or [],
        metadata=metadata or {},
    )


def parse_semantic_category(category_str: str) -> SemanticCategory:
    """Parse semantic category from string."""
    try:
        return SemanticCategory(category_str.lower())
    except ValueError:
        return SemanticCategory.UNKNOWN


def parse_responsibility_markers(markers: List[str]) -> Set[ResponsibilityMarker]:
    """Parse responsibility markers from string list."""
    result = set()
    for marker_str in markers:
        try:
            result.add(ResponsibilityMarker(marker_str.lower()))
        except ValueError:
            continue  # Skip unknown markers
    return result


def parse_block_type(block_type_str: str) -> BlockType:
    """Parse block type from string."""
    try:
        return BlockType(block_type_str.lower())
    except ValueError:
        return BlockType.STATEMENT_GROUP


def parse_dependency_resolution(resolution_str: str) -> DependencyResolution:
    """Parse dependency resolution from string."""
    try:
        return DependencyResolution(resolution_str.lower())
    except ValueError:
        return DependencyResolution.UNKNOWN

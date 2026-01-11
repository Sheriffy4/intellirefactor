"""
Data models for expert refactoring analysis.

Defines the core data structures used throughout the expert analysis system.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class CallType(str, Enum):
    """Types of method calls."""
    DIRECT = "direct"
    INDIRECT = "indirect"
    RECURSIVE = "recursive"
    PROPERTY = "property"


class TestCategory(str, Enum):
    """Categories of characterization tests."""
    TYPICAL = "typical"
    EDGE_CASE = "edge_case"
    ERROR_CASE = "error_case"
    BOUNDARY = "boundary"


class RiskLevel(str, Enum):
    """Risk levels for refactoring operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CallNode:
    """Represents a method/function in the call graph."""
    method_name: str
    class_name: Optional[str] = None
    line_number: int = 0
    end_line_number: int = 0
    complexity: int = 0
    is_public: bool = True
    is_property: bool = False
    parameters: List[str] = field(default_factory=list)
    return_type: Optional[str] = None
    docstring: Optional[str] = None


@dataclass
class CallEdge:
    """Represents a call relationship between methods."""
    caller: str
    callee: str
    call_type: CallType = CallType.DIRECT
    frequency: int = 1
    line_number: int = 0
    context: Optional[str] = None


@dataclass
class Cycle:
    """Represents a cycle in the call graph."""
    nodes: List[str]
    cycle_type: str = "simple"  # simple, complex, self-recursive
    risk_level: RiskLevel = RiskLevel.MEDIUM


@dataclass
class ComplexityMetrics:
    """Complexity metrics for call graph."""
    cyclomatic_complexity: int = 0
    call_depth: int = 0
    fan_in: Dict[str, int] = field(default_factory=dict)
    fan_out: Dict[str, int] = field(default_factory=dict)
    coupling_score: float = 0.0
    cohesion_score: float = 0.0


@dataclass
class CallGraph:
    """Complete call graph representation."""
    nodes: List[CallNode] = field(default_factory=list)
    edges: List[CallEdge] = field(default_factory=list)
    cycles: List[Cycle] = field(default_factory=list)
    metrics: Optional[ComplexityMetrics] = None
    
    def get_node(self, method_name: str) -> Optional[CallNode]:
        """Get node by method name."""
        for node in self.nodes:
            if node.method_name == method_name:
                return node
        return None
    
    def get_callers(self, method_name: str) -> List[str]:
        """Get all methods that call the given method."""
        return [edge.caller for edge in self.edges if edge.callee == method_name]
    
    def get_callees(self, method_name: str) -> List[str]:
        """Get all methods called by the given method."""
        # [IR_DELEGATED] Auto-generated wrapper (functional decomposition)
        from intellirefactor.unified.analysis import expert_get_callers as __ir_unified_expert_get_callers
        return __ir_unified_expert_get_callers(self, method_name)


@dataclass
class ExternalCaller:
    """Represents external usage of the module."""
    file_path: str
    line_number: int
    import_statement: str
    used_symbols: List[str] = field(default_factory=list)
    usage_frequency: int = 1
    context: Optional[str] = None


@dataclass
class UsageAnalysis:
    """Analysis of how the module is used externally."""
    total_callers: int = 0
    most_used_symbols: List[Tuple[str, int]] = field(default_factory=list)
    usage_patterns: Dict[str, int] = field(default_factory=dict)
    critical_dependencies: List[str] = field(default_factory=list)


@dataclass
class ImpactAssessment:
    """Assessment of breaking change impact."""
    affected_files: List[str] = field(default_factory=list)
    risk_level: RiskLevel = RiskLevel.LOW
    breaking_changes: List[str] = field(default_factory=list)
    migration_effort: str = "low"  # low, medium, high
    recommendations: List[str] = field(default_factory=list)


@dataclass
class MethodGroup:
    """Group of methods with high cohesion."""
    methods: List[str]
    shared_attributes: List[str]
    cohesion_score: float
    extraction_recommendation: str
    suggested_class_name: Optional[str] = None


@dataclass
class CohesionMatrix:
    """Matrix showing method-attribute relationships."""
    methods: List[str]
    attributes: List[str]
    matrix: List[List[float]]  # method x attribute access matrix
    cohesion_scores: Dict[str, float] = field(default_factory=dict)
    suggested_groups: List[MethodGroup] = field(default_factory=list)
    
    def get_cohesion_score(self, method: str) -> float:
        """Get cohesion score for a method."""
        return self.cohesion_scores.get(method, 0.0)


@dataclass
class BehavioralContract:
    """Behavioral contract for a method."""
    method_name: str
    class_name: Optional[str] = None
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    invariants: List[str] = field(default_factory=list)
    side_effects: List[str] = field(default_factory=list)
    performance_constraints: List[str] = field(default_factory=list)
    error_conditions: List[str] = field(default_factory=list)


@dataclass
class DependencyInterface:
    """Interface of an external dependency."""
    module_name: str
    used_methods: List[str] = field(default_factory=list)
    used_attributes: List[str] = field(default_factory=list)
    import_style: str = "unknown"  # import, from_import, relative
    criticality: RiskLevel = RiskLevel.LOW
    version_constraints: Optional[str] = None


@dataclass
class InterfaceUsage:
    """Analysis of dependency interface usage."""
    total_dependencies: int = 0
    critical_interfaces: List[DependencyInterface] = field(default_factory=list)
    unused_imports: List[str] = field(default_factory=list)
    potential_violations: List[str] = field(default_factory=list)


@dataclass
class CharacterizationTest:
    """A characterization test case."""
    method_name: str
    class_name: Optional[str] = None
    input_params: Dict[str, Any] = field(default_factory=dict)
    expected_output: Any = None
    test_category: TestCategory = TestCategory.TYPICAL
    priority: int = 1
    description: str = ""
    setup_code: Optional[str] = None
    teardown_code: Optional[str] = None


@dataclass
class DiscoveryResult:
    """Result of test discovery analysis."""
    existing_test_files: List[str] = field(default_factory=list)
    coverage_analysis: Dict[str, float] = field(default_factory=dict)
    missing_tests: List[str] = field(default_factory=list)
    test_quality_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)


@dataclass
class DuplicateFragment:
    """A fragment of duplicated code."""
    content: str
    locations: List[Tuple[str, int, int]]  # (file, start_line, end_line)
    similarity_score: float
    extraction_suggestion: str
    estimated_savings: int  # lines of code


@dataclass
class GitChangePattern:
    """Pattern of changes in Git history."""
    files_changed_together: List[Tuple[str, str]]
    change_frequency: Dict[str, int] = field(default_factory=dict)
    hotspots: List[str] = field(default_factory=list)
    hidden_dependencies: List[Tuple[str, str]] = field(default_factory=list)


@dataclass
class ExpertAnalysisResult:
    """Complete result of expert refactoring analysis."""
    target_file: str
    timestamp: str
    
    # Core analyses
    call_graph: Optional[CallGraph] = None
    external_callers: List[ExternalCaller] = field(default_factory=list)
    usage_analysis: Optional[UsageAnalysis] = None
    cohesion_matrix: Optional[CohesionMatrix] = None
    behavioral_contracts: List[BehavioralContract] = field(default_factory=list)
    dependency_interfaces: List[DependencyInterface] = field(default_factory=list)
    
    # Test analysis
    test_discovery: Optional[DiscoveryResult] = None
    characterization_tests: List[CharacterizationTest] = field(default_factory=list)
    
    # Code quality
    duplicate_fragments: List[DuplicateFragment] = field(default_factory=list)
    git_patterns: Optional[GitChangePattern] = None
    
    # Impact assessment
    impact_assessment: Optional[ImpactAssessment] = None
    compatibility_constraints: List[str] = field(default_factory=list)
    
    # Summary metrics
    analysis_quality_score: float = 0.0
    recommendations: List[str] = field(default_factory=list)
    risk_assessment: RiskLevel = RiskLevel.LOW
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        # This would be implemented to handle dataclass serialization
        # For now, return a basic structure
        return {
            "target_file": self.target_file,
            "timestamp": self.timestamp,
            "analysis_quality_score": self.analysis_quality_score,
            "risk_assessment": self.risk_assessment.value,
            "recommendations": self.recommendations,
            # Add other fields as needed
        }
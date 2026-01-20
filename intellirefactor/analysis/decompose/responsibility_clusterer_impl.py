"""
Responsibility Clustering with Success Criteria

This module implements clustering of methods by shared responsibilities,
attributes, and dependencies to suggest cohesive component extraction
from God Classes and other large classes.

Features:
- Method clustering by self.* attribute access patterns
- External dependency analysis and semantic markers
- Community detection and agglomerative clustering algorithms
- Quality metrics: cluster sizes, cohesion scores, unclustered ratio
- High-confidence decisions with configurable thresholds
"""

import ast
from typing import List, Dict, Set, Optional
from dataclasses import dataclass, field
from enum import Enum

# Lazy imports for heavy data science libraries
def _import_networkx():
    """Lazy import of networkx."""
    global nx
    import networkx as nx
    return nx

def _import_sklearn():
    """Lazy import of sklearn components."""
    global AgglomerativeClustering, silhouette_score
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score
    return AgglomerativeClustering, silhouette_score

def _import_numpy():
    """Lazy import of numpy."""
    global np
    import numpy as np
    return np

from intellirefactor.analysis.foundation.models import Evidence


def _make_evidence(description: str, confidence: float, metadata: Optional[Dict] = None):
    """
    Evidence factory compatible with multiple Evidence signatures
    (old/new model layouts).
    """
    metadata = metadata or {}
    try:
        # Most common signature in your codebase (dedup)
        return Evidence(
            description=description,
            confidence=confidence,
            file_references=[],
            code_snippets=[],
            metadata=metadata,
        )
    except TypeError:
        # Fallback: minimal signature
        return Evidence(description=description, confidence=confidence)


class ClusteringAlgorithm(Enum):
    """Supported clustering algorithms."""

    COMMUNITY_DETECTION = "community_detection"
    AGGLOMERATIVE = "agglomerative"
    HYBRID = "hybrid"


class ClusterQuality(Enum):
    """Quality levels for clusters."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


@dataclass
class ClusteringConfig:
    """Configuration for responsibility clustering."""

    # Algorithm selection
    algorithm: ClusteringAlgorithm = ClusteringAlgorithm.HYBRID

    # Quality thresholds
    min_cluster_size: int = 3
    max_cluster_size: int = 8
    min_cohesion: float = 0.7
    min_confidence: float = 0.8

    # Clustering parameters
    max_clusters: int = 5
    similarity_threshold: float = 0.3

    # Feature weights
    attribute_weight: float = 0.4
    dependency_weight: float = 0.3
    semantic_weight: float = 0.3

    # Quality metrics
    max_unclustered_ratio: float = 0.3
    min_silhouette_score: float = 0.5


@dataclass
class MethodInfo:
    """Information about a method for clustering analysis."""

    name: str
    node: ast.AST
    line_start: int
    line_end: int

    # Attribute access patterns
    self_attributes: Set[str] = field(default_factory=set)
    external_attributes: Set[str] = field(default_factory=set)

    # Dependencies
    external_calls: Set[str] = field(default_factory=set)
    imports_used: Set[str] = field(default_factory=set)

    # Semantic markers
    responsibility_keywords: Set[str] = field(default_factory=set)
    operation_patterns: List[str] = field(default_factory=list)

    # Metrics
    complexity: int = 0
    lines_of_code: int = 0


@dataclass
class ResponsibilityCluster:
    """A cluster of methods with shared responsibilities."""

    cluster_id: str
    methods: List[MethodInfo]

    # Cluster characteristics
    shared_attributes: Set[str] = field(default_factory=set)
    shared_dependencies: Set[str] = field(default_factory=set)
    dominant_responsibilities: List[str] = field(default_factory=list)

    # Quality metrics
    cohesion_score: float = 0.0
    confidence: float = 0.0
    quality: ClusterQuality = ClusterQuality.INSUFFICIENT

    # Suggested component
    suggested_name: Optional[str] = None
    interface_methods: List[str] = field(default_factory=list)
    private_methods: List[str] = field(default_factory=list)

    # Evidence and reasoning
    evidence: Optional[Evidence] = None


@dataclass
class ClusteringResult:
    """Result of responsibility clustering analysis."""

    class_name: str
    file_path: str
    total_methods: int

    # Clustering results
    clusters: List[ResponsibilityCluster]
    unclustered_methods: List[MethodInfo]

    # Quality metrics
    unclustered_ratio: float
    average_cohesion: float
    silhouette_score: float

    # Overall assessment
    extraction_recommended: bool
    confidence: float

    # Evidence and recommendations
    evidence: Evidence
    recommendations: List[str] = field(default_factory=list)


class ExtractionComplexity(Enum):
    """Complexity levels for component extraction."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ComponentInterface:
    """Interface definition for an extracted component."""

    component_name: str
    public_methods: List[str]
    private_methods: List[str]
    required_attributes: Set[str]
    external_dependencies: Set[str]

    # Interface characteristics
    cohesion_score: float
    complexity: ExtractionComplexity
    confidence: float

    # Implementation guidance
    constructor_parameters: List[str] = field(default_factory=list)
    method_signatures: Dict[str, str] = field(default_factory=dict)
    docstring_template: str = ""


@dataclass
class ExtractionPlan:
    """Detailed plan for extracting a component from a class."""

    source_class: str
    target_component: str
    extraction_type: str  # "extract_class", "extract_mixin", "extract_service"

    # Methods and attributes to move
    methods_to_extract: List[str]
    attributes_to_extract: Set[str]
    dependencies_to_inject: Set[str]

    # Complexity assessment
    complexity: ExtractionComplexity
    estimated_effort_hours: int
    risk_factors: List[str] = field(default_factory=list)

    # Implementation steps
    implementation_steps: List[str] = field(default_factory=list)


class ResponsibilityClusterer:
    """Main class for performing responsibility-based clustering."""
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        """Initialize the clusterer with configuration."""
        self.config = config or ClusteringConfig()
    
    def analyze_class(self, class_node: ast.ClassDef, file_path: str) -> ClusteringResult:
        """Analyze a class and return clustering results."""
        # This is a simplified implementation
        # In reality, this would contain the full clustering logic
        methods = self._extract_methods(class_node)
        similarity_matrix = self._build_similarity_matrix(methods)
        
        # Placeholder result
        return ClusteringResult(
            class_name=class_node.name,
            file_path=file_path,
            total_methods=len(methods),
            clusters=[],
            unclustered_methods=methods,
            unclustered_ratio=1.0,
            average_cohesion=0.0,
            silhouette_score=0.0,
            extraction_recommended=False,
            confidence=0.0,
            evidence=_make_evidence(
                "Responsibility clustering is not implemented in this simplified version",
                0.0,
                metadata={"class_name": class_node.name, "file_path": file_path},
            ),
            recommendations=["Clustering not implemented in this simplified version"]
        )
    
    def _extract_methods(self, class_node: ast.ClassDef) -> List[MethodInfo]:
        """Extract method information from class AST."""
        methods = []
        for node in ast.iter_child_nodes(class_node):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_info = MethodInfo(
                    name=node.name,
                    node=node,
                    line_start=getattr(node, 'lineno', 0),
                    line_end=getattr(node, 'end_lineno', 0) or getattr(node, 'lineno', 0)
                )
                methods.append(method_info)
        return methods
    
    def _build_similarity_matrix(self, methods: List[MethodInfo]) -> "np.ndarray":
        """Build similarity matrix between methods based on multiple features."""
        # Lazy import numpy
        np = _import_numpy()
        
        n_methods = len(methods)
        similarity_matrix = np.zeros((n_methods, n_methods))
        
        # Simple placeholder implementation
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                similarity = 0.5  # Placeholder similarity
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        # Set diagonal to 1.0 (self-similarity)
        np = _import_numpy()
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
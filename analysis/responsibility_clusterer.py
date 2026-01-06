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
import re
from typing import List, Dict, Set, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import networkx as nx
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import numpy as np
from collections import defaultdict, Counter

from .models import Evidence


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
    node: ast.FunctionDef
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
    testing_requirements: List[str] = field(default_factory=list)
    
    # Quality metrics
    confidence: float = 0.0
    expected_cohesion_improvement: float = 0.0


@dataclass
class ComponentSuggestion:
    """High-level suggestion for component extraction."""
    
    suggestion_type: str  # "decision" or "suggestion"
    priority: str  # "high", "medium", "low"
    
    # Core information
    cluster: ResponsibilityCluster
    interface: ComponentInterface
    extraction_plan: ExtractionPlan
    
    # Decision rationale
    rationale: str
    benefits: List[str] = field(default_factory=list)
    trade_offs: List[str] = field(default_factory=list)
    
    # Supporting evidence
    evidence: Evidence = None
    confidence: float = 0.0


class ResponsibilityClusterer:
    """
    Clusters methods by shared responsibilities and suggests component extraction.
    
    Uses multiple clustering algorithms and quality metrics to identify
    cohesive groups of methods that can be extracted into separate components.
    """
    
    def __init__(self, config: Optional[ClusteringConfig] = None):
        """Initialize the responsibility clusterer."""
        self.config = config or ClusteringConfig()
        
        # Responsibility keywords for semantic analysis
        self.responsibility_keywords = {
            'data': ['load', 'save', 'read', 'write', 'parse', 'serialize', 'deserialize', 'fetch', 'store'],
            'validation': ['validate', 'check', 'verify', 'ensure', 'assert', 'confirm', 'test'],
            'calculation': ['calculate', 'compute', 'process', 'transform', 'convert', 'analyze'],
            'communication': ['send', 'receive', 'request', 'response', 'notify', 'broadcast', 'publish'],
            'formatting': ['format', 'render', 'display', 'print', 'show', 'present', 'export'],
            'logging': ['log', 'debug', 'info', 'warn', 'error', 'trace', 'audit'],
            'configuration': ['config', 'setting', 'option', 'parameter', 'preference', 'setup'],
            'caching': ['cache', 'store', 'retrieve', 'invalidate', 'refresh', 'expire'],
            'security': ['authenticate', 'authorize', 'encrypt', 'decrypt', 'hash', 'sign', 'verify'],
            'workflow': ['start', 'stop', 'pause', 'resume', 'execute', 'run', 'trigger', 'schedule']
        }
        
        # Operation patterns for semantic similarity
        self.operation_patterns = [
            'create', 'update', 'delete', 'get', 'set', 'add', 'remove',
            'find', 'search', 'filter', 'sort', 'group', 'merge', 'split'
        ]
    
    def cluster_class_methods(self, source_code: str, class_name: str, file_path: str) -> Optional[ClusteringResult]:
        """
        Cluster methods in a class by shared responsibilities.
        
        Args:
            source_code: Python source code containing the class
            class_name: Name of the class to analyze
            file_path: Path to the source file
            
        Returns:
            ClusteringResult with clusters and quality metrics, or None if analysis fails
        """
        try:
            tree = ast.parse(source_code)
            
            # Find the target class
            class_node = None
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == class_name:
                    class_node = node
                    break
            
            if not class_node:
                return None
            
            # Extract method information
            methods = self._extract_method_info(class_node, source_code)
            
            if len(methods) < self.config.min_cluster_size:
                return self._create_insufficient_result(class_name, file_path, methods)
            
            # Build similarity matrix
            similarity_matrix = self._build_similarity_matrix(methods)
            
            # Perform clustering
            clusters = self._perform_clustering(methods, similarity_matrix)
            
            # Calculate quality metrics
            result = self._evaluate_clustering_quality(
                class_name, file_path, methods, clusters, similarity_matrix
            )
            
            return result
            
        except Exception as e:
            return None
    
    def _extract_method_info(self, class_node: ast.ClassDef, source_code: str) -> List[MethodInfo]:
        """Extract detailed information about each method in the class."""
        methods = []
        source_lines = source_code.split('\n')
        
        for node in class_node.body:
            if not isinstance(node, ast.FunctionDef):
                continue
            
            # Skip special methods
            if node.name.startswith('__') and node.name.endswith('__'):
                continue
            
            method_info = MethodInfo(
                name=node.name,
                node=node,
                line_start=node.lineno,
                line_end=node.end_lineno or node.lineno,
                lines_of_code=node.end_lineno - node.lineno + 1 if node.end_lineno else 1
            )
            
            # Analyze method content
            self._analyze_method_attributes(method_info, node)
            self._analyze_method_dependencies(method_info, node)
            self._analyze_method_semantics(method_info, node)
            self._calculate_method_complexity(method_info, node)
            
            methods.append(method_info)
        
        return methods
    
    def _analyze_method_attributes(self, method_info: MethodInfo, node: ast.FunctionDef):
        """Analyze attribute access patterns in the method."""
        for child in ast.walk(node):
            if isinstance(child, ast.Attribute):
                if isinstance(child.value, ast.Name) and child.value.id == 'self':
                    method_info.self_attributes.add(child.attr)
                elif isinstance(child.value, ast.Name):
                    method_info.external_attributes.add(f"{child.value.id}.{child.attr}")
    
    def _analyze_method_dependencies(self, method_info: MethodInfo, node: ast.FunctionDef):
        """Analyze external dependencies and function calls."""
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    method_info.external_calls.add(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    if isinstance(child.func.value, ast.Name) and child.func.value.id != 'self':
                        method_info.external_calls.add(f"{child.func.value.id}.{child.func.attr}")
    
    def _analyze_method_semantics(self, method_info: MethodInfo, node: ast.FunctionDef):
        """Analyze semantic markers and responsibility keywords."""
        method_name = node.name.lower()
        
        # Check responsibility keywords
        for responsibility, keywords in self.responsibility_keywords.items():
            for keyword in keywords:
                if keyword in method_name:
                    method_info.responsibility_keywords.add(responsibility)
        
        # Check operation patterns
        for pattern in self.operation_patterns:
            if pattern in method_name:
                method_info.operation_patterns.append(pattern)
    
    def _calculate_method_complexity(self, method_info: MethodInfo, node: ast.FunctionDef):
        """Calculate cyclomatic complexity of the method."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        
        method_info.complexity = complexity
    
    def _build_similarity_matrix(self, methods: List[MethodInfo]) -> np.ndarray:
        """Build similarity matrix between methods based on multiple features."""
        n_methods = len(methods)
        similarity_matrix = np.zeros((n_methods, n_methods))
        
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                similarity = self._calculate_method_similarity(methods[i], methods[j])
                similarity_matrix[i][j] = similarity
                similarity_matrix[j][i] = similarity
        
        # Set diagonal to 1.0 (self-similarity)
        np.fill_diagonal(similarity_matrix, 1.0)
        
        return similarity_matrix
    
    def _calculate_method_similarity(self, method1: MethodInfo, method2: MethodInfo) -> float:
        """Calculate similarity between two methods using multiple features."""
        
        # Attribute similarity
        attr_similarity = self._jaccard_similarity(
            method1.self_attributes, method2.self_attributes
        )
        
        # Dependency similarity
        dep_similarity = self._jaccard_similarity(
            method1.external_calls, method2.external_calls
        )
        
        # Semantic similarity
        semantic_similarity = self._jaccard_similarity(
            method1.responsibility_keywords, method2.responsibility_keywords
        )
        
        # Weighted combination
        total_similarity = (
            attr_similarity * self.config.attribute_weight +
            dep_similarity * self.config.dependency_weight +
            semantic_similarity * self.config.semantic_weight
        )
        
        return total_similarity
    
    def _jaccard_similarity(self, set1: Set[str], set2: Set[str]) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def _perform_clustering(self, methods: List[MethodInfo], similarity_matrix: np.ndarray) -> List[List[int]]:
        """Perform clustering using the configured algorithm."""
        
        if self.config.algorithm == ClusteringAlgorithm.COMMUNITY_DETECTION:
            return self._community_detection_clustering(methods, similarity_matrix)
        elif self.config.algorithm == ClusteringAlgorithm.AGGLOMERATIVE:
            return self._agglomerative_clustering(methods, similarity_matrix)
        else:  # HYBRID
            return self._hybrid_clustering(methods, similarity_matrix)
    
    def _community_detection_clustering(self, methods: List[MethodInfo], similarity_matrix: np.ndarray) -> List[List[int]]:
        """Perform clustering using community detection on method graph."""
        
        # Create graph from similarity matrix
        G = nx.Graph()
        n_methods = len(methods)
        
        # Add nodes
        for i in range(n_methods):
            G.add_node(i, name=methods[i].name)
        
        # Add edges above similarity threshold
        for i in range(n_methods):
            for j in range(i + 1, n_methods):
                if similarity_matrix[i][j] > self.config.similarity_threshold:
                    G.add_edge(i, j, weight=similarity_matrix[i][j])
        
        # Perform community detection
        try:
            import networkx.algorithms.community as nx_comm
            communities = nx_comm.greedy_modularity_communities(G)
            return [list(community) for community in communities]
        except ImportError:
            # Fallback to simple connected components
            return [list(component) for component in nx.connected_components(G)]
    
    def _agglomerative_clustering(self, methods: List[MethodInfo], similarity_matrix: np.ndarray) -> List[List[int]]:
        """Perform agglomerative clustering on similarity matrix."""
        
        # Convert similarity to distance
        distance_matrix = 1.0 - similarity_matrix
        
        # Determine optimal number of clusters
        best_n_clusters = self._find_optimal_clusters(distance_matrix)
        
        # Perform clustering
        clustering = AgglomerativeClustering(
            n_clusters=best_n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # Group methods by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(cluster_labels):
            clusters[label].append(i)
        
        return list(clusters.values())
    
    def _hybrid_clustering(self, methods: List[MethodInfo], similarity_matrix: np.ndarray) -> List[List[int]]:
        """Perform hybrid clustering combining both approaches."""
        
        # Try community detection first
        community_clusters = self._community_detection_clustering(methods, similarity_matrix)
        
        # If community detection produces good results, use it
        if self._evaluate_cluster_quality(community_clusters, similarity_matrix) > 0.6:
            return community_clusters
        
        # Otherwise, fall back to agglomerative clustering
        return self._agglomerative_clustering(methods, similarity_matrix)
    
    def _find_optimal_clusters(self, distance_matrix: np.ndarray) -> int:
        """Find optimal number of clusters using silhouette analysis."""
        n_methods = distance_matrix.shape[0]
        max_clusters = min(self.config.max_clusters, n_methods - 1)
        
        best_score = -1
        best_n_clusters = 2
        
        for n_clusters in range(2, max_clusters + 1):
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            
            cluster_labels = clustering.fit_predict(distance_matrix)
            
            # Calculate silhouette score
            try:
                score = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
                if score > best_score:
                    best_score = score
                    best_n_clusters = n_clusters
            except ValueError:
                continue
        
        return best_n_clusters
    
    def _evaluate_cluster_quality(self, clusters: List[List[int]], similarity_matrix: np.ndarray) -> float:
        """Evaluate overall quality of clustering."""
        if not clusters:
            return 0.0
        
        total_quality = 0.0
        total_methods = sum(len(cluster) for cluster in clusters)
        
        for cluster in clusters:
            if len(cluster) < 2:
                continue
            
            # Calculate intra-cluster similarity
            intra_similarity = 0.0
            pairs = 0
            
            for i in range(len(cluster)):
                for j in range(i + 1, len(cluster)):
                    intra_similarity += similarity_matrix[cluster[i]][cluster[j]]
                    pairs += 1
            
            if pairs > 0:
                cluster_quality = intra_similarity / pairs
                total_quality += cluster_quality * len(cluster)
        
        return total_quality / total_methods if total_methods > 0 else 0.0
    
    def _evaluate_clustering_quality(self, class_name: str, file_path: str, 
                                   methods: List[MethodInfo], clusters: List[List[int]], 
                                   similarity_matrix: np.ndarray) -> ClusteringResult:
        """Evaluate clustering quality and create result."""
        
        # Create responsibility clusters
        responsibility_clusters = []
        unclustered_methods = []
        
        for i, cluster_indices in enumerate(clusters):
            if len(cluster_indices) < self.config.min_cluster_size:
                # Add to unclustered methods
                for idx in cluster_indices:
                    unclustered_methods.append(methods[idx])
                continue
            
            cluster_methods = [methods[idx] for idx in cluster_indices]
            cluster = self._create_responsibility_cluster(f"cluster_{i}", cluster_methods, similarity_matrix, cluster_indices)
            
            if cluster.quality != ClusterQuality.INSUFFICIENT:
                responsibility_clusters.append(cluster)
            else:
                unclustered_methods.extend(cluster_methods)
        
        # Calculate overall metrics
        total_methods = len(methods)
        unclustered_ratio = len(unclustered_methods) / total_methods
        
        # Calculate average cohesion
        if responsibility_clusters:
            average_cohesion = sum(c.cohesion_score for c in responsibility_clusters) / len(responsibility_clusters)
        else:
            average_cohesion = 0.0
        
        # Calculate silhouette score
        try:
            if len(clusters) > 1:
                cluster_labels = np.zeros(total_methods)
                for i, cluster_indices in enumerate(clusters):
                    for idx in cluster_indices:
                        cluster_labels[idx] = i
                
                distance_matrix = 1.0 - similarity_matrix
                silhouette = silhouette_score(distance_matrix, cluster_labels, metric='precomputed')
            else:
                silhouette = 0.0
        except (ValueError, ZeroDivisionError):
            silhouette = 0.0
        
        # Determine if extraction is recommended
        extraction_recommended = (
            len(responsibility_clusters) >= 2 and
            unclustered_ratio <= self.config.max_unclustered_ratio and
            average_cohesion >= self.config.min_cohesion and
            silhouette >= self.config.min_silhouette_score
        )
        
        # Calculate overall confidence
        confidence = self._calculate_overall_confidence(
            responsibility_clusters, unclustered_ratio, average_cohesion, silhouette
        )
        
        # Generate evidence
        evidence = self._generate_clustering_evidence(
            class_name, responsibility_clusters, unclustered_ratio, average_cohesion, silhouette
        )
        
        # Generate recommendations
        recommendations = self._generate_clustering_recommendations(
            responsibility_clusters, unclustered_ratio, extraction_recommended
        )
        
        return ClusteringResult(
            class_name=class_name,
            file_path=file_path,
            total_methods=total_methods,
            clusters=responsibility_clusters,
            unclustered_methods=unclustered_methods,
            unclustered_ratio=unclustered_ratio,
            average_cohesion=average_cohesion,
            silhouette_score=silhouette,
            extraction_recommended=extraction_recommended,
            confidence=confidence,
            evidence=evidence,
            recommendations=recommendations
        )
    
    def _create_responsibility_cluster(self, cluster_id: str, methods: List[MethodInfo], 
                                     similarity_matrix: np.ndarray, indices: List[int]) -> ResponsibilityCluster:
        """Create a responsibility cluster with quality assessment."""
        
        # Find shared attributes and dependencies (attributes used by at least 50% of methods)
        if methods:
            all_attributes = {}
            all_dependencies = {}
            
            for method in methods:
                for attr in method.self_attributes:
                    all_attributes[attr] = all_attributes.get(attr, 0) + 1
                for dep in method.external_calls:
                    all_dependencies[dep] = all_dependencies.get(dep, 0) + 1
            
            threshold = max(1, len(methods) // 2)  # At least 50% of methods
            shared_attributes = {attr for attr, count in all_attributes.items() if count >= threshold}
            shared_dependencies = {dep for dep, count in all_dependencies.items() if count >= threshold}
        else:
            shared_attributes = set()
            shared_dependencies = set()
        
        # Find dominant responsibilities
        all_responsibilities = []
        for method in methods:
            all_responsibilities.extend(method.responsibility_keywords)
        
        responsibility_counts = Counter(all_responsibilities)
        dominant_responsibilities = [resp for resp, count in responsibility_counts.most_common(3)]
        
        # Calculate cohesion score
        cohesion_score = self._calculate_cluster_cohesion(indices, similarity_matrix)
        
        # Determine quality
        quality = self._assess_cluster_quality(methods, cohesion_score)
        
        # Calculate confidence
        confidence = self._calculate_cluster_confidence(methods, cohesion_score, shared_attributes, shared_dependencies)
        
        # Generate suggested component name
        suggested_name = self._generate_component_name(dominant_responsibilities, shared_attributes)
        
        # Categorize methods as interface or private
        interface_methods, private_methods = self._categorize_cluster_methods(methods)
        
        # Generate evidence
        evidence = self._generate_cluster_evidence(cluster_id, methods, cohesion_score, shared_attributes, shared_dependencies)
        
        return ResponsibilityCluster(
            cluster_id=cluster_id,
            methods=methods,
            shared_attributes=shared_attributes,
            shared_dependencies=shared_dependencies,
            dominant_responsibilities=dominant_responsibilities,
            cohesion_score=cohesion_score,
            confidence=confidence,
            quality=quality,
            suggested_name=suggested_name,
            interface_methods=interface_methods,
            private_methods=private_methods,
            evidence=evidence
        )
    
    def _calculate_cluster_cohesion(self, indices: List[int], similarity_matrix: np.ndarray) -> float:
        """Calculate cohesion score for a cluster."""
        if len(indices) < 2:
            return 1.0
        
        total_similarity = 0.0
        pairs = 0
        
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                total_similarity += similarity_matrix[indices[i]][indices[j]]
                pairs += 1
        
        return total_similarity / pairs if pairs > 0 else 0.0
    
    def _assess_cluster_quality(self, methods: List[MethodInfo], cohesion_score: float) -> ClusterQuality:
        """Assess the quality of a cluster."""
        cluster_size = len(methods)
        
        if cluster_size < self.config.min_cluster_size:
            return ClusterQuality.INSUFFICIENT
        
        if (cluster_size <= self.config.max_cluster_size and 
            cohesion_score >= self.config.min_cohesion):
            return ClusterQuality.HIGH
        elif cohesion_score >= self.config.min_cohesion * 0.8:
            return ClusterQuality.MEDIUM
        elif cohesion_score >= self.config.min_cohesion * 0.6:
            return ClusterQuality.LOW
        else:
            return ClusterQuality.INSUFFICIENT
    
    def _calculate_cluster_confidence(self, methods: List[MethodInfo], cohesion_score: float,
                                    shared_attributes: Set[str], shared_dependencies: Set[str]) -> float:
        """Calculate confidence score for a cluster."""
        
        # Base confidence from cohesion
        confidence = cohesion_score
        
        # Boost confidence for shared attributes
        if shared_attributes:
            confidence += 0.1 * min(len(shared_attributes), 3)
        
        # Boost confidence for shared dependencies
        if shared_dependencies:
            confidence += 0.05 * min(len(shared_dependencies), 2)
        
        # Boost confidence for appropriate cluster size
        cluster_size = len(methods)
        if self.config.min_cluster_size <= cluster_size <= self.config.max_cluster_size:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _generate_component_name(self, responsibilities: List[str], attributes: Set[str]) -> str:
        """Generate a suggested component name based on responsibilities and attributes."""
        
        if responsibilities:
            primary_responsibility = responsibilities[0]
            return f"{primary_responsibility.title()}Handler"
        
        if attributes:
            # Try to infer from attribute names
            common_prefixes = self._find_common_prefixes(attributes)
            if common_prefixes:
                return f"{common_prefixes[0].title()}Manager"
        
        return "ExtractedComponent"
    
    def _find_common_prefixes(self, attributes: Set[str]) -> List[str]:
        """Find common prefixes in attribute names."""
        if not attributes:
            return []
        
        # Simple prefix detection
        prefixes = []
        for attr in attributes:
            parts = attr.split('_')
            if len(parts) > 1:
                prefixes.append(parts[0])
        
        if prefixes:
            prefix_counts = Counter(prefixes)
            return [prefix for prefix, count in prefix_counts.most_common(2)]
        
        return []
    
    def _categorize_cluster_methods(self, methods: List[MethodInfo]) -> Tuple[List[str], List[str]]:
        """Categorize methods as interface (public) or private."""
        interface_methods = []
        private_methods = []
        
        for method in methods:
            if method.name.startswith('_'):
                private_methods.append(method.name)
            else:
                interface_methods.append(method.name)
        
        return interface_methods, private_methods
    
    def _calculate_overall_confidence(self, clusters: List[ResponsibilityCluster], 
                                    unclustered_ratio: float, average_cohesion: float, 
                                    silhouette_score: float) -> float:
        """Calculate overall confidence in the clustering result."""
        
        if not clusters:
            return 0.0
        
        # Base confidence from cluster quality
        high_quality_clusters = sum(1 for c in clusters if c.quality == ClusterQuality.HIGH)
        cluster_confidence = high_quality_clusters / len(clusters)
        
        # Penalty for high unclustered ratio
        unclustered_penalty = max(0, unclustered_ratio - self.config.max_unclustered_ratio)
        
        # Combine factors
        confidence = (
            cluster_confidence * 0.4 +
            average_cohesion * 0.3 +
            max(0, silhouette_score) * 0.2 +
            max(0, 1.0 - unclustered_penalty * 2) * 0.1
        )
        
        return min(confidence, 1.0)
    
    def _generate_clustering_evidence(self, class_name: str, clusters: List[ResponsibilityCluster],
                                    unclustered_ratio: float, average_cohesion: float, 
                                    silhouette_score: float) -> Evidence:
        """Generate evidence for the clustering analysis."""
        
        description = f"Responsibility clustering analysis for {class_name}"
        
        if clusters:
            description += f" identified {len(clusters)} cohesive clusters"
        else:
            description += " found no cohesive clusters"
        
        metadata = {
            'total_clusters': len(clusters),
            'unclustered_ratio': unclustered_ratio,
            'average_cohesion': average_cohesion,
            'silhouette_score': silhouette_score,
            'cluster_qualities': [c.quality.value for c in clusters],
            'cluster_sizes': [len(c.methods) for c in clusters]
        }
        
        confidence = self._calculate_overall_confidence(clusters, unclustered_ratio, average_cohesion, silhouette_score)
        
        return Evidence(
            description=description,
            confidence=confidence,
            code_snippets=[],
            metadata=metadata
        )
    
    def _generate_cluster_evidence(self, cluster_id: str, methods: List[MethodInfo],
                                 cohesion_score: float, shared_attributes: Set[str], 
                                 shared_dependencies: Set[str]) -> Evidence:
        """Generate evidence for a specific cluster."""
        
        method_names = [m.name for m in methods]
        description = f"Cluster {cluster_id} with {len(methods)} methods: {', '.join(method_names)}"
        
        metadata = {
            'cluster_size': len(methods),
            'cohesion_score': cohesion_score,
            'shared_attributes': list(shared_attributes),
            'shared_dependencies': list(shared_dependencies),
            'method_names': method_names
        }
        
        return Evidence(
            description=description,
            confidence=cohesion_score,
            code_snippets=[f"def {name}(...):" for name in method_names[:3]],
            metadata=metadata
        )
    
    def _generate_clustering_recommendations(self, clusters: List[ResponsibilityCluster],
                                           unclustered_ratio: float, extraction_recommended: bool) -> List[str]:
        """Generate recommendations based on clustering results."""
        recommendations = []
        
        if extraction_recommended:
            recommendations.append("Extract identified clusters into separate components")
            
            for cluster in clusters:
                if cluster.quality == ClusterQuality.HIGH:
                    recommendations.append(f"Extract {cluster.suggested_name} with methods: {', '.join([m.name for m in cluster.methods])}")
        
        if unclustered_ratio > self.config.max_unclustered_ratio:
            recommendations.append("Consider manual review of unclustered methods for additional extraction opportunities")
        
        if not clusters:
            recommendations.append("No clear clustering patterns found - consider alternative refactoring approaches")
        
        recommendations.append("Use composition or delegation to maintain loose coupling between extracted components")
        
        return recommendations
    
    def _create_insufficient_result(self, class_name: str, file_path: str, methods: List[MethodInfo]) -> ClusteringResult:
        """Create result for classes with insufficient methods for clustering."""
        
        evidence = Evidence(
            description=f"Class {class_name} has only {len(methods)} methods - insufficient for clustering analysis",
            confidence=1.0,
            code_snippets=[],
            metadata={'method_count': len(methods), 'min_required': self.config.min_cluster_size}
        )
        
        return ClusteringResult(
            class_name=class_name,
            file_path=file_path,
            total_methods=len(methods),
            clusters=[],
            unclustered_methods=methods,
            unclustered_ratio=1.0,
            average_cohesion=0.0,
            silhouette_score=0.0,
            extraction_recommended=False,
            confidence=0.0,
            evidence=evidence,
            recommendations=["Class is too small for responsibility clustering - consider other refactoring approaches"]
        )
    
    # Component Suggestion Generation Methods (Task 7.2)
    
    def generate_component_suggestions(self, clustering_result: ClusteringResult) -> List[ComponentSuggestion]:
        """
        Generate component extraction suggestions from clustering results.
        
        Args:
            clustering_result: Result from cluster_class_methods
            
        Returns:
            List of ComponentSuggestion objects with interfaces and extraction plans
        """
        suggestions = []
        
        for cluster in clustering_result.clusters:
            if cluster.quality == ClusterQuality.HIGH and cluster.confidence >= self.config.min_confidence:
                # Generate interface definition
                interface = self._generate_component_interface(cluster, clustering_result.class_name)
                
                # Generate extraction plan
                extraction_plan = self._generate_extraction_plan(cluster, clustering_result.class_name, clustering_result.file_path)
                
                # Determine suggestion type based on confidence
                suggestion_type = "decision" if cluster.confidence >= 0.8 else "suggestion"
                priority = self._determine_priority(cluster, interface, extraction_plan)
                
                # Generate rationale and benefits
                rationale = self._generate_extraction_rationale(cluster, interface)
                benefits = self._generate_extraction_benefits(cluster, interface)
                trade_offs = self._generate_extraction_trade_offs(cluster, extraction_plan)
                
                # Create evidence for the suggestion
                evidence = self._generate_suggestion_evidence(cluster, interface, extraction_plan)
                
                suggestion = ComponentSuggestion(
                    suggestion_type=suggestion_type,
                    priority=priority,
                    cluster=cluster,
                    interface=interface,
                    extraction_plan=extraction_plan,
                    rationale=rationale,
                    benefits=benefits,
                    trade_offs=trade_offs,
                    evidence=evidence,
                    confidence=cluster.confidence
                )
                
                suggestions.append(suggestion)
        
        # Sort by priority and confidence
        suggestions.sort(key=lambda s: (
            {"high": 3, "medium": 2, "low": 1}[s.priority],
            s.confidence
        ), reverse=True)
        
        return suggestions
    
    def _generate_component_interface(self, cluster: ResponsibilityCluster, source_class: str) -> ComponentInterface:
        """Generate interface definition for a cluster."""
        
        # Determine required attributes from method usage
        required_attributes = cluster.shared_attributes.copy()
        for method in cluster.methods:
            required_attributes.update(method.self_attributes)
        
        # Determine external dependencies
        external_dependencies = cluster.shared_dependencies.copy()
        for method in cluster.methods:
            external_dependencies.update(method.external_calls)
        
        # Assess extraction complexity
        complexity = self._assess_extraction_complexity(cluster, required_attributes, external_dependencies)
        
        # Generate method signatures
        method_signatures = {}
        for method in cluster.methods:
            if method.name in cluster.interface_methods:
                method_signatures[method.name] = self._generate_method_signature(method)
        
        # Generate constructor parameters
        constructor_parameters = self._generate_constructor_parameters(required_attributes, external_dependencies)
        
        # Generate docstring template
        docstring_template = self._generate_component_docstring(cluster, source_class)
        
        return ComponentInterface(
            component_name=cluster.suggested_name,
            public_methods=cluster.interface_methods,
            private_methods=cluster.private_methods,
            required_attributes=required_attributes,
            external_dependencies=external_dependencies,
            cohesion_score=cluster.cohesion_score,
            complexity=complexity,
            confidence=cluster.confidence,
            constructor_parameters=constructor_parameters,
            method_signatures=method_signatures,
            docstring_template=docstring_template
        )
    
    def _generate_extraction_plan(self, cluster: ResponsibilityCluster, source_class: str, file_path: str) -> ExtractionPlan:
        """Generate detailed extraction plan for a cluster."""
        
        # Determine extraction type based on cluster characteristics
        extraction_type = self._determine_extraction_type(cluster)
        
        # Collect methods and attributes to extract
        methods_to_extract = [m.name for m in cluster.methods]
        attributes_to_extract = cluster.shared_attributes.copy()
        for method in cluster.methods:
            attributes_to_extract.update(method.self_attributes)
        
        # Determine dependencies to inject
        dependencies_to_inject = cluster.shared_dependencies.copy()
        for method in cluster.methods:
            dependencies_to_inject.update(method.external_calls)
        
        # Assess complexity and effort
        complexity = self._assess_extraction_complexity(cluster, attributes_to_extract, dependencies_to_inject)
        estimated_effort = self._estimate_extraction_effort(cluster, complexity)
        
        # Identify risk factors
        risk_factors = self._identify_extraction_risks(cluster, attributes_to_extract, dependencies_to_inject)
        
        # Generate implementation steps
        implementation_steps = self._generate_implementation_steps(cluster, extraction_type)
        
        # Generate testing requirements
        testing_requirements = self._generate_testing_requirements(cluster, extraction_type)
        
        # Calculate expected improvements
        expected_cohesion_improvement = self._calculate_expected_cohesion_improvement(cluster)
        
        return ExtractionPlan(
            source_class=source_class,
            target_component=cluster.suggested_name,
            extraction_type=extraction_type,
            methods_to_extract=methods_to_extract,
            attributes_to_extract=attributes_to_extract,
            dependencies_to_inject=dependencies_to_inject,
            complexity=complexity,
            estimated_effort_hours=estimated_effort,
            risk_factors=risk_factors,
            implementation_steps=implementation_steps,
            testing_requirements=testing_requirements,
            confidence=cluster.confidence,
            expected_cohesion_improvement=expected_cohesion_improvement
        )
    
    def _assess_extraction_complexity(self, cluster: ResponsibilityCluster, 
                                    attributes: Set[str], dependencies: Set[str]) -> ExtractionComplexity:
        """Assess the complexity of extracting a cluster."""
        
        complexity_score = 0
        
        # Method count factor
        method_count = len(cluster.methods)
        if method_count > 8:
            complexity_score += 3
        elif method_count > 5:
            complexity_score += 2
        elif method_count > 3:
            complexity_score += 1
        
        # Attribute dependencies factor
        if len(attributes) > 5:
            complexity_score += 2
        elif len(attributes) > 2:
            complexity_score += 1
        
        # External dependencies factor
        if len(dependencies) > 3:
            complexity_score += 2
        elif len(dependencies) > 1:
            complexity_score += 1
        
        # Cohesion factor (lower cohesion = higher complexity)
        if cluster.cohesion_score < 0.5:
            complexity_score += 2
        elif cluster.cohesion_score < 0.7:
            complexity_score += 1
        
        # Map score to complexity level
        if complexity_score >= 7:
            return ExtractionComplexity.VERY_HIGH
        elif complexity_score >= 5:
            return ExtractionComplexity.HIGH
        elif complexity_score >= 3:
            return ExtractionComplexity.MEDIUM
        else:
            return ExtractionComplexity.LOW
    
    def _determine_extraction_type(self, cluster: ResponsibilityCluster) -> str:
        """Determine the best extraction type for a cluster."""
        
        # Analyze dominant responsibilities
        dominant_resp = cluster.dominant_responsibilities[0] if cluster.dominant_responsibilities else ""
        
        # Service-oriented responsibilities
        if dominant_resp in ['communication', 'data', 'caching']:
            return "extract_service"
        
        # Mixin-oriented responsibilities
        elif dominant_resp in ['validation', 'formatting', 'logging']:
            return "extract_mixin"
        
        # Default to class extraction
        else:
            return "extract_class"
    
    def _estimate_extraction_effort(self, cluster: ResponsibilityCluster, complexity: ExtractionComplexity) -> int:
        """Estimate effort in hours for extraction."""
        
        base_hours = len(cluster.methods) * 2  # 2 hours per method
        
        complexity_multiplier = {
            ExtractionComplexity.LOW: 1.0,
            ExtractionComplexity.MEDIUM: 1.5,
            ExtractionComplexity.HIGH: 2.0,
            ExtractionComplexity.VERY_HIGH: 3.0
        }
        
        return int(base_hours * complexity_multiplier[complexity])
    
    def _identify_extraction_risks(self, cluster: ResponsibilityCluster, 
                                 attributes: Set[str], dependencies: Set[str]) -> List[str]:
        """Identify potential risks in extraction."""
        
        risks = []
        
        if len(attributes) > 5:
            risks.append("High attribute coupling may require significant refactoring")
        
        if len(dependencies) > 3:
            risks.append("Multiple external dependencies may complicate testing")
        
        if cluster.cohesion_score < 0.6:
            risks.append("Lower cohesion may indicate unclear component boundaries")
        
        if len(cluster.methods) > 10:
            risks.append("Large number of methods may indicate over-extraction")
        
        # Check for complex method interactions
        complex_methods = [m for m in cluster.methods if m.complexity > 5]
        if len(complex_methods) > 2:
            risks.append("Multiple complex methods may increase extraction difficulty")
        
        return risks
    
    def _generate_implementation_steps(self, cluster: ResponsibilityCluster, extraction_type: str) -> List[str]:
        """Generate step-by-step implementation plan."""
        
        steps = [
            f"1. Create new {extraction_type.replace('extract_', '')} '{cluster.suggested_name}'",
            "2. Move identified methods to new component",
            "3. Extract required attributes as constructor parameters or properties",
            "4. Update method calls to use new component instance",
            "5. Add dependency injection for external dependencies",
            "6. Update imports and module structure",
            "7. Run tests to verify functionality preservation",
            "8. Update documentation and type hints"
        ]
        
        if extraction_type == "extract_service":
            steps.insert(2, "2.5. Implement service interface for loose coupling")
        
        return steps
    
    def _generate_testing_requirements(self, cluster: ResponsibilityCluster, extraction_type: str) -> List[str]:
        """Generate testing requirements for extraction."""
        
        requirements = [
            "Unit tests for all extracted methods",
            "Integration tests for component interaction",
            "Regression tests to ensure no functionality loss"
        ]
        
        if extraction_type == "extract_service":
            requirements.append("Service contract tests for interface compliance")
        
        if len(cluster.methods) > 5:
            requirements.append("Performance tests to verify no degradation")
        
        return requirements
    
    def _calculate_expected_cohesion_improvement(self, cluster: ResponsibilityCluster) -> float:
        """Calculate expected cohesion improvement from extraction."""
        
        # Estimate improvement based on cluster quality
        if cluster.quality == ClusterQuality.HIGH:
            return min(0.3, 1.0 - cluster.cohesion_score)  # Up to 30% improvement
        else:
            return min(0.2, 1.0 - cluster.cohesion_score)  # Up to 20% improvement
    
    def _determine_priority(self, cluster: ResponsibilityCluster, 
                          interface: ComponentInterface, plan: ExtractionPlan) -> str:
        """Determine priority level for extraction suggestion."""
        
        priority_score = 0
        
        # High confidence and quality
        if cluster.confidence >= 0.8 and cluster.quality == ClusterQuality.HIGH:
            priority_score += 3
        
        # Low complexity
        if interface.complexity in [ExtractionComplexity.LOW, ExtractionComplexity.MEDIUM]:
            priority_score += 2
        
        # High cohesion
        if cluster.cohesion_score >= 0.7:
            priority_score += 2
        
        # Clear responsibilities
        if len(cluster.dominant_responsibilities) >= 2:
            priority_score += 1
        
        if priority_score >= 6:
            return "high"
        elif priority_score >= 4:
            return "medium"
        else:
            return "low"
    
    def _generate_extraction_rationale(self, cluster: ResponsibilityCluster, interface: ComponentInterface) -> str:
        """Generate rationale for extraction suggestion."""
        
        rationale = f"Extract {interface.component_name} component with {len(cluster.methods)} methods "
        rationale += f"sharing {len(interface.required_attributes)} attributes and "
        rationale += f"{len(interface.external_dependencies)} external dependencies. "
        rationale += f"Cluster shows {cluster.cohesion_score:.2f} cohesion score and "
        rationale += f"{cluster.confidence:.2f} confidence level."
        
        return rationale
    
    def _generate_extraction_benefits(self, cluster: ResponsibilityCluster, interface: ComponentInterface) -> List[str]:
        """Generate list of extraction benefits."""
        
        benefits = [
            f"Improved cohesion through focused {', '.join(cluster.dominant_responsibilities)} responsibilities",
            f"Reduced class complexity by extracting {len(cluster.methods)} methods",
            "Enhanced testability through isolated component testing",
            "Better separation of concerns and single responsibility principle"
        ]
        
        if interface.complexity == ExtractionComplexity.LOW:
            benefits.append("Low extraction complexity minimizes implementation risk")
        
        if cluster.cohesion_score >= 0.7:
            benefits.append("High cluster cohesion ensures meaningful component boundaries")
        
        return benefits
    
    def _generate_extraction_trade_offs(self, cluster: ResponsibilityCluster, plan: ExtractionPlan) -> List[str]:
        """Generate list of extraction trade-offs."""
        
        trade_offs = []
        
        if plan.complexity in [ExtractionComplexity.HIGH, ExtractionComplexity.VERY_HIGH]:
            trade_offs.append(f"High extraction complexity requires {plan.estimated_effort_hours} hours of effort")
        
        if len(plan.dependencies_to_inject) > 2:
            trade_offs.append("Multiple dependency injections may increase component coupling")
        
        if len(plan.risk_factors) > 0:
            trade_offs.append(f"Extraction risks: {', '.join(plan.risk_factors[:2])}")
        
        trade_offs.append("Additional component increases overall system complexity")
        
        return trade_offs
    
    def _generate_suggestion_evidence(self, cluster: ResponsibilityCluster, 
                                    interface: ComponentInterface, plan: ExtractionPlan) -> Evidence:
        """Generate evidence supporting the extraction suggestion."""
        
        description = f"Component extraction suggestion for {interface.component_name} "
        description += f"based on {cluster.cohesion_score:.2f} cohesion and "
        description += f"{cluster.confidence:.2f} confidence scores"
        
        code_snippets = [f"def {method}(...):" for method in interface.public_methods[:3]]
        
        metadata = {
            'component_name': interface.component_name,
            'method_count': len(cluster.methods),
            'cohesion_score': cluster.cohesion_score,
            'confidence': cluster.confidence,
            'complexity': interface.complexity.value,
            'estimated_effort': plan.estimated_effort_hours,
            'extraction_type': plan.extraction_type,
            'dominant_responsibilities': cluster.dominant_responsibilities
        }
        
        return Evidence(
            description=description,
            confidence=cluster.confidence,
            code_snippets=code_snippets,
            metadata=metadata
        )
    
    def _generate_method_signature(self, method: MethodInfo) -> str:
        """Generate method signature for interface definition."""
        return f"def {method.name}(self, ...) -> Any"
    
    def _generate_constructor_parameters(self, attributes: Set[str], dependencies: Set[str]) -> List[str]:
        """Generate constructor parameters for component."""
        
        params = []
        
        # Add essential attributes as parameters
        for attr in sorted(attributes):
            if not attr.startswith('_'):  # Skip private attributes
                params.append(f"{attr}: Any")
        
        # Add key dependencies as parameters
        for dep in sorted(dependencies):
            if '.' not in dep:  # Simple dependencies only
                params.append(f"{dep}_service: Any")
        
        return params[:5]  # Limit to 5 parameters
    
    def _generate_component_docstring(self, cluster: ResponsibilityCluster, source_class: str) -> str:
        """Generate docstring template for component."""
        
        responsibilities = ', '.join(cluster.dominant_responsibilities)
        
        docstring = f'"""\n'
        docstring += f'{cluster.suggested_name} component extracted from {source_class}.\n\n'
        docstring += f'Handles {responsibilities} responsibilities with {len(cluster.methods)} methods.\n'
        docstring += f'Cohesion score: {cluster.cohesion_score:.2f}\n'
        docstring += f'Confidence: {cluster.confidence:.2f}\n'
        docstring += f'"""'
        
        return docstring
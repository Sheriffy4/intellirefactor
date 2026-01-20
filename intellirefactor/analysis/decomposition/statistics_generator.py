"""
Statistics Generator Module

Generates statistics and recommendations from functional decomposition analysis.
Calculates metrics, cluster benefits, and provides actionable recommendations.

Extracted from DecompositionAnalyzer god class to improve modularity.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .models import (
    ProjectFunctionalMap,
    SimilarityCluster,
    CanonicalizationPlan,
    RecommendationType,
)


class StatisticsGenerator:
    """
    Generates statistics and recommendations from analysis results.
    
    Provides metrics calculation, cluster benefit scoring, and
    actionable recommendations for refactoring opportunities.
    """

    def __init__(self):
        """Initialize the StatisticsGenerator."""
        pass

    def generate_statistics(
        self,
        functional_map: ProjectFunctionalMap,
        clusters: List[SimilarityCluster],
        plans: List[CanonicalizationPlan],
    ) -> Dict[str, Any]:
        """
        Generate comprehensive statistics from analysis results.

        Args:
            functional_map: Project functional map with blocks and capabilities
            clusters: List of similarity clusters
            plans: List of canonicalization plans

        Returns:
            Dictionary containing various statistics and metrics
        """
        if not functional_map:
            return {}

        category_counts: Dict[str, int] = {}
        for block in functional_map.blocks.values():
            key = f"{block.category}:{block.subcategory}"
            category_counts[key] = category_counts.get(key, 0) + 1

        cluster_stats = {
            "merge_candidates": sum(1 for c in clusters if c.recommendation == RecommendationType.MERGE),
            "extract_base_candidates": sum(1 for c in clusters if c.recommendation == RecommendationType.EXTRACT_BASE),
            "wrap_only_candidates": sum(1 for c in clusters if c.recommendation == RecommendationType.WRAP_ONLY),
            "keep_separate": sum(1 for c in clusters if c.recommendation == RecommendationType.KEEP_SEPARATE),
        }

        risk_stats = {
            "low_risk": sum(1 for c in clusters if c.risk_level.value == "LOW"),
            "medium_risk": sum(1 for c in clusters if c.risk_level.value == "MEDIUM"),
            "high_risk": sum(1 for c in clusters if c.risk_level.value == "HIGH"),
        }

        return {
            "total_blocks": functional_map.total_blocks,
            "total_capabilities": functional_map.total_capabilities,
            "total_clusters": functional_map.total_clusters,
            "resolution_rate": functional_map.resolution_rate,
            "resolution_rate_internal": functional_map.resolution_rate_internal,
            "resolution_rate_actionable": functional_map.resolution_rate_actionable,
            "external_calls_count": functional_map.external_calls_count,
            "dynamic_attribute_calls_count": functional_map.dynamic_attribute_calls_count,
            "category_distribution": category_counts,
            "cluster_recommendations": cluster_stats,
            "risk_distribution": risk_stats,
            "total_plans": len(plans),
        }

    def generate_recommendations(
        self,
        functional_map: ProjectFunctionalMap,
        clusters: List[SimilarityCluster],
    ) -> List[str]:
        """
        Generate actionable recommendations from analysis results.

        Args:
            functional_map: Project functional map with blocks and capabilities
            clusters: List of similarity clusters

        Returns:
            List of recommendation strings
        """
        if not functional_map:
            return ["Run analysis first to get recommendations"]

        recommendations: List[str] = []

        if clusters:
            high_priority = [c for c in clusters if c.risk_level.value == "LOW" and c.avg_similarity >= 0.8]
            if high_priority:
                recommendations.append(f"Found {len(high_priority)} high-priority, low-risk consolidation opportunities")

        complex_blocks = [b for b in functional_map.blocks.values() if b.cyclomatic > 15]
        if complex_blocks:
            recommendations.append(f"Consider refactoring {len(complex_blocks)} highly complex blocks (complexity > 15)")

        if clusters:
            category_clusters: Dict[str, int] = {}
            for cluster in clusters:
                key = f"{cluster.category}:{cluster.subcategory}"
                category_clusters[key] = category_clusters.get(key, 0) + 1
            problematic = [k for k, cnt in category_clusters.items() if cnt >= 3]
            if problematic:
                recommendations.append(f"Focus on categories with most duplication: {', '.join(problematic[:3])}")

        internal_rate = getattr(functional_map, "resolution_rate_internal", functional_map.resolution_rate)
        actionable_rate = getattr(functional_map, "resolution_rate_actionable", internal_rate)
        external_count = getattr(functional_map, "external_calls_count", 0)
        dynamic_count = getattr(functional_map, "dynamic_attribute_calls_count", 0)

        recommendations.append(
            f"Call resolution: {actionable_rate:.1%} actionable, {internal_rate:.1%} internal. "
            f"External calls: {external_count}, Dynamic attribute calls: {dynamic_count}"
        )
        return recommendations

    def calculate_cluster_benefit(self, cluster: SimilarityCluster) -> float:
        """
        Calculate benefit score for a cluster.

        The benefit score considers:
        - Number of blocks in cluster
        - Average similarity
        - Recommendation type
        - Risk level
        - Effort class

        Args:
            cluster: Similarity cluster to evaluate

        Returns:
            Benefit score (higher is better)
        """
        benefit = 0.0
        benefit += len(cluster.blocks) * 2
        benefit += cluster.avg_similarity * 10

        if cluster.recommendation == RecommendationType.MERGE:
            benefit += 5
        elif cluster.recommendation == RecommendationType.EXTRACT_BASE:
            benefit += 3
        elif cluster.recommendation == RecommendationType.WRAP_ONLY:
            benefit += 1

        if cluster.risk_level.value == "LOW":
            benefit += 3
        elif cluster.risk_level.value == "MEDIUM":
            benefit += 1

        effort_bonus = {"XS": 5, "S": 4, "M": 2, "L": 1, "XL": 0}
        benefit += effort_bonus.get(cluster.effort_class.value, 0)
        return benefit

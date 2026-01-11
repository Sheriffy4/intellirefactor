"""
Functional Clusterer

Groups similar functional blocks into clusters for consolidation planning.
Uses similarity scores to create clusters with recommendations for refactoring.
"""

import logging
import uuid
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

from .models import (
    FunctionalBlock,
    SimilarityCluster,
    RecommendationType,
    RiskLevel,
    EffortClass,
    DecompositionConfig,
)
from .similarity import SimilarityCalculator


logger = logging.getLogger(__name__)


class FunctionalClusterer:
    """
    Groups functional blocks into similarity clusters for consolidation.

    Uses configurable thresholds to determine cluster recommendations:
    - >= merge_threshold: MERGE/WRAP (strong similarity)
    - >= extract_threshold: EXTRACT_BASE (common base + differences)
    - < extract_threshold: KEEP_SEPARATE
    """

    def __init__(self, config: Optional[DecompositionConfig] = None):
        self.config = config or DecompositionConfig.default()
        # FIX: pass self.config, not the raw (possibly None) argument
        self.similarity_calculator = SimilarityCalculator(self.config)
        self.logger = logger

    def cluster_blocks(self, blocks: List[FunctionalBlock]) -> List[SimilarityCluster]:
        """
        Group functional blocks into similarity clusters.
        """
        if not blocks:
            return []

        category_groups = self._group_by_category(blocks)

        all_clusters: List[SimilarityCluster] = []
        for category_key, category_blocks in category_groups.items():
            if len(category_blocks) < 2:
                continue

            clusters = self._cluster_category_blocks(category_key, category_blocks)
            all_clusters.extend(clusters)

        self.logger.info(f"Created {len(all_clusters)} similarity clusters from {len(blocks)} blocks")
        return all_clusters

    def _group_by_category(self, blocks: List[FunctionalBlock]) -> Dict[str, List[FunctionalBlock]]:
        """Group blocks by category+subcategory for separate clustering."""
        groups: Dict[str, List[FunctionalBlock]] = defaultdict(list)

        for block in blocks:
            cat = (block.category or "uncategorized").strip() or "uncategorized"
            sub = (block.subcategory or "generic").strip() or "generic"
            key = f"{cat}:{sub}"
            groups[key].append(block)

        return dict(groups)

    def _cluster_category_blocks(self, category_key: str, blocks: List[FunctionalBlock]) -> List[SimilarityCluster]:
        """Cluster blocks within a single category."""
        if len(blocks) < 2:
            return []

        similarity_matrix = self.similarity_calculator.calculate_similarity_matrix(blocks)
        clusters = self._find_similarity_clusters(blocks, similarity_matrix)

        result_clusters: List[SimilarityCluster] = []
        for cluster_blocks in clusters:
            if len(cluster_blocks) < 2:
                continue

            cluster = self._create_similarity_cluster(category_key, cluster_blocks, similarity_matrix)

            # Intentionally exclude KEEP_SEPARATE to avoid noisy output
            if cluster.recommendation != RecommendationType.KEEP_SEPARATE:
                result_clusters.append(cluster)

        return result_clusters

    def _find_similarity_clusters(
        self,
        blocks: List[FunctionalBlock],
        similarity_matrix: Dict[Tuple[str, str], float],
    ) -> List[List[FunctionalBlock]]:
        """
        Find clusters using similarity-based connected components.

        FIX: Use extract_threshold for edges to reduce "chaining" components
        that later average out to KEEP_SEPARATE.
        """
        adjacency: Dict[str, Set[str]] = defaultdict(set)
        block_dict = {block.id: block for block in blocks}

        edge_threshold = self.config.extract_threshold  # instead of separate_threshold

        for (id1, id2), similarity in similarity_matrix.items():
            if similarity >= edge_threshold:
                adjacency[id1].add(id2)
                adjacency[id2].add(id1)

        visited: Set[str] = set()
        clusters: List[List[FunctionalBlock]] = []

        for block in blocks:
            if block.id in visited:
                continue

            component = self._dfs_cluster(block.id, adjacency, visited, block_dict)
            if len(component) < 2:
                continue

            if len(component) > self.config.max_component_size:
                sub_clusters = self._secondary_clustering(component, similarity_matrix)
                clusters.extend(sub_clusters)
            else:
                clusters.append(component)

        return clusters

    def _secondary_clustering(
        self,
        large_component: List[FunctionalBlock],
        similarity_matrix: Dict[Tuple[str, str], float],
    ) -> List[List[FunctionalBlock]]:
        """Apply secondary clustering to large components using stricter thresholds."""
        sub_clusters = self._cluster_with_threshold(
            large_component, similarity_matrix, self.config.extract_threshold
        )

        final_clusters: List[List[FunctionalBlock]] = []
        for cluster in sub_clusters:
            if len(cluster) > max(2, self.config.max_component_size // 2):
                strict_clusters = self._cluster_with_threshold(
                    cluster, similarity_matrix, self.config.merge_threshold
                )
                final_clusters.extend(strict_clusters)
            else:
                final_clusters.append(cluster)

        return final_clusters

    def _cluster_with_threshold(
        self,
        blocks: List[FunctionalBlock],
        similarity_matrix: Dict[Tuple[str, str], float],
        threshold: float,
    ) -> List[List[FunctionalBlock]]:
        """Cluster blocks with a specific similarity threshold."""
        if len(blocks) < 2:
            return [blocks] if blocks else []

        adjacency: Dict[str, Set[str]] = defaultdict(set)
        block_dict = {block.id: block for block in blocks}

        for (id1, id2), similarity in similarity_matrix.items():
            if id1 in block_dict and id2 in block_dict and similarity >= threshold:
                adjacency[id1].add(id2)
                adjacency[id2].add(id1)

        visited: Set[str] = set()
        clusters: List[List[FunctionalBlock]] = []

        for block in blocks:
            if block.id in visited:
                continue
            cluster = self._dfs_cluster(block.id, adjacency, visited, block_dict)
            if len(cluster) >= 2:
                clusters.append(cluster)

        return clusters

    def _dfs_cluster(
        self,
        start_id: str,
        adjacency: Dict[str, Set[str]],
        visited: Set[str],
        block_dict: Dict[str, FunctionalBlock],
    ) -> List[FunctionalBlock]:
        """DFS to find connected component (cluster)."""
        cluster: List[FunctionalBlock] = []
        stack = [start_id]

        while stack:
            node_id = stack.pop()
            if node_id in visited:
                continue

            visited.add(node_id)
            b = block_dict.get(node_id)
            if b is not None:
                cluster.append(b)

            for neighbor_id in adjacency.get(node_id, set()):
                if neighbor_id not in visited:
                    stack.append(neighbor_id)

        return cluster

    def _create_similarity_cluster(
        self,
        category_key: str,
        blocks: List[FunctionalBlock],
        similarity_matrix: Dict[Tuple[str, str], float],
    ) -> SimilarityCluster:
        """Create a SimilarityCluster from a group of blocks."""
        main_category, subcategory = (category_key.split(":", 1) + ["generic"])[:2]
        main_category = (main_category or "uncategorized").strip() or "uncategorized"
        subcategory = (subcategory or "generic").strip() or "generic"

        cluster_similarities: Dict[Tuple[str, str], float] = {}
        similarities: List[float] = []

        for i, block1 in enumerate(blocks):
            for j in range(i + 1, len(blocks)):
                block2 = blocks[j]
                key = tuple(sorted([block1.id, block2.id]))
                sim = similarity_matrix.get(key, 0.0)
                cluster_similarities[key] = sim
                similarities.append(sim)

        avg_similarity = (sum(similarities) / len(similarities)) if similarities else 0.0

        recommendation = self._determine_recommendation(avg_similarity, blocks)
        canonical_candidate = self._find_canonical_candidate(blocks, similarity_matrix)
        proposed_target = self._generate_proposed_target(main_category, subcategory, canonical_candidate)
        risk_level = self._assess_risk_level(blocks, avg_similarity)
        effort_class = self._assess_effort_class(blocks, recommendation)
        notes = self._generate_cluster_notes(blocks, avg_similarity, recommendation)

        return SimilarityCluster(
            id=str(uuid.uuid4()),
            category=main_category,
            subcategory=subcategory,
            blocks=[block.id for block in blocks],
            similarity=cluster_similarities,
            avg_similarity=avg_similarity,
            recommendation=recommendation,
            canonical_candidate=canonical_candidate.id,
            proposed_target=proposed_target,
            risk_level=risk_level,
            effort_class=effort_class,
            notes=notes,
        )

    def _determine_recommendation(self, avg_similarity: float, blocks: List[FunctionalBlock]) -> RecommendationType:
        """
        Determine consolidation recommendation.

        Fix: dunder-handling was duplicated and contradictory.
        """
        if self._has_dunder_methods(blocks):
            # safest: do not consolidate dunder methods at all
            return RecommendationType.KEEP_SEPARATE

        method_names = [b.method_name.lower() for b in blocks]

        if self._has_modal_mix(method_names):
            return RecommendationType.KEEP_SEPARATE

        if self._has_antonym_patterns(blocks):
            if avg_similarity >= self.config.extract_threshold:
                return RecommendationType.EXTRACT_BASE
            return RecommendationType.KEEP_SEPARATE

        if avg_similarity >= self.config.merge_threshold:
            if self.config.require_name_similarity_for_merge:
                name_similarity = self._calculate_name_similarity(blocks)
                if name_similarity < self.config.min_name_similarity_for_merge:
                    return RecommendationType.EXTRACT_BASE

            if self._signatures_compatible(blocks):
                return RecommendationType.MERGE
            return RecommendationType.WRAP_ONLY

        if avg_similarity >= self.config.extract_threshold:
            return RecommendationType.EXTRACT_BASE

        return RecommendationType.KEEP_SEPARATE

    def _signatures_compatible(self, blocks: List[FunctionalBlock]) -> bool:
        """Check if block signatures are compatible for merging."""
        if len(blocks) < 2:
            return True

        def param_count(sig: str) -> Optional[int]:
            if not sig:
                return None
            if "(" not in sig or ")" not in sig:
                return None
            inside = sig.split("(", 1)[1].rsplit(")", 1)[0].strip()
            if not inside:
                return 0
            return inside.count(",") + 1

        counts: List[int] = []
        for b in blocks:
            c = param_count(b.signature)
            if c is not None:
                counts.append(c)

        # If we can't parse any signatures, don't block MERGE
        if not counts:
            return True

        return (max(counts) - min(counts)) <= 2

    def _find_canonical_candidate(
        self,
        blocks: List[FunctionalBlock],
        similarity_matrix: Dict[Tuple[str, str], float],
    ) -> FunctionalBlock:
        """Find the best candidate to be the canonical implementation."""
        if len(blocks) == 1:
            return blocks[0]

        scores: Dict[str, float] = {}

        for block in blocks:
            score = 0.0

            fan_in = len(block.called_by)
            score += fan_in * 0.3

            complexity_score = max(0.0, 20.0 - float(block.cyclomatic)) / 20.0
            score += complexity_score * 0.2

            size_score = 1.0 - abs(float(block.loc) - 20.0) / 50.0
            size_score = max(0.0, size_score)
            score += size_score * 0.2

            sims: List[float] = []
            for other in blocks:
                if other.id == block.id:
                    continue
                key = tuple(sorted([block.id, other.id]))
                sims.append(similarity_matrix.get(key, 0.0))
            if sims:
                score += (sum(sims) / len(sims)) * 0.3

            scores[block.id] = score

        best_id = max(scores, key=scores.get)
        return next(b for b in blocks if b.id == best_id)

    def _generate_proposed_target(self, category: str, subcategory: str, canonical_block: FunctionalBlock) -> str:
        """Generate proposed target location with normalized names."""
        if category == "parsing":
            target_module = "unified/parsers.py"
        elif category == "validation":
            target_module = "unified/validators.py"
        elif category == "telemetry":
            target_module = "unified/telemetry.py"
        elif category == "persistence":
            target_module = "unified/storage.py"
        elif category == "orchestration":
            target_module = "unified/orchestration.py"
        else:
            target_module = f"unified/{category}.py"

        base_name = canonical_block.method_name.lstrip("_").replace("__", "_")

        if subcategory and subcategory != "generic" and not canonical_block.method_name.startswith("_"):
            func_name = f"{subcategory}_{base_name}"
        else:
            func_name = base_name

        return f"{target_module}::{func_name}"

    def _assess_risk_level(self, blocks: List[FunctionalBlock], avg_similarity: float) -> RiskLevel:
        total_callers = sum(len(b.called_by) for b in blocks)
        max_complexity = max((b.cyclomatic for b in blocks), default=0)

        if avg_similarity < 0.6:
            return RiskLevel.HIGH
        if total_callers > 10 or max_complexity > 15:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW

    def _assess_effort_class(self, blocks: List[FunctionalBlock], recommendation: RecommendationType) -> EffortClass:
        total_loc = sum(b.loc for b in blocks)
        block_count = len(blocks)

        if recommendation == RecommendationType.KEEP_SEPARATE:
            return EffortClass.XS
        if recommendation == RecommendationType.WRAP_ONLY:
            return EffortClass.S
        if recommendation == RecommendationType.MERGE:
            if block_count <= 2 and total_loc <= 50:
                return EffortClass.S
            if block_count <= 4 and total_loc <= 100:
                return EffortClass.M
            return EffortClass.L

        # EXTRACT_BASE
        return EffortClass.M if total_loc <= 100 else EffortClass.L

    def _generate_cluster_notes(
        self,
        blocks: List[FunctionalBlock],
        avg_similarity: float,
        recommendation: RecommendationType,
    ) -> List[str]:
        notes: List[str] = []
        notes.append(f"Cluster contains {len(blocks)} blocks with {avg_similarity:.2f} average similarity")

        if recommendation == RecommendationType.MERGE:
            notes.append("High similarity suggests these blocks can be merged into a single implementation")
        elif recommendation == RecommendationType.EXTRACT_BASE:
            notes.append("Moderate similarity suggests extracting common base with specialized variants")
        elif recommendation == RecommendationType.WRAP_ONLY:
            notes.append("Similar functionality but different signatures - consider wrapper pattern")
        else:
            notes.append("Low similarity - blocks should remain separate")

        complexities = [b.cyclomatic for b in blocks]
        if complexities and max(complexities) > 10:
            notes.append("Contains complex methods - consider simplification during consolidation")

        sizes = [b.loc for b in blocks]
        if sizes and max(sizes) > 50:
            notes.append("Contains large methods - may need decomposition")

        total_callers = sum(len(b.called_by) for b in blocks)
        if total_callers > 5:
            notes.append(f"High usage ({total_callers} callers) - ensure backward compatibility")

        return notes

    def _has_dunder_methods(self, blocks: List[FunctionalBlock]) -> bool:
        return any(b.method_name.startswith("__") and b.method_name.endswith("__") for b in blocks)

    def _has_antonym_patterns(self, blocks: List[FunctionalBlock]) -> bool:
        antonym_pairs = [
            ("enable", "disable"), ("start", "stop"), ("open", "close"),
            ("create", "delete"), ("add", "remove"), ("inc", "dec"),
            ("begin", "end"), ("enter", "exit"), ("connect", "disconnect"),
            ("lock", "unlock"), ("show", "hide"), ("on", "off"),
            ("get", "set"), ("import", "export"), ("load", "save"),
            ("push", "pop"), ("put", "take"), ("send", "receive"),
            ("encode", "decode"), ("compress", "decompress"),
            ("serialize", "deserialize"), ("pack", "unpack"),
        ]

        method_names = [b.method_name.lower() for b in blocks]

        for word1, word2 in antonym_pairs:
            has_word1 = any(word1 in n for n in method_names)
            has_word2 = any(word2 in n for n in method_names)
            if has_word1 and has_word2:
                return True

        if self._has_variant_mix(method_names):
            return True

        if self._has_modal_mix(method_names):
            return True

        for i, name1 in enumerate(method_names):
            for j in range(i + 1, len(method_names)):
                name2 = method_names[j]
                if "_no_" in name1 and name1.replace("_no_", "_") == name2:
                    return True
                if "_no_" in name2 and name2.replace("_no_", "_") == name1:
                    return True
                if ("_strict" in name1 and name1.replace("_strict", "") == name2) or (
                    "_strict" in name2 and name2.replace("_strict", "") == name1
                ):
                    return True

        return False

    def _has_variant_mix(self, method_names: List[str]) -> bool:
        variants = ["_no_signal", "_strict", "_unsafe", "_legacy", "_fast", "_slow", "_v2"]

        for variant in variants:
            has_variant = any(variant in name for name in method_names)
            has_non_variant = any(variant not in name for name in method_names)
            if has_variant and has_non_variant:
                return True

        return False

    def _has_modal_mix(self, method_names: List[str]) -> bool:
        modal_prefixes = ["can_", "is_", "has_"]

        for prefix in modal_prefixes:
            has_modal = any(name.startswith(prefix) for name in method_names)
            has_non_modal = any(not name.startswith(prefix) for name in method_names)
            if has_modal and has_non_modal:
                return True

        return False

    def _calculate_name_similarity(self, blocks: List[FunctionalBlock]) -> float:
        if len(blocks) < 2:
            return 1.0

        from difflib import SequenceMatcher

        sims: List[float] = []
        for i, b1 in enumerate(blocks):
            for j in range(i + 1, len(blocks)):
                b2 = blocks[j]
                sims.append(SequenceMatcher(None, b1.method_name.lower(), b2.method_name.lower()).ratio())

        return (sum(sims) / len(sims)) if sims else 0.0
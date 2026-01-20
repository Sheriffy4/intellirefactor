"""
Method analysis and grouping logic for AutoRefactor.

This module provides the MethodAnalyzer class that handles method grouping,
cohesion analysis, and risk assessment for refactoring operations.

The MethodAnalyzer is responsible for:
- Grouping methods by responsibility using keyword matching
- Clustering methods by cohesion (shared attributes/calls)
- Assessing extraction risk for individual methods
- Identifying private method dependencies

This module replaces the method analysis logic that was previously embedded
in the AutoRefactor class, improving separation of concerns and testability.

Classes:
    MethodAnalyzer: Main analyzer for method grouping and risk assessment

Example:
    >>> analyzer = MethodAnalyzer(
    ...     responsibility_keywords={'validation': ['validate', 'check']},
    ...     min_methods_for_extraction=2
    ... )
    >>> groups = analyzer.group_methods_by_responsibility(methods, module_names)
"""

from __future__ import annotations

import ast
import logging
from typing import Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# Import types from auto_refactor (will be available after full refactoring)
try:
    from .auto_refactor import MethodInfo
except ImportError:
    # Temporary fallback during refactoring
    from typing import NamedTuple
    from enum import Enum, auto

    class DecoratorType(Enum):
        NONE = auto()
        STATICMETHOD = auto()
        CLASSMETHOD = auto()
        PROPERTY = auto()

    class MethodInfo(NamedTuple):
        node: ast.FunctionDef
        name: str
        is_async: bool
        decorator_type: DecoratorType
        called_methods: FrozenSet[str]
        used_attributes: FrozenSet[str]
        used_names: FrozenSet[str]
        dangerous_reasons: FrozenSet[str]
        module_level_deps: FrozenSet[str]
        bare_self_used: bool


class MethodAnalyzer:
    """
    Analyzes and groups methods by responsibility and cohesion.

    This class provides sophisticated method analysis capabilities including:
    - Keyword-based responsibility grouping
    - Cohesion-based clustering using Jaccard similarity
    - Risk assessment for method extraction
    - Private method dependency tracking

    The analyzer uses a multi-stage approach:
    1. Group methods by responsibility keywords
    2. Cluster remaining methods by cohesion
    3. Assess extraction risks
    4. Assign private methods to groups

    Attributes:
        responsibility_keywords: Dict mapping group names to keyword lists
        cohesion_cluster_other: Whether to cluster ungrouped methods
        cohesion_similarity_threshold: Minimum similarity for clustering (0.0-1.0)
        cohesion_stop_features: Stop words to ignore in cohesion analysis
        min_methods_for_extraction: Minimum methods required for a group

    Example:
        >>> analyzer = MethodAnalyzer(
        ...     responsibility_keywords={'storage': ['load', 'save', 'file']},
        ...     cohesion_similarity_threshold=0.3,
        ...     min_methods_for_extraction=2
        ... )
        >>> groups, private_groups, unextracted, init, dunder, dangerous = \\
        ...     analyzer.group_methods_by_responsibility(methods, module_names)
    """

    def __init__(
        self,
        responsibility_keywords: Dict[str, List[str]],
        cohesion_cluster_other: bool = True,
        cohesion_similarity_threshold: float = 0.30,
        cohesion_stop_features: FrozenSet[str] = frozenset(),
        min_methods_for_extraction: int = 1,
    ):
        """
        Initialize MethodAnalyzer.

        Args:
            responsibility_keywords: Mapping of responsibility names to keyword lists
            cohesion_cluster_other: Whether to cluster "other" methods by cohesion
            cohesion_similarity_threshold: Jaccard similarity threshold for clustering
            cohesion_stop_features: Features to ignore in cohesion analysis
            min_methods_for_extraction: Minimum methods required to extract a group
        """
        self.responsibility_keywords = responsibility_keywords
        self.cohesion_cluster_other = cohesion_cluster_other
        self.cohesion_similarity_threshold = cohesion_similarity_threshold
        self.cohesion_stop_features = cohesion_stop_features
        self.min_methods_for_extraction = min_methods_for_extraction

    def is_public_extractable(self, method: MethodInfo) -> bool:
        """
        Check if a method is public and safe to extract.

        Args:
            method: Method to check

        Returns:
            True if method is public and has no dangerous patterns
        """
        return (not method.name.startswith("_")) and (
            len(method.dangerous_reasons) == 0
        )

    def group_methods_by_responsibility(
        self,
        class_node: ast.ClassDef,
        all_methods: List[MethodInfo],
        init_method: Optional[MethodInfo] = None,
        dunder_methods: Optional[List[MethodInfo]] = None,
    ) -> Tuple[
        Dict[str, List[MethodInfo]],  # method_groups
        Dict[str, List[MethodInfo]],  # private_by_group
        List[MethodInfo],  # unextracted
        Dict[str, Set[str]],  # dangerous_methods
    ]:
        """
        Group methods by responsibility using keyword matching and cohesion.

        Args:
            class_node: AST node of the class being analyzed
            all_methods: List of all methods (excluding __init__ and dunder methods)
            init_method: The __init__ method if present
            dunder_methods: List of dunder methods

        Returns:
            Tuple of (method_groups, private_by_group, unextracted, dangerous_methods)
        """
        # Separate public and private methods
        public_methods = [m for m in all_methods if not m.name.startswith("_")]
        private_methods = [
            m
            for m in all_methods
            if m.name.startswith("_") and not m.name.startswith("__")
        ]

        # Track dangerous methods
        dangerous_methods: Dict[str, Set[str]] = {}
        for m in all_methods:
            if m.dangerous_reasons:
                dangerous_methods[m.name] = set(m.dangerous_reasons)

        # Initialize groups
        groups: Dict[str, List[MethodInfo]] = {
            k: [] for k in self.responsibility_keywords
        }
        other: List[MethodInfo] = []

        # Group public methods by keyword matching
        for m in public_methods:
            name = m.name.lower()
            scores: Dict[str, int] = {}
            for group, words in self.responsibility_keywords.items():
                score = sum(1 for w in words if w in name)
                if score:
                    scores[group] = score
            if scores:
                best = max(scores, key=lambda k: scores[k])
                groups[best].append(m)
            else:
                other.append(m)

        # Handle "other" methods with cohesion clustering
        if other:
            if (
                self.cohesion_cluster_other
                and len(other) >= self.min_methods_for_extraction * 2
            ):
                clusters = self.cluster_methods_by_cohesion(
                    other, self.cohesion_similarity_threshold
                )
                for idx, cluster in enumerate(clusters, start=1):
                    groups[f"misc_{idx}"] = cluster
            else:
                groups["other"] = other

        # Remove empty groups
        groups = {k: v for k, v in groups.items() if v}

        # Determine which groups have enough extractable methods
        extract_groups: Set[str] = set()
        for g, ms in groups.items():
            extractable = [m for m in ms if self.is_public_extractable(m)]
            if len(extractable) >= self.min_methods_for_extraction:
                extract_groups.add(g)

        # Move methods from non-extractable groups to unextracted
        unextracted: List[MethodInfo] = []
        for g in list(groups.keys()):
            if g not in extract_groups:
                unextracted.extend(groups[g])
                del groups[g]

        # Assign private methods to groups
        private_by_group = self.assign_private_to_groups(groups, private_methods)

        # Add unassigned private methods to unextracted
        assigned_private: Set[str] = set()
        for ms in private_by_group.values():
            assigned_private.update(m.name for m in ms)

        for pm in private_methods:
            if pm.name not in assigned_private:
                unextracted.append(pm)

        return groups, private_by_group, unextracted, dangerous_methods

    def cluster_methods_by_cohesion(
        self,
        methods: List[MethodInfo],
        threshold: float,
    ) -> List[List[MethodInfo]]:
        """
        Cluster methods by cohesion using Jaccard similarity.

        Args:
            methods: Methods to cluster
            threshold: Jaccard similarity threshold (0.0 to 1.0)

        Returns:
            List of method clusters, sorted by size (largest first)
        """
        # Extract features for each method
        feats: Dict[str, Set[str]] = {}
        for m in methods:
            f = set(m.used_attributes) | set(m.called_methods)
            f -= set(self.cohesion_stop_features)
            feats[m.name] = f

        def jaccard(a: Set[str], b: Set[str]) -> float:
            """Calculate Jaccard similarity between two sets."""
            u = a | b
            return 0.0 if not u else (len(a & b) / len(u))

        # Build adjacency graph
        names = [m.name for m in methods]
        adj: Dict[str, Set[str]] = {n: set() for n in names}

        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if jaccard(feats[a], feats[b]) >= threshold:
                    adj[a].add(b)
                    adj[b].add(a)

        # Find connected components (clusters)
        seen: Set[str] = set()
        clusters: List[List[str]] = []

        for n in names:
            if n in seen:
                continue
            stack = [n]
            comp: List[str] = []
            seen.add(n)
            while stack:
                cur = stack.pop()
                comp.append(cur)
                for nb in adj[cur]:
                    if nb not in seen:
                        seen.add(nb)
                        stack.append(nb)
            clusters.append(comp)

        # Sort clusters by size (largest first)
        clusters.sort(key=len, reverse=True)

        # Convert method names back to MethodInfo objects
        by_name = {m.name: m for m in methods}
        return [[by_name[n] for n in cluster] for cluster in clusters]

    def assign_private_to_groups(
        self,
        public_groups: Dict[str, List[MethodInfo]],
        private_methods: List[MethodInfo],
    ) -> Dict[str, List[MethodInfo]]:
        """
        Assign private methods to public method groups based on call relationships.

        Uses transitive closure to find all private methods called by each group.

        Args:
            public_groups: Groups of public methods
            private_methods: List of private methods to assign

        Returns:
            Mapping of group names to lists of assigned private methods
        """
        private_names = {m.name for m in private_methods}
        private_by_name = {m.name: m for m in private_methods}

        def safe_private(m: MethodInfo) -> bool:
            """Check if private method is safe to extract."""
            return len(m.dangerous_reasons) == 0

        # Initialize result with empty sets
        result: Dict[str, Set[str]] = {g: set() for g in public_groups}

        # First pass: direct calls from public methods
        for g, methods in public_groups.items():
            for m in methods:
                called_priv = set(m.called_methods) & private_names
                for pn in called_priv:
                    pm = private_by_name.get(pn)
                    if pm and safe_private(pm):
                        result[g].add(pn)

        # Transitive closure: find private methods called by other private methods
        changed = True
        limit = len(private_methods) + 1
        it = 0
        while changed and it < limit:
            it += 1
            changed = False
            for g, assigned in result.items():
                new_calls: Set[str] = set()
                for pn in list(assigned):
                    pm = private_by_name.get(pn)
                    if not pm:
                        continue
                    trans = (set(pm.called_methods) & private_names) - assigned
                    for t in trans:
                        tm = private_by_name.get(t)
                        if tm and safe_private(tm):
                            new_calls.add(t)
                if new_calls:
                    assigned |= new_calls
                    changed = True

        # Convert sets to sorted lists of MethodInfo
        return {
            g: [private_by_name[n] for n in sorted(ns)]
            for g, ns in result.items()
            if ns
        }

    def assess_risk(
        self,
        components: List[str],
        method_groups: Dict[str, List[MethodInfo]],
        private_by_group: Dict[str, List[MethodInfo]],
        dangerous_methods: Dict[str, Set[str]],
    ) -> str:
        """
        Assess refactoring risk level based on complexity metrics.

        Args:
            components: List of component names to be extracted
            method_groups: Groups of public methods
            private_by_group: Private methods assigned to each group
            dangerous_methods: Methods with dangerous patterns

        Returns:
            Risk level: "low", "medium", or "high"
        """
        # Too many components is high risk
        if len(components) > 8:
            return "high"

        dangerous_count = len(dangerous_methods)

        # Calculate cross-group dependencies
        all_private_names: Set[str] = set()
        for ms in private_by_group.values():
            all_private_names.update(m.name for m in ms)

        cross_deps = 0
        for g, ms in method_groups.items():
            group_priv = {m.name for m in private_by_group.get(g, [])}
            for m in ms:
                external = (set(m.called_methods) & all_private_names) - group_priv
                cross_deps += len(external)

        # Risk assessment
        if dangerous_count > 10 or cross_deps > 6 or len(components) > 5:
            return "high"
        if dangerous_count > 0 or cross_deps > 0 or len(components) > 2:
            return "medium"
        return "low"

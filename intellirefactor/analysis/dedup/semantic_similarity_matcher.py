"""
Semantic Similarity Matcher for IntelliRefactor

This module implements semantic similarity matching that goes beyond structural
clone detection to find functionally similar methods based on operation sequences,
semantic patterns, and behavioral similarity.
"""

import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum

from intellirefactor.analysis.foundation.models import (
    Evidence,
    DeepMethodInfo,
    SemanticCategory,
    ResponsibilityMarker,
)


class SimilarityType(Enum):
    """Types of semantic similarity."""

    STRUCTURAL = "structural"  # Same AST structure, different names
    FUNCTIONAL = "functional"  # Same operation sequence/pattern
    BEHAVIORAL = "behavioral"  # Same side effects and responsibilities
    HYBRID = "hybrid"  # Multiple similarity types


class OperationPattern(Enum):
    """Common operation patterns in methods."""

    VALIDATE_TRANSFORM_RETURN = "validate_transform_return"
    FETCH_PROCESS_STORE = "fetch_process_store"
    CHECK_EXECUTE_NOTIFY = "check_execute_notify"
    LOAD_VALIDATE_SAVE = "load_validate_save"
    PARSE_TRANSFORM_FORMAT = "parse_transform_format"
    AUTHENTICATE_AUTHORIZE_EXECUTE = "authenticate_authorize_execute"
    ACQUIRE_PROCESS_RELEASE = "acquire_process_release"
    UNKNOWN = "unknown"


@dataclass
class OperationSequence:
    """Represents a sequence of operations performed by a method."""

    operations: List[str]
    pattern: OperationPattern
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "operations": list(self.operations),
            "pattern": self.pattern.value,
            "confidence": float(self.confidence),
            "metadata": self.metadata,
        }


@dataclass
class SimilarityMatch:
    """Represents a semantic similarity match between methods."""

    method1: DeepMethodInfo
    method2: DeepMethodInfo
    similarity_type: SimilarityType
    similarity_score: float
    confidence: float
    evidence: Evidence
    merge_strategy: Optional[str] = None
    common_operations: List[str] = field(default_factory=list)
    differences: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError(
                f"Similarity score must be between 0.0 and 1.0, got {self.similarity_score}"
            )
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"Confidence must be between 0.0 and 1.0, got {self.confidence}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method1": self.method1.to_dict() if hasattr(self.method1, "to_dict") else str(self.method1),
            "method2": self.method2.to_dict() if hasattr(self.method2, "to_dict") else str(self.method2),
            "similarity_type": self.similarity_type.value,
            "similarity_score": float(self.similarity_score),
            "confidence": float(self.confidence),
            "evidence": self.evidence.to_dict() if hasattr(self.evidence, "to_dict") else self.evidence,
            "merge_strategy": self.merge_strategy,
            "common_operations": list(self.common_operations),
            "differences": list(self.differences),
        }


class SemanticSimilarityMatcher:
    """
    Matches methods based on semantic similarity using operation sequences,
    structural patterns, and behavioral analysis.
    """

    def __init__(
        self,
        structural_threshold: float = 0.8,
        functional_threshold: float = 0.7,
        behavioral_threshold: float = 0.6,
        min_confidence: float = 0.5,
    ):
        """
        Initialize the semantic similarity matcher.

        Args:
            structural_threshold: Minimum similarity for structural matches
            functional_threshold: Minimum similarity for functional matches
            behavioral_threshold: Minimum similarity for behavioral matches
            min_confidence: Minimum confidence threshold for matches
        """
        self.structural_threshold = structural_threshold
        self.functional_threshold = functional_threshold
        self.behavioral_threshold = behavioral_threshold
        self.min_confidence = min_confidence
        self.logger = logging.getLogger(__name__)

        # Cache for operation sequences
        self._operation_cache: Dict[str, OperationSequence] = {}

    def clear_cache(self) -> None:
        """Clear the operation sequence cache."""
        self._operation_cache.clear()

    def find_similar_methods(
        self,
        methods: List[DeepMethodInfo],
        target_method: Optional[DeepMethodInfo] = None,
        similarity_types: Optional[Set[SimilarityType]] = None,
    ) -> List[SimilarityMatch]:
        """
        Find semantically similar methods.

        Args:
            methods: List of methods to analyze
            target_method: Optional specific method to find similarities for
            similarity_types: Types of similarity to search for

        Returns:
            List of similarity matches sorted by similarity score
        """
        if not methods:
            return []

        similarity_types = similarity_types or {
            SimilarityType.STRUCTURAL,
            SimilarityType.FUNCTIONAL,
            SimilarityType.BEHAVIORAL,
        }
        matches = []

        # If target method specified, find similarities to it
        if target_method:
            for method in methods:
                if method.qualified_name != target_method.qualified_name:
                    match = self._compare_methods(target_method, method, similarity_types)
                    if match and match.confidence >= self.min_confidence:
                        matches.append(match)
        else:
            # Find all pairwise similarities
            for i, method1 in enumerate(methods):
                for method2 in methods[i + 1 :]:
                    match = self._compare_methods(method1, method2, similarity_types)
                    if match and match.confidence >= self.min_confidence:
                        matches.append(match)

        # Sort by similarity score (descending)
        matches.sort(key=lambda m: m.similarity_score, reverse=True)
        return matches

    def _compare_methods(
        self,
        method1: DeepMethodInfo,
        method2: DeepMethodInfo,
        similarity_types: Set[SimilarityType],
    ) -> Optional[SimilarityMatch]:
        """Compare two methods for semantic similarity."""
        similarities = {}

        # Structural similarity
        if SimilarityType.STRUCTURAL in similarity_types:
            structural_score = self._calculate_structural_similarity(method1, method2)
            if structural_score >= self.structural_threshold:
                similarities[SimilarityType.STRUCTURAL] = structural_score

        # Functional similarity
        if SimilarityType.FUNCTIONAL in similarity_types:
            functional_score = self._calculate_functional_similarity(method1, method2)
            if functional_score >= self.functional_threshold:
                similarities[SimilarityType.FUNCTIONAL] = functional_score

        # Behavioral similarity
        if SimilarityType.BEHAVIORAL in similarity_types:
            behavioral_score = self._calculate_behavioral_similarity(method1, method2)
            if behavioral_score >= self.behavioral_threshold:
                similarities[SimilarityType.BEHAVIORAL] = behavioral_score

        if not similarities:
            return None

        # Determine primary similarity type and overall score
        if len(similarities) > 1:
            primary_type = SimilarityType.HYBRID
            # Weighted average for hybrid similarity
            weights = {
                SimilarityType.STRUCTURAL: 0.4,
                SimilarityType.FUNCTIONAL: 0.4,
                SimilarityType.BEHAVIORAL: 0.2,
            }

            weighted_sum = sum(similarities[t] * weights.get(t, 0.3) for t in similarities)
            total_weight = sum(weights.get(t, 0.3) for t in similarities)

            overall_score = weighted_sum / total_weight if total_weight > 0 else 0.0
        else:
            primary_type = next(iter(similarities.keys()))
            overall_score = next(iter(similarities.values()))

        # Calculate confidence based on multiple factors
        confidence = self._calculate_match_confidence(method1, method2, similarities)

        if confidence < self.min_confidence:
            return None

        # Generate evidence
        evidence = self._generate_similarity_evidence(
            method1, method2, similarities, primary_type, confidence
        )

        # Analyze common operations and differences
        common_ops, differences = self._analyze_method_differences(method1, method2)

        # Suggest merge strategy
        merge_strategy = self._suggest_merge_strategy(method1, method2, primary_type, overall_score)

        return SimilarityMatch(
            method1=method1,
            method2=method2,
            similarity_type=primary_type,
            similarity_score=overall_score,
            confidence=confidence,
            evidence=evidence,
            merge_strategy=merge_strategy,
            common_operations=common_ops,
            differences=differences,
        )

    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

    def _jaccard_similarity_empty_as_unknown(
        self, set1: Set, set2: Set, empty_score: float = 0.0
    ) -> float:
        """Calculate Jaccard similarity, treating empty sets as unknown/no match."""
        if not set1 and not set2:
            return empty_score
        return self._jaccard_similarity(set1, set2)

    def _calculate_structural_similarity(
        self, method1: DeepMethodInfo, method2: DeepMethodInfo
    ) -> float:
        """Calculate structural similarity based on AST fingerprints."""
        fp1 = method1.ast_fingerprint
        fp2 = method2.ast_fingerprint

        # Handle None/empty fingerprints
        if not fp1 and not fp2:
            return 0.5  # Both missing implies uncertainty
        if not fp1 or not fp2:
            return 0.0

        if fp1 == fp2:
            return 1.0

        # Calculate similarity based on fingerprint comparison
        fp1_parts = set(fp1.split("_"))
        fp2_parts = set(fp2.split("_"))

        return self._jaccard_similarity(fp1_parts, fp2_parts)

    def _calculate_functional_similarity(
        self, method1: DeepMethodInfo, method2: DeepMethodInfo
    ) -> float:
        """Calculate functional similarity based on operation sequences."""
        # Get operation sequences for both methods
        seq1 = self._extract_operation_sequence(method1)
        seq2 = self._extract_operation_sequence(method2)

        # Filter unknown operations to avoid false positives
        ops1 = {
            op.strip().lower() for op in seq1.operations if op and op.strip().lower() != "unknown"
        }
        ops2 = {
            op.strip().lower() for op in seq2.operations if op and op.strip().lower() != "unknown"
        }

        if not ops1 and not ops2:
            return 0.0

        # Compare operation sequences
        if seq1.pattern == seq2.pattern and seq1.pattern != OperationPattern.UNKNOWN:
            # Same operation pattern - high similarity
            base_score = 0.8

            # Compare actual operations
            ops_similarity = self._jaccard_similarity(ops1, ops2)
            return base_score + (0.2 * ops_similarity)

        # Different patterns - compare operations directly
        return self._jaccard_similarity(ops1, ops2)

    def _calculate_behavioral_similarity(
        self, method1: DeepMethodInfo, method2: DeepMethodInfo
    ) -> float:
        """Calculate behavioral similarity based on responsibilities and side effects."""
        # 1. Compare semantic categories
        if method1.semantic_category == method2.semantic_category:
            category_score = 1.0 if method1.semantic_category != SemanticCategory.UNKNOWN else 0.5
        else:
            category_score = 0.0

        # 2. Compare responsibility markers
        markers1 = method1.responsibility_markers or set()
        markers2 = method2.responsibility_markers or set()
        # Treat empty markers as unknown/no match to avoid false positives
        resp_score = self._jaccard_similarity_empty_as_unknown(markers1, markers2, empty_score=0.0)

        # 3. Compare side effects
        se1 = method1.side_effects or set()
        se2 = method2.side_effects or set()
        # Empty side effects means pure function, so match is 1.0
        side_effect_score = self._jaccard_similarity(se1, se2)

        # Weighted average
        w_cat, w_resp, w_se = 0.6, 0.2, 0.2
        return (category_score * w_cat + resp_score * w_resp + side_effect_score * w_se) / (
            w_cat + w_resp + w_se
        )

    def _extract_operation_sequence(self, method: DeepMethodInfo) -> OperationSequence:
        """Extract operation sequence from method."""
        # Include fingerprint in cache key to handle method changes
        cache_key = f"{method.qualified_name}:{method.ast_fingerprint or ''}"
        if cache_key in self._operation_cache:
            return self._operation_cache[cache_key]

        # Use operation signature if available
        if method.operation_signature:
            operations = [
                op.strip().lower() for op in method.operation_signature.split("->") if op.strip()
            ]
            pattern = self._identify_operation_pattern(operations)
            confidence = 0.9  # High confidence from existing signature
        else:
            # Extract from method characteristics
            operations = self._infer_operations_from_method(method)
            pattern = self._identify_operation_pattern(operations)
            confidence = 0.6  # Lower confidence from inference

        markers = method.responsibility_markers or set()

        sequence = OperationSequence(
            operations=operations,
            pattern=pattern,
            confidence=confidence,
            metadata={
                "method_name": method.name,
                "semantic_category": method.semantic_category.value,
                "responsibilities": [r.value for r in markers],
            },
        )

        self._operation_cache[cache_key] = sequence
        return sequence

    def _infer_operations_from_method(self, method: DeepMethodInfo) -> List[str]:
        """Infer operations from method characteristics."""
        operations = []

        # Infer from semantic category
        if method.semantic_category == SemanticCategory.VALIDATION:
            operations.append("validate")
        elif method.semantic_category == SemanticCategory.TRANSFORMATION:
            operations.append("transform")
        elif method.semantic_category == SemanticCategory.PERSISTENCE:
            operations.extend(["load", "save"])
        elif method.semantic_category == SemanticCategory.COMPUTATION:
            operations.append("compute")

        # Infer from responsibility markers
        markers = method.responsibility_markers or set()
        for responsibility in markers:
            if responsibility == ResponsibilityMarker.VALIDATION:
                operations.append("validate")
            elif responsibility == ResponsibilityMarker.DATA_ACCESS:
                operations.append("fetch")
            elif responsibility == ResponsibilityMarker.BUSINESS_LOGIC:
                operations.append("process")
            elif responsibility == ResponsibilityMarker.FORMATTING:
                operations.append("format")
            elif responsibility == ResponsibilityMarker.PERSISTENCE:
                operations.append("store")

        # Infer from side effects
        side_effects = method.side_effects or set()
        if "file_io" in side_effects:
            operations.extend(["read", "write"])
        if "network" in side_effects:
            operations.extend(["fetch", "send"])
        if "state_modification" in side_effects:
            operations.append("modify")

        # Infer from method name patterns
        name_lower = method.name.lower()
        if "check" in name_lower:
            operations.append("check")
        elif "validate" in name_lower:
            operations.append("validate")

        if "transform" in name_lower or "convert" in name_lower:
            operations.append("transform")
        if "process" in name_lower:
            operations.append("process")
        if "save" in name_lower or "store" in name_lower:
            operations.append("store")
        if "load" in name_lower or "get" in name_lower or "fetch" in name_lower:
            operations.append("fetch")
        if "format" in name_lower:
            operations.append("format")

        # Remove duplicates while preserving order using dict.fromkeys
        unique_operations = list(dict.fromkeys(operations))

        return unique_operations or ["unknown"]

    def _identify_operation_pattern(self, operations: List[str]) -> OperationPattern:
        """Identify the operation pattern from a sequence of operations."""
        ops_set = set(operations)
        ops_str = "->".join(operations).lower()

        # Check specific patterns first
        if {"load", "validate", "save"}.issubset(ops_set):
            return OperationPattern.LOAD_VALIDATE_SAVE

        if {"fetch", "process", "store"}.issubset(ops_set):
            return OperationPattern.FETCH_PROCESS_STORE

        if {"check", "execute"}.issubset(ops_set) and ("notify" in ops_str or "send" in ops_str):
            return OperationPattern.CHECK_EXECUTE_NOTIFY

        if {"parse", "transform", "format"}.issubset(ops_set):
            return OperationPattern.PARSE_TRANSFORM_FORMAT

        if {"authenticate", "authorize", "execute"}.issubset(ops_set):
            return OperationPattern.AUTHENTICATE_AUTHORIZE_EXECUTE

        if len([op for op in operations if op in ["acquire", "lock", "release", "unlock"]]) >= 2:
            return OperationPattern.ACQUIRE_PROCESS_RELEASE

        # Check general patterns last
        if ("validate" in ops_set or "check" in ops_set) and (
            "transform" in ops_set or "process" in ops_set
        ):
            return OperationPattern.VALIDATE_TRANSFORM_RETURN

        return OperationPattern.UNKNOWN

    def _calculate_match_confidence(
        self,
        method1: DeepMethodInfo,
        method2: DeepMethodInfo,
        similarities: Dict[SimilarityType, float],
    ) -> float:
        """Calculate confidence in the similarity match."""
        confidence = 0.0

        # Base confidence from similarity scores (weighted more heavily)
        avg_similarity = sum(similarities.values()) / len(similarities)
        confidence += avg_similarity * 0.6  # Increased weight

        # Confidence from method characteristics
        if method1.confidence >= 0.8 and method2.confidence >= 0.8:
            confidence += 0.15
        elif method1.confidence >= 0.6 and method2.confidence >= 0.6:
            confidence += 0.1

        # Confidence from complexity similarity
        if method1.cyclomatic_complexity > 0 and method2.cyclomatic_complexity > 0:
            complexity_ratio = min(
                method1.cyclomatic_complexity, method2.cyclomatic_complexity
            ) / max(method1.cyclomatic_complexity, method2.cyclomatic_complexity)
            confidence += complexity_ratio * 0.1

        # Confidence from size similarity
        if method1.lines_of_code > 0 and method2.lines_of_code > 0:
            size_ratio = min(method1.lines_of_code, method2.lines_of_code) / max(
                method1.lines_of_code, method2.lines_of_code
            )
            confidence += size_ratio * 0.1

        # Confidence from multiple similarity types
        if len(similarities) > 1:
            confidence += 0.05

        return min(1.0, confidence)

    def _generate_similarity_evidence(
        self,
        method1: DeepMethodInfo,
        method2: DeepMethodInfo,
        similarities: Dict[SimilarityType, float],
        primary_type: SimilarityType,
        confidence: float,
    ) -> Evidence:
        """Generate evidence for the similarity match."""
        # Filter out None file references
        file_refs = [
            ref for ref in [method1.file_reference, method2.file_reference] if ref is not None
        ]

        code_snippets = [
            f"Method: {method1.qualified_name}",
            f"Signature: {method1.signature}",
            f"Method: {method2.qualified_name}",
            f"Signature: {method2.signature}",
        ]

        # Build description
        similarity_types = list(similarities.keys())
        if len(similarity_types) == 1:
            desc = f"{primary_type.value.title()} similarity detected"
        else:
            types_str = ", ".join(t.value for t in similarity_types)
            desc = f"Hybrid similarity detected ({types_str})"

        # Add similarity scores to metadata
        markers1 = method1.responsibility_markers or set()
        markers2 = method2.responsibility_markers or set()

        metadata = {
            "similarity_type": primary_type.value,
            "similarity_scores": {t.value: score for t, score in similarities.items()},
            "method1_category": method1.semantic_category.value,
            "method2_category": method2.semantic_category.value,
            "method1_responsibilities": [r.value for r in markers1],
            "method2_responsibilities": [r.value for r in markers2],
        }

        return Evidence(
            description=desc,
            confidence=confidence,
            file_references=file_refs,
            code_snippets=code_snippets,
            metadata=metadata,
        )

    def _analyze_method_differences(
        self, method1: DeepMethodInfo, method2: DeepMethodInfo
    ) -> Tuple[List[str], List[str]]:
        """Analyze common operations and differences between methods."""
        # Get operation sequences
        seq1 = self._extract_operation_sequence(method1)
        seq2 = self._extract_operation_sequence(method2)

        ops1 = set(seq1.operations)
        ops2 = set(seq2.operations)

        common_operations = sorted(list(ops1 & ops2))

        differences = []

        # Operations only in method1
        only_in_1 = ops1 - ops2
        if only_in_1:
            differences.append(f"Only in {method1.name}: {', '.join(only_in_1)}")

        # Operations only in method2
        only_in_2 = ops2 - ops1
        if only_in_2:
            differences.append(f"Only in {method2.name}: {', '.join(only_in_2)}")

        # Complexity differences
        if abs(method1.cyclomatic_complexity - method2.cyclomatic_complexity) > 2:
            differences.append(
                f"Complexity: {method1.name}={method1.cyclomatic_complexity}, {method2.name}={method2.cyclomatic_complexity}"
            )

        # Size differences
        if abs(method1.lines_of_code - method2.lines_of_code) > 5:
            differences.append(
                f"Size: {method1.name}={method1.lines_of_code} LOC, {method2.name}={method2.lines_of_code} LOC"
            )

        # Side effect differences
        se1 = method1.side_effects or set()
        se2 = method2.side_effects or set()
        side_effects_1 = se1 - se2
        side_effects_2 = se2 - se1

        if side_effects_1:
            differences.append(f"Side effects only in {method1.name}: {', '.join(side_effects_1)}")
        if side_effects_2:
            differences.append(f"Side effects only in {method2.name}: {', '.join(side_effects_2)}")

        return common_operations, differences

    def _suggest_merge_strategy(
        self,
        method1: DeepMethodInfo,
        method2: DeepMethodInfo,
        similarity_type: SimilarityType,
        similarity_score: float,
    ) -> str:
        """Suggest a merge strategy based on similarity analysis."""
        if similarity_score >= 0.9:
            if similarity_type == SimilarityType.STRUCTURAL:
                return "Extract common method - methods are nearly identical"
            elif similarity_type == SimilarityType.FUNCTIONAL:
                return "Create template method - same functionality with parameter differences"
            elif similarity_type == SimilarityType.BEHAVIORAL:
                return "Unify behavior - methods have same responsibilities"
            else:
                return (
                    "Full merge candidate - methods are highly similar across multiple dimensions"
                )

        elif similarity_score >= 0.8:
            if similarity_type == SimilarityType.FUNCTIONAL:
                return "Parameterize differences - extract common logic with parameters"
            elif similarity_type == SimilarityType.BEHAVIORAL:
                return "Create shared interface - methods have similar behavior patterns"
            else:
                return "Partial merge - extract common parts into shared utility"

        elif similarity_score >= 0.7:
            return "Consider refactoring - methods share significant similarities"

        else:
            return "Monitor for patterns - methods have some similarities worth tracking"

    def get_similarity_statistics(self, matches: List[SimilarityMatch]) -> Dict[str, Any]:
        """Get statistics about similarity matches."""
        if not matches:
            return {
                "total_matches": 0,
                "by_type": {},
                "by_score_range": {},
                "average_similarity": 0.0,
                "average_confidence": 0.0,
            }

        # Group by similarity type
        by_type = {}
        for match in matches:
            stype = match.similarity_type.value
            by_type[stype] = by_type.get(stype, 0) + 1

        # Group by score ranges
        by_score_range = {
            "high (≥0.8)": len([m for m in matches if m.similarity_score >= 0.8]),
            "medium (0.6-0.8)": len([m for m in matches if 0.6 <= m.similarity_score < 0.8]),
            "low (<0.6)": len([m for m in matches if m.similarity_score < 0.6]),
        }

        # Calculate averages
        avg_similarity = sum(m.similarity_score for m in matches) / len(matches)
        avg_confidence = sum(m.confidence for m in matches) / len(matches)

        return {
            "total_matches": len(matches),
            "by_type": by_type,
            "by_score_range": by_score_range,
            "average_similarity": round(avg_similarity, 3),
            "average_confidence": round(avg_confidence, 3),
            "methods_involved": len(
                set(m.method1.qualified_name for m in matches)
                | set(m.method2.qualified_name for m in matches)
            ),
        }

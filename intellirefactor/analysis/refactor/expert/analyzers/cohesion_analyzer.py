"""
Cohesion Matrix Analyzer for expert refactoring analysis.

Analyzes method-attribute relationships to determine class cohesion
and suggest optimal boundaries for class decomposition.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..models import CohesionMatrix, MethodGroup

logger = logging.getLogger(__name__)


class CohesionMatrixAnalyzer:
    """Analyzes class cohesion through method-attribute relationships."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)

    def build_cohesion_matrix(self, class_ast: ast.ClassDef) -> CohesionMatrix:
        logger.info("Building cohesion matrix for class %s", class_ast.name)

        methods = self._extract_methods(class_ast)
        attributes = self._extract_attributes(class_ast)

        matrix = self._build_matrix(methods, attributes, class_ast)
        cohesion_scores = self._calculate_cohesion_scores(methods, attributes, matrix)
        suggested_groups = self._suggest_method_groups(methods, attributes, class_ast)

        cm = CohesionMatrix(
            methods=methods,
            attributes=attributes,
            matrix=matrix,
            cohesion_scores=cohesion_scores,
            suggested_groups=suggested_groups,
        )

        logger.info("Cohesion matrix built: %d methods, %d attributes", len(methods), len(attributes))
        return cm

    def export_detailed_cohesion_matrix(self, matrix: CohesionMatrix) -> Dict[str, Any]:
        method_analysis: Dict[str, Dict[str, Any]] = {}

        for i, method_name in enumerate(matrix.methods):
            accessed = [matrix.attributes[j] for j, v in enumerate(matrix.matrix[i]) if v > 0]
            cohesion_score = float(matrix.cohesion_scores.get(method_name, 0.0))

            if cohesion_score == 0.0:
                recommendation = "МОЖНО вынести/сделать @staticmethod: не использует self.*"
            elif cohesion_score < 0.25:
                recommendation = "Слабо связан с состоянием класса — кандидат на извлечение"
            elif cohesion_score > 0.7:
                recommendation = "СИЛЬНО связан — оставить в классе"
            else:
                recommendation = "Умеренно связан — зависит от контекста"

            method_analysis[method_name] = {
                "reads": accessed,
                "writes": [],  # точный write требует отдельного RW-анализа (см. ниже)
                "cohesion": cohesion_score,
                "recommendation": recommendation,
                "attributes_used": len(accessed),
                "total_attributes": len(matrix.attributes),
            }

        attribute_analysis: Dict[str, Dict[str, Any]] = {}
        for j, attr_name in enumerate(matrix.attributes):
            used_by = []
            total_usage = 0.0

            for i, method_name in enumerate(matrix.methods):
                access_count = matrix.matrix[i][j]
                if access_count > 0:
                    used_by.append({"method": method_name, "access_count": access_count})
                    total_usage += access_count

            attribute_analysis[attr_name] = {
                "used_by_methods": used_by,
                "total_usage": total_usage,
                "method_count": len(used_by),
                "usage_density": len(used_by) / len(matrix.methods) if matrix.methods else 0.0,
            }

        extraction_groups = [
            {
                "methods": g.methods,
                "shared_attributes": g.shared_attributes,
                "cohesion_score": g.cohesion_score,
                "recommendation": g.extraction_recommendation,
                "potential_class_name": self._suggest_class_name(g.methods, g.shared_attributes),
            }
            for g in matrix.suggested_groups
        ]

        avg = (
            sum(matrix.cohesion_scores.values()) / len(matrix.cohesion_scores)
            if matrix.cohesion_scores
            else 0.0
        )

        class_metrics = {
            "total_methods": len(matrix.methods),
            "total_attributes": len(matrix.attributes),
            "average_cohesion": avg,
            "high_cohesion_methods": sum(1 for s in matrix.cohesion_scores.values() if s > 0.7),
            "low_cohesion_methods": sum(1 for s in matrix.cohesion_scores.values() if s < 0.25),
            "extraction_candidates": sum(1 for s in matrix.cohesion_scores.values() if s < 0.25),
        }

        return {
            "cohesion_matrix": {
                "methods": matrix.methods,
                "attributes": matrix.attributes,
                "matrix": matrix.matrix,
                "method_analysis": method_analysis,
                "attribute_analysis": attribute_analysis,
            },
            "extraction_recommendations": extraction_groups,
            "class_metrics": class_metrics,
            "detailed_recommendations": self._generate_detailed_recommendations(
                method_analysis, attribute_analysis, class_metrics
            ),
        }

    # -------------------------
    # Internals
    # -------------------------

    def _extract_methods(self, class_ast: ast.ClassDef) -> List[str]:
        return [n.name for n in class_ast.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]

    def _extract_attributes(self, class_ast: ast.ClassDef) -> List[str]:
        """
        FIX: support AnnAssign (self.x: T = ...), Assign, AugAssign, Delete.
        """
        attrs: Set[str] = set()

        def add_target(t: ast.AST) -> None:
            if isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name) and t.value.id == "self":
                attrs.add(t.attr)

        for node in ast.walk(class_ast):
            if isinstance(node, ast.Assign):
                for t in node.targets:
                    add_target(t)
            elif isinstance(node, ast.AnnAssign):
                add_target(node.target)
            elif isinstance(node, ast.AugAssign):
                add_target(node.target)
            elif isinstance(node, ast.Delete):
                for t in node.targets:
                    add_target(t)

        return sorted(attrs)

    def _find_method_node(self, class_ast: ast.ClassDef, method_name: str) -> Optional[ast.AST]:
        for n in class_ast.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == method_name:
                return n
        return None

    def _count_attr_access(self, method_node: ast.AST, attr_name: str) -> int:
        count = 0
        for n in ast.walk(method_node):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "self":
                if n.attr == attr_name:
                    count += 1
        return count

    def _build_matrix(self, methods: List[str], attributes: List[str], class_ast: ast.ClassDef) -> List[List[float]]:
        matrix: List[List[float]] = [[0.0 for _ in attributes] for _ in methods]
        for i, m in enumerate(methods):
            mn = self._find_method_node(class_ast, m)
            if not mn:
                continue
            for j, a in enumerate(attributes):
                matrix[i][j] = float(self._count_attr_access(mn, a))
        return matrix

    def _calculate_cohesion_scores(self, methods: List[str], attributes: List[str], matrix: List[List[float]]) -> Dict[str, float]:
        scores: Dict[str, float] = {}
        if not attributes:
            return {m: 0.0 for m in methods}

        total = len(attributes)
        for i, m in enumerate(methods):
            touched = sum(1 for v in matrix[i] if v > 0)
            scores[m] = touched / total
        return scores

    def _suggest_method_groups(self, methods: List[str], attributes: List[str], class_ast: ast.ClassDef) -> List[MethodGroup]:
        """
        Better grouping: Jaccard similarity between attribute sets.
        """
        if not methods or not attributes:
            return []

        method_sets: Dict[str, Set[str]] = {}
        for m in methods:
            mn = self._find_method_node(class_ast, m)
            s: Set[str] = set()
            if mn:
                for a in attributes:
                    if self._count_attr_access(mn, a) > 0:
                        s.add(a)
            method_sets[m] = s

        def jaccard(a: Set[str], b: Set[str]) -> float:
            if not a and not b:
                return 1.0
            if not a or not b:
                return 0.0
            return len(a & b) / len(a | b)

        # union-find clustering
        parent = {m: m for m in methods}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        TH = 0.6
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                m1, m2 = methods[i], methods[j]
                if jaccard(method_sets[m1], method_sets[m2]) >= TH and (method_sets[m1] or method_sets[m2]):
                    union(m1, m2)

        clusters: Dict[str, List[str]] = {}
        for m in methods:
            clusters.setdefault(find(m), []).append(m)

        groups: List[MethodGroup] = []
        for ms in clusters.values():
            if len(ms) < 2:
                continue
            shared = set.intersection(*(method_sets[m] for m in ms)) if ms else set()
            score = len(shared) / len(attributes) if attributes else 0.0
            groups.append(
                MethodGroup(
                    methods=sorted(ms),
                    shared_attributes=sorted(shared),
                    cohesion_score=score,
                    extraction_recommendation=f"Consider extracting {len(ms)} methods sharing {len(shared)} attributes (Jaccard clustering)",
                )
            )
        return groups

    def _suggest_class_name(self, methods: List[str], attributes: List[str]) -> str:
        words = []
        for m in methods:
            words.extend([w for w in m.split("_") if len(w) > 2])
        for a in attributes:
            words.extend([w for w in a.split("_") if len(w) > 2])
        if not words:
            return "ExtractedClass"
        freq: Dict[str, int] = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        return f"{max(freq.items(), key=lambda x: x[1])[0].title()}Handler"

    def _generate_detailed_recommendations(self, method_analysis: Dict[str, Any], attribute_analysis: Dict[str, Any], class_metrics: Dict[str, Any]) -> List[str]:
        recs: List[str] = []
        low = [m for m, d in method_analysis.items() if float(d["cohesion"]) < 0.25]
        if low:
            recs.append(f"Consider extracting {len(low)} low-cohesion methods: {', '.join(low[:5])}{'...' if len(low) > 5 else ''}")

        unused = [a for a, d in attribute_analysis.items() if int(d["method_count"]) == 0]
        if unused:
            recs.append(f"Remove {len(unused)} unused attributes: {', '.join(unused)}")

        avg = float(class_metrics.get("average_cohesion", 0.0))
        if avg < 0.35:
            recs.append("Class has low overall cohesion - consider major refactoring")
        return recs
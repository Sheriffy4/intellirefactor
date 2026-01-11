"""
Cohesion Matrix Analyzer for expert refactoring analysis.

Analyzes method-attribute relationships to determine class cohesion
and suggest optimal boundaries for class decomposition.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional

from ..models import (
    CohesionMatrix,
    MethodGroup,
)

logger = logging.getLogger(__name__)


class CohesionMatrixAnalyzer:
    """Analyzes class cohesion through method-attribute relationships."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)

    def build_cohesion_matrix(self, class_ast: ast.ClassDef) -> CohesionMatrix:
        """
        Build cohesion matrix for a class.
        
        Args:
            class_ast: AST node of the class to analyze
            
        Returns:
            CohesionMatrix with method-attribute relationships
        """
        logger.info(f"Building cohesion matrix for class {class_ast.name}")
        
        # Extract methods and attributes
        methods = self._extract_methods(class_ast)
        attributes = self._extract_attributes(class_ast)
        
        # Build the matrix
        matrix = self._build_matrix(methods, attributes, class_ast)
        
        # Calculate cohesion scores
        cohesion_scores = self._calculate_cohesion_scores(methods, attributes, matrix)
        
        # Suggest method groups
        suggested_groups = self._suggest_method_groups(methods, attributes, matrix)
        
        cohesion_matrix = CohesionMatrix(
            methods=methods,
            attributes=attributes,
            matrix=matrix,
            cohesion_scores=cohesion_scores,
            suggested_groups=suggested_groups
        )
        
        logger.info(f"Cohesion matrix built: {len(methods)} methods, {len(attributes)} attributes")
        return cohesion_matrix

    def export_detailed_cohesion_matrix(self, matrix: CohesionMatrix) -> Dict[str, any]:
        """
        Export detailed cohesion matrix as requested by experts.
        
        Returns:
            Dictionary with method-attribute relationships and extraction recommendations
        """
        # Create detailed method analysis
        method_analysis = {}
        for i, method_name in enumerate(matrix.methods):
            reads = []
            writes = []
            
            # Analyze what each method reads and writes
            for j, attr_name in enumerate(matrix.attributes):
                access_count = matrix.matrix[i][j]
                if access_count > 0:
                    # For now, assume all access is reading
                    # In a more sophisticated version, we'd distinguish read vs write
                    reads.append(attr_name)
            
            cohesion_score = matrix.cohesion_scores.get(method_name, 0.0)
            
            # Determine extraction recommendation
            if cohesion_score == 0.0:
                recommendation = "МОЖНО вынести в отдельный класс/статик (не использует self атрибуты)"
            elif cohesion_score < 0.3:
                recommendation = "Слабо связан, кандидат на извлечение"
            elif cohesion_score > 0.7:
                recommendation = "СИЛЬНО связан, оставить в классе"
            else:
                recommendation = "Умеренно связан, решение зависит от контекста"
            
            method_analysis[method_name] = {
                "reads": reads,
                "writes": writes,  # TODO: Implement write detection
                "cohesion": cohesion_score,
                "recommendation": recommendation,
                "attributes_used": len(reads),
                "total_attributes": len(matrix.attributes)
            }
        
        # Create attribute usage analysis
        attribute_analysis = {}
        for j, attr_name in enumerate(matrix.attributes):
            used_by_methods = []
            total_usage = 0
            
            for i, method_name in enumerate(matrix.methods):
                access_count = matrix.matrix[i][j]
                if access_count > 0:
                    used_by_methods.append({
                        "method": method_name,
                        "access_count": access_count
                    })
                    total_usage += access_count
            
            attribute_analysis[attr_name] = {
                "used_by_methods": used_by_methods,
                "total_usage": total_usage,
                "method_count": len(used_by_methods),
                "usage_density": len(used_by_methods) / len(matrix.methods) if matrix.methods else 0.0
            }
        
        # Generate extraction groups
        extraction_groups = []
        for group in matrix.suggested_groups:
            extraction_groups.append({
                "methods": group.methods,
                "shared_attributes": group.shared_attributes,
                "cohesion_score": group.cohesion_score,
                "recommendation": group.extraction_recommendation,
                "potential_class_name": self._suggest_class_name(group.methods, group.shared_attributes)
            })
        
        # Calculate overall class metrics
        class_metrics = {
            "total_methods": len(matrix.methods),
            "total_attributes": len(matrix.attributes),
            "average_cohesion": sum(matrix.cohesion_scores.values()) / len(matrix.cohesion_scores) if matrix.cohesion_scores else 0.0,
            "high_cohesion_methods": sum(1 for score in matrix.cohesion_scores.values() if score > 0.7),
            "low_cohesion_methods": sum(1 for score in matrix.cohesion_scores.values() if score < 0.3),
            "extraction_candidates": sum(1 for score in matrix.cohesion_scores.values() if score < 0.3)
        }
        
        return {
            "cohesion_matrix": {
                "methods": matrix.methods,
                "attributes": matrix.attributes,
                "matrix": matrix.matrix,
                "method_analysis": method_analysis,
                "attribute_analysis": attribute_analysis
            },
            "extraction_recommendations": extraction_groups,
            "class_metrics": class_metrics,
            "detailed_recommendations": self._generate_detailed_recommendations(method_analysis, attribute_analysis, class_metrics)
        }

    def _suggest_class_name(self, methods: List[str], attributes: List[str]) -> str:
        """Suggest a name for an extracted class based on methods and attributes."""
        # Simple heuristic based on common patterns
        method_words = []
        for method in methods:
            method_words.extend(method.split('_'))
        
        attr_words = []
        for attr in attributes:
            attr_words.extend(attr.split('_'))
        
        # Find common themes
        all_words = method_words + attr_words
        word_counts = {}
        for word in all_words:
            if len(word) > 2:  # Skip short words
                word_counts[word] = word_counts.get(word, 0) + 1
        
        if word_counts:
            most_common = max(word_counts.items(), key=lambda x: x[1])
            return f"{most_common[0].title()}Handler"
        else:
            return "ExtractedClass"

    def _generate_detailed_recommendations(self, method_analysis: Dict, attribute_analysis: Dict, class_metrics: Dict) -> List[str]:
        """Generate detailed recommendations for class refactoring."""
        recommendations = []
        
        # Analyze low cohesion methods
        low_cohesion_methods = [
            method for method, data in method_analysis.items() 
            if data["cohesion"] < 0.3
        ]
        
        if low_cohesion_methods:
            recommendations.append(
                f"Consider extracting {len(low_cohesion_methods)} low-cohesion methods: {', '.join(low_cohesion_methods[:3])}{'...' if len(low_cohesion_methods) > 3 else ''}"
            )
        
        # Analyze unused attributes
        unused_attributes = [
            attr for attr, data in attribute_analysis.items()
            if data["method_count"] == 0
        ]
        
        if unused_attributes:
            recommendations.append(
                f"Remove {len(unused_attributes)} unused attributes: {', '.join(unused_attributes)}"
            )
        
        # Analyze highly used attributes
        core_attributes = [
            attr for attr, data in attribute_analysis.items()
            if data["usage_density"] > 0.7
        ]
        
        if core_attributes:
            recommendations.append(
                f"Core attributes that should stay in main class: {', '.join(core_attributes)}"
            )
        
        # Overall class assessment
        if class_metrics["average_cohesion"] < 0.4:
            recommendations.append("Class has low overall cohesion - consider major refactoring")
        elif class_metrics["average_cohesion"] > 0.7:
            recommendations.append("Class has good cohesion - minimal refactoring needed")
        
        return recommendations

    def _extract_methods(self, class_ast: ast.ClassDef) -> List[str]:
        """Extract method names from the class."""
        methods = []
        for node in class_ast.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods.append(node.name)
        return methods

    def _extract_attributes(self, class_ast: ast.ClassDef) -> List[str]:
        """Extract instance attributes from the class."""
        attributes = set()
        
        # Look for self.attribute assignments
        for node in ast.walk(class_ast):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute):
                        if isinstance(target.value, ast.Name) and target.value.id == 'self':
                            attributes.add(target.attr)
        
        return list(attributes)

    def _build_matrix(self, methods: List[str], attributes: List[str], class_ast: ast.ClassDef) -> List[List[float]]:
        """Build the method-attribute access matrix."""
        matrix = [[0.0 for _ in attributes] for _ in methods]
        
        # For each method, check which attributes it accesses
        for method_idx, method_name in enumerate(methods):
            method_node = self._find_method_node(class_ast, method_name)
            if method_node:
                for attr_idx, attr_name in enumerate(attributes):
                    access_count = self._count_attribute_access(method_node, attr_name)
                    matrix[method_idx][attr_idx] = float(access_count)
        
        return matrix

    def _find_method_node(self, class_ast: ast.ClassDef, method_name: str) -> Optional[ast.FunctionDef]:
        """Find the AST node for a specific method."""
        for node in class_ast.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
                return node
        return None

    def _count_attribute_access(self, method_node: ast.FunctionDef, attr_name: str) -> int:
        """Count how many times a method accesses an attribute."""
        count = 0
        for node in ast.walk(method_node):
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == 'self' and node.attr == attr_name:
                    count += 1
        return count

    def _calculate_cohesion_scores(self, methods: List[str], attributes: List[str], matrix: List[List[float]]) -> Dict[str, float]:
        """Calculate cohesion scores for each method."""
        scores = {}
        
        for method_idx, method_name in enumerate(methods):
            if not attributes:
                scores[method_name] = 0.0
                continue
            
            # Calculate cohesion as the ratio of accessed attributes
            accessed_attrs = sum(1 for access in matrix[method_idx] if access > 0)
            cohesion_score = accessed_attrs / len(attributes)
            scores[method_name] = cohesion_score
        
        return scores

    def _suggest_method_groups(self, methods: List[str], attributes: List[str], matrix: List[List[float]]) -> List[MethodGroup]:
        """Suggest method groups based on shared attribute access."""
        # This is a simplified grouping algorithm
        # In practice, you'd use more sophisticated clustering
        
        groups = []
        
        # Group methods that access similar attributes
        method_signatures = {}
        for method_idx, method_name in enumerate(methods):
            signature = tuple(1 if access > 0 else 0 for access in matrix[method_idx])
            if signature not in method_signatures:
                method_signatures[signature] = []
            method_signatures[signature].append(method_name)
        
        # Create groups for signatures with multiple methods
        for signature, group_methods in method_signatures.items():
            if len(group_methods) > 1:
                shared_attrs = [attr for attr_idx, attr in enumerate(attributes) if signature[attr_idx] == 1]
                cohesion_score = sum(signature) / len(attributes) if attributes else 0.0
                
                group = MethodGroup(
                    methods=group_methods,
                    shared_attributes=shared_attrs,
                    cohesion_score=cohesion_score,
                    extraction_recommendation=f"Consider extracting {len(group_methods)} methods that work with {len(shared_attrs)} shared attributes"
                )
                groups.append(group)
        
        return groups

    def suggest_class_decomposition(self, matrix: CohesionMatrix) -> List[MethodGroup]:
        """
        Suggest how to decompose a class based on cohesion analysis.
        
        Args:
            matrix: CohesionMatrix to analyze
            
        Returns:
            List of suggested method groups for extraction
        """
        # Return the already calculated suggested groups
        return matrix.suggested_groups

    def calculate_cohesion_metrics(self, matrix: CohesionMatrix) -> Dict[str, float]:
        """
        Calculate various cohesion metrics.
        
        Args:
            matrix: CohesionMatrix to analyze
            
        Returns:
            Dictionary of cohesion metrics
        """
        metrics = {}
        
        # Overall class cohesion (average method cohesion)
        if matrix.cohesion_scores:
            metrics['average_cohesion'] = sum(matrix.cohesion_scores.values()) / len(matrix.cohesion_scores)
        else:
            metrics['average_cohesion'] = 0.0
        
        # Method distribution
        metrics['total_methods'] = len(matrix.methods)
        metrics['total_attributes'] = len(matrix.attributes)
        
        # Cohesion distribution
        high_cohesion_methods = sum(1 for score in matrix.cohesion_scores.values() if score > 0.7)
        low_cohesion_methods = sum(1 for score in matrix.cohesion_scores.values() if score < 0.3)
        
        metrics['high_cohesion_methods'] = high_cohesion_methods
        metrics['low_cohesion_methods'] = low_cohesion_methods
        
        # Suggested groups metrics
        metrics['suggested_groups'] = len(matrix.suggested_groups)
        
        return metrics
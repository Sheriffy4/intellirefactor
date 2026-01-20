"""
Concrete Duplication Analyzer for expert refactoring analysis.

Finds concrete duplicate code fragments with extraction suggestions.
"""

from __future__ import annotations

import ast
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from ..models import DuplicateFragment

logger = logging.getLogger(__name__)


class ConcreteDeduplicationAnalyzer:
    """Finds concrete duplicate code fragments."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)
        self.min_duplicate_lines = 3  # Minimum lines for duplicate detection

    def find_concrete_duplicates(self) -> List[DuplicateFragment]:
        """
        Find concrete duplicate code fragments.
        
        Returns:
            List of DuplicateFragment objects
        """
        logger.info("Finding concrete duplicate code fragments...")
        
        # Read and parse the target module
        try:
            content = self.target_module.read_text(encoding='utf-8')
            lines = content.splitlines()
        except (OSError, UnicodeDecodeError) as e:
            logger.error(f"Error reading {self.target_module}: {e}")
            return []
        
        # Find duplicates using different strategies
        duplicates = []
        
        # Strategy 1: Line-based duplicates
        line_duplicates = self._find_line_based_duplicates(lines)
        duplicates.extend(line_duplicates)
        
        # Strategy 2: AST-based duplicates (structural similarity)
        try:
            tree = ast.parse(content)
            ast_duplicates = self._find_ast_based_duplicates(tree, lines)
            duplicates.extend(ast_duplicates)
        except SyntaxError:
            logger.warning("Could not parse AST for structural duplicate detection")
        
        # Remove duplicates and sort by estimated savings
        unique_duplicates = self._deduplicate_fragments(duplicates)
        unique_duplicates.sort(key=lambda x: x.estimated_savings, reverse=True)
        
        logger.info(f"Found {len(unique_duplicates)} concrete duplicate fragments")
        return unique_duplicates

    def export_detailed_duplicates(self, duplicates: List[DuplicateFragment]) -> Dict[str, Any]:
        """
        Export detailed duplicate information as requested by experts.
        
        Returns:
            Dictionary with specific locations, similarity scores, and extraction suggestions
        """
        detailed_duplicates = []
        
        for i, duplicate in enumerate(duplicates):
            # Extract actual code content for each location
            locations_with_code = []
            for file_path, start_line, end_line in duplicate.locations:
                try:
                    if file_path == str(self.target_module):
                        content = self.target_module.read_text(encoding='utf-8')
                        lines = content.splitlines()
                        code_fragment = '\n'.join(lines[start_line-1:end_line])
                    else:
                        # Handle other files if needed
                        code_fragment = "Code from external file"
                    
                    locations_with_code.append({
                        "file": file_path,
                        "start_line": start_line,
                        "end_line": end_line,
                        "lines_count": end_line - start_line + 1,
                        "code": code_fragment
                    })
                except Exception as e:
                    logger.warning(f"Could not extract code for {file_path}:{start_line}-{end_line}: {e}")
                    locations_with_code.append({
                        "file": file_path,
                        "start_line": start_line,
                        "end_line": end_line,
                        "lines_count": end_line - start_line + 1,
                        "code": "Could not extract code"
                    })
            
            # Categorize the duplicate pattern
            pattern_type = self._categorize_duplicate_pattern(duplicate.content)
            
            detailed_duplicate = {
                "id": i + 1,
                "locations": locations_with_code,
                "similarity": duplicate.similarity_score,
                "pattern": pattern_type,
                "extraction_suggestion": duplicate.extraction_suggestion,
                "estimated_savings": duplicate.estimated_savings,
                "occurrences": len(duplicate.locations),
                "content_preview": duplicate.content[:200] + "..." if len(duplicate.content) > 200 else duplicate.content
            }
            detailed_duplicates.append(detailed_duplicate)
        
        # Group duplicates by pattern type
        patterns_summary = {}
        total_savings = 0
        
        for dup in detailed_duplicates:
            pattern = dup["pattern"]
            if pattern not in patterns_summary:
                patterns_summary[pattern] = {
                    "count": 0,
                    "total_savings": 0,
                    "examples": []
                }
            
            patterns_summary[pattern]["count"] += 1
            patterns_summary[pattern]["total_savings"] += dup["estimated_savings"]
            total_savings += dup["estimated_savings"]
            
            if len(patterns_summary[pattern]["examples"]) < 3:
                patterns_summary[pattern]["examples"].append({
                    "id": dup["id"],
                    "locations": len(dup["locations"]),
                    "savings": dup["estimated_savings"]
                })
        
        return {
            "duplicates": detailed_duplicates,
            "summary": {
                "total_duplicates": len(detailed_duplicates),
                "total_savings": total_savings,
                "patterns": patterns_summary
            },
            "recommendations": self._generate_duplicate_recommendations(detailed_duplicates)
        }

    def _categorize_duplicate_pattern(self, content: str) -> str:
        """Categorize the type of duplicate pattern."""
        content_lower = content.lower()
        
        if 'sni' in content_lower and ('parse' in content_lower or 'extract' in content_lower):
            return "SNI parsing logic"
        elif 'parameter' in content_lower and ('normal' in content_lower or 'valid' in content_lower):
            return "Parameter normalization"
        elif 'log' in content_lower and ('operation' in content_lower or 'debug' in content_lower):
            return "Logging operations"
        elif 'attack' in content_lower and ('dispatch' in content_lower or 'execute' in content_lower):
            return "Attack execution logic"
        elif 'error' in content_lower and ('handle' in content_lower or 'exception' in content_lower):
            return "Error handling"
        elif 'config' in content_lower or 'setting' in content_lower:
            return "Configuration handling"
        elif 'import' in content_lower:
            return "Import statements"
        elif 'def ' in content_lower:
            return "Method definitions"
        else:
            return "General code pattern"

    def _generate_duplicate_recommendations(self, duplicates: List[Dict]) -> List[str]:
        """Generate specific recommendations for handling duplicates."""
        recommendations = []
        
        # Group by pattern type
        pattern_counts = {}
        for dup in duplicates:
            pattern = dup["pattern"]
            pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Generate pattern-specific recommendations
        for pattern, count in pattern_counts.items():
            if pattern == "SNI parsing logic":
                recommendations.append(f"Extract SNI parsing into a dedicated utility class ({count} duplicates found)")
            elif pattern == "Parameter normalization":
                recommendations.append(f"Create a centralized parameter validation module ({count} duplicates found)")
            elif pattern == "Logging operations":
                recommendations.append(f"Implement a logging decorator or utility function ({count} duplicates found)")
            elif pattern == "Attack execution logic":
                recommendations.append(f"Refactor attack execution into a strategy pattern ({count} duplicates found)")
            elif pattern == "Error handling":
                recommendations.append(f"Create common exception handling utilities ({count} duplicates found)")
            else:
                recommendations.append(f"Consider extracting common {pattern.lower()} ({count} duplicates found)")
        
        # Add general recommendations
        total_savings = sum(dup["estimated_savings"] for dup in duplicates)
        if total_savings > 500:
            recommendations.append(f"High duplication detected: {total_savings} lines could be saved through refactoring")
        
        return recommendations

    def _find_line_based_duplicates(self, lines: List[str]) -> List[DuplicateFragment]:
        """Find duplicates based on exact line matching."""
        duplicates = []
        
        # Create sliding windows of different sizes
        for window_size in range(self.min_duplicate_lines, min(20, len(lines) // 2)):
            window_locations = {}
            
            for start_line in range(len(lines) - window_size + 1):
                # Extract window content (normalized)
                window_lines = []
                for i in range(window_size):
                    line = lines[start_line + i].strip()
                    if line and not line.startswith('#'):  # Skip empty lines and comments
                        window_lines.append(line)
                
                if len(window_lines) < self.min_duplicate_lines:
                    continue
                
                # Create hash of the window
                window_content = '\n'.join(window_lines)
                # usedforsecurity kwarg is not supported on all Python builds (can raise TypeError).
                try:
                    window_hash = hashlib.md5(
                        window_content.encode("utf-8"), usedforsecurity=False
                    ).hexdigest()  # type: ignore[arg-type]
                except TypeError:
                    window_hash = hashlib.md5(window_content.encode("utf-8")).hexdigest()
                
                if window_hash not in window_locations:
                    window_locations[window_hash] = []
                
                window_locations[window_hash].append({
                    'start_line': start_line + 1,  # 1-based line numbers
                    'end_line': start_line + window_size,
                    'content': window_content
                })
            
            # Find windows that appear multiple times
            for window_hash, locations in window_locations.items():
                if len(locations) > 1:
                    # Create duplicate fragment
                    fragment = DuplicateFragment(
                        content=locations[0]['content'],
                        locations=[(str(self.target_module), loc['start_line'], loc['end_line']) 
                                 for loc in locations],
                        similarity_score=1.0,  # Exact match
                        extraction_suggestion=self._suggest_extraction(locations[0]['content'], len(locations)),
                        estimated_savings=window_size * (len(locations) - 1)
                    )
                    duplicates.append(fragment)
        
        return duplicates

    def _find_ast_based_duplicates(self, tree: ast.Module, lines: List[str]) -> List[DuplicateFragment]:
        """Find duplicates based on AST structural similarity."""
        duplicates = []
        
        # Extract function/method bodies for comparison
        function_bodies = self._extract_function_bodies(tree)
        
        # Compare function bodies for structural similarity
        for i, (name1, body1, lines1) in enumerate(function_bodies):
            for j, (name2, body2, lines2) in enumerate(function_bodies[i+1:], i+1):
                similarity = self._calculate_ast_similarity(body1, body2)
                
                if similarity > 0.8:  # High structural similarity
                    # Extract the actual code content
                    content1 = self._extract_lines_content(lines, lines1[0]-1, lines1[1])
                    content2 = self._extract_lines_content(lines, lines2[0]-1, lines2[1])
                    
                    fragment = DuplicateFragment(
                        content=f"Similar structure in {name1} and {name2}:\n{content1}",
                        locations=[
                            (str(self.target_module), lines1[0], lines1[1]),
                            (str(self.target_module), lines2[0], lines2[1])
                        ],
                        similarity_score=similarity,
                        extraction_suggestion=f"Extract common logic from {name1} and {name2}",
                        estimated_savings=min(lines1[1] - lines1[0], lines2[1] - lines2[0])
                    )
                    duplicates.append(fragment)
        
        return duplicates

    def _extract_function_bodies(self, tree: ast.Module) -> List[Tuple[str, List[ast.stmt], Tuple[int, int]]]:
        """Extract function bodies for comparison."""
        bodies = []
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                start_line = getattr(node, 'lineno', 0)
                end_line = getattr(node, 'end_lineno', start_line)
                
                bodies.append((
                    node.name,
                    node.body,
                    (start_line, end_line)
                ))
        
        return bodies

    def _calculate_ast_similarity(self, body1: List[ast.stmt], body2: List[ast.stmt]) -> float:
        """Calculate structural similarity between two AST bodies."""
        if not body1 or not body2:
            return 0.0
        
        # Simple similarity based on statement types
        types1 = [type(stmt).__name__ for stmt in body1]
        types2 = [type(stmt).__name__ for stmt in body2]
        
        # Calculate Jaccard similarity
        set1 = set(types1)
        set2 = set(types2)
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0

    def _extract_lines_content(self, lines: List[str], start: int, end: int) -> str:
        """Extract content from specific line range."""
        if start < 0 or end > len(lines):
            return ""
        
        return '\n'.join(lines[start:end])

    def _suggest_extraction(self, content: str, occurrence_count: int) -> str:
        """Suggest how to extract the duplicate code."""
        lines = content.split('\n')
        
        # Analyze the content to suggest extraction strategy
        if any('def ' in line for line in lines):
            return f"Extract method (appears {occurrence_count} times)"
        elif any('class ' in line for line in lines):
            return f"Extract class or refactor inheritance (appears {occurrence_count} times)"
        elif any('import ' in line or 'from ' in line for line in lines):
            return f"Consolidate imports (appears {occurrence_count} times)"
        elif len(lines) > 10:
            return f"Extract large code block into method (appears {occurrence_count} times)"
        else:
            return f"Extract common code pattern (appears {occurrence_count} times)"

    def _deduplicate_fragments(self, duplicates: List[DuplicateFragment]) -> List[DuplicateFragment]:
        """Remove overlapping duplicate fragments."""
        if not duplicates:
            return []
        
        # Sort by estimated savings (descending)
        sorted_duplicates = sorted(duplicates, key=lambda x: x.estimated_savings, reverse=True)
        
        unique_duplicates = []
        used_ranges = set()
        
        for duplicate in sorted_duplicates:
            # Check if this duplicate overlaps with already selected ones
            overlaps = False
            for file_path, start_line, end_line in duplicate.locations:
                range_key = (file_path, start_line, end_line)
                if any(self._ranges_overlap(range_key, used_range) for used_range in used_ranges):
                    overlaps = True
                    break
            
            if not overlaps:
                unique_duplicates.append(duplicate)
                for file_path, start_line, end_line in duplicate.locations:
                    used_ranges.add((file_path, start_line, end_line))
        
        return unique_duplicates

    def _ranges_overlap(self, range1: Tuple[str, int, int], range2: Tuple[str, int, int]) -> bool:
        """Check if two line ranges overlap."""
        file1, start1, end1 = range1
        file2, start2, end2 = range2
        
        if file1 != file2:
            return False
        
        return not (end1 < start2 or end2 < start1)

    def analyze_duplication_patterns(self, duplicates: List[DuplicateFragment]) -> Dict[str, Any]:
        """
        Analyze patterns in code duplication.
        
        Args:
            duplicates: List of duplicate fragments
            
        Returns:
            Dictionary with duplication analysis
        """
        if not duplicates:
            return {
                'total_duplicates': 0,
                'total_savings': 0,
                'patterns': []
            }
        
        total_savings = sum(dup.estimated_savings for dup in duplicates)
        
        # Categorize by extraction suggestion
        patterns = {}
        for dup in duplicates:
            suggestion_type = dup.extraction_suggestion.split('(')[0].strip()
            if suggestion_type not in patterns:
                patterns[suggestion_type] = {
                    'count': 0,
                    'total_savings': 0,
                    'examples': []
                }
            
            patterns[suggestion_type]['count'] += 1
            patterns[suggestion_type]['total_savings'] += dup.estimated_savings
            if len(patterns[suggestion_type]['examples']) < 3:
                patterns[suggestion_type]['examples'].append({
                    'content_preview': dup.content[:100] + '...' if len(dup.content) > 100 else dup.content,
                    'locations': len(dup.locations),
                    'savings': dup.estimated_savings
                })
        
        return {
            'total_duplicates': len(duplicates),
            'total_savings': total_savings,
            'patterns': patterns
        }
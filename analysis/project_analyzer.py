# analysis/project_analyzer.py

import ast
import logging
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Any, Set

from ..config import AnalysisConfig
from ..interfaces import (
    BaseProjectAnalyzer, 
    GenericAnalysisResult
)


class ProjectAnalyzer(BaseProjectAnalyzer):
    """
    Analyzes overall project structure and identifies refactoring opportunities.
    
    Extracted from recon project and made generic for any Python project.
    Implements the generic ProjectAnalyzerProtocol for compatibility with different project types.
    """
    
    # Default exclude patterns for source file discovery
    DEFAULT_EXCLUDE_NAMES = [
        '__pycache__', '.git', '.venv', 'venv', 'build', 'dist',
        'node_modules', '.idea', '.vscode', '.tox', '.nox'
    ]
    
    def __init__(self, config: Optional[AnalysisConfig] = None, project_root: str = "."):
        """Initialize the project analyzer with configuration."""
        self.config = config or AnalysisConfig()
        # Resolve project root immediately to avoid relative path issues later
        try:
            self.project_root = Path(project_root).resolve()
        except Exception:
            self.project_root = Path(project_root)
        self.logger = logging.getLogger(__name__)
        
    def _get_setting(self, name: str, default: Any) -> Any:
        """Get setting with fallback chain: analysis_settings -> config -> default."""
        # Try analysis_settings first
        settings = getattr(self.config, 'analysis_settings', None)
        if settings is not None and hasattr(settings, name):
            return getattr(settings, name)
        
        # Try top-level config
        if hasattr(self.config, name):
            return getattr(self.config, name)
            
        # Fallback to default
        return default

    def _is_excluded_by_pattern(self, path: Path, patterns: List[str]) -> bool:
        """Check if path matches any exclude pattern (handles glob patterns correctly)."""
        str_path = str(path)
        
        for pattern in patterns:
            # Паттерны с ** проверяем как подстроки
            if '**' in pattern:
                clean_pattern = pattern.replace('**/', '').replace('/**', '')
                if clean_pattern in str_path:
                    return True
            else:
                # Простые паттерны через fnmatch
                if path.match(pattern):
                    return True
        return False

    def identify_source_files(self, project_path: Union[str, Path]) -> List[str]:
        """Identify source files in the project."""
        try:
            project_path = Path(project_path).resolve()
        except Exception:
            project_path = Path(project_path)
            
        source_files = []
        
        # Get include patterns
        include_patterns = self._get_setting('include_patterns', ['**/*.py'])
        
        # Build exclude patterns (config + defaults)
        raw_excludes = self._get_setting('exclude_patterns', [])
        exclude_patterns = list(raw_excludes)
        
        for default_exclude in self.DEFAULT_EXCLUDE_NAMES:
            # Добавляем варианты с **/ и /** для покрытия всех случаев
            if f'**/{default_exclude}/**' not in exclude_patterns:
                exclude_patterns.append(f'**/{default_exclude}/**')
            if f'**/{default_exclude}' not in exclude_patterns:
                exclude_patterns.append(f'**/{default_exclude}')
        
        # Find files matching include patterns
        for pattern in include_patterns:
            try:
                files_found = list(project_path.glob(pattern))
                source_files.extend(files_found)
            except Exception as e:
                self.logger.warning(f"Error globbing pattern '{pattern}': {e}")
        
        # Filter out excluded files
        final_source_files = []
        for file_path in source_files:
            if not file_path.is_file():
                continue
            
            # Используем корректную проверку паттернов
            if not self._is_excluded_by_pattern(file_path, exclude_patterns):
                final_source_files.append(str(file_path))
        
        return list(set(final_source_files))
    
    def _safe_relpath(self, filepath: Path) -> str:
        """Safely calculate relative path, falling back to absolute if not relative."""
        try:
            return str(filepath.resolve().relative_to(self.project_root))
        except Exception:
            # Catch ValueError, OSError, etc.
            return str(filepath)

    def _safe_asdict(self, obj: Any) -> Any:
        """Safely convert object to dict if it is a dataclass, otherwise return __dict__ or repr."""
        if is_dataclass(obj):
            return asdict(obj)
        if hasattr(obj, '__dict__'):
            return {k: v for k, v in obj.__dict__.items() if not k.startswith('_')}
        return str(obj)

    def analyze_file(self, filepath: Path) -> GenericAnalysisResult:
        """Analyze a single Python file."""
        # Initialize defaults for error handling context
        rel_path = str(filepath)
        
        try:
            filepath = Path(filepath)
            try:
                # Resolve path safely inside try block to handle permission/symlink errors
                filepath = filepath.resolve()
            except Exception:
                # If resolve fails, keep original path object
                pass

            rel_path = self._safe_relpath(filepath)

            # utf-8-sig handles BOM if present
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                content = f.read()
                
            # Parse AST with filename for better error reporting
            tree = ast.parse(content, filename=str(filepath))
            
            # Basic metrics
            lines_count = len(content.splitlines())
            
            # Walk tree once to gather nodes
            nodes = list(ast.walk(tree))
            classes = [n for n in nodes if isinstance(n, ast.ClassDef)]
            
            # Categorize functions/methods
            all_func_defs = [n for n in nodes if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            
            # Identify methods inside classes using IDs for robustness
            class_method_ids = set()
            for cls in classes:
                for item in cls.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_method_ids.add(id(item))
            
            # Identify top-level module functions
            module_level_funcs = [
                n for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            
            # Calculate counts
            total_callables = len(all_func_defs)
            class_methods_count = len(class_method_ids)
            module_functions_count = len(module_level_funcs)
            nested_functions_count = total_callables - class_methods_count - module_functions_count
            
            # Complexity analysis
            complexity_score = self._calculate_complexity(tree)
            
            # Responsibility analysis (based on class methods)
            responsibilities_count = self._count_responsibilities(classes)
            
            # Dependency analysis
            dependencies_nodes_count = len([n for n in nodes if isinstance(n, (ast.Import, ast.ImportFrom))])
            unique_dependencies = self._count_unique_dependencies(tree)
            
            # Issue identification
            issues = []
            recommendations = []
            
            # Get settings using helper
            large_file_threshold = self._get_setting('large_file_threshold', 500)
            complexity_threshold = self._get_setting('complexity_threshold', 15.0)
            responsibilities_threshold = self._get_setting('responsibilities_threshold', 5)
            god_object_threshold = self._get_setting('god_object_threshold', 15)

            if lines_count > large_file_threshold:
                issues.append(f"Large file: {lines_count} lines (threshold: {large_file_threshold})")
                recommendations.append("Consider splitting into multiple modules")
                
            if complexity_score > complexity_threshold:
                issues.append(f"High complexity: {complexity_score:.1f} (threshold: {complexity_threshold})")
                recommendations.append("Extract methods into separate components")
                
            if responsibilities_count > responsibilities_threshold:
                issues.append(f"Multiple responsibilities: {responsibilities_count} (threshold: {responsibilities_threshold})")
                recommendations.append("Apply Single Responsibility Principle")
                
            # Check for God Objects
            for cls in classes:
                class_methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                if len(class_methods) > god_object_threshold:
                    issues.append(f"God Object: class {cls.name} has {len(class_methods)} methods (threshold: {god_object_threshold})")
                    recommendations.append(f"Split class {cls.name} into multiple components")
            
            # Create generic analysis result
            return GenericAnalysisResult(
                success=True,
                project_path=str(self.project_root),
                analysis_type="file_analysis",
                data={
                    "filepath": rel_path,
                    "lines_count": lines_count,
                    "classes_count": len(classes),
                    "total_callables": total_callables, # Renamed for clarity
                    "methods_count": total_callables,   # Backward compatibility alias
                    "functions_count": total_callables, # Backward compatibility alias
                    "module_functions_count": module_functions_count,
                    "class_methods_count": class_methods_count,
                    "nested_functions_count": max(0, nested_functions_count),
                    "responsibilities_count": responsibilities_count,
                    "dependencies_count": dependencies_nodes_count,
                    "unique_dependencies_count": len(unique_dependencies),
                    "test_coverage": 0.0,
                },
                metrics={
                    "complexity_score": float(complexity_score),
                    "lines_count": float(lines_count),
                    "classes_count": float(len(classes)),
                    "methods_count": float(total_callables),
                },
                issues=issues,
                recommendations=recommendations,
                metadata={
                    "analyzer": "ProjectAnalyzer",
                    "version": "1.4.0"
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing file {filepath}: {e}")
            return GenericAnalysisResult(
                success=False,
                project_path=str(self.project_root),
                analysis_type="file_analysis",
                data={
                    "filepath": rel_path,
                    "lines_count": 0,
                    "classes_count": 0,
                    "methods_count": 0,
                    "functions_count": 0,
                    "module_functions_count": 0,
                    "class_methods_count": 0,
                    "nested_functions_count": 0,
                    "responsibilities_count": 0,
                    "dependencies_count": 0,
                    "unique_dependencies_count": 0,
                    "test_coverage": 0.0,
                },
                metrics={},
                issues=[f"Analysis error: {str(e)}"],
                recommendations=[],
                metadata={
                    "analyzer": "ProjectAnalyzer",
                    "version": "1.4.0",
                    "error_type": type(e).__name__
                },
                timestamp=datetime.now()
            )
            
    def _calculate_complexity(self, tree: ast.AST) -> float:
        """Calculate simplified cyclomatic complexity."""
        complexity = 1
        
        for node in ast.walk(tree):
            # Control flow statements
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.IfExp)):
                complexity += 1
            # Exception handling
            elif isinstance(node, ast.Try):
                complexity += len(node.handlers)
                if node.orelse:
                    complexity += 1
            # Boolean operators (and/or)
            elif isinstance(node, ast.BoolOp):
                # Each operator adds a path (values - 1)
                complexity += max(1, len(node.values) - 1)
            # Comprehensions (each generator adds a loop)
            elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                complexity += len(node.generators)
                
        return float(complexity)
        
    def _count_responsibilities(self, classes: List[ast.ClassDef]) -> int:
        """Count responsibilities based on class methods."""
        if not classes:
            return 1  # Default to 1 if no classes found
            
        max_responsibilities = 1
        for cls in classes:
            methods = [n for n in cls.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            
            # Group methods by prefixes to determine responsibilities
            prefixes: Set[str] = set()
            for method in methods:
                # Skip dunder and private methods for responsibility counting
                name = method.name.lstrip('_')
                if name and '_' in name:
                    prefix = name.split('_', 1)[0]
                    if prefix:  # Ensure prefix is not empty
                        prefixes.add(prefix)
            
            # Heuristic: max of prefixes OR raw method count division
            # Ensure at least 1 responsibility if methods exist
            responsibilities = max(1, len(prefixes), (len(methods) + 4) // 5)
            max_responsibilities = max(max_responsibilities, responsibilities)
            
        return max_responsibilities

    def _count_unique_dependencies(self, tree: ast.AST) -> Set[str]:
        """Count unique import roots."""
        roots: Set[str] = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                for alias in n.names:
                    roots.add(alias.name.split(".", 1)[0])
            elif isinstance(n, ast.ImportFrom):
                if n.module:
                    roots.add(n.module.split(".", 1)[0])
        return roots
        
    def analyze_project(self, project_path: Union[str, Path]) -> GenericAnalysisResult:
        """
        Analyze an entire project and return comprehensive analysis results.
        
        Args:
            project_path: Path to the project to analyze
            
        Returns:
            GenericAnalysisResult containing comprehensive analysis results
        """
        try:
            project_path = Path(project_path).resolve()
        except Exception:
            project_path = Path(project_path)
            
        self.project_root = project_path
        
        if not project_path.exists():
            return GenericAnalysisResult(
                success=False,
                project_path=str(project_path),
                analysis_type="project_analysis",
                data={},
                metrics={},
                issues=[f"Project path does not exist: {project_path}"],
                recommendations=[],
                metadata={},
                timestamp=datetime.now()
            )

        if not project_path.is_dir():
            return GenericAnalysisResult(
                success=False,
                project_path=str(project_path),
                analysis_type="project_analysis",
                data={},
                metrics={},
                issues=[f"Project path is not a directory: {project_path}"],
                recommendations=[],
                metadata={},
                timestamp=datetime.now()
            )
        
        try:
            self.logger.info(f"Starting project analysis for: {project_path}")
            
            # Get project structure using base class method
            project_structure = self.get_project_structure(project_path)
            
            # Find all Python source files
            source_files = self.identify_source_files(project_path)
            
            self.logger.info(f"Found {len(source_files)} Python files for analysis")
            
            # Analyze each file
            file_analyses = []
            large_files = []
            complex_files = []
            god_objects = []
            total_lines = 0
            
            # Get settings using helper
            large_file_threshold = self._get_setting('large_file_threshold', 500)
            complexity_threshold = self._get_setting('complexity_threshold', 15.0)
            min_candidate_size = self._get_setting('min_candidate_size', 100)
            max_candidates = self._get_setting('max_candidates', 10)

            for i, file_path in enumerate(source_files):
                if i % 10 == 0:
                    self.logger.info(f"Analyzed {i}/{len(source_files)} files...")
                    
                analysis = self.analyze_file(Path(file_path))
                file_analyses.append(analysis)
                
                # Extract data from generic analysis result
                lines_count = analysis.data.get('lines_count', 0)
                complexity_score = analysis.metrics.get('complexity_score', 0.0)
                
                total_lines += lines_count
                
                # Categorize problematic files
                rel_path = analysis.data.get('filepath', str(file_path))
                
                if lines_count > large_file_threshold:
                    large_files.append(rel_path)
                if complexity_score > complexity_threshold:
                    complex_files.append(rel_path)
                if any("God Object" in issue for issue in analysis.issues):
                    god_objects.append(rel_path)
                    
            # Identify refactoring candidates
            refactoring_candidates = [
                analysis for analysis in file_analyses
                if len(analysis.issues) > 0 and analysis.data.get('lines_count', 0) > min_candidate_size
            ]
            
            # Sort by priority (more issues * file size = higher priority)
            refactoring_candidates.sort(
                key=lambda x: len(x.issues) * x.data.get('lines_count', 0), reverse=True
            )
            
            # Generate overall recommendations
            overall_recommendations = self._generate_overall_recommendations(
                file_analyses, large_files, complex_files, god_objects
            )
            
            # Calculate automation potential
            automation_potential = self._calculate_automation_potential(refactoring_candidates)
            
            # Convert file analyses to legacy format for backward compatibility
            legacy_candidates = []
            for analysis in refactoring_candidates[:max_candidates]:
                legacy_candidates.append({
                    'filepath': analysis.data.get('filepath', ''),
                    'lines_count': analysis.data.get('lines_count', 0),
                    'classes_count': analysis.data.get('classes_count', 0),
                    'methods_count': analysis.data.get('methods_count', 0),
                    'complexity_score': analysis.metrics.get('complexity_score', 0.0),
                    'responsibilities_count': analysis.data.get('responsibilities_count', 0),
                    'dependencies_count': analysis.data.get('dependencies_count', 0),
                    'test_coverage': analysis.data.get('test_coverage', 0.0),
                    'issues': analysis.issues,
                    'recommendations': analysis.recommendations
                })
            
            # Construct analysis data with reduced duplication
            # We keep canonical fields at top level and only include necessary legacy structures
            analysis_data = {
                'project_structure': self._safe_asdict(project_structure),
                'total_files': len(source_files),
                'total_lines': total_lines,
                'large_files': large_files,
                'complex_files': complex_files,
                'god_objects': god_objects,
                'refactoring_candidates': legacy_candidates,
                'overall_recommendations': overall_recommendations,
                'automation_potential': automation_potential,
                # Legacy support - only include if strictly necessary for backward compatibility
                'legacy_analysis': {
                    'refactoring_candidates': legacy_candidates,
                    'overall_recommendations': overall_recommendations,
                    'automation_potential': automation_potential
                }
            }
            
            metrics = {
                'files_analyzed': float(len(source_files)),
                'total_lines': float(total_lines),
                'large_files_ratio': len(large_files) / len(source_files) if source_files else 0.0,
                'complex_files_ratio': len(complex_files) / len(source_files) if source_files else 0.0,
                'god_objects_ratio': len(god_objects) / len(source_files) if source_files else 0.0,
                'automation_potential': automation_potential
            }
            
            return GenericAnalysisResult(
                success=True,
                project_path=str(project_path),
                analysis_type="project_analysis",
                data=analysis_data,
                metrics=metrics,
                issues=[],
                recommendations=overall_recommendations,
                metadata={
                    'analyzer_version': '1.4.0',
                    'config_used': self._safe_asdict(self.config),
                    'analysis_timestamp': datetime.now().isoformat()
                },
                timestamp=datetime.now()
            )
            
        except Exception as e:
            self.logger.error(f"Project analysis failed: {e}")
            return GenericAnalysisResult(
                success=False,
                project_path=str(project_path),
                analysis_type="project_analysis",
                data={},
                metrics={},
                issues=[str(e)],
                recommendations=[],
                metadata={'error': str(e)},
                timestamp=datetime.now()
            )
        
    def _generate_overall_recommendations(self, file_analyses: List[GenericAnalysisResult],
                                        large_files: List[str], complex_files: List[str],
                                        god_objects: List[str]) -> List[str]:
        """Generate overall recommendations for the project."""
        recommendations = []
        
        if len(large_files) > 5:
            recommendations.append(
                f"Found {len(large_files)} large files. "
                "Consider applying 'Split Monolithic Configuration' pattern"
            )
            
        if len(complex_files) > 3:
            recommendations.append(
                f"Found {len(complex_files)} complex files. "
                "Consider applying 'Extract Method to Component Class' pattern"
            )
            
        if len(god_objects) > 0:
            recommendations.append(
                f"Found {len(god_objects)} God Objects. "
                "Critical: apply refactoring using DI patterns"
            )
            
        # Additional recommendations based on overall project health
        total_issues = sum(len(analysis.issues) for analysis in file_analyses)
        if total_issues > 20:
            recommendations.append(
                "High number of issues detected. Consider creating a Facade "
                "to maintain backward compatibility during major refactoring"
            )
                
        return recommendations
        
    def _calculate_automation_potential(self, candidates: List[GenericAnalysisResult]) -> float:
        """Calculate automation potential for refactoring."""
        if not candidates:
            return 0.0
            
        # Base potential on types of issues
        automation_scores = []
        
        for candidate in candidates:
            score = 0.0
            
            # Issues that are easily automatable
            for issue in candidate.issues:
                if "Large file" in issue:
                    score += 0.8  # File splitting is easily automatable
                elif "High complexity" in issue:
                    score += 0.6  # Method extraction is partially automatable
                elif "God Object" in issue:
                    score += 0.4  # Requires more manual work
                elif "Multiple responsibilities" in issue:
                    score += 0.7  # Responsibility separation is automatable
                    
            automation_scores.append(min(score, 1.0))
            
        return sum(automation_scores) / len(automation_scores) if automation_scores else 0.0
        
    def generate_report(self, analysis: GenericAnalysisResult, output_file: str = None) -> str:
        """Generate a comprehensive analysis report."""
        
        # Extract data, preferring top-level but falling back to legacy if needed
        data = analysis.data
        legacy_data = data.get('legacy_analysis', {})
        
        # Use analysis timestamp if available, else current time
        analysis_date = analysis.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        # Helper to get data from either location
        def get_val(key, default=None):
            if key in data:
                return data[key]
            return legacy_data.get(key, default)
        
        total_files = get_val('total_files', 0)
        total_lines = get_val('total_lines', 0)
        automation_potential = get_val('automation_potential', 0.0)
        
        report = f"""
# Project Analysis Report

**Analysis Date**: {analysis_date}
**Total Files**: {total_files}
**Total Lines of Code**: {total_lines:,}
**Automation Potential**: {automation_potential:.1%}

## Priority Refactoring Candidates

"""
        
        candidates = get_val('refactoring_candidates', [])
        for i, candidate in enumerate(candidates, 1):
            report += f"""
### {i}. `{candidate.get('filepath', 'Unknown')}`
- **Lines of Code**: {candidate.get('lines_count', 0)}
- **Classes**: {candidate.get('classes_count', 0)}
- **Methods**: {candidate.get('methods_count', 0)}
- **Complexity**: {candidate.get('complexity_score', 0.0):.2f}
- **Issues**: {len(candidate.get('issues', []))}

**Identified Issues**:
"""
            for issue in candidate.get('issues', []):
                report += f"- [!] {issue}\n"
                
            report += "\n**Recommendations**:\n"
            for rec in candidate.get('recommendations', []):
                report += f"- [+] {rec}\n"
                
        large_files = get_val('large_files', [])
        complex_files = get_val('complex_files', [])
        god_objects = get_val('god_objects', [])
        
        large_file_threshold = self._get_setting('large_file_threshold', 500)
        complexity_threshold = self._get_setting('complexity_threshold', 15.0)

        report += f"""

## Overall Statistics

- **Large Files** (>{large_file_threshold} lines): {len(large_files)}
- **Complex Files** (> {complexity_threshold} complexity): {len(complex_files)}
- **God Objects**: {len(god_objects)}

## Overall Recommendations

"""
        
        for rec in get_val('overall_recommendations', []):
            report += f"- {rec}\n"
            
        if large_files:
            report += f"""

## Large Files Requiring Attention

"""
            for file_path in large_files[:5]:  # Top 5
                report += f"- `{file_path}`\n"
                
        if god_objects:
            report += f"""

## God Objects (Critical Priority)

"""
            for file_path in god_objects:
                report += f"- `{file_path}`\n"
                
        report += f"""

## Next Steps

1. **Start with God Objects** - they have the highest improvement potential
2. **Apply refactoring patterns**:
   - Extract Method to Component Class
   - Constructor Injection with Interfaces  
   - Split Monolithic Configuration
3. **Use automation** where possible (potential: {automation_potential:.1%})
4. **Create tests** before refactoring for safety

---
*Report generated automatically based on code analysis*
"""
        
        if output_file:
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report)
                self.logger.info(f"Report saved to {output_file}")
            except Exception as e:
                self.logger.error(f"Failed to save report to {output_file}: {e}")
            
        return report
        
    


def create_project_analyzer(config: Optional[AnalysisConfig] = None, project_root: str = ".") -> ProjectAnalyzer:
    """Factory function to create ProjectAnalyzer instance."""
    return ProjectAnalyzer(config, project_root)
"""
Test Quality Analyzer for expert refactoring analysis.

Analyzes test usefulness vs noise to identify high-quality tests
and problematic test patterns.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class QualityAnalyzer:
    """Analyzes test quality and identifies signal vs noise."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)

    def analyze_test_quality(self, test_files: List[str]) -> Dict[str, any]:
        """
        Analyze test quality across multiple test files.
        
        Args:
            test_files: List of test file paths
            
        Returns:
            Dictionary with test quality analysis
        """
        logger.info("Analyzing test quality and signal vs noise...")
        
        quality_analysis = {
            'overall_score': 0.0,
            'file_analyses': [],
            'quality_issues': [],
            'signal_vs_noise': {},
            'recommendations': []
        }
        
        total_score = 0.0
        valid_files = 0
        
        # Analyze each test file
        for test_file in test_files:
            file_analysis = self._analyze_test_file_quality(test_file)
            if file_analysis:
                quality_analysis['file_analyses'].append(file_analysis)
                total_score += file_analysis['quality_score']
                valid_files += 1
        
        # Calculate overall score
        if valid_files > 0:
            quality_analysis['overall_score'] = total_score / valid_files
        
        # Aggregate quality issues
        quality_analysis['quality_issues'] = self._aggregate_quality_issues(quality_analysis['file_analyses'])
        
        # Calculate signal vs noise
        quality_analysis['signal_vs_noise'] = self._calculate_signal_vs_noise(quality_analysis['file_analyses'])
        
        # Generate recommendations
        quality_analysis['recommendations'] = self._generate_quality_recommendations(quality_analysis)
        
        return quality_analysis

    def export_detailed_test_quality(self, analysis: Dict[str, any]) -> Dict[str, any]:
        """
        Export detailed test quality analysis as requested by experts.
        
        Returns:
            Dictionary with signal/noise analysis and quality metrics
        """
        # Categorize tests by quality
        high_quality_tests = []
        low_quality_tests = []
        noisy_tests = []
        
        for file_analysis in analysis['file_analyses']:
            file_name = file_analysis['file']
            
            for test_analysis in file_analysis['test_analyses']:
                test_info = {
                    'file': file_name,
                    'test_name': test_analysis['name'],
                    'quality_score': test_analysis['quality_score'],
                    'issues': test_analysis['issues'],
                    'signal_strength': test_analysis['signal_strength']
                }
                
                if test_analysis['quality_score'] >= 80:
                    high_quality_tests.append(test_info)
                elif test_analysis['quality_score'] <= 40:
                    low_quality_tests.append(test_info)
                
                if test_analysis['signal_strength'] <= 30:
                    noisy_tests.append(test_info)
        
        # Identify problematic patterns
        problematic_patterns = self._identify_problematic_patterns(analysis['file_analyses'])
        
        # Generate quality metrics
        quality_metrics = self._generate_quality_metrics(analysis)
        
        return {
            'test_quality_distribution': {
                'high_quality': len(high_quality_tests),
                'medium_quality': len(analysis['file_analyses']) - len(high_quality_tests) - len(low_quality_tests),
                'low_quality': len(low_quality_tests),
                'detailed_high_quality': high_quality_tests,
                'detailed_low_quality': low_quality_tests
            },
            'signal_vs_noise_analysis': {
                'overall_signal_ratio': analysis['signal_vs_noise'].get('signal_ratio', 0.0),
                'noisy_tests': noisy_tests,
                'signal_strength_distribution': analysis['signal_vs_noise'].get('distribution', {}),
                'noise_sources': analysis['signal_vs_noise'].get('noise_sources', [])
            },
            'problematic_patterns': problematic_patterns,
            'quality_metrics': quality_metrics,
            'improvement_recommendations': self._generate_improvement_recommendations(analysis)
        }

    def _analyze_test_file_quality(self, test_file: str) -> Optional[Dict[str, any]]:
        """Analyze quality of a single test file."""
        test_path = self.project_root / test_file
        
        if not test_path.exists():
            return None
        
        try:
            content = test_path.read_text(encoding='utf-8')
            tree = ast.parse(content)
        except (OSError, SyntaxError) as e:
            logger.warning(f"Could not parse {test_file}: {e}")
            return None
        
        file_analysis = {
            'file': test_file,
            'quality_score': 0.0,
            'test_analyses': [],
            'file_issues': [],
            'metrics': {}
        }
        
        # Find all test functions and classes
        test_functions = self._find_test_functions(tree)
        test_classes = self._find_test_classes(tree)
        
        # Analyze each test function
        for test_func in test_functions:
            test_analysis = self._analyze_test_function_quality(test_func, content)
            file_analysis['test_analyses'].append(test_analysis)
        
        # Analyze test classes
        for test_class in test_classes:
            class_methods = self._find_test_methods_in_class(test_class)
            for method in class_methods:
                test_analysis = self._analyze_test_function_quality(method, content)
                test_analysis['class'] = test_class.name
                file_analysis['test_analyses'].append(test_analysis)
        
        # Calculate file-level metrics
        file_analysis['metrics'] = self._calculate_file_metrics(tree, content)
        
        # Calculate overall file quality score
        if file_analysis['test_analyses']:
            total_score = sum(test['quality_score'] for test in file_analysis['test_analyses'])
            file_analysis['quality_score'] = total_score / len(file_analysis['test_analyses'])
        
        # Identify file-level issues
        file_analysis['file_issues'] = self._identify_file_level_issues(tree, content, file_analysis['metrics'])
        
        return file_analysis

    def _find_test_functions(self, tree: ast.Module) -> List[ast.FunctionDef]:
        """Find all test functions in the module."""
        test_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                # Make sure it's not inside a class
                parent_classes = [p for p in ast.walk(tree) if isinstance(p, ast.ClassDef) and node in ast.walk(p)]
                if not parent_classes:
                    test_functions.append(node)
        
        return test_functions

    def _find_test_classes(self, tree: ast.Module) -> List[ast.ClassDef]:
        """Find all test classes in the module."""
        return [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name.startswith('Test')]

    def _find_test_methods_in_class(self, test_class: ast.ClassDef) -> List[ast.FunctionDef]:
        """Find test methods within a test class."""
        return [node for node in test_class.body if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')]

    def _analyze_test_function_quality(self, test_func: ast.FunctionDef, file_content: str) -> Dict[str, any]:
        """Analyze the quality of a single test function."""
        analysis = {
            'name': test_func.name,
            'line': getattr(test_func, 'lineno', 0),
            'quality_score': 0.0,
            'signal_strength': 0.0,
            'issues': [],
            'positive_indicators': [],
            'metrics': {}
        }
        
        # Calculate basic metrics
        analysis['metrics'] = self._calculate_test_metrics(test_func)
        
        # Check for quality indicators
        quality_score = 50.0  # Base score
        signal_strength = 50.0  # Base signal strength
        
        # Positive indicators
        if ast.get_docstring(test_func):
            quality_score += 10
            signal_strength += 10
            analysis['positive_indicators'].append('Has docstring')
        
        # Check for meaningful assertions
        assertions = self._find_assertions(test_func)
        if assertions:
            quality_score += min(len(assertions) * 5, 20)
            signal_strength += min(len(assertions) * 5, 20)
            analysis['positive_indicators'].append(f'Has {len(assertions)} assertions')
        else:
            quality_score -= 30
            signal_strength -= 30
            analysis['issues'].append('No assertions found')
        
        # Check for proper setup/teardown
        if self._has_setup_teardown(test_func):
            quality_score += 10
            analysis['positive_indicators'].append('Has setup/teardown')
        
        # Check for mocking
        if self._uses_mocking(test_func):
            quality_score += 10
            signal_strength += 10
            analysis['positive_indicators'].append('Uses mocking')
        
        # Negative indicators
        if self._has_empty_assertions(test_func):
            quality_score -= 20
            signal_strength -= 30
            analysis['issues'].append('Has empty or trivial assertions')
        
        if self._has_exception_swallowing(test_func):
            quality_score -= 25
            signal_strength -= 25
            analysis['issues'].append('Swallows exceptions (except: pass)')
        
        if self._has_wrong_types_in_test(test_func):
            quality_score -= 15
            signal_strength -= 20
            analysis['issues'].append('Uses wrong parameter types')
        
        if self._is_too_complex(test_func):
            quality_score -= 10
            signal_strength -= 10
            analysis['issues'].append('Test is too complex')
        
        if self._has_hardcoded_values(test_func):
            quality_score -= 5
            signal_strength -= 5
            analysis['issues'].append('Has hardcoded values')
        
        if self._lacks_edge_case_testing(test_func):
            quality_score -= 10
            analysis['issues'].append('Lacks edge case testing')
        
        # Ensure scores are within bounds
        analysis['quality_score'] = max(0, min(100, quality_score))
        analysis['signal_strength'] = max(0, min(100, signal_strength))
        
        return analysis

    def _calculate_test_metrics(self, test_func: ast.FunctionDef) -> Dict[str, any]:
        """Calculate metrics for a test function."""
        metrics = {
            'lines_of_code': 0,
            'cyclomatic_complexity': 1,
            'assertion_count': 0,
            'mock_count': 0,
            'parameter_count': len(test_func.args.args)
        }
        
        # Count lines (approximate)
        if hasattr(test_func, 'end_lineno') and hasattr(test_func, 'lineno'):
            metrics['lines_of_code'] = test_func.end_lineno - test_func.lineno + 1
        
        # Calculate complexity
        for node in ast.walk(test_func):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                metrics['cyclomatic_complexity'] += 1
        
        # Count assertions and mocks
        metrics['assertion_count'] = len(self._find_assertions(test_func))
        metrics['mock_count'] = len(self._find_mocks(test_func))
        
        return metrics

    def _find_assertions(self, test_func: ast.FunctionDef) -> List[ast.stmt]:
        """Find assertion statements in a test function."""
        assertions = []
        
        for node in ast.walk(test_func):
            if isinstance(node, ast.Assert):
                assertions.append(node)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id.startswith('assert'):
                    assertions.append(node)
                elif isinstance(node.func, ast.Attribute) and node.func.attr.startswith('assert'):
                    assertions.append(node)
        
        return assertions

    def _find_mocks(self, test_func: ast.FunctionDef) -> List[ast.stmt]:
        """Find mock usage in a test function."""
        mocks = []
        
        for node in ast.walk(test_func):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and 'mock' in node.func.id.lower():
                    mocks.append(node)
                elif isinstance(node.func, ast.Attribute) and 'mock' in node.func.attr.lower():
                    mocks.append(node)
        
        return mocks

    def _has_setup_teardown(self, test_func: ast.FunctionDef) -> bool:
        """Check if test has proper setup/teardown."""
        # Look for setup/teardown method calls or fixture usage
        for node in ast.walk(test_func):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['setUp', 'tearDown', 'setup_method', 'teardown_method']:
                        return True
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['setUp', 'tearDown', 'setup_method', 'teardown_method']:
                        return True
        
        return False

    def _uses_mocking(self, test_func: ast.FunctionDef) -> bool:
        """Check if test uses mocking."""
        return len(self._find_mocks(test_func)) > 0

    def _has_empty_assertions(self, test_func: ast.FunctionDef) -> bool:
        """Check for empty or trivial assertions."""
        for node in ast.walk(test_func):
            if isinstance(node, ast.Assert):
                if isinstance(node.test, ast.Constant) and node.test.value is True:
                    return True
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == 'assertTrue':
                    if (node.args and isinstance(node.args[0], ast.Constant) 
                        and node.args[0].value is True):
                        return True
        
        return False

    def _has_exception_swallowing(self, test_func: ast.FunctionDef) -> bool:
        """Check for exception swallowing (except: pass)."""
        for node in ast.walk(test_func):
            if isinstance(node, ast.ExceptHandler):
                if (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
                    return True
        
        return False

    def _has_wrong_types_in_test(self, test_func: ast.FunctionDef) -> bool:
        """Check for wrong parameter types in test calls."""
        # This is a heuristic - look for obvious type mismatches
        for node in ast.walk(test_func):
            if isinstance(node, ast.Call):
                for arg in node.args:
                    if isinstance(arg, ast.List) and len(arg.elts) == 0:
                        # Empty list passed where string might be expected
                        return True
                    elif isinstance(arg, ast.Dict) and len(arg.keys) == 0:
                        # Empty dict passed where string might be expected
                        return True
        
        return False

    def _is_too_complex(self, test_func: ast.FunctionDef) -> bool:
        """Check if test is too complex."""
        complexity = 1
        for node in ast.walk(test_func):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        
        return complexity > 5  # Threshold for test complexity

    def _has_hardcoded_values(self, test_func: ast.FunctionDef) -> bool:
        """Check for excessive hardcoded values."""
        hardcoded_count = 0
        
        for node in ast.walk(test_func):
            if isinstance(node, ast.Constant):
                if isinstance(node.value, (str, int, float)) and node.value not in [True, False, None]:
                    hardcoded_count += 1
        
        return hardcoded_count > 10  # Threshold for too many hardcoded values

    def _lacks_edge_case_testing(self, test_func: ast.FunctionDef) -> bool:
        """Check if test lacks edge case testing."""
        # Look for edge case indicators
        edge_case_indicators = ['null', 'none', 'empty', 'zero', 'negative', 'max', 'min', 'edge']
        
        test_name = test_func.name.lower()
        docstring = ast.get_docstring(test_func) or ""
        
        return not any(indicator in test_name or indicator in docstring.lower() 
                      for indicator in edge_case_indicators)

    def _calculate_file_metrics(self, tree: ast.Module, content: str) -> Dict[str, any]:
        """Calculate file-level metrics."""
        metrics = {
            'total_lines': len(content.splitlines()),
            'test_function_count': 0,
            'test_class_count': 0,
            'import_count': 0,
            'assertion_density': 0.0
        }
        
        total_assertions = 0
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                metrics['test_function_count'] += 1
                total_assertions += len(self._find_assertions(node))
            elif isinstance(node, ast.ClassDef) and node.name.startswith('Test'):
                metrics['test_class_count'] += 1
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                metrics['import_count'] += 1
        
        # Calculate assertion density
        if metrics['test_function_count'] > 0:
            metrics['assertion_density'] = total_assertions / metrics['test_function_count']
        
        return metrics

    def _identify_file_level_issues(self, tree: ast.Module, content: str, metrics: Dict[str, any]) -> List[str]:
        """Identify file-level quality issues."""
        issues = []
        
        if metrics['test_function_count'] == 0:
            issues.append('No test functions found')
        
        if metrics['assertion_density'] < 1.0:
            issues.append('Low assertion density - tests may not be thorough')
        
        if metrics['total_lines'] > 1000:
            issues.append('Very large test file - consider splitting')
        
        if metrics['import_count'] > 20:
            issues.append('Too many imports - may indicate complex dependencies')
        
        # Check for missing test organization
        if metrics['test_function_count'] > 10 and metrics['test_class_count'] == 0:
            issues.append('Many test functions without class organization')
        
        return issues

    def _aggregate_quality_issues(self, file_analyses: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Aggregate quality issues across all files."""
        issue_counts = {}
        
        for file_analysis in file_analyses:
            # File-level issues
            for issue in file_analysis['file_issues']:
                key = f"file_level: {issue}"
                issue_counts[key] = issue_counts.get(key, 0) + 1
            
            # Test-level issues
            for test_analysis in file_analysis['test_analyses']:
                for issue in test_analysis['issues']:
                    key = f"test_level: {issue}"
                    issue_counts[key] = issue_counts.get(key, 0) + 1
        
        # Convert to list of issues with counts
        aggregated_issues = [
            {'issue': issue, 'count': count, 'severity': self._determine_issue_severity(issue)}
            for issue, count in issue_counts.items()
        ]
        
        # Sort by severity and count
        aggregated_issues.sort(key=lambda x: (x['severity'], -x['count']))
        
        return aggregated_issues

    def _determine_issue_severity(self, issue: str) -> int:
        """Determine severity of an issue (1=high, 2=medium, 3=low)."""
        high_severity = ['No assertions found', 'Swallows exceptions']
        medium_severity = ['Has empty or trivial assertions', 'Uses wrong parameter types']
        
        issue_text = issue.split(': ', 1)[-1]  # Remove prefix
        
        if any(high in issue_text for high in high_severity):
            return 1
        elif any(medium in issue_text for medium in medium_severity):
            return 2
        else:
            return 3

    def _calculate_signal_vs_noise(self, file_analyses: List[Dict[str, any]]) -> Dict[str, any]:
        """Calculate signal vs noise ratio across all tests."""
        total_tests = 0
        total_signal = 0.0
        signal_distribution = {'high': 0, 'medium': 0, 'low': 0}
        noise_sources = []
        
        for file_analysis in file_analyses:
            for test_analysis in file_analysis['test_analyses']:
                total_tests += 1
                signal_strength = test_analysis['signal_strength']
                total_signal += signal_strength
                
                if signal_strength >= 70:
                    signal_distribution['high'] += 1
                elif signal_strength >= 40:
                    signal_distribution['medium'] += 1
                else:
                    signal_distribution['low'] += 1
                    
                    # Identify noise sources
                    if signal_strength < 30:
                        noise_sources.append({
                            'test': test_analysis['name'],
                            'file': file_analysis['file'],
                            'signal_strength': signal_strength,
                            'issues': test_analysis['issues']
                        })
        
        signal_ratio = (total_signal / total_tests) if total_tests > 0 else 0.0
        
        return {
            'signal_ratio': signal_ratio,
            'distribution': signal_distribution,
            'noise_sources': noise_sources,
            'total_tests_analyzed': total_tests
        }

    def _identify_problematic_patterns(self, file_analyses: List[Dict[str, any]]) -> List[Dict[str, any]]:
        """Identify problematic patterns in tests."""
        patterns = []
        
        # Pattern 1: Tests with no assertions
        no_assertion_tests = []
        for file_analysis in file_analyses:
            for test_analysis in file_analysis['test_analyses']:
                if 'No assertions found' in test_analysis['issues']:
                    no_assertion_tests.append(f"{file_analysis['file']}::{test_analysis['name']}")
        
        if no_assertion_tests:
            patterns.append({
                'pattern': 'Tests without assertions',
                'count': len(no_assertion_tests),
                'severity': 'high',
                'examples': no_assertion_tests[:5],
                'recommendation': 'Add meaningful assertions to verify behavior'
            })
        
        # Pattern 2: Exception swallowing
        exception_swallowing_tests = []
        for file_analysis in file_analyses:
            for test_analysis in file_analysis['test_analyses']:
                if 'Swallows exceptions' in test_analysis['issues']:
                    exception_swallowing_tests.append(f"{file_analysis['file']}::{test_analysis['name']}")
        
        if exception_swallowing_tests:
            patterns.append({
                'pattern': 'Exception swallowing (except: pass)',
                'count': len(exception_swallowing_tests),
                'severity': 'high',
                'examples': exception_swallowing_tests[:5],
                'recommendation': 'Replace with specific exception handling or proper assertions'
            })
        
        # Pattern 3: Trivial assertions
        trivial_assertion_tests = []
        for file_analysis in file_analyses:
            for test_analysis in file_analysis['test_analyses']:
                if 'Has empty or trivial assertions' in test_analysis['issues']:
                    trivial_assertion_tests.append(f"{file_analysis['file']}::{test_analysis['name']}")
        
        if trivial_assertion_tests:
            patterns.append({
                'pattern': 'Trivial assertions (assert True)',
                'count': len(trivial_assertion_tests),
                'severity': 'medium',
                'examples': trivial_assertion_tests[:5],
                'recommendation': 'Replace with meaningful assertions that verify actual behavior'
            })
        
        return patterns

    def _generate_quality_metrics(self, analysis: Dict[str, any]) -> Dict[str, any]:
        """Generate overall quality metrics."""
        metrics = {
            'overall_quality_score': analysis['overall_score'],
            'total_files_analyzed': len(analysis['file_analyses']),
            'total_tests_analyzed': 0,
            'quality_distribution': {'high': 0, 'medium': 0, 'low': 0},
            'common_issues': [],
            'best_practices_adherence': 0.0
        }
        
        # Count tests and quality distribution
        for file_analysis in analysis['file_analyses']:
            metrics['total_tests_analyzed'] += len(file_analysis['test_analyses'])
            
            for test_analysis in file_analysis['test_analyses']:
                score = test_analysis['quality_score']
                if score >= 70:
                    metrics['quality_distribution']['high'] += 1
                elif score >= 40:
                    metrics['quality_distribution']['medium'] += 1
                else:
                    metrics['quality_distribution']['low'] += 1
        
        # Identify most common issues
        issue_counts = {}
        for issue_info in analysis['quality_issues']:
            issue_text = issue_info['issue'].split(': ', 1)[-1]
            issue_counts[issue_text] = issue_info['count']
        
        metrics['common_issues'] = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate best practices adherence
        total_tests = metrics['total_tests_analyzed']
        high_quality_tests = metrics['quality_distribution']['high']
        metrics['best_practices_adherence'] = (high_quality_tests / total_tests * 100) if total_tests > 0 else 0.0
        
        return metrics

    def _generate_quality_recommendations(self, analysis: Dict[str, any]) -> List[str]:
        """Generate recommendations for improving test quality."""
        recommendations = []
        
        overall_score = analysis['overall_score']
        signal_ratio = analysis['signal_vs_noise'].get('signal_ratio', 0.0)
        
        if overall_score < 50:
            recommendations.append("Overall test quality is low - comprehensive test review needed")
        
        if signal_ratio < 50:
            recommendations.append("High noise-to-signal ratio - focus on meaningful assertions")
        
        # Issue-specific recommendations
        for issue_info in analysis['quality_issues'][:3]:  # Top 3 issues
            issue_text = issue_info['issue']
            count = issue_info['count']
            
            if 'No assertions found' in issue_text:
                recommendations.append(f"Add assertions to {count} tests that lack verification")
            elif 'Swallows exceptions' in issue_text:
                recommendations.append(f"Fix {count} tests that swallow exceptions")
            elif 'trivial assertions' in issue_text:
                recommendations.append(f"Replace trivial assertions in {count} tests")
        
        return recommendations

    def _generate_improvement_recommendations(self, analysis: Dict[str, any]) -> List[str]:
        """Generate specific improvement recommendations."""
        recommendations = []
        
        # Based on signal vs noise analysis
        noise_sources = analysis['signal_vs_noise'].get('noise_sources', [])
        if noise_sources:
            recommendations.append(f"Refactor or remove {len(noise_sources)} noisy tests with low signal strength")
        
        # Based on problematic patterns
        for pattern in analysis.get('problematic_patterns', []):
            if pattern['severity'] == 'high':
                recommendations.append(f"High priority: Fix {pattern['count']} tests with {pattern['pattern']}")
            else:
                recommendations.append(f"Medium priority: Improve {pattern['count']} tests with {pattern['pattern']}")
        
        # General recommendations
        quality_metrics = analysis.get('quality_metrics', {})
        if quality_metrics.get('best_practices_adherence', 0) < 60:
            recommendations.append("Improve adherence to testing best practices")
        
        total_tests = quality_metrics.get('total_tests_analyzed', 0)
        high_quality = quality_metrics.get('quality_distribution', {}).get('high', 0)
        
        if total_tests > 0 and (high_quality / total_tests) < 0.3:
            recommendations.append("Focus on creating more high-quality tests with clear assertions and proper setup")
        
        return recommendations
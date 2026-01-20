"""
Test Discovery Analyzer for expert refactoring analysis.

Discovers existing tests and analyzes test coverage to assess
refactoring safety and identify missing test cases.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Set

from ..models import TestDiscoveryResult

logger = logging.getLogger(__name__)


class TestDiscoveryAnalyzer:
    """Discovers and analyzes existing tests for the target module."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)
        
        # Common test directory patterns
        self.test_patterns = [
            'test_*.py',
            '*_test.py',
            'tests.py',
        ]
        
        # Common test directory names
        self.test_dirs = [
            'tests',
            'test',
            'testing',
        ]

    def find_existing_tests(self) -> TestDiscoveryResult:
        """
        Find existing test files related to the target module.
        
        Returns:
            TestDiscoveryResult with test discovery information
        """
        logger.info("Discovering existing tests...")
        
        # Find test files
        test_files = self._find_test_files()
        
        # Analyze coverage for found tests
        coverage_analysis = self._analyze_test_coverage(test_files)
        
        # Identify missing tests
        missing_tests = self._identify_missing_tests(coverage_analysis)
        
        # Calculate test quality score
        quality_score = self._calculate_test_quality_score(test_files, coverage_analysis)
        
        # Generate recommendations
        recommendations = self._generate_test_recommendations(test_files, coverage_analysis, missing_tests)
        
        result = TestDiscoveryResult(
            existing_test_files=test_files,
            coverage_analysis=coverage_analysis,
            missing_tests=missing_tests,
            test_quality_score=quality_score,
            recommendations=recommendations
        )
        
        logger.info(f"Test discovery: {len(test_files)} test files, quality score {quality_score:.1f}")
        return result

    def export_detailed_test_analysis(self, result: TestDiscoveryResult) -> Dict[str, Any]:
        """
        Export detailed test analysis as requested by experts.
        
        Returns:
            Dictionary with specific test files, missing methods, and test structure
        """
        # Analyze test file structure
        test_file_details = []
        for test_file in result.existing_test_files:
            test_path = self.project_root / test_file
            file_analysis = self._analyze_test_file_structure(test_path)
            test_file_details.append({
                "file": test_file,
                "analysis": file_analysis
            })
        
        # Get detailed missing test information
        target_symbols = self._extract_target_symbols()
        detailed_missing_tests = self._get_detailed_missing_tests(target_symbols, result.existing_test_files)
        
        # Categorize test types
        test_categories = self._categorize_existing_tests(test_file_details)
        
        # Generate test structure recommendations
        structure_recommendations = self._generate_test_structure_recommendations(test_file_details, detailed_missing_tests)
        
        return {
            "test_files": {
                "total_files": len(result.existing_test_files),
                "files": result.existing_test_files,
                "detailed_analysis": test_file_details
            },
            "missing_test_coverage": {
                "total_missing": len(result.missing_tests),
                "missing_methods": result.missing_tests,
                "detailed_missing": detailed_missing_tests,
                "coverage_gaps": self._identify_coverage_gaps(target_symbols, result.existing_test_files)
            },
            "test_structure": {
                "categories": test_categories,
                "integration_tests": self._find_integration_tests(test_file_details),
                "unit_tests": self._find_unit_tests(test_file_details)
            },
            "quality_assessment": {
                "overall_score": result.test_quality_score,
                "coverage_percentage": self._calculate_overall_coverage_percentage(result.coverage_analysis),
                "quality_issues": self._identify_quality_issues(test_file_details),
                "recommendations": structure_recommendations
            }
        }

    def _analyze_test_file_structure(self, test_file: Path) -> Dict[str, Any]:
        """Analyze the structure of a single test file."""
        analysis = {
            "test_functions": [],
            "test_classes": [],
            "imports": [],
            "fixtures": [],
            "mocks": [],
            "assertions": 0,
            "file_size_lines": 0
        }
        
        try:
            content = test_file.read_text(encoding='utf-8')
            lines = content.splitlines()
            analysis["file_size_lines"] = len(lines)
            
            tree = ast.parse(content)
            
            # Find test functions and classes
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    if node.name.startswith('test_'):
                        analysis["test_functions"].append({
                            "name": node.name,
                            "line": getattr(node, 'lineno', 0),
                            "docstring": ast.get_docstring(node),
                            "parameters": [arg.arg for arg in node.args.args]
                        })
                    elif 'fixture' in [d.id for d in node.decorator_list if isinstance(d, ast.Name)]:
                        analysis["fixtures"].append(node.name)
                
                elif isinstance(node, ast.ClassDef):
                    if node.name.startswith('Test'):
                        test_methods = [
                            method.name for method in node.body 
                            if isinstance(method, ast.FunctionDef) and method.name.startswith('test_')
                        ]
                        analysis["test_classes"].append({
                            "name": node.name,
                            "line": getattr(node, 'lineno', 0),
                            "test_methods": test_methods,
                            "method_count": len(test_methods)
                        })
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            analysis["imports"].append(alias.name)
                    else:
                        module = node.module or ""
                        for alias in node.names:
                            analysis["imports"].append(f"{module}.{alias.name}")
                
                # Count assertions (simplified)
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id.startswith('assert'):
                        analysis["assertions"] += 1
                    elif isinstance(node.func, ast.Attribute) and node.func.attr.startswith('assert'):
                        analysis["assertions"] += 1
            
            # Check for mock usage
            mock_indicators = ['mock', 'Mock', 'patch', 'MagicMock']
            for indicator in mock_indicators:
                if indicator in content:
                    analysis["mocks"].append(indicator)
        
        except (OSError, SyntaxError) as e:
            analysis["error"] = str(e)
        
        return analysis

    def _get_detailed_missing_tests(self, target_symbols: Set[str], test_files: List[str]) -> List[Dict[str, Any]]:
        """Get detailed information about missing tests."""
        detailed_missing = []
        
        # Find what's covered
        covered_symbols = set()
        for test_file in test_files:
            test_path = self.project_root / test_file
            covered = self._analyze_test_file_coverage(test_path, target_symbols)
            covered_symbols.update(covered)
        
        # Analyze each missing symbol
        for symbol in target_symbols - covered_symbols:
            symbol_info = self._analyze_symbol_for_testing(symbol)
            detailed_missing.append({
                "symbol": symbol,
                "type": symbol_info["type"],
                "complexity": symbol_info["complexity"],
                "parameters": symbol_info["parameters"],
                "suggested_tests": symbol_info["suggested_tests"],
                "priority": symbol_info["priority"]
            })
        
        # Sort by priority
        detailed_missing.sort(key=lambda x: x["priority"])
        
        return detailed_missing

    def _analyze_symbol_for_testing(self, symbol: str) -> Dict[str, Any]:
        """Analyze a symbol to determine what tests it needs."""
        symbol_info = {
            "type": "function",
            "complexity": 1,
            "parameters": [],
            "suggested_tests": [],
            "priority": 3
        }
        
        try:
            content = self.target_module.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Find the symbol in the AST
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == symbol or symbol.endswith(f".{node.name}"):
                        symbol_info["type"] = "method" if "." in symbol else "function"
                        symbol_info["parameters"] = [arg.arg for arg in node.args.args]
                        symbol_info["complexity"] = self._calculate_test_complexity(node)
                        symbol_info["suggested_tests"] = self._suggest_tests_for_function(node)
                        symbol_info["priority"] = self._calculate_test_priority(node)
                        break
                elif isinstance(node, ast.ClassDef):
                    if node.name == symbol:
                        symbol_info["type"] = "class"
                        symbol_info["suggested_tests"] = ["test_initialization", "test_public_methods"]
                        symbol_info["priority"] = 2
                        break
        
        except (OSError, SyntaxError):
            pass
        
        return symbol_info

    def _calculate_test_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate how complex testing this function would be."""
        complexity = 1
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
        
        return complexity

    def _suggest_tests_for_function(self, func_node: ast.FunctionDef) -> List[str]:
        """Suggest specific tests for a function."""
        suggestions = []
        
        # Basic test
        suggestions.append(f"test_{func_node.name}_basic")
        
        # Parameter-based tests
        if func_node.args.args:
            suggestions.append(f"test_{func_node.name}_with_parameters")
            suggestions.append(f"test_{func_node.name}_invalid_parameters")
        
        # Complexity-based tests
        has_conditions = any(isinstance(node, ast.If) for node in ast.walk(func_node))
        if has_conditions:
            suggestions.append(f"test_{func_node.name}_edge_cases")
        
        has_exceptions = any(isinstance(node, ast.ExceptHandler) for node in ast.walk(func_node))
        if has_exceptions:
            suggestions.append(f"test_{func_node.name}_error_handling")
        
        return suggestions

    def _calculate_test_priority(self, func_node: ast.FunctionDef) -> int:
        """Calculate testing priority (1=highest, 3=lowest)."""
        # Public methods have higher priority
        if not func_node.name.startswith('_'):
            priority = 1
        else:
            priority = 3
        
        # Complex functions have higher priority
        complexity = self._calculate_test_complexity(func_node)
        if complexity > 3:
            priority = max(1, priority - 1)
        
        return priority

    def _identify_coverage_gaps(self, target_symbols: Set[str], test_files: List[str]) -> List[Dict[str, Any]]:
        """Identify specific coverage gaps."""
        gaps = []
        
        # Analyze coverage by category
        functions = [s for s in target_symbols if '.' not in s and not s[0].isupper()]
        classes = [s for s in target_symbols if '.' not in s and s[0].isupper()]
        methods = [s for s in target_symbols if '.' in s]
        
        categories = {
            "functions": functions,
            "classes": classes,
            "methods": methods
        }
        
        for category, symbols in categories.items():
            if symbols:
                covered = 0
                for test_file in test_files:
                    test_path = self.project_root / test_file
                    covered_in_file = self._analyze_test_file_coverage(test_path, set(symbols))
                    covered += len(covered_in_file)
                
                coverage_pct = (covered / len(symbols)) * 100 if symbols else 100
                gaps.append({
                    "category": category,
                    "total_symbols": len(symbols),
                    "covered_symbols": covered,
                    "coverage_percentage": coverage_pct,
                    "gap_severity": "high" if coverage_pct < 30 else "medium" if coverage_pct < 70 else "low"
                })
        
        return gaps

    def _categorize_existing_tests(self, test_file_details: List[Dict]) -> Dict[str, int]:
        """Categorize existing tests by type."""
        categories = {
            "unit_tests": 0,
            "integration_tests": 0,
            "fixture_tests": 0,
            "mock_tests": 0
        }
        
        for file_detail in test_file_details:
            analysis = file_detail["analysis"]
            
            # Count test functions
            categories["unit_tests"] += len(analysis["test_functions"])
            
            # Count fixture usage
            if analysis["fixtures"]:
                categories["fixture_tests"] += len(analysis["fixtures"])
            
            # Count mock usage
            if analysis["mocks"]:
                categories["mock_tests"] += 1
            
            # Heuristic for integration tests
            if any("integration" in tf["name"].lower() for tf in analysis["test_functions"]):
                categories["integration_tests"] += 1
        
        return categories

    def _find_integration_tests(self, test_file_details: List[Dict]) -> List[str]:
        """Find integration tests."""
        integration_tests = []
        
        for file_detail in test_file_details:
            analysis = file_detail["analysis"]
            for test_func in analysis["test_functions"]:
                if "integration" in test_func["name"].lower() or "end_to_end" in test_func["name"].lower():
                    integration_tests.append(f"{file_detail['file']}::{test_func['name']}")
        
        return integration_tests

    def _find_unit_tests(self, test_file_details: List[Dict]) -> List[str]:
        """Find unit tests."""
        unit_tests = []
        
        for file_detail in test_file_details:
            analysis = file_detail["analysis"]
            for test_func in analysis["test_functions"]:
                if "integration" not in test_func["name"].lower():
                    unit_tests.append(f"{file_detail['file']}::{test_func['name']}")
        
        return unit_tests

    def _calculate_overall_coverage_percentage(self, coverage_analysis: Dict[str, float]) -> float:
        """Calculate overall coverage percentage."""
        if not coverage_analysis:
            return 0.0
        
        return sum(coverage_analysis.values()) / len(coverage_analysis)

    def _identify_quality_issues(self, test_file_details: List[Dict]) -> List[str]:
        """Identify quality issues in test files."""
        issues = []
        
        for file_detail in test_file_details:
            file_name = file_detail["file"]
            analysis = file_detail["analysis"]
            
            # Check for common issues
            if len(analysis["test_functions"]) == 0 and len(analysis["test_classes"]) == 0:
                issues.append(f"{file_name}: No test functions or classes found")
            
            if analysis["assertions"] == 0:
                issues.append(f"{file_name}: No assertions found")
            
            if len(analysis["test_functions"]) > 50:
                issues.append(f"{file_name}: Too many test functions ({len(analysis['test_functions'])}) - consider splitting")
            
            if analysis["file_size_lines"] > 1000:
                issues.append(f"{file_name}: Large test file ({analysis['file_size_lines']} lines) - consider refactoring")
            
            # Check for missing docstrings
            functions_without_docs = [
                tf["name"] for tf in analysis["test_functions"] 
                if not tf["docstring"]
            ]
            if len(functions_without_docs) > 5:
                issues.append(f"{file_name}: Many test functions lack docstrings")
        
        return issues

    def _generate_test_structure_recommendations(self, test_file_details: List[Dict], detailed_missing_tests: List[Dict]) -> List[str]:
        """Generate recommendations for test structure improvements."""
        recommendations = []
        
        # Analyze current structure
        total_test_functions = sum(len(fd["analysis"]["test_functions"]) for fd in test_file_details)
        total_test_classes = sum(len(fd["analysis"]["test_classes"]) for fd in test_file_details)
        
        # Structure recommendations
        if len(test_file_details) == 1 and total_test_functions > 20:
            recommendations.append("Consider splitting large test file into multiple files by functionality")
        
        if total_test_classes == 0 and total_test_functions > 10:
            recommendations.append("Consider organizing tests into test classes for better structure")
        
        # Missing test recommendations
        high_priority_missing = [t for t in detailed_missing_tests if t["priority"] == 1]
        if high_priority_missing:
            recommendations.append(f"Add tests for {len(high_priority_missing)} high-priority methods before refactoring")
        
        # Coverage recommendations
        if len(detailed_missing_tests) > 10:
            recommendations.append("Significant test coverage gaps - create comprehensive test plan")
        
        return recommendations

    def _find_test_files(self) -> List[str]:
        """Find test files that might test the target module."""
        test_files = []
        
        # Get the module name for matching
        module_name = self.target_module.stem
        
        # Search in common test directories
        for test_dir_name in self.test_dirs:
            test_dir = self.project_root / test_dir_name
            if test_dir.exists():
                test_files.extend(self._search_test_dir(test_dir, module_name))
        
        # Search in the same directory as the target module
        target_dir = self.target_module.parent
        test_files.extend(self._search_test_dir(target_dir, module_name))
        
        # Search recursively in the project
        for py_file in self.project_root.rglob("*.py"):
            if self._is_test_file(py_file) and self._might_test_module(py_file, module_name):
                rel_path = str(py_file.relative_to(self.project_root))
                if rel_path not in test_files:
                    test_files.append(rel_path)
        
        return test_files

    def _search_test_dir(self, test_dir: Path, module_name: str) -> List[str]:
        """Search for test files in a specific directory."""
        test_files = []
        
        if not test_dir.exists():
            return test_files
        
        for pattern in self.test_patterns:
            for test_file in test_dir.glob(pattern):
                if self._might_test_module(test_file, module_name):
                    rel_path = str(test_file.relative_to(self.project_root))
                    test_files.append(rel_path)
        
        return test_files

    def _is_test_file(self, file_path: Path) -> bool:
        """Check if a file is likely a test file."""
        name = file_path.name.lower()
        return (name.startswith('test_') or 
                name.endswith('_test.py') or 
                name == 'tests.py' or
                'test' in file_path.parts)

    def _might_test_module(self, test_file: Path, module_name: str) -> bool:
        """Check if a test file might test the target module."""
        # Check filename
        if module_name.lower() in test_file.name.lower():
            return True
        
        # Check file content for imports
        try:
            content = test_file.read_text(encoding='utf-8')
            return module_name in content
        except (OSError, UnicodeDecodeError):
            return False

    def _analyze_test_coverage(self, test_files: List[str]) -> Dict[str, float]:
        """Analyze test coverage for the target module."""
        coverage = {}
        
        if not test_files:
            return coverage
        
        # Extract methods/functions from target module
        target_symbols = self._extract_target_symbols()
        
        # For each test file, check what it covers
        for test_file in test_files:
            test_path = self.project_root / test_file
            covered_symbols = self._analyze_test_file_coverage(test_path, target_symbols)
            
            # Calculate coverage percentage
            if target_symbols:
                coverage_pct = len(covered_symbols) / len(target_symbols) * 100
                coverage[test_file] = coverage_pct
            else:
                coverage[test_file] = 0.0
        
        return coverage

    def _extract_target_symbols(self) -> Set[str]:
        """Extract public methods and functions from the target module."""
        symbols = set()
        
        try:
            content = self.target_module.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not node.name.startswith('_'):  # Public functions
                        symbols.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    if not node.name.startswith('_'):  # Public classes
                        symbols.add(node.name)
                        # Add public methods
                        for method in node.body:
                            if isinstance(method, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                if not method.name.startswith('_'):
                                    symbols.add(f"{node.name}.{method.name}")
        
        except (OSError, SyntaxError) as e:
            logger.warning(f"Error extracting symbols from {self.target_module}: {e}")
        
        return symbols

    def _analyze_test_file_coverage(self, test_file: Path, target_symbols: Set[str]) -> Set[str]:
        """Analyze what symbols a test file covers."""
        covered = set()
        
        try:
            content = test_file.read_text(encoding='utf-8')
            
            # Simple heuristic: if a symbol name appears in the test file, it's "covered"
            for symbol in target_symbols:
                if symbol in content:
                    covered.add(symbol)
        
        except (OSError, UnicodeDecodeError):
            pass
        
        return covered

    def _identify_missing_tests(self, coverage_analysis: Dict[str, float]) -> List[str]:
        """Identify methods/functions that lack test coverage."""
        target_symbols = self._extract_target_symbols()
        
        if not coverage_analysis:
            return list(target_symbols)
        
        # Find symbols that aren't covered by any test
        all_covered = set()
        for test_file in coverage_analysis:
            test_path = self.project_root / test_file
            covered = self._analyze_test_file_coverage(test_path, target_symbols)
            all_covered.update(covered)
        
        missing = target_symbols - all_covered
        return list(missing)

    def _calculate_test_quality_score(self, test_files: List[str], coverage_analysis: Dict[str, float]) -> float:
        """Calculate overall test quality score."""
        if not test_files:
            return 0.0
        
        # Base score for having tests
        score = 30.0
        
        # Coverage score (up to 50 points)
        if coverage_analysis:
            avg_coverage = sum(coverage_analysis.values()) / len(coverage_analysis)
            score += min(avg_coverage * 0.5, 50.0)
        
        # Number of test files (up to 20 points)
        test_file_score = min(len(test_files) * 5, 20.0)
        score += test_file_score
        
        return min(score, 100.0)

    def _generate_test_recommendations(self, test_files: List[str], coverage_analysis: Dict[str, float], missing_tests: List[str]) -> List[str]:
        """Generate recommendations for improving test coverage."""
        recommendations = []
        
        if not test_files:
            recommendations.append("No test files found - create comprehensive test suite before refactoring")
            recommendations.append(f"Create test file: test_{self.target_module.stem}.py")
        else:
            # Coverage recommendations
            if coverage_analysis:
                avg_coverage = sum(coverage_analysis.values()) / len(coverage_analysis)
                if avg_coverage < 50:
                    recommendations.append(f"Low test coverage ({avg_coverage:.1f}%) - add more tests before refactoring")
            
            # Missing test recommendations
            if missing_tests:
                if len(missing_tests) <= 5:
                    recommendations.append(f"Add tests for: {', '.join(missing_tests)}")
                else:
                    recommendations.append(f"Add tests for {len(missing_tests)} uncovered methods/functions")
        
        # Quality recommendations
        if len(test_files) == 1:
            recommendations.append("Consider organizing tests into multiple files by functionality")
        
        return recommendations

    def assess_test_quality(self, test_files: List[str]) -> Dict[str, Any]:
        """
        Assess the quality of existing tests.
        
        Args:
            test_files: List of test file paths
            
        Returns:
            Dictionary with quality assessment
        """
        assessment = {
            'total_files': len(test_files),
            'has_tests': len(test_files) > 0,
            'test_patterns': [],
            'quality_issues': []
        }
        
        for test_file in test_files:
            test_path = self.project_root / test_file
            try:
                content = test_path.read_text(encoding='utf-8')
                tree = ast.parse(content)
                
                # Check for test patterns
                test_functions = [node.name for node in ast.walk(tree) 
                                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_')]
                
                if test_functions:
                    assessment['test_patterns'].append({
                        'file': test_file,
                        'test_count': len(test_functions),
                        'test_names': test_functions[:5]  # First 5 test names
                    })
                
                # Check for quality issues
                if len(test_functions) == 0:
                    assessment['quality_issues'].append(f"No test functions found in {test_file}")
                
            except (OSError, SyntaxError) as e:
                assessment['quality_issues'].append(f"Cannot parse {test_file}: {e}")
        
        return assessment
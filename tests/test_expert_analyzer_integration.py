"""
Integration tests for ExpertRefactoringAnalyzer.

These tests verify the complete workflow with real or realistic mock data.
"""

import pytest
import json
from pathlib import Path
from unittest.mock import Mock, patch
import ast

from intellirefactor.analysis.expert.expert_analyzer import ExpertRefactoringAnalyzer
from intellirefactor.analysis.expert.models import ExpertAnalysisResult, RiskLevel


# ============ Fixtures ============

@pytest.fixture
def temp_project(tmp_path):
    """Create a temporary project structure."""
    # Create project directory
    project_root = tmp_path / "test_project"
    project_root.mkdir()
    
    # Create a sample Python module
    module_path = project_root / "sample_module.py"
    module_path.write_text('''
"""Sample module for testing."""

import os
import json
from typing import List, Optional


class SampleClass:
    """A sample class for testing cohesion analysis."""
    
    def __init__(self, name: str):
        self.name = name
        self.data = []
    
    def process_data(self, items: List[str]) -> int:
        """Process a list of items."""
        self.data = items
        return len(items)
    
    def get_name(self) -> str:
        """Get the name."""
        return self.name
    
    def save_to_file(self, path: str) -> None:
        """Save data to file."""
        with open(path, 'w') as f:
            json.dump({'name': self.name, 'data': self.data}, f)


def standalone_function(x: int, y: int) -> int:
    """A standalone function."""
    return x + y


def another_function(items: List[str]) -> str:
    """Another function that calls SampleClass."""
    obj = SampleClass("test")
    obj.process_data(items)
    return obj.get_name()
''', encoding='utf-8')
    
    # Create a test file
    tests_dir = project_root / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_sample.py"
    test_file.write_text('''
"""Tests for sample_module."""

import pytest
from sample_module import SampleClass, standalone_function


def test_standalone_function():
    assert standalone_function(2, 3) == 5


def test_sample_class_init():
    obj = SampleClass("test")
    assert obj.name == "test"
''', encoding='utf-8')
    
    # Create an external file that imports sample_module
    external_file = project_root / "external_user.py"
    external_file.write_text('''
"""External file that uses sample_module."""

from sample_module import SampleClass, another_function

def use_sample():
    obj = SampleClass("external")
    return obj.process_data(["a", "b", "c"])
''', encoding='utf-8')
    
    return {
        'root': project_root,
        'module': module_path,
        'tests_dir': tests_dir,
        'external': external_file,
    }


@pytest.fixture
def mock_all_analyzers():
    """Create mocks for all analyzers."""
    def create_analyzer_mocks():
        mocks = {}
        
        # Call graph analyzer
        call_graph_mock = Mock()
        call_graph_mock.analyze_internal_calls.return_value = Mock(
            nodes=[Mock(name='method1'), Mock(name='method2')],
            edges=[Mock()],
            cycles=[]
        )
        call_graph_mock.export_detailed_call_graph.return_value = {
            'call_graph': {'total_relationships': 5}
        }
        mocks['call_graph'] = call_graph_mock
        
        # Caller analyzer
        caller_mock = Mock()
        caller_mock.find_external_callers.return_value = [
            Mock(file='external.py', line=10)
        ]
        caller_mock.analyze_usage_patterns.return_value = Mock(
            total_callers=3,
            most_used_symbols=[('SampleClass', 5), ('func', 2)]
        )
        caller_mock.export_detailed_external_usage.return_value = {
            'files_summary': {'total_files': 1}
        }
        mocks['caller'] = caller_mock
        
        # Cohesion analyzer
        cohesion_mock = Mock()
        cohesion_mock.build_cohesion_matrix.return_value = Mock(
            methods=['__init__', 'process_data'],
            attributes=['name', 'data']
        )
        cohesion_mock.export_detailed_cohesion_matrix.return_value = {
            'cohesion_score': 0.75
        }
        mocks['cohesion'] = cohesion_mock
        
        # Contract analyzer
        contract_mock = Mock()
        contract_mock.extract_contracts_from_docstrings.return_value = [
            Mock(method='process_data', preconditions=[], postconditions=[])
        ]
        mocks['contract'] = contract_mock
        
        # Dependency analyzer
        dependency_mock = Mock()
        dependency_mock.extract_dependency_interfaces.return_value = [
            Mock(module='os'), Mock(module='json')
        ]
        dependency_mock.analyze_import_dependencies.return_value = {
            'cycles': [],
            'external_imports': ['os', 'json'],
            'internal_imports': []
        }
        dependency_mock.extract_external_dependency_contracts.return_value = {}
        mocks['dependency'] = dependency_mock
        
        # Test analyzer
        test_mock = Mock()
        test_mock.find_existing_tests.return_value = Mock(
            existing_test_files=['test_sample.py'],
            test_quality_score=60.0,
            missing_tests=['save_to_file']
        )
        test_mock.export_detailed_test_analysis.return_value = {
            'missing_test_coverage': {'total_missing': 1}
        }
        mocks['test'] = test_mock
        
        # Characterization generator
        char_mock = Mock()
        char_mock.generate_characterization_tests.return_value = [
            Mock(method_name='process_data', test_category=Mock(value='typical'))
        ]
        char_mock.export_detailed_characterization_tests.return_value = {
            'summary': {'total_tests': 5}
        }
        mocks['characterization'] = char_mock
        
        # Duplicate analyzer
        duplicate_mock = Mock()
        duplicate_mock.find_concrete_duplicates.return_value = []
        duplicate_mock.export_detailed_duplicates.return_value = {
            'summary': {'total_duplicates': 0}
        }
        mocks['duplicate'] = duplicate_mock
        
        # Git analyzer
        git_mock = Mock()
        git_mock.analyze_change_patterns.return_value = Mock(
            hot_files=['sample_module.py'],
            change_frequency=5
        )
        mocks['git'] = git_mock
        
        # Compatibility analyzer
        compat_mock = Mock()
        compat_mock.assess_breaking_change_impact.return_value = Mock(risk='low')
        compat_mock.determine_compatibility_constraints.return_value = [
            'Maintain backward compatibility'
        ]
        mocks['compatibility'] = compat_mock
        
        # Exception contract analyzer
        exc_mock = Mock()
        exc_mock.analyze_exception_contracts.return_value = {}
        exc_mock.export_detailed_exception_contracts.return_value = {}
        mocks['exception_contract'] = exc_mock
        
        # Data schema analyzer
        schema_mock = Mock()
        schema_mock.analyze_data_schemas.return_value = {}
        schema_mock.export_detailed_data_schemas.return_value = {}
        mocks['data_schema'] = schema_mock
        
        # Optional dependencies analyzer
        opt_mock = Mock()
        opt_mock.analyze_optional_dependencies.return_value = []
        opt_mock.export_detailed_optional_dependencies.return_value = {}
        mocks['optional_dependencies'] = opt_mock
        
        # Golden traces extractor
        traces_mock = Mock()
        traces_mock.extract_golden_traces.return_value = []
        traces_mock.export_detailed_golden_traces.return_value = {}
        mocks['golden_traces'] = traces_mock
        
        # Test quality analyzer
        quality_mock = Mock()
        quality_mock.analyze_test_quality.return_value = Mock(score=70.0)
        quality_mock.export_detailed_test_quality.return_value = {'score': 70.0}
        mocks['test_quality'] = quality_mock
        
        return mocks
    
    return create_analyzer_mocks


# ============ Integration Tests ============

class TestExpertAnalyzerInitialization:
    """Tests for analyzer initialization."""
    
    def test_init_creates_output_directory(self, temp_project):
        """Test that initialization creates output directory."""
        output_dir = temp_project['root'] / "custom_output"
        
        analyzer = ExpertRefactoringAnalyzer(
            str(temp_project['root']),
            str(temp_project['module']),
            str(output_dir)
        )
        
        assert output_dir.exists()
    
    def test_init_default_output_directory(self, temp_project):
        """Test default output directory creation."""
        analyzer = ExpertRefactoringAnalyzer(
            str(temp_project['root']),
            str(temp_project['module'])
        )
        
        expected_dir = temp_project['root'] / "expert_analysis"
        assert expected_dir.exists()
    
    def test_init_stores_paths_correctly(self, temp_project):
        """Test that paths are stored correctly."""
        analyzer = ExpertRefactoringAnalyzer(
            str(temp_project['root']),
            str(temp_project['module'])
        )
        
        assert analyzer.project_root == temp_project['root']
        assert analyzer.target_module == temp_project['module']
    
    def test_init_creates_timestamp(self, temp_project):
        """Test that timestamp is created."""
        analyzer = ExpertRefactoringAnalyzer(
            str(temp_project['root']),
            str(temp_project['module'])
        )
        
        assert analyzer.timestamp is not None
        assert len(analyzer.timestamp) == 15  # YYYYMMDD_HHMMSS


class TestExpertAnalyzerASTHandling:
    """Tests for AST parsing and caching."""
    
    def test_get_ast_parses_valid_module(self, temp_project):
        """Test AST parsing of valid module."""
        analyzer = ExpertRefactoringAnalyzer(
            str(temp_project['root']),
            str(temp_project['module'])
        )
        
        ast_result = analyzer._get_ast()
        
        assert ast_result is not None
        assert isinstance(ast_result, ast.Module)
    
    def test_get_ast_caches_result(self, temp_project):
        """Test that AST is cached."""
        analyzer = ExpertRefactoringAnalyzer(
            str(temp_project['root']),
            str(temp_project['module'])
        )
        
        ast1 = analyzer._get_ast()
        ast2 = analyzer._get_ast()
        
        assert ast1 is ast2  # Same object
    
    def test_get_ast_handles_syntax_error(self, temp_project):
        """Test handling of syntax errors."""
        bad_file = temp_project['root'] / "bad_syntax.py"
        bad_file.write_text("def broken(:\n    pass", encoding='utf-8')
        
        analyzer = ExpertRefactoringAnalyzer(
            str(temp_project['root']),
            str(bad_file)
        )
        
        ast_result = analyzer._get_ast()
        
        assert ast_result is None
    
    def test_get_ast_handles_missing_file(self, temp_project):
        """Test handling of missing file."""
        missing = temp_project['root'] / "nonexistent.py"
        
        analyzer = ExpertRefactoringAnalyzer(
            str(temp_project['root']),
            str(missing)
        )
        
        ast_result = analyzer._get_ast()
        
        assert ast_result is None


class TestExpertAnalyzerMainAnalysis:
    """Tests for main analysis workflow."""
    
    def test_analyze_returns_result_object(self, temp_project, mock_all_analyzers):
        """Test that analysis returns ExpertAnalysisResult."""
        with patch.object(
            ExpertRefactoringAnalyzer,
            '_initialize_analyzers',
            return_value=mock_all_analyzers()
        ):
            analyzer = ExpertRefactoringAnalyzer(
                str(temp_project['root']),
                str(temp_project['module'])
            )
            
            result = analyzer.analyze_for_expert_refactoring()
            
            assert isinstance(result, ExpertAnalysisResult)
            assert result.target_file == str(temp_project['module'])
    
    def test_analyze_handles_parse_failure(self, temp_project):
        """Test analysis with unparseable file."""
        bad_file = temp_project['root'] / "bad.py"
        bad_file.write_text("def broken(:\n    pass", encoding='utf-8')
        
        analyzer = ExpertRefactoringAnalyzer(
            str(temp_project['root']),
            str(bad_file)
        )
        
        result = analyzer.analyze_for_expert_refactoring()
        
        assert result.risk_assessment == RiskLevel.HIGH
        assert any("Cannot parse" in r for r in result.recommendations)
    
    def test_analyze_calculates_quality_score(self, temp_project, mock_all_analyzers):
        """Test that quality score is calculated."""
        with patch.object(
            ExpertRefactoringAnalyzer,
            '_initialize_analyzers',
            return_value=mock_all_analyzers()
        ):
            analyzer = ExpertRefactoringAnalyzer(
                str(temp_project['root']),
                str(temp_project['module'])
            )
            
            result = analyzer.analyze_for_expert_refactoring()
            
            assert result.analysis_quality_score >= 0
            assert result.analysis_quality_score <= 100
    
    def test_analyze_assesses_risk(self, temp_project, mock_all_analyzers):
        """Test that risk is assessed."""
        with patch.object(
            ExpertRefactoringAnalyzer,
            '_initialize_analyzers',
            return_value=mock_all_analyzers()
        ):
            analyzer = ExpertRefactoringAnalyzer(
                str(temp_project['root']),
                str(temp_project['module'])
            )
            
            result = analyzer.analyze_for_expert_refactoring()
            
            assert result.risk_assessment is not None
            assert isinstance(result.risk_assessment, RiskLevel)
    
    def test_analyze_pipeline_failures_become_recommendations(
        self, temp_project, mock_all_analyzers
    ):
        """Test that pipeline failures are added as recommendations."""
        mocks = mock_all_analyzers()
        mocks['call_graph'].analyze_internal_calls.side_effect = ValueError("Test error")
        
        with patch.object(
            ExpertRefactoringAnalyzer,
            '_initialize_analyzers',
            return_value=mocks
        ):
            analyzer = ExpertRefactoringAnalyzer(
                str(temp_project['root']),
                str(temp_project['module'])
            )
            
            result = analyzer.analyze_for_expert_refactoring()
            
            # Should have recommendation about failed call graph
            assert any("Call graph" in r or "dependency" in r.lower() 
                      for r in result.recommendations)


class TestExpertAnalyzerDetailedExport:
    """Tests for detailed data export."""
    
    def test_export_returns_dict(self, temp_project, mock_all_analyzers):
        """Test that export returns a dictionary."""
        with patch.object(
            ExpertRefactoringAnalyzer,
            '_initialize_analyzers',
            return_value=mock_all_analyzers()
        ):
            analyzer = ExpertRefactoringAnalyzer(
                str(temp_project['root']),
                str(temp_project['module'])
            )
            
            data = analyzer.export_detailed_expert_data()
            
            assert isinstance(data, dict)
    
    def test_export_contains_summary(self, temp_project, mock_all_analyzers):
        """Test that export contains summary."""
        with patch.object(
            ExpertRefactoringAnalyzer,
            '_initialize_analyzers',
            return_value=mock_all_analyzers()
        ):
            analyzer = ExpertRefactoringAnalyzer(
                str(temp_project['root']),
                str(temp_project['module'])
            )
            
            data = analyzer.export_detailed_expert_data()
            
            assert 'summary' in data
            assert 'target_file' in data['summary']
            assert 'timestamp' in data['summary']
            assert 'analysis_quality_score' in data['summary']
    
    def test_export_contains_expert_recommendations(
        self, temp_project, mock_all_analyzers
    ):
        """Test that export contains expert recommendations."""
        with patch.object(
            ExpertRefactoringAnalyzer,
            '_initialize_analyzers',
            return_value=mock_all_analyzers()
        ):
            analyzer = ExpertRefactoringAnalyzer(
                str(temp_project['root']),
                str(temp_project['module'])
            )
            
            data = analyzer.export_detailed_expert_data()
            
            assert 'expert_recommendations' in data
            assert 'expert_1_requirements' in data['expert_recommendations']
            assert 'expert_2_requirements' in data['expert_recommendations']
    
    def test_export_contains_acceptance_criteria(
        self, temp_project, mock_all_analyzers
    ):
        """Test that export contains acceptance criteria."""
        with patch.object(
            ExpertRefactoringAnalyzer,
            '_initialize_analyzers',
            return_value=mock_all_analyzers()
        ):
            analyzer = ExpertRefactoringAnalyzer(
                str(temp_project['root']),
                str(temp_project['module'])
            )
            
            data = analyzer.export_detailed_expert_data()
            
            assert 'acceptance_criteria' in data
            assert 'behavioral_invariants' in data['acceptance_criteria']
    
    def test_export_handles_parse_failure(self, temp_project):
        """Test export with unparseable file."""
        bad_file = temp_project['root'] / "bad.py"
        bad_file.write_text("def broken(:\n    pass", encoding='utf-8')
        
        analyzer = ExpertRefactoringAnalyzer(
            str(temp_project['root']),
            str(bad_file)
        )
        
        data = analyzer.export_detailed_expert_data()
        
        assert 'error' in data


class TestExpertAnalyzerReportGeneration:
    """Tests for report generation."""
    
    def test_generate_report_creates_files(self, temp_project, mock_all_analyzers):
        """Test that report generation creates files."""
        with patch.object(
            ExpertRefactoringAnalyzer,
            '_initialize_analyzers',
            return_value=mock_all_analyzers()
        ):
            analyzer = ExpertRefactoringAnalyzer(
                str(temp_project['root']),
                str(temp_project['module'])
            )
            
            output_dir = temp_project['root'] / "reports"
            report_path = analyzer.generate_expert_report(str(output_dir))
            
            assert Path(report_path).exists()
            assert report_path.endswith('.md')
            
            # Check JSON also exists
            json_files = list(output_dir.glob("*.json"))
            assert len(json_files) >= 1
    
    def test_generate_report_json_valid(self, temp_project, mock_all_analyzers):
        """Test that generated JSON is valid."""
        with patch.object(
            ExpertRefactoringAnalyzer,
            '_initialize_analyzers',
            return_value=mock_all_analyzers()
        ):
            analyzer = ExpertRefactoringAnalyzer(
                str(temp_project['root']),
                str(temp_project['module'])
            )
            
            output_dir = temp_project['root'] / "reports"
            analyzer.generate_expert_report(str(output_dir))
            
            json_file = list(output_dir.glob("*.json"))[0]
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            assert 'target_file' in data
    
    def test_generate_report_markdown_contains_sections(
        self, temp_project, mock_all_analyzers
    ):
        """Test that Markdown contains expected sections."""
        with patch.object(
            ExpertRefactoringAnalyzer,
            '_initialize_analyzers',
            return_value=mock_all_analyzers()
        ):
            analyzer = ExpertRefactoringAnalyzer(
                str(temp_project['root']),
                str(temp_project['module'])
            )
            
            output_dir = temp_project['root'] / "reports"
            report_path = analyzer.generate_expert_report(str(output_dir))
            
            with open(report_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            assert "# Expert Refactoring Analysis Report" in content
            assert "**Target Module:**" in content
            assert "## Next Steps" in content
    
    def test_generate_report_uses_default_output_dir(
        self, temp_project, mock_all_analyzers
    ):
        """Test report generation with default output directory."""
        with patch.object(
            ExpertRefactoringAnalyzer,
            '_initialize_analyzers',
            return_value=mock_all_analyzers()
        ):
            analyzer = ExpertRefactoringAnalyzer(
                str(temp_project['root']),
                str(temp_project['module'])
            )
            
            report_path = analyzer.generate_expert_report()
            
            assert Path(report_path).exists()
            assert "expert_analysis" in report_path


class TestExpertAnalyzerEndToEnd:
    """End-to-end tests without mocks."""
    
    @pytest.mark.slow
    def test_full_analysis_real_file(self, temp_project):
        """Test full analysis on a real Python file."""
        # This test uses real analyzers
        # It may fail if analyzers have issues
        try:
            analyzer = ExpertRefactoringAnalyzer(
                str(temp_project['root']),
                str(temp_project['module'])
            )
            
            result = analyzer.analyze_for_expert_refactoring()
            
            # Basic sanity checks
            assert result.target_file == str(temp_project['module'])
            assert result.analysis_quality_score >= 0
            assert result.risk_assessment is not None
            
        except ImportError:
            pytest.skip("Some analyzers not available")
    
    @pytest.mark.slow
    def test_full_export_real_file(self, temp_project):
        """Test full export on a real Python file."""
        try:
            analyzer = ExpertRefactoringAnalyzer(
                str(temp_project['root']),
                str(temp_project['module'])
            )
            
            data = analyzer.export_detailed_expert_data()
            
            assert 'summary' in data
            assert 'expert_recommendations' in data
            
        except ImportError:
            pytest.skip("Some analyzers not available")


class TestExpertAnalyzerEdgeCases:
    """Edge case tests."""
    
    def test_empty_file(self, temp_project):
        """Test analysis of empty file."""
        empty_file = temp_project['root'] / "empty.py"
        empty_file.write_text("", encoding='utf-8')
        
        analyzer = ExpertRefactoringAnalyzer(
            str(temp_project['root']),
            str(empty_file)
        )
        
        result = analyzer.analyze_for_expert_refactoring()
        
        # Should complete without error
        assert result.target_file == str(empty_file)
    
    def test_file_with_only_comments(self, temp_project):
        """Test analysis of file with only comments."""
        comment_file = temp_project['root'] / "comments.py"
        comment_file.write_text("# Just a comment\n# Another comment", encoding='utf-8')
        
        analyzer = ExpertRefactoringAnalyzer(
            str(temp_project['root']),
            str(comment_file)
        )
        
        result = analyzer.analyze_for_expert_refactoring()
        
        assert result.target_file == str(comment_file)
    
    def test_unicode_content(self, temp_project):
        """Test analysis of file with unicode content."""
        unicode_file = temp_project['root'] / "unicode.py"
        unicode_file.write_text('''
"""Модуль с юникодом."""

def функция():
    """Функция с русским названием."""
    return "Привет мир! 你好世界"
''', encoding='utf-8')
        
        analyzer = ExpertRefactoringAnalyzer(
            str(temp_project['root']),
            str(unicode_file)
        )
        
        ast_result = analyzer._get_ast()
        
        assert ast_result is not None
    
    def test_large_class(self, temp_project):
        """Test analysis with large class."""
        large_file = temp_project['root'] / "large.py"
        
        # Generate a class with many methods
        methods = "\n".join([
            f"    def method_{i}(self): return {i}"
            for i in range(50)
        ])
        
        large_file.write_text(f'''
class LargeClass:
{methods}
''', encoding='utf-8')
        
        analyzer = ExpertRefactoringAnalyzer(
            str(temp_project['root']),
            str(large_file)
        )
        
        ast_result = analyzer._get_ast()
        
        assert ast_result is not None
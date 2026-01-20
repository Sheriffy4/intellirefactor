"""
Tests for analysis_pipeline module.
"""

import pytest
from unittest.mock import Mock
import ast

from intellirefactor.analysis.expert.analysis_pipeline import (
    AnalysisStep,
    Priority,
    StepResult,
    PipelineResult,
    AnalysisPipelineExecutor,
    ANALYSIS_STEPS,
)
from intellirefactor.analysis.expert.safe_access import SafeAnalyzerRegistry


@pytest.fixture
def mock_ast():
    """Create a mock AST module."""
    return ast.parse("def test(): pass")


@pytest.fixture
def mock_analyzers():
    """Create mock analyzers."""
    call_graph = Mock()
    call_graph.analyze_internal_calls.return_value = {'nodes': [], 'edges': []}
    
    caller = Mock()
    caller.find_external_callers.return_value = []
    
    test = Mock()
    test.find_existing_tests.return_value = Mock(existing_test_files=[])
    
    return SafeAnalyzerRegistry({
        'call_graph': call_graph,
        'caller': caller,
        'test': test,
    })


@pytest.fixture
def executor(mock_analyzers):
    """Create a pipeline executor."""
    context = {'target_module_str': 'test.py'}
    return AnalysisPipelineExecutor(mock_analyzers, context)


class TestAnalysisStep:
    """Tests for AnalysisStep dataclass."""
    
    def test_create_step(self):
        """Test creating an analysis step."""
        step = AnalysisStep(
            name="Test",
            analyzer_key="test",
            method_name="analyze",
            result_attr="test_result",
            failure_message="test failed",
            priority=Priority.P0_CRITICAL
        )
        
        assert step.name == "Test"
        assert step.requires_ast == True  # Default
    
    def test_step_with_extra_args(self):
        """Test step with extra args factory."""
        step = AnalysisStep(
            name="Test",
            analyzer_key="test",
            method_name="analyze",
            result_attr="test_result",
            failure_message="",
            priority=Priority.P1_IMPORTANT,
            requires_ast=False,
            extra_args_factory=lambda ctx: (ctx['target_module_str'],)
        )
        
        args = step.extra_args_factory({'target_module_str': 'module.py'})
        assert args == ('module.py',)


class TestPriority:
    """Tests for Priority enum."""
    
    def test_priority_ordering(self):
        """Test that priorities are correctly ordered."""
        assert Priority.P0_CRITICAL < Priority.P1_IMPORTANT
        assert Priority.P1_IMPORTANT < Priority.P2_ENHANCEMENT
    
    def test_priority_values(self):
        """Test priority integer values."""
        assert Priority.P0_CRITICAL.value == 0
        assert Priority.P1_IMPORTANT.value == 1
        assert Priority.P2_ENHANCEMENT.value == 2


class TestStepResult:
    """Tests for StepResult dataclass."""
    
    def test_successful_result(self):
        """Test creating a successful result."""
        step = Mock()
        result = StepResult(step=step, success=True, value={'data': 'value'})
        
        assert result.success is True
        assert result.value == {'data': 'value'}
        assert result.error is None
    
    def test_failed_result(self):
        """Test creating a failed result."""
        step = Mock()
        error = ValueError("Test error")
        result = StepResult(step=step, success=False, error=error)
        
        assert result.success is False
        assert result.value is None
        assert result.error is error


class TestPipelineResult:
    """Tests for PipelineResult dataclass."""
    
    def test_empty_result(self):
        """Test empty pipeline result."""
        result = PipelineResult()
        
        assert result.results == {}
        assert result.failures == []
        assert result.step_results == []
    
    def test_success_rate_empty(self):
        """Test success rate with no steps."""
        result = PipelineResult()
        assert result.success_rate == 0.0
    
    def test_success_rate_calculation(self):
        """Test success rate calculation."""
        result = PipelineResult()
        result.step_results = [
            StepResult(step=Mock(), success=True),
            StepResult(step=Mock(), success=True),
            StepResult(step=Mock(), success=False),
            StepResult(step=Mock(), success=True),
        ]
        
        assert result.success_rate == 75.0


class TestAnalysisPipelineExecutor:
    """Tests for AnalysisPipelineExecutor class."""
    
    def test_execute_step_success(self, executor, mock_ast):
        """Test successful step execution."""
        step = AnalysisStep(
            name="Call graph",
            analyzer_key="call_graph",
            method_name="analyze_internal_calls",
            result_attr="call_graph",
            failure_message="failed",
            priority=Priority.P0_CRITICAL
        )
        
        result = executor.execute_step(step, mock_ast)
        
        assert result.success is True
        assert result.value == {'nodes': [], 'edges': []}
    
    def test_execute_step_missing_analyzer(self, executor, mock_ast):
        """Test step execution with missing analyzer."""
        step = AnalysisStep(
            name="Missing",
            analyzer_key="nonexistent",
            method_name="analyze",
            result_attr="result",
            failure_message="failed",
            priority=Priority.P0_CRITICAL
        )
        
        result = executor.execute_step(step, mock_ast)
        
        assert result.success is False
        assert isinstance(result.error, KeyError)
    
    def test_execute_step_missing_method(self, executor, mock_ast):
        """Test step execution with missing method."""
        step = AnalysisStep(
            name="Missing method",
            analyzer_key="call_graph",
            method_name="nonexistent_method",
            result_attr="result",
            failure_message="failed",
            priority=Priority.P0_CRITICAL
        )
        
        result = executor.execute_step(step, mock_ast)
        
        assert result.success is False
        assert isinstance(result.error, AttributeError)
    
    def test_execute_step_with_exception(self, mock_analyzers, mock_ast):
        """Test step execution when method raises exception."""
        mock_analyzers._analyzers['call_graph'].analyze_internal_calls.side_effect = ValueError("Error")
        
        executor = AnalysisPipelineExecutor(
            mock_analyzers,
            {'target_module_str': 'test.py'}
        )
        
        step = AnalysisStep(
            name="Call graph",
            analyzer_key="call_graph",
            method_name="analyze_internal_calls",
            result_attr="call_graph",
            failure_message="failed",
            priority=Priority.P0_CRITICAL
        )
        
        result = executor.execute_step(step, mock_ast)
        
        assert result.success is False
        assert isinstance(result.error, ValueError)
    
    def test_execute_step_without_ast(self, executor, mock_ast):
        """Test step execution without AST requirement."""
        step = AnalysisStep(
            name="Test discovery",
            analyzer_key="test",
            method_name="find_existing_tests",
            result_attr="test_discovery",
            failure_message="failed",
            priority=Priority.P1_IMPORTANT,
            requires_ast=False
        )
        
        result = executor.execute_step(step, mock_ast)
        
        assert result.success is True
        # Method should be called without AST
        executor.analyzers._analyzers['test'].find_existing_tests.assert_called_once_with()
    
    def test_execute_pipeline_all_steps(self, executor, mock_ast):
        """Test executing full pipeline."""
        steps = [
            AnalysisStep(
                name="Step 1",
                analyzer_key="call_graph",
                method_name="analyze_internal_calls",
                result_attr="result1",
                failure_message="",
                priority=Priority.P0_CRITICAL
            ),
            AnalysisStep(
                name="Step 2",
                analyzer_key="test",
                method_name="find_existing_tests",
                result_attr="result2",
                failure_message="",
                priority=Priority.P1_IMPORTANT,
                requires_ast=False
            ),
        ]
        
        result = executor.execute_pipeline(mock_ast, steps=steps)
        
        assert 'result1' in result.results
        assert 'result2' in result.results
        assert len(result.step_results) == 2
        assert result.success_rate == 100.0
    
    def test_execute_pipeline_priority_filtering(self, executor, mock_ast):
        """Test pipeline respects max_priority."""
        steps = [
            AnalysisStep(
                name="P0 Step",
                analyzer_key="call_graph",
                method_name="analyze_internal_calls",
                result_attr="p0_result",
                failure_message="",
                priority=Priority.P0_CRITICAL
            ),
            AnalysisStep(
                name="P2 Step",
                analyzer_key="test",
                method_name="find_existing_tests",
                result_attr="p2_result",
                failure_message="",
                priority=Priority.P2_ENHANCEMENT,
                requires_ast=False
            ),
        ]
        
        result = executor.execute_pipeline(
            mock_ast,
            steps=steps,
            max_priority=Priority.P0_CRITICAL
        )
        
        assert 'p0_result' in result.results
        assert 'p2_result' not in result.results
    
    def test_execute_pipeline_failures_collected(self, mock_analyzers, mock_ast):
        """Test that failures are collected in pipeline result."""
        mock_analyzers._analyzers['call_graph'].analyze_internal_calls.side_effect = ValueError("Error")
        
        executor = AnalysisPipelineExecutor(
            mock_analyzers,
            {'target_module_str': 'test.py'}
        )
        
        steps = [
            AnalysisStep(
                name="Failing Step",
                analyzer_key="call_graph",
                method_name="analyze_internal_calls",
                result_attr="result",
                failure_message="check manually",
                priority=Priority.P0_CRITICAL
            ),
        ]
        
        result = executor.execute_pipeline(mock_ast, steps=steps)
        
        assert len(result.failures) == 1
        assert "check manually" in result.failures[0]


class TestDefaultAnalysisSteps:
    """Tests for default ANALYSIS_STEPS configuration."""
    
    def test_steps_exist(self):
        """Test that default steps are defined."""
        assert len(ANALYSIS_STEPS) > 0
    
    def test_all_steps_have_required_fields(self):
        """Test all steps have required fields."""
        for step in ANALYSIS_STEPS:
            assert step.name
            assert step.analyzer_key
            assert step.method_name
            assert step.result_attr
            assert isinstance(step.priority, Priority)
    
    def test_p0_steps_exist(self):
        """Test that P0 critical steps exist."""
        p0_steps = [s for s in ANALYSIS_STEPS if s.priority == Priority.P0_CRITICAL]
        assert len(p0_steps) >= 3  # At least call_graph, caller, contract
    
    def test_steps_have_unique_result_attrs(self):
        """Test that result attributes are unique."""
        attrs = [s.result_attr for s in ANALYSIS_STEPS]
        assert len(attrs) == len(set(attrs))
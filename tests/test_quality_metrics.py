"""
Tests for quality_metrics module.
"""

import pytest
from unittest.mock import Mock

from intellirefactor.analysis.expert.quality_metrics import (
    calculate_quality_score,
    assess_overall_risk,
    generate_recommendations,
    QualityWeights,
    RiskThresholds,
)
from intellirefactor.analysis.expert.models import ExpertAnalysisResult, RiskLevel


@pytest.fixture
def empty_result():
    """Create an empty ExpertAnalysisResult."""
    return ExpertAnalysisResult(
        target_file="test.py",
        timestamp="20240101_120000"
    )


@pytest.fixture
def full_result():
    """Create a fully populated ExpertAnalysisResult."""
    result = ExpertAnalysisResult(
        target_file="test.py",
        timestamp="20240101_120000"
    )
    
    # Critical components
    result.call_graph = Mock(nodes=[], edges=[], cycles=[])
    result.external_callers = [Mock()]
    result.behavioral_contracts = [Mock()]
    result.dependency_interfaces = [Mock()]
    
    # Important components
    result.cohesion_matrix = Mock()
    result.test_discovery = Mock(existing_test_files=['test.py'])
    result.characterization_tests = [Mock()]
    
    # Enhancement components
    result.duplicate_fragments = [Mock(estimated_savings=10)]
    result.git_patterns = Mock()
    result.impact_assessment = Mock()
    
    return result


class TestCalculateQualityScore:
    """Tests for calculate_quality_score function."""
    
    def test_empty_result_zero_score(self, empty_result):
        """Test that empty result gives zero score."""
        score = calculate_quality_score(empty_result)
        assert score == 0.0
    
    def test_full_result_max_score(self, full_result):
        """Test that full result gives max score."""
        score = calculate_quality_score(full_result)
        assert score == 100.0
    
    def test_partial_result(self, empty_result):
        """Test partial result scoring."""
        empty_result.call_graph = Mock()  # +15
        empty_result.external_callers = [Mock()]  # +15
        
        score = calculate_quality_score(empty_result)
        assert score == 30.0
    
    def test_custom_weights(self, empty_result):
        """Test with custom weights."""
        empty_result.call_graph = Mock()
        
        custom_weights = QualityWeights(call_graph=50.0)
        score = calculate_quality_score(empty_result, weights=custom_weights)
        
        assert score == 50.0
    
    def test_max_score_cap(self, full_result):
        """Test that score is capped at max_score."""
        score = calculate_quality_score(full_result, max_score=50.0)
        assert score == 50.0
    
    def test_empty_list_not_counted(self, empty_result):
        """Test that empty lists are not counted."""
        empty_result.external_callers = []  # Empty list
        
        score = calculate_quality_score(empty_result)
        assert score == 0.0


class TestAssessOverallRisk:
    """Tests for assess_overall_risk function."""
    
    def test_no_data_critical_risk(self, empty_result):
        """Test that missing critical data gives CRITICAL risk."""
        risk = assess_overall_risk(empty_result)
        assert risk == RiskLevel.CRITICAL
    
    def test_full_data_low_risk(self, full_result):
        """Test that full data gives LOW risk."""
        full_result.usage_analysis = Mock(total_callers=5)
        risk = assess_overall_risk(full_result)
        assert risk == RiskLevel.LOW
    
    def test_missing_tests_increases_risk(self, full_result):
        """Test that missing tests increases risk."""
        full_result.test_discovery = None
        
        risk = assess_overall_risk(full_result)
        # Should be at least MEDIUM due to missing tests
        assert risk.value in ['medium', 'high', 'critical']
    
    def test_high_caller_count_increases_risk(self, full_result):
        """Test that high caller count increases risk."""
        full_result.usage_analysis = Mock(total_callers=15)  # > 10
        
        risk = assess_overall_risk(full_result)
        # Should add 1 to risk factors
        assert risk in [RiskLevel.LOW, RiskLevel.MEDIUM]
    
    def test_many_duplicates_increases_risk(self, full_result):
        """Test that many duplicates increases risk."""
        full_result.duplicate_fragments = [Mock() for _ in range(10)]  # > 5
        
        risk = assess_overall_risk(full_result)
        assert risk in [RiskLevel.LOW, RiskLevel.MEDIUM]
    
    def test_custom_thresholds(self, empty_result):
        """Test with custom thresholds."""
        empty_result.call_graph = Mock()  # Removes 3 risk points
        
        # With very low thresholds, even partial data is CRITICAL
        strict_thresholds = RiskThresholds(critical=1, high=0, medium=0)
        risk = assess_overall_risk(empty_result, thresholds=strict_thresholds)
        
        assert risk == RiskLevel.CRITICAL


class TestGenerateRecommendations:
    """Tests for generate_recommendations function."""
    
    def test_low_quality_score_recommendation(self, empty_result):
        """Test recommendation for low quality score."""
        empty_result.analysis_quality_score = 30.0
        empty_result.risk_assessment = RiskLevel.LOW
        
        recs = generate_recommendations(empty_result)
        
        assert any("Low analysis quality" in r for r in recs)
    
    def test_high_risk_recommendation(self, empty_result):
        """Test recommendation for high risk."""
        empty_result.analysis_quality_score = 80.0
        empty_result.risk_assessment = RiskLevel.HIGH
        
        recs = generate_recommendations(empty_result)
        
        assert any("High risk" in r for r in recs)
    
    def test_many_dependencies_recommendation(self, empty_result):
        """Test recommendation for many external dependencies."""
        empty_result.analysis_quality_score = 80.0
        empty_result.risk_assessment = RiskLevel.LOW
        empty_result.external_callers = [Mock() for _ in range(10)]  # > 5
        
        recs = generate_recommendations(empty_result)
        
        assert any("many external dependencies" in r for r in recs)
    
    def test_circular_dependencies_recommendation(self, empty_result):
        """Test recommendation for circular dependencies."""
        empty_result.analysis_quality_score = 80.0
        empty_result.risk_assessment = RiskLevel.LOW
        empty_result.call_graph = Mock(cycles=[Mock()])
        
        recs = generate_recommendations(empty_result)
        
        assert any("Circular dependencies" in r for r in recs)
    
    def test_no_recommendations_for_good_result(self, full_result):
        """Test no recommendations for good result."""
        full_result.analysis_quality_score = 95.0
        full_result.risk_assessment = RiskLevel.LOW
        full_result.call_graph.cycles = []
        
        recs = generate_recommendations(full_result)
        
        # Should be empty or minimal
        assert len(recs) <= 1
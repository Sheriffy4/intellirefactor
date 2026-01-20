"""
Test regression test for expert analysis MVP.

This test verifies that the expert analysis pipeline produces consistent,
valid output that meets the MVP requirements.
"""

import json
from pathlib import Path

from intellirefactor.analysis.expert.expert_analyzer import ExpertRefactoringAnalyzer


def test_expert_analysis_mvp():
    """Test the expert analysis MVP functionality."""
    
    # Setup
    fixture_path = Path(__file__).parent / "fixtures" / "sample_project"
    target_file = fixture_path / "sample_module.py"
    
    # Verify fixtures exist
    assert fixture_path.exists(), f"Fixture path {fixture_path} does not exist"
    assert target_file.exists(), f"Target file {target_file} does not exist"
    
    # Run expert analysis
    analyzer = ExpertRefactoringAnalyzer(
        project_root=str(fixture_path),
        target_module=str(target_file)
    )
    
    # Basic analysis
    result = analyzer.analyze_for_expert_refactoring()
    assert result is not None, "Analysis should return a result"
    
    # Export detailed data
    detailed_data = analyzer.export_detailed_expert_data()
    assert isinstance(detailed_data, dict), "Detailed data should be a dictionary"
    
    # Generate reports
    output_dir = Path("test_regression_output")
    output_dir.mkdir(exist_ok=True)
    
    report_path = analyzer.generate_expert_report(str(output_dir))
    assert Path(report_path).exists(), "Report file should be created"
    
    # Verify JSON report exists and is valid
    json_files = list(output_dir.glob("expert_analysis_*.json"))
    assert len(json_files) > 0, "Should create JSON report file"
    
    # Validate JSON structure
    json_file = json_files[0]
    with open(json_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # Check basic structure
    required_fields = ["target_file", "timestamp", "analysis_quality_score"]
    for field in required_fields:
        assert field in json_data, f"Missing required field: {field}"
    
    # Check quality score is reasonable
    quality_score = json_data["analysis_quality_score"]
    assert isinstance(quality_score, (int, float)), "Quality score should be numeric"
    assert 0 <= quality_score <= 100, "Quality score should be between 0-100"
    
    # Verify deterministic file scanning
    python_files = list(fixture_path.rglob("*.py"))
    # Should find at least the sample files (excluding __pycache__)
    python_file_count = len([f for f in python_files if "__pycache__" not in str(f)])
    assert python_file_count >= 2, f"Should find at least 2 Python files, found {python_file_count}"
    
    print("✅ MVP regression test passed!")
    print(f"   - Found {python_file_count} Python files")
    print(f"   - Quality score: {quality_score}")
    print(f"   - Generated report: {report_path}")


if __name__ == "__main__":
    test_expert_analysis_mvp()
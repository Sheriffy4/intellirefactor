# intellirefactor/analysis/expert/analysis_report_utils.py
"""
Утилиты генерации отчетов для ExpertAnalysisResult.

Отличается от report_generator.py:
- report_generator.py работает с ExpertReport (findings-based)
- Этот модуль работает с ExpertAnalysisResult (analysis-based)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from .models import ExpertAnalysisResult, RiskLevel
from .safe_access import safe_get_nested


def generate_expert_recommendations(detailed_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Генерация рекомендаций для экспертов на основе детальных данных анализа.
    
    Args:
        detailed_data: Детальные данные анализа
        
    Returns:
        Словарь с рекомендациями по категориям
    """
    recommendations = {
        'expert_1_requirements': [],
        'expert_2_requirements': [],
        'general_recommendations': []
    }
    
    # Expert 1: Call graph
    if 'call_graph' in detailed_data:
        total = safe_get_nested(
            detailed_data, 
            'call_graph', 'call_graph', 'total_relationships',
            default=0
        )
        recommendations['expert_1_requirements'].append(
            f"✅ Complete call graph available: {total} relationships documented"
        )
    
    # Expert 1: External usage
    if 'external_usage' in detailed_data:
        total_files = safe_get_nested(
            detailed_data,
            'external_usage', 'files_summary', 'total_files',
            default=0
        )
        recommendations['expert_1_requirements'].append(
            f"✅ External usage detailed: {total_files} files with specific line locations"
        )
    
    # Expert 1: Duplicates
    if 'duplicates' in detailed_data:
        total = safe_get_nested(
            detailed_data,
            'duplicates', 'summary', 'total_duplicates',
            default=0
        )
        recommendations['expert_1_requirements'].append(
            f"✅ Code duplicates detailed: {total} fragments with exact locations"
        )
    
    # Expert 1: Cohesion
    if 'cohesion_matrix' in detailed_data:
        recommendations['expert_1_requirements'].append(
            "✅ Cohesion matrix available: method-attribute relationships with extraction recommendations"
        )
    
    # Expert 2: Characterization tests
    if 'characterization_tests' in detailed_data:
        total = safe_get_nested(
            detailed_data,
            'characterization_tests', 'summary', 'total_tests',
            default=0
        )
        recommendations['expert_2_requirements'].append(
            f"✅ Executable characterization tests: {total} test cases with mocks and fixtures"
        )
    
    # Expert 2: Missing coverage
    if 'test_analysis' in detailed_data:
        missing = safe_get_nested(
            detailed_data,
            'test_analysis', 'missing_test_coverage', 'total_missing',
            default=0
        )
        recommendations['expert_2_requirements'].append(
            f"✅ Missing test coverage detailed: {missing} specific methods identified"
        )
    
    # General
    recommendations['general_recommendations'] = [
        "All expert requirements have been addressed with detailed data",
        "Use the exported JSON data for comprehensive refactoring planning",
        "Follow the phased approach: 1) Resolve cycles, 2) Add tests, 3) Refactor systematically"
    ]
    
    return recommendations


def generate_acceptance_criteria(detailed_data: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Генерация критериев приемки для рефакторинга.
    
    Args:
        detailed_data: Детальные данные анализа
        
    Returns:
        Словарь с критериями по категориям
    """
    criteria = {
        'behavioral_invariants': [
            "Attack recipe structure and order must be preserved",
            "Parameter validation must occur before any processing",
            "All existing public method signatures must remain unchanged",
            "Exception types and messages should remain consistent",
            "State transitions must follow the same sequence"
        ],
        'api_compatibility': [],
        'performance_requirements': [
            "Refactored code should not be significantly slower than original",
            "Memory usage patterns should remain similar",
            "No new performance bottlenecks should be introduced"
        ],
        'error_handling': [],
        'logging_requirements': [
            "All existing log levels and messages must be preserved",
            "Log format should remain consistent for external parsing",
            "Debug information should not be reduced"
        ]
    }
    
    # API compatibility based on external usage
    if 'external_usage' in detailed_data:
        total = safe_get_nested(
            detailed_data,
            'external_usage', 'files_summary', 'total_files',
            default=0
        )
        if total > 0:
            criteria['api_compatibility'] = [
                f"All {total} external files must continue to work without changes",
                "Public method return types must remain compatible",
                "Parameter order and types must be preserved for public methods"
            ]
    
    # Error handling
    if 'exception_contracts' in detailed_data:
        criteria['error_handling'] = [
            "All existing exception types must be preserved",
            "Error conditions that previously raised exceptions must continue to do so",
            "Error messages should remain informative and consistent"
        ]
    
    return criteria


def generate_analysis_markdown_report(result: ExpertAnalysisResult) -> str:
    """
    Генерация Markdown отчета из результатов анализа ExpertAnalysisResult.
    
    Отличается от ExpertReportGenerator.to_markdown():
    - Работает с ExpertAnalysisResult, а не с ExpertReport
    - Содержит секции специфичные для refactoring analysis
    
    Args:
        result: Результат анализа
        
    Returns:
        Markdown-текст отчета
    """
    lines = [
        "# Expert Refactoring Analysis Report",
        "",
        f"**Target Module:** {result.target_file}",
        f"**Analysis Date:** {result.timestamp}",
        f"**Quality Score:** {result.analysis_quality_score:.1f}/100",
        f"**Risk Level:** {_format_risk_level(result.risk_assessment)}",
        "",
        "## Executive Summary",
        "",
    ]
    
    # Recommendations
    if result.recommendations:
        lines.extend(["### Key Recommendations", ""])
        for rec in result.recommendations:
            lines.append(f"- {rec}")
        lines.append("")
    
    # Call Graph
    lines.extend(_generate_call_graph_section(result))
    
    # External Usage
    lines.extend(_generate_external_usage_section(result))
    
    # Test Analysis
    lines.extend(_generate_test_section(result))
    
    # Characterization Tests
    lines.extend(_generate_characterization_section(result))
    
    # Duplicates
    lines.extend(_generate_duplicate_section(result))
    
    # Compatibility
    lines.extend(_generate_compatibility_section(result))
    
    # Footer
    lines.extend([
        "## Next Steps",
        "",
        "1. Review all high-risk findings above",
        "2. Create/run characterization tests",
        "3. Address circular dependencies",
        "4. Plan refactoring in small, safe steps",
        "5. Monitor external usage during changes",
        "",
        "---",
        f"*Report generated by ExpertRefactoringAnalyzer at {datetime.now().isoformat()}*",
    ])
    
    return "\n".join(lines)


# ============ Helper functions ============

def _format_risk_level(risk: RiskLevel) -> str:
    """Format risk level for display."""
    if risk is None:
        return "UNKNOWN"
    return risk.value.upper()


def _generate_call_graph_section(result: ExpertAnalysisResult) -> List[str]:
    """Generate call graph section."""
    if not result.call_graph:
        return []
    
    cg = result.call_graph
    lines = [
        "## Call Graph Analysis",
        "",
        f"- **Methods:** {len(cg.nodes)}",
        f"- **Call Relationships:** {len(cg.edges)}",
        f"- **Cycles Detected:** {len(cg.cycles)}",
        "",
    ]
    
    if cg.cycles:
        lines.extend(["### Circular Dependencies", ""])
        for cycle in cg.cycles:
            nodes_str = ' → '.join(cycle.nodes)
            lines.append(f"- {nodes_str} (Risk: {cycle.risk_level.value})")
        lines.append("")
    
    return lines


def _generate_external_usage_section(result: ExpertAnalysisResult) -> List[str]:
    """Generate external usage section."""
    if not result.external_callers:
        return []
    
    lines = [
        "## External Usage Analysis",
        "",
        f"- **External Files:** {len(result.external_callers)}",
    ]
    
    if result.usage_analysis:
        ua = result.usage_analysis
        lines.append(f"- **Total Callers:** {ua.total_callers}")
        
        if ua.most_used_symbols:
            lines.extend(["", "### Most Used Symbols", ""])
            for symbol, count in ua.most_used_symbols[:5]:
                lines.append(f"- `{symbol}`: {count} uses")
    
    lines.append("")
    return lines


def _generate_test_section(result: ExpertAnalysisResult) -> List[str]:
    """Generate test analysis section."""
    if not result.test_discovery:
        return []
    
    td = result.test_discovery
    return [
        "## Test Coverage Analysis",
        "",
        f"- **Test Files Found:** {len(td.existing_test_files)}",
        f"- **Quality Score:** {td.test_quality_score:.1f}/100",
        f"- **Missing Tests:** {len(td.missing_tests)}",
        "",
    ]


def _generate_characterization_section(result: ExpertAnalysisResult) -> List[str]:
    """Generate characterization tests section."""
    if not result.characterization_tests:
        return []
    
    lines = [
        "## Characterization Tests",
        "",
        f"Generated {len(result.characterization_tests)} characterization test cases:",
        "",
    ]
    
    # Group by category
    by_category: Dict[str, List] = {}
    for test in result.characterization_tests:
        category = test.test_category.value
        by_category.setdefault(category, []).append(test)
    
    for category, tests in by_category.items():
        lines.extend([f"### {category.title()} Cases ({len(tests)})", ""])
        for test in tests[:3]:  # Show first 3
            method = f"{test.class_name}.{test.method_name}" if test.class_name else test.method_name
            lines.append(f"- `{method}`: {test.description}")
        lines.append("")
    
    return lines


def _generate_duplicate_section(result: ExpertAnalysisResult) -> List[str]:
    """Generate duplicate analysis section."""
    if not result.duplicate_fragments:
        return []
    
    total_savings = sum(f.estimated_savings for f in result.duplicate_fragments)
    
    return [
        "## Code Duplication Analysis",
        "",
        f"Found {len(result.duplicate_fragments)} duplicate code fragments:",
        "",
        f"**Potential Savings:** {total_savings} lines of code",
        "",
    ]


def _generate_compatibility_section(result: ExpertAnalysisResult) -> List[str]:
    """Generate compatibility section."""
    if not result.compatibility_constraints:
        return []
    
    lines = ["## Compatibility Constraints", ""]
    for constraint in result.compatibility_constraints:
        lines.append(f"- {constraint}")
    lines.append("")
    
    return lines
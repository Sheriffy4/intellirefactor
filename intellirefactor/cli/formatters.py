"""
Output formatters for CLI commands.

Provides formatting functions for various analysis results including
clone detection, similarity analysis, unused code, audit results, and
architectural smells. Supports multiple output formats: text, JSON, HTML.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


def format_clone_detection_results(
    clone_groups,
    statistics,
    format_type: str,
    show_code: bool = False,
    group_by: str = "type",
) -> str:
    """Format clone detection results for output."""
    from intellirefactor.analysis.dedup.block_clone_detector import ExtractionStrategy

    if format_type == "json":
        # Convert clone groups to JSON-serializable format
        json_groups = []
        for group in clone_groups:
            json_group = {
                "group_id": group.group_id,
                "clone_type": group.clone_type.value,
                "similarity_score": group.similarity_score,
                "extraction_strategy": (
                    group.extraction_strategy.value if group.extraction_strategy else None
                ),
                "extraction_confidence": group.extraction_confidence,
                "instance_count": len(group.instances),
                "instances": [],
            }

            for instance in group.instances:
                json_instance = {
                    "file_path": instance.file_path,
                    "line_start": instance.line_start,
                    "line_end": instance.line_end,
                    "lines_of_code": instance.block_info.lines_of_code,
                    "statement_count": instance.block_info.statement_count,
                    "similarity_score": instance.similarity_score,
                    "extraction_feasibility": instance.extraction_feasibility,
                }
                json_group["instances"].append(json_instance)

            json_groups.append(json_group)

        return json.dumps({"clone_groups": json_groups, "statistics": statistics}, indent=2)

    elif format_type == "html":
        return _format_clone_detection_html(clone_groups, statistics)

    else:  # text format
        return _format_clone_detection_text(clone_groups, statistics, show_code, group_by)


def _format_clone_detection_html(clone_groups, statistics) -> str:
    """Format clone detection results as HTML."""
    from intellirefactor.analysis.dedup.block_clone_detector import ExtractionStrategy

    html_parts = []
    html_parts.append("<html><head><title>Clone Detection Results</title>")
    html_parts.append("<style>")
    html_parts.append("body { font-family: Arial, sans-serif; margin: 20px; }")
    html_parts.append(".clone-group { border: 1px solid #ccc; margin: 10px 0; padding: 10px; }")
    html_parts.append(".instance { background: #f5f5f5; margin: 5px 0; padding: 5px; }")
    html_parts.append(".code { background: #f0f0f0; padding: 10px; font-family: monospace; }")
    html_parts.append("</style></head><body>")

    html_parts.append("<h1>Clone Detection Results</h1>")
    html_parts.append(
        f"<p>Found {len(clone_groups)} clone groups with {statistics['total_instances']} total instances</p>"
    )

    for group in clone_groups:
        html_parts.append('<div class="clone-group">')
        html_parts.append(f"<h3>{group.clone_type.value.title()} Clone Group</h3>")
        html_parts.append(
            f"<p>Similarity: {group.similarity_score:.3f}, Instances: {len(group.instances)}</p>"
        )

        if (
            group.extraction_strategy
            and group.extraction_strategy != ExtractionStrategy.NO_EXTRACTION
        ):
            html_parts.append(
                f"<p><strong>Recommended:</strong> {group.extraction_strategy.value.replace('_', ' ').title()} (confidence: {group.extraction_confidence:.2f})</p>"
            )

        for instance in group.instances:
            html_parts.append('<div class="instance">')
            html_parts.append(
                f"<strong>{instance.file_path}:{instance.line_start}-{instance.line_end}</strong>"
            )
            html_parts.append(
                f" ({instance.block_info.lines_of_code} LOC, {instance.block_info.statement_count} statements)"
            )
            html_parts.append("</div>")

        html_parts.append("</div>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def _format_clone_detection_text(clone_groups, statistics, show_code, group_by) -> str:
    """Format clone detection results as text."""
    from intellirefactor.analysis.dedup.block_clone_detector import ExtractionStrategy

    output = []
    output.append("Clone Detection Results")
    output.append("=" * 50)
    output.append(
        f"Found {len(clone_groups)} clone groups with {statistics['total_instances']} total instances"
    )
    output.append(f"Files affected: {statistics['files_affected']}")
    output.append(f"Average similarity: {statistics['average_similarity']}")
    output.append("")

    # Group results based on group_by parameter
    if group_by == "type":
        grouped = {}
        for group in clone_groups:
            clone_type = group.clone_type.value
            if clone_type not in grouped:
                grouped[clone_type] = []
            grouped[clone_type].append(group)
    elif group_by == "file":
        grouped = {}
        for group in clone_groups:
            files = set(inst.file_path for inst in group.instances)
            key = f"{len(files)} files"
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(group)
    else:  # similarity
        grouped = {}
        for group in clone_groups:
            similarity_range = f"{int(group.similarity_score * 10) * 10}%-{int(group.similarity_score * 10) * 10 + 9}%"
            if similarity_range not in grouped:
                grouped[similarity_range] = []
            grouped[similarity_range].append(group)

    for group_key, groups in grouped.items():
        output.append(f"{group_key.title()} ({len(groups)} groups):")
        output.append("-" * 40)

        for group in groups:
            output.append(f"  Clone Group: {group.clone_type.value.title()}")
            output.append(f"  Similarity: {group.similarity_score:.3f}")
            output.append(f"  Instances: {len(group.instances)}")

            if (
                group.extraction_strategy
                and group.extraction_strategy != ExtractionStrategy.NO_EXTRACTION
            ):
                output.append(
                    f"  Recommended: {group.extraction_strategy.value.replace('_', ' ').title()} (confidence: {group.extraction_confidence:.2f})"
                )

            output.append("  Locations:")
            for instance in group.instances:
                output.append(
                    f"    {instance.file_path}:{instance.line_start}-{instance.line_end} ({instance.block_info.lines_of_code} LOC)"
                )

            if show_code and group.instances:
                # Show code snippet from first instance
                first_instance = group.instances[0]
                try:
                    with open(first_instance.file_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                        start_idx = max(0, first_instance.line_start - 1)
                        end_idx = min(len(lines), first_instance.line_end)
                        code_lines = lines[start_idx:end_idx]

                        output.append("  Code snippet:")
                        for i, line in enumerate(code_lines, start=first_instance.line_start):
                            output.append(f"    {i:4d}: {line.rstrip()}")
                except Exception:
                    output.append("  Code snippet: <unable to read file>")

            output.append("")

        output.append("")

    return "\n".join(output)


def format_similarity_results(
    results,
    format_type: str,
    show_evidence: bool = False,
    show_differences: bool = False,
    show_merge_recommendations: bool = False,
) -> str:
    """Format semantic similarity results for output."""
    if format_type == "json":
        return json.dumps(results, indent=2, default=str)

    if format_type == "html":
        return _format_similarity_html(results, show_evidence, show_differences, show_merge_recommendations)

    # Text format (default)
    return _format_similarity_text(results, show_evidence, show_differences, show_merge_recommendations)


def _format_similarity_html(results, show_evidence, show_differences, show_merge_recommendations) -> str:
    """Format similarity results as HTML."""
    output = ["<html><head><title>Semantic Similarity Results</title></head><body>"]
    output.append("<h1>Semantic Similarity Analysis Results</h1>")

    if not results:
        output.append("<p>No similar methods found.</p>")
    else:
        for result in results:
            output.append(f"<h2>Target Method: {result['target_method']['name']}</h2>")
            output.append(
                f"<p><strong>File:</strong> {result['target_method']['file_path']}</p>"
            )
            output.append(
                f"<p><strong>Lines:</strong> {result['target_method']['line_start']}-{result['target_method']['line_end']}</p>"
            )

            if result["similar_methods"]:
                output.append("<h3>Similar Methods:</h3>")
                output.append("<ul>")

                for similar in result["similar_methods"]:
                    output.append("<li>")
                    output.append(f"<strong>{similar['method']['name']}</strong> ")
                    output.append(f"(Score: {similar['similarity_score']:.3f}, ")
                    output.append(f"Confidence: {similar['confidence']:.3f})<br>")
                    output.append(f"File: {similar['method']['file_path']}<br>")
                    output.append(
                        f"Lines: {similar['method']['line_start']}-{similar['method']['line_end']}<br>"
                    )
                    output.append(f"Type: {similar['similarity_type']}")

                    if show_evidence and similar.get("evidence"):
                        output.append("<br><strong>Evidence:</strong>")
                        output.append("<ul>")
                        for evidence in similar["evidence"]:
                            output.append(f"<li>{evidence}</li>")
                        output.append("</ul>")

                    if show_differences and similar.get("differences"):
                        output.append("<br><strong>Differences:</strong>")
                        output.append("<ul>")
                        for diff in similar["differences"]:
                            output.append(f"<li>{diff}</li>")
                        output.append("</ul>")

                    if show_merge_recommendations and similar.get("merge_recommendation"):
                        rec = similar["merge_recommendation"]
                        output.append(
                            f"<br><strong>Merge Strategy:</strong> {rec['strategy']}<br>"
                        )
                        output.append(f"<strong>Effort:</strong> {rec['effort']}<br>")
                        output.append(f"<strong>Description:</strong> {rec['description']}")

                    output.append("</li>")

                output.append("</ul>")
            else:
                output.append("<p>No similar methods found for this target.</p>")

            output.append("<hr>")

    output.append("</body></html>")
    return "\n".join(output)


def _format_similarity_text(results, show_evidence, show_differences, show_merge_recommendations) -> str:
    """Format similarity results as text."""
    output = []
    output.append("=== Semantic Similarity Analysis Results ===")
    output.append("")

    if not results:
        output.append("No similar methods found.")
        return "\n".join(output)

    for result in results:
        output.append(f"Target Method: {result['target_method']['name']}")
        output.append(f"  File: {result['target_method']['file_path']}")
        output.append(
            f"  Lines: {result['target_method']['line_start']}-{result['target_method']['line_end']}"
        )
        output.append("")

        if result["similar_methods"]:
            output.append("Similar Methods:")

            for i, similar in enumerate(result["similar_methods"], 1):
                output.append(f"  {i}. {similar['method']['name']}")
                output.append(f"     File: {similar['method']['file_path']}")
                output.append(
                    f"     Lines: {similar['method']['line_start']}-{similar['method']['line_end']}"
                )
                output.append(f"     Similarity Score: {similar['similarity_score']:.3f}")
                output.append(f"     Confidence: {similar['confidence']:.3f}")
                output.append(f"     Type: {similar['similarity_type']}")

                if show_evidence and similar.get("evidence"):
                    output.append("     Evidence:")
                    for evidence in similar["evidence"]:
                        output.append(f"       - {evidence}")

                if show_differences and similar.get("differences"):
                    output.append("     Differences:")
                    for diff in similar["differences"]:
                        output.append(f"       - {diff}")

                if show_merge_recommendations and similar.get("merge_recommendation"):
                    rec = similar["merge_recommendation"]
                    output.append(f"     Merge Strategy: {rec['strategy']}")
                    output.append(f"     Effort: {rec['effort']}")
                    output.append(f"     Description: {rec['description']}")

                output.append("")
        else:
            output.append("  No similar methods found for this target.")
            output.append("")

        output.append("-" * 60)
        output.append("")

    return "\n".join(output)


def format_unused_code_results(
    analysis_result,
    format_type: str,
    show_evidence: bool = False,
    show_usage: bool = False,
    group_by: str = "type",
) -> str:
    """Format unused code detection results for output."""

    def _sev_text(sev) -> str:
        # поддерживает и Severity enum, и legacy-строки
        return getattr(sev, "value", str(sev))

    if format_type == "json":
        return json.dumps(analysis_result.to_dict(), indent=2)

    elif format_type == "html":
        return _format_unused_html(analysis_result, show_evidence, _sev_text)

    else:  # text format
        return _format_unused_text(analysis_result, show_evidence, show_usage, group_by, _sev_text)


def _format_unused_html(analysis_result, show_evidence, _sev_text) -> str:
    """Format unused code results as HTML."""
    html_parts = []
    html_parts.append("<html><head><title>Unused Code Detection Results</title>")
    html_parts.append("<style>")
    html_parts.append("body { font-family: Arial, sans-serif; margin: 20px; }")
    html_parts.append(".finding { border: 1px solid #ccc; margin: 10px 0; padding: 10px; }")
    html_parts.append(".evidence { background: #f5f5f5; margin: 5px 0; padding: 5px; }")
    html_parts.append(".usage { background: #f0f0f0; padding: 5px; font-family: monospace; }")
    html_parts.append("</style></head><body>")

    html_parts.append("<h1>Unused Code Detection Results</h1>")
    html_parts.append(f"<p>Found {len(analysis_result.findings)} unused code findings</p>")
    html_parts.append(f"<p>Entry points: {', '.join(analysis_result.entry_points)}</p>")

    for finding in analysis_result.findings:
        html_parts.append('<div class="finding">')
        html_parts.append(
            f"<h3>{finding.unused_type.value.replace('_', ' ').title()}: {finding.symbol_name}</h3>"
        )
        html_parts.append(
            f"<p><strong>File:</strong> {finding.file_path}:{finding.line_start}-{finding.line_end}</p>"
        )
        html_parts.append(
            f"<p><strong>Confidence:</strong> {finding.confidence:.3f} ({finding.confidence_level.value})</p>"
        )
        html_parts.append(f"<p><strong>Severity:</strong> {_sev_text(finding.severity)}</p>")

        if show_evidence:
            html_parts.append('<div class="evidence">')
            html_parts.append(f"<strong>Evidence:</strong> {finding.evidence.description}")
            html_parts.append("</div>")

        html_parts.append("</div>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def _format_unused_text(analysis_result, show_evidence, show_usage, group_by, _sev_text) -> str:
    """Format unused code results as text."""
    output = []
    output.append("Unused Code Detection Results")
    output.append("=" * 50)
    output.append(f"Found {len(analysis_result.findings)} unused code findings")
    output.append(f"Entry points: {', '.join(analysis_result.entry_points)}")
    output.append(
        f"Analysis confidence threshold: {analysis_result.analysis_metadata.get('confidence_threshold', 'N/A')}"
    )
    output.append("")

    # Statistics
    stats = analysis_result.statistics
    output.append("Statistics:")
    output.append(f"  Total findings: {stats['total_findings']}")
    output.append(f"  Files analyzed: {stats['total_files_analyzed']}")
    output.append(f"  High confidence: {stats['findings_by_confidence']['high']}")
    output.append(f"  Medium confidence: {stats['findings_by_confidence']['medium']}")
    output.append(f"  Low confidence: {stats['findings_by_confidence']['low']}")
    output.append("")

    # Group results based on group_by parameter
    if group_by == "type":
        grouped = {}
        for finding in analysis_result.findings:
            unused_type = finding.unused_type.value
            if unused_type not in grouped:
                grouped[unused_type] = []
            grouped[unused_type].append(finding)
    elif group_by == "file":
        grouped = {}
        for finding in analysis_result.findings:
            file_path = finding.file_path
            if file_path not in grouped:
                grouped[file_path] = []
            grouped[file_path].append(finding)
    else:  # confidence
        grouped = {"high": [], "medium": [], "low": []}
        for finding in analysis_result.findings:
            grouped[finding.confidence_level.value].append(finding)

    for group_key, findings in grouped.items():
        if not findings:
            continue

        output.append(f"{group_key.replace('_', ' ').title()} ({len(findings)} findings):")
        output.append("-" * 40)

        for finding in findings:
            output.append(f"  Symbol: {finding.symbol_name}")
            output.append(f"  Type: {finding.unused_type.value.replace('_', ' ')}")
            output.append(
                f"  Location: {finding.file_path}:{finding.line_start}-{finding.line_end}"
            )
            output.append(
                f"  Confidence: {finding.confidence:.3f} ({finding.confidence_level.value})"
            )
            output.append(f"  Severity: {_sev_text(finding.severity)}")

            if show_evidence:
                output.append(f"  Evidence: {finding.evidence.description}")
                if finding.evidence.metadata:
                    for key, value in finding.evidence.metadata.items():
                        if key not in ["dynamic_patterns"]:  # Skip verbose metadata
                            output.append(f"    {key}: {value}")

            if show_usage and finding.usage_references:
                output.append(f"  Usage references ({len(finding.usage_references)}):")
                for ref in finding.usage_references[:3]:  # Show first 3
                    output.append(
                        f"    {ref.file_path}:{ref.line_number} ({ref.pattern.value})"
                    )
                if len(finding.usage_references) > 3:
                    output.append(f"    ... and {len(finding.usage_references) - 3} more")

            if finding.dynamic_usage_indicators:
                output.append(
                    f"  Dynamic usage indicators: {len(finding.dynamic_usage_indicators)}"
                )

            output.append("")

        output.append("")

    return "\n".join(output)


def format_audit_results(audit_result, format_type: str) -> str:
    """Format audit results for output."""
    from intellirefactor.analysis.foundation.models import Severity

    if format_type == "json":
        return json.dumps(audit_result.to_dict(), indent=2)

    elif format_type == "html":
        return _format_audit_html(audit_result)

    else:  # text format
        return _format_audit_text(audit_result)


def _format_audit_html(audit_result) -> str:
    """Format audit results as HTML."""
    from intellirefactor.analysis.foundation.models import Severity

    html_parts = []
    html_parts.append("<html><head><title>Project Audit Results</title>")
    html_parts.append("<style>")
    html_parts.append("body { font-family: Arial, sans-serif; margin: 20px; }")
    html_parts.append(".finding { border: 1px solid #ccc; margin: 10px 0; padding: 10px; }")
    html_parts.append(".critical { border-color: #d32f2f; background: #ffebee; }")
    html_parts.append(".high { border-color: #f57c00; background: #fff3e0; }")
    html_parts.append(".medium { border-color: #fbc02d; background: #fffde7; }")
    html_parts.append(".low { border-color: #388e3c; background: #e8f5e8; }")
    html_parts.append(".evidence { background: #f5f5f5; margin: 5px 0; padding: 5px; }")
    html_parts.append("</style></head><body>")

    html_parts.append("<h1>Project Audit Results</h1>")
    html_parts.append(f"<p><strong>Project:</strong> {audit_result.project_path}</p>")
    html_parts.append(f"<p><strong>Total Findings:</strong> {len(audit_result.findings)}</p>")
    html_parts.append(
        f"<p><strong>Files Analyzed:</strong> {audit_result.statistics.files_analyzed}</p>"
    )
    html_parts.append(
        f"<p><strong>Analysis Time:</strong> {audit_result.statistics.analysis_time_seconds:.2f}s</p>"
    )

    # Group findings by severity
    for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
        severity_findings = audit_result.get_findings_by_severity(severity)
        if not severity_findings:
            continue

        html_parts.append(
            f"<h2>{severity.value.title()} Priority ({len(severity_findings)} findings)</h2>"
        )

        for finding in severity_findings:
            html_parts.append(f'<div class="finding {severity.value}">')
            html_parts.append(f"<h3>{finding.title}</h3>")
            html_parts.append(
                f"<p><strong>Type:</strong> {finding.finding_type.value.replace('_', ' ').title()}</p>"
            )
            html_parts.append(
                f"<p><strong>Location:</strong> {finding.file_path}:{finding.line_start}-{finding.line_end}</p>"
            )
            html_parts.append(f"<p><strong>Confidence:</strong> {finding.confidence:.3f}</p>")
            html_parts.append(f"<p><strong>Description:</strong> {finding.description}</p>")

            if finding.evidence.description:
                html_parts.append('<div class="evidence">')
                html_parts.append(f"<strong>Evidence:</strong> {finding.evidence.description}")
                html_parts.append("</div>")

            if finding.recommendations:
                html_parts.append("<p><strong>Recommendations:</strong></p>")
                html_parts.append("<ul>")
                for rec in finding.recommendations:
                    html_parts.append(f"<li>{rec}</li>")
                html_parts.append("</ul>")

            html_parts.append("</div>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)


def _format_audit_text(audit_result) -> str:
    """Format audit results as text."""
    from intellirefactor.analysis.foundation.models import Severity

    output = []
    output.append("Project Audit Results")
    output.append("=" * 50)
    output.append(f"Project: {audit_result.project_path}")
    output.append(f"Total findings: {len(audit_result.findings)}")
    output.append(f"Files analyzed: {audit_result.statistics.files_analyzed}")
    output.append(f"Analysis time: {audit_result.statistics.analysis_time_seconds:.2f}s")
    output.append("")

    # Statistics
    stats = audit_result.statistics
    output.append("Statistics:")

    # Severity distribution
    output.append("  Findings by Severity:")
    for severity_key, count in stats.findings_by_severity.items():
        if count > 0:
            emoji = {
                "critical": "[CRITICAL]",
                "high": "[HIGH]",
                "medium": "[MEDIUM]",
                "low": "[LOW]",
                "info": "[INFO]",
            }.get(severity_key, "")
            output.append(f"    {emoji} {severity_key.title()}: {count}")
    output.append("")

    # Type distribution
    output.append("  Findings by Type:")
    for finding_type, count in stats.findings_by_type.items():
        if count > 0:
            output.append(f"    {finding_type.replace('_', ' ').title()}: {count}")
    output.append("")

    # Confidence distribution
    output.append("  Confidence Distribution:")
    conf_dist = stats.confidence_distribution
    output.append(f"    High (>=80%): {conf_dist.get('high', 0)}")
    output.append(f"    Medium (50-79%): {conf_dist.get('medium', 0)}")
    output.append(f"    Low (<50%): {conf_dist.get('low', 0)}")
    output.append("")

    # Summary by analysis type
    if audit_result.index_result:
        output.append("Index Analysis:")
        idx = audit_result.index_result
        output.append(f"  Files processed: {idx.files_processed}")
        output.append(f"  Symbols found: {idx.symbols_found}")
        output.append(f"  Build time: {idx.build_time_seconds:.2f}s")
        output.append("")

    if audit_result.clone_groups:
        output.append("Duplicate Code Analysis:")
        output.append(f"  Clone groups found: {len(audit_result.clone_groups)}")
        total_instances = sum(len(group.instances) for group in audit_result.clone_groups)
        output.append(f"  Total instances: {total_instances}")
        output.append("")

    if audit_result.unused_result:
        output.append("Unused Code Analysis:")
        unused_stats = audit_result.unused_result.statistics
        output.append(f"  Unused items found: {unused_stats.get('total_findings', 0)}")
        output.append(f"  Entry points: {len(audit_result.unused_result.entry_points)}")
        output.append("")

    # Detailed findings by severity
    for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW]:
        severity_findings = audit_result.get_findings_by_severity(severity)
        if not severity_findings:
            continue

        emoji = {
            "critical": "[CRITICAL]",
            "high": "[HIGH]",
            "medium": "[MEDIUM]",
            "low": "[LOW]",
        }.get(severity.value, "")
        output.append(
            f"{emoji} {severity.value.title()} Priority Findings ({len(severity_findings)}):"
        )
        output.append("-" * 40)

        for finding in severity_findings:
            output.append(f"  {finding.title}")
            output.append(f"  Type: {finding.finding_type.value.replace('_', ' ')}")
            output.append(
                f"  Location: {Path(finding.file_path).name}:{finding.line_start}-{finding.line_end}"
            )
            output.append(f"  Confidence: {finding.confidence:.1%}")
            output.append(f"  Description: {finding.description}")

            if finding.recommendations:
                output.append("  Recommendations:")
                for rec in finding.recommendations[:2]:  # Show first 2
                    output.append(f"    - {rec}")
                if len(finding.recommendations) > 2:
                    output.append(f"    ... and {len(finding.recommendations) - 2} more")

            if finding.related_findings:
                output.append(f"  Related: {', '.join(finding.related_findings)}")

            output.append("")

        output.append("")

    # Summary recommendations
    if len(audit_result.findings) > 0:
        output.append("Next Steps:")
        critical_count = stats.findings_by_severity.get("critical", 0)
        high_count = stats.findings_by_severity.get("high", 0)

        if critical_count > 0:
            output.append(
                f"  1. [CRITICAL] Address {critical_count} critical issues immediately"
            )
        if high_count > 0:
            output.append(f"  2. [HIGH] Plan to fix {high_count} high priority issues")

        output.append("  3. [MEDIUM] Review medium priority issues for future sprints")
        output.append("  4. [LOW] Consider low priority improvements when time permits")
        output.append("")
        output.append("Use --emit-spec to generate detailed Requirements.md document")
    else:
        output.append("[SUCCESS] Excellent! No issues found in your codebase.")
        output.append("Your project appears to be well-maintained and follows good practices.")

    output.append("")

    return "\n".join(output)


def format_smells_text(smells, args, analysis_time, processed_files) -> str:
    """Format smell detection results as text."""
    output = []

    # Header
    output.append("Architectural Smell Detection Results")
    output.append("=" * 50)
    output.append(f"Project: {args.project_path}")
    output.append(f"Total smells: {len(smells)}")
    output.append(f"Files analyzed: {processed_files}")
    output.append(f"Analysis time: {analysis_time:.2f}s")
    output.append("")

    if not smells:
        output.append("[SUCCESS] No architectural smells detected!")
        output.append("Your code appears to follow good architectural practices.")
        return "\n".join(output)

    # Statistics
    output.append("Statistics:")

    # By severity
    severity_counts = {}
    for smell in smells:
        severity_counts[smell.severity.value] = severity_counts.get(smell.severity.value, 0) + 1

    output.append("  By Severity:")
    for severity in ["critical", "high", "medium", "low"]:
        count = severity_counts.get(severity, 0)
        if count > 0:
            icon = {
                "critical": "CRITICAL",
                "high": "HIGH",
                "medium": "MEDIUM",
                "low": "LOW",
            }[severity]
            output.append(f"    {icon}: {count}")

    # By type
    type_counts = {}
    for smell in smells:
        type_counts[smell.smell_type.value] = type_counts.get(smell.smell_type.value, 0) + 1

    output.append("  By Type:")
    for smell_type, count in type_counts.items():
        output.append(f"    {smell_type.replace('_', ' ').title()}: {count}")

    output.append("")

    # Group smells
    if args.group_by == "file":
        grouped_smells = {}
        for smell in smells:
            file_path = smell.file_path
            if file_path not in grouped_smells:
                grouped_smells[file_path] = []
            grouped_smells[file_path].append(smell)

        for file_path, file_smells in grouped_smells.items():
            output.append(f"File: {file_path}")
            output.append("-" * 40)
            for smell in file_smells:
                output.extend(_format_single_smell_text(smell, args))
            output.append("")

    elif args.group_by == "type":
        grouped_smells = {}
        for smell in smells:
            smell_type = smell.smell_type.value
            if smell_type not in grouped_smells:
                grouped_smells[smell_type] = []
            grouped_smells[smell_type].append(smell)

        for smell_type, type_smells in grouped_smells.items():
            output.append(f"Smell Type: {smell_type.replace('_', ' ').title()}")
            output.append("-" * 40)
            for smell in type_smells:
                output.extend(_format_single_smell_text(smell, args))
            output.append("")

    elif args.group_by == "severity":
        severity_order = ["critical", "high", "medium", "low"]
        for severity_key in severity_order:
            severity_smells = [s for s in smells if s.severity.value == severity_key]
            if severity_smells:
                output.append(f"Severity: {severity_key.upper()}")
                output.append("-" * 40)
                for smell in severity_smells:
                    output.extend(_format_single_smell_text(smell, args))
                output.append("")

    return "\n".join(output)


def _format_single_smell_text(smell, args) -> List[str]:
    """Format a single smell for text output."""
    lines = []

    # Basic info
    lines.append(f"  {smell.smell_type.value.replace('_', ' ').title()}")
    lines.append(f"  Symbol: {smell.symbol_name}")
    lines.append(f"  Location: {smell.file_path}:{smell.line_start}-{smell.line_end}")
    lines.append(f"  Severity: {smell.severity.value.upper()}")
    lines.append(f"  Confidence: {smell.confidence:.1%}")

    # Metrics
    if smell.metrics:
        lines.append("  Metrics:")
        for key, value in smell.metrics.items():
            if isinstance(value, float):
                lines.append(f"    {key}: {value:.3f}")
            else:
                lines.append(f"    {key}: {value}")

    # Evidence
    if args.show_evidence and smell.evidence:
        lines.append("  Evidence:")
        lines.append(f"    {smell.evidence.description}")
        if smell.evidence.metadata:
            for key, value in smell.evidence.metadata.items():
                if key not in ["responsibilities"]:  # Skip complex metadata
                    lines.append(f"    {key}: {value}")

    # Recommendations
    if args.show_recommendations and smell.recommendations:
        lines.append("  Recommendations:")
        for rec in smell.recommendations:
            lines.append(f"    - {rec}")

    # Decomposition strategy for God Classes
    if smell.decomposition_strategy:
        lines.append(f"  Decomposition Strategy: {smell.decomposition_strategy}")

    lines.append("")
    return lines


def format_smells_json(smells, args, analysis_time, processed_files) -> str:
    """Format smell detection results as JSON."""
    result = {
        "project_path": args.project_path,
        "analysis_metadata": {
            "total_smells": len(smells),
            "files_analyzed": processed_files,
            "analysis_time_seconds": analysis_time,
            "min_confidence": args.min_confidence,
            "include_patterns": args.include_patterns,
            "exclude_patterns": args.exclude_patterns,
        },
        "statistics": {"by_severity": {}, "by_type": {}},
        "smells": [],
    }

    # Calculate statistics
    for smell in smells:
        severity = smell.severity.value
        smell_type = smell.smell_type.value

        result["statistics"]["by_severity"][severity] = (
            result["statistics"]["by_severity"].get(severity, 0) + 1
        )
        result["statistics"]["by_type"][smell_type] = (
            result["statistics"]["by_type"].get(smell_type, 0) + 1
        )

    # Convert smells to dictionaries
    for smell in smells:
        smell_dict = {
            "smell_type": smell.smell_type.value,
            "severity": smell.severity.value,
            "confidence": smell.confidence,
            "file_path": smell.file_path,
            "symbol_name": smell.symbol_name,
            "line_start": smell.line_start,
            "line_end": smell.line_end,
            "metrics": smell.metrics,
            "evidence": {
                "description": smell.evidence.description,
                "confidence": smell.evidence.confidence,
                "metadata": smell.evidence.metadata,
            },
            "recommendations": smell.recommendations,
        }

        if smell.decomposition_strategy:
            smell_dict["decomposition_strategy"] = smell.decomposition_strategy

        if smell.extraction_candidates:
            smell_dict["extraction_candidates"] = smell.extraction_candidates

        result["smells"].append(smell_dict)

    return json.dumps(result, indent=2)


def format_smells_html(smells, args, analysis_time, processed_files) -> str:
    """Format smell detection results as HTML."""
    html_parts = []
    html_parts.append("<!DOCTYPE html>")
    html_parts.append("<html><head><title>Architectural Smell Detection Results</title>")
    html_parts.append("<style>")
    html_parts.append("body { font-family: Arial, sans-serif; margin: 20px; }")
    html_parts.append(".header { background-color: #f5f5f5; padding: 15px; border-radius: 5px; }")
    html_parts.append(".statistics { margin: 20px 0; }")
    html_parts.append(".smell { border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }")
    html_parts.append(".critical { border-left: 5px solid #d32f2f; }")
    html_parts.append(".high { border-left: 5px solid #f57c00; }")
    html_parts.append(".medium { border-left: 5px solid #fbc02d; }")
    html_parts.append(".low { border-left: 5px solid #388e3c; }")
    html_parts.append(".metrics { background-color: #f9f9f9; padding: 10px; margin: 10px 0; }")
    html_parts.append(".recommendations { background-color: #e3f2fd; padding: 10px; margin: 10px 0; }")
    html_parts.append("ul { margin: 5px 0; }")
    html_parts.append("</style></head><body>")
    
    html_parts.append('<div class="header">')
    html_parts.append("<h1>Architectural Smell Detection Results</h1>")
    html_parts.append(f"<p><strong>Project:</strong> {args.project_path}</p>")
    html_parts.append(f"<p><strong>Total smells:</strong> {len(smells)}</p>")
    html_parts.append(f"<p><strong>Files analyzed:</strong> {processed_files}</p>")
    html_parts.append(f"<p><strong>Analysis time:</strong> {analysis_time:.2f}s</p>")
    html_parts.append("</div>")

    if not smells:
        html_parts.append('<div class="success">')
        html_parts.append("<h2>Success!</h2>")
        html_parts.append("<p>No architectural smells detected. Your code appears to follow good architectural practices.</p>")
        html_parts.append("</div>")
        html_parts.append("</body></html>")
        return "\n".join(html_parts)

    # Add smells
    for smell in smells:
        severity_class = smell.severity.value
        html_parts.append(f'<div class="smell {severity_class}">')
        html_parts.append(f"<h3>{smell.smell_type.value.replace('_', ' ').title()}: {smell.symbol_name}</h3>")
        html_parts.append(f"<p><strong>Location:</strong> {smell.file_path}:{smell.line_start}-{smell.line_end}</p>")
        html_parts.append(f"<p><strong>Severity:</strong> {smell.severity.value.upper()}</p>")
        html_parts.append(f"<p><strong>Confidence:</strong> {smell.confidence:.1%}</p>")
        
        if smell.metrics:
            html_parts.append('<div class="metrics"><strong>Metrics:</strong><ul>')
            for key, value in smell.metrics.items():
                html_parts.append(f"<li>{key}: {value}</li>")
            html_parts.append("</ul></div>")
        
        if smell.recommendations:
            html_parts.append('<div class="recommendations"><strong>Recommendations:</strong><ul>')
            for rec in smell.recommendations:
                html_parts.append(f"<li>{rec}</li>")
            html_parts.append("</ul></div>")
        
        html_parts.append("</div>")

    html_parts.append("</body></html>")
    return "\n".join(html_parts)

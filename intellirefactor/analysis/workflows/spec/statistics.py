"""
Statistics and Summary Generators

Generates executive summary and statistics sections for specification documents.
"""

from typing import List

from ..audit_models import AuditResult


class ExecutiveSummaryGenerator:
    """Generates executive summary section."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate executive summary section.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for the executive summary
        """
        content = []
        content.append("## Executive Summary")
        content.append("")

        stats = audit_result.statistics

        if stats.total_findings == 0:
            content.append("✅ **Excellent!** No significant issues found in the codebase.")
            content.append("")
            content.append("The project appears to be well-maintained with:")
            content.append("- No duplicate code blocks detected")
            content.append("- No unused code identified")
            content.append("- Clean project structure")
            content.append("")
        else:
            content.append(
                f"📊 **Analysis Results:** Found {stats.total_findings} issues across {stats.files_analyzed} files."
            )
            content.append("")

            # Priority breakdown
            critical_count = stats.findings_by_severity.get("critical", 0)
            high_count = stats.findings_by_severity.get("high", 0)
            medium_count = stats.findings_by_severity.get("medium", 0)
            low_count = stats.findings_by_severity.get("low", 0)

            if critical_count > 0:
                content.append(
                    f"🚨 **{critical_count} Critical Issues** require immediate attention"
                )
            if high_count > 0:
                content.append(f"⚠️ **{high_count} High Priority Issues** should be addressed soon")
            if medium_count > 0:
                content.append(
                    f"📋 **{medium_count} Medium Priority Issues** for future improvement"
                )
            if low_count > 0:
                content.append(f"💡 **{low_count} Low Priority Issues** for consideration")

            content.append("")

            # Key areas for improvement
            content.append("**Key Areas for Improvement:**")

            duplicate_count = stats.findings_by_type.get("duplicate_block", 0)
            unused_count = stats.findings_by_type.get("unused_code", 0)
            quality_count = stats.findings_by_type.get("quality_issue", 0)

            if duplicate_count > 0:
                content.append(
                    f"- **Code Duplication:** {duplicate_count} duplicate code blocks found"
                )
            if unused_count > 0:
                content.append(f"- **Unused Code:** {unused_count} unused code elements identified")
            if quality_count > 0:
                content.append(f"- **Code Quality:** {quality_count} quality issues detected")

            content.append("")

        return content


class StatisticsGenerator:
    """Generates statistics section."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate statistics section.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for the statistics section
        """
        content = []
        content.append("## Analysis Statistics")
        content.append("")

        stats = audit_result.statistics

        content.append("| Metric | Value |")
        content.append("|--------|-------|")
        content.append(f"| Total Findings | {stats.total_findings} |")
        content.append(f"| Files Analyzed | {stats.files_analyzed} |")
        content.append(f"| Analysis Time | {stats.analysis_time_seconds:.2f}s |")
        content.append("")

        # Severity distribution
        content.append("### Findings by Severity")
        content.append("")
        for severity_key in ["critical", "high", "medium", "low", "info"]:
            count = stats.findings_by_severity.get(severity_key, 0)
            if count > 0:
                emoji = {
                    "critical": "🚨",
                    "high": "⚠️",
                    "medium": "📋",
                    "low": "💡",
                    "info": "ℹ️",
                }[severity_key]
                content.append(f"- {emoji} **{severity_key.title()}:** {count} findings")
        content.append("")

        # Type distribution
        content.append("### Findings by Type")
        content.append("")
        for finding_type, count in stats.findings_by_type.items():
            if count > 0:
                type_name = finding_type.replace("_", " ").title()
                content.append(f"- **{type_name}:** {count} findings")
        content.append("")

        # Confidence distribution
        content.append("### Confidence Distribution")
        content.append("")
        high_conf = stats.confidence_distribution.get("high", 0)
        medium_conf = stats.confidence_distribution.get("medium", 0)
        low_conf = stats.confidence_distribution.get("low", 0)

        content.append(f"- **High Confidence (≥80%):** {high_conf} findings")
        content.append(f"- **Medium Confidence (50-79%):** {medium_conf} findings")
        content.append(f"- **Low Confidence (<50%):** {low_conf} findings")
        content.append("")

        return content

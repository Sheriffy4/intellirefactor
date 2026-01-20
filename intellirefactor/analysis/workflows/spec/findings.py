"""
Finding Section Generators

Generates sections for critical findings, duplicates, unused code, and quality issues.
"""

from typing import List
from pathlib import Path

from ..audit_models import AuditFinding


class CriticalFindingsGenerator:
    """Generates critical findings section."""

    @staticmethod
    def generate(critical_findings: List[AuditFinding]) -> List[str]:
        """
        Generate critical findings section.

        Args:
            critical_findings: List of critical audit findings

        Returns:
            List of markdown lines for critical findings
        """
        content = []
        content.append("## 🚨 Critical Issues")
        content.append("")
        content.append(
            "These issues require immediate attention as they may impact system stability or functionality."
        )
        content.append("")

        for i, finding in enumerate(critical_findings, 1):
            content.append(f"### {i}. {finding.title}")
            content.append("")
            content.append(
                f"**File:** `{finding.file_path}:{finding.line_start}-{finding.line_end}`"
            )
            content.append(f"**Confidence:** {finding.confidence:.1%}")
            content.append("")
            content.append(f"**Description:** {finding.description}")
            content.append("")
            content.append(f"**Evidence:** {finding.evidence.description}")
            content.append("")

            if finding.recommendations:
                content.append("**Recommended Actions:**")
                for rec in finding.recommendations:
                    content.append(f"- {rec}")
                content.append("")

            if finding.related_findings:
                content.append(f"**Related Issues:** {', '.join(finding.related_findings)}")
                content.append("")

        return content


class HighPriorityGenerator:
    """Generates high priority findings section."""

    @staticmethod
    def generate(high_priority: List[AuditFinding]) -> List[str]:
        """
        Generate high priority findings section.

        Args:
            high_priority: List of high priority audit findings

        Returns:
            List of markdown lines for high priority findings
        """
        content = []
        content.append("## ⚠️ High Priority Issues")
        content.append("")
        content.append("These issues should be addressed in the next development cycle.")
        content.append("")

        # Group by type for better organization
        by_type = {}
        for finding in high_priority:
            finding_type = finding.finding_type.value
            if finding_type not in by_type:
                by_type[finding_type] = []
            by_type[finding_type].append(finding)

        for finding_type, findings in by_type.items():
            type_name = finding_type.replace("_", " ").title()
            content.append(f"### {type_name} ({len(findings)} issues)")
            content.append("")

            for finding in findings:
                content.append(
                    f"- **{finding.title}** in `{Path(finding.file_path).name}:{finding.line_start}`"
                )
                content.append(f"  - {finding.description}")
                content.append(f"  - Confidence: {finding.confidence:.1%}")
                if finding.recommendations:
                    content.append(f"  - Action: {finding.recommendations[0]}")
                content.append("")

        return content


class DuplicateCodeGenerator:
    """Generates duplicate code analysis section."""

    @staticmethod
    def generate(duplicate_findings: List[AuditFinding]) -> List[str]:
        """
        Generate duplicate code section.

        Args:
            duplicate_findings: List of duplicate code findings

        Returns:
            List of markdown lines for duplicate code analysis
        """
        content = []
        content.append("## 🔄 Code Duplication Analysis")
        content.append("")

        if not duplicate_findings:
            content.append("✅ No duplicate code blocks detected.")
            content.append("")
            return content

        content.append(
            f"Found {len(duplicate_findings)} duplicate code blocks that should be refactored."
        )
        content.append("")

        # Group by clone type
        by_clone_type = {}
        for finding in duplicate_findings:
            clone_type = finding.metadata.get("clone_type", "unknown")
            if clone_type not in by_clone_type:
                by_clone_type[clone_type] = []
            by_clone_type[clone_type].append(finding)

        for clone_type, findings in by_clone_type.items():
            content.append(f"### {clone_type.title()} Clones ({len(findings)} groups)")
            content.append("")

            for finding in findings:
                instances = finding.evidence.metadata.get("all_instances", [])
                content.append(f"**{finding.title}**")
                content.append(f"- Similarity: {finding.confidence:.1%}")
                content.append(f"- Instances: {len(instances)}")
                content.append("- Locations:")
                for instance in instances[:5]:  # Show first 5
                    content.append(f"  - `{instance}`")
                if len(instances) > 5:
                    content.append(f"  - ... and {len(instances) - 5} more")

                if finding.recommendations:
                    content.append("- Recommended action:")
                    content.append(f"  - {finding.recommendations[0]}")
                content.append("")

        # Summary recommendations
        content.append("### Refactoring Strategy")
        content.append("")
        content.append("1. **Start with exact clones** - These are the easiest to refactor")
        content.append(
            "2. **Extract common methods** - Create shared functions for duplicate logic"
        )
        content.append("3. **Use parameters** - Make extracted methods flexible with parameters")
        content.append("4. **Add tests** - Ensure behavior is preserved during refactoring")
        content.append("5. **Review structural clones** - May require more complex refactoring")
        content.append("")

        return content


class UnusedCodeGenerator:
    """Generates unused code analysis section."""

    @staticmethod
    def generate(unused_findings: List[AuditFinding]) -> List[str]:
        """
        Generate unused code section.

        Args:
            unused_findings: List of unused code findings

        Returns:
            List of markdown lines for unused code analysis
        """
        content = []
        content.append("## 🗑️ Unused Code Analysis")
        content.append("")

        if not unused_findings:
            content.append("✅ No unused code detected.")
            content.append("")
            return content

        content.append(f"Found {len(unused_findings)} potentially unused code elements.")
        content.append("")

        # Group by unused type
        by_unused_type = {}
        for finding in unused_findings:
            unused_type = finding.metadata.get("unused_type", "unknown")
            if unused_type not in by_unused_type:
                by_unused_type[unused_type] = []
            by_unused_type[unused_type].append(finding)

        for unused_type, findings in by_unused_type.items():
            type_name = unused_type.replace("_", " ").title()
            content.append(f"### {type_name} ({len(findings)} items)")
            content.append("")

            # Group by confidence for better prioritization
            high_conf = [f for f in findings if f.confidence >= 0.8]
            medium_conf = [f for f in findings if 0.5 <= f.confidence < 0.8]
            low_conf = [f for f in findings if f.confidence < 0.5]

            if high_conf:
                content.append("**High Confidence (Safe to Remove):**")
                for finding in high_conf:
                    content.append(
                        f"- `{finding.metadata.get('symbol_name', 'Unknown')}` in `{Path(finding.file_path).name}:{finding.line_start}`"
                    )
                content.append("")

            if medium_conf:
                content.append("**Medium Confidence (Review Before Removing):**")
                for finding in medium_conf:
                    content.append(
                        f"- `{finding.metadata.get('symbol_name', 'Unknown')}` in `{Path(finding.file_path).name}:{finding.line_start}`"
                    )
                content.append("")

            if low_conf:
                content.append("**Low Confidence (Investigate Further):**")
                for finding in low_conf:
                    content.append(
                        f"- `{finding.metadata.get('symbol_name', 'Unknown')}` in `{Path(finding.file_path).name}:{finding.line_start}`"
                    )
                content.append("")

        # Cleanup strategy
        content.append("### Cleanup Strategy")
        content.append("")
        content.append("1. **Start with high-confidence findings** - These are safest to remove")
        content.append("2. **Review medium-confidence items** - May need additional investigation")
        content.append("3. **Be cautious with low-confidence items** - May be used dynamically")
        content.append("4. **Check for API compatibility** - Don't remove public interfaces")
        content.append("5. **Run tests after removal** - Ensure functionality is preserved")
        content.append("")

        return content


class QualityPerformanceGenerator:
    """Generates quality and performance issues section."""

    @staticmethod
    def generate(
        quality_findings: List[AuditFinding],
        performance_findings: List[AuditFinding],
    ) -> List[str]:
        """
        Generate quality and performance issues section.

        Args:
            quality_findings: List of quality issue findings
            performance_findings: List of performance issue findings

        Returns:
            List of markdown lines for quality and performance issues
        """
        content = []
        content.append("## 🔧 Quality & Performance Issues")
        content.append("")

        if quality_findings:
            content.append(f"### Quality Issues ({len(quality_findings)} items)")
            content.append("")
            for finding in quality_findings:
                content.append(f"- **{finding.title}**")
                content.append(f"  - {finding.description}")
                content.append(
                    f"  - Location: `{Path(finding.file_path).name}:{finding.line_start}`"
                )
                if finding.recommendations:
                    content.append(f"  - Action: {finding.recommendations[0]}")
                content.append("")

        if performance_findings:
            content.append(f"### Performance Issues ({len(performance_findings)} items)")
            content.append("")
            for finding in performance_findings:
                content.append(f"- **{finding.title}**")
                content.append(f"  - {finding.description}")
                content.append(
                    f"  - Location: `{Path(finding.file_path).name}:{finding.line_start}`"
                )
                if finding.recommendations:
                    content.append(f"  - Action: {finding.recommendations[0]}")
                content.append("")

        return content

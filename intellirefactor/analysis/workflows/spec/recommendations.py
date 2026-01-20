"""
Recommendation Generators for Specification Documents

This module provides generators for implementation recommendations
based on audit findings and statistics.
"""

from typing import List

from ..audit_models import AuditResult


class ImplementationRecommendationsGenerator:
    """Generates implementation recommendations section for Requirements.md."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate implementation recommendations section.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for implementation recommendations
        """
        content = []
        content.append("## 📋 Implementation Plan")
        content.append("")

        stats = audit_result.statistics

        if stats.total_findings == 0:
            content.append("🎉 **No action required!** Your codebase is in excellent shape.")
            content.append("")
            content.append("**Maintenance Recommendations:**")
            content.append("- Continue regular code reviews")
            content.append("- Run periodic audits to maintain code quality")
            content.append("- Consider adding automated quality checks to CI/CD")
            content.append("")
            return content

        # Phase-based implementation plan
        critical_count = stats.findings_by_severity.get("critical", 0)
        high_count = stats.findings_by_severity.get("high", 0)
        medium_count = stats.findings_by_severity.get("medium", 0)

        phase = 1

        if critical_count > 0:
            content.append(f"### Phase {phase}: Critical Issues (Immediate)")
            content.append("")
            content.append("**Timeline:** 1-2 days")
            content.append("**Priority:** 🚨 Critical")
            content.append(f"**Items:** {critical_count} issues")
            content.append("")
            content.append("**Actions:**")
            content.append("- Address all critical issues immediately")
            content.append("- Focus on system stability and functionality")
            content.append("- Test thoroughly after each fix")
            content.append("")
            phase += 1

        if high_count > 0:
            content.append(f"### Phase {phase}: High Priority Issues")
            content.append("")
            content.append("**Timeline:** 1-2 weeks")
            content.append("**Priority:** ⚠️ High")
            content.append(f"**Items:** {high_count} issues")
            content.append("")
            content.append("**Actions:**")
            content.append("- Plan refactoring work for duplicate code")
            content.append("- Remove high-confidence unused code")
            content.append("- Address quality issues affecting maintainability")
            content.append("")
            phase += 1

        if medium_count > 0:
            content.append(f"### Phase {phase}: Medium Priority Issues")
            content.append("")
            content.append("**Timeline:** 1-2 months")
            content.append("**Priority:** 📋 Medium")
            content.append(f"**Items:** {medium_count} issues")
            content.append("")
            content.append("**Actions:**")
            content.append("- Refactor remaining duplicate code")
            content.append("- Review and clean up medium-confidence unused code")
            content.append("- Improve code organization and structure")
            content.append("")

        # General recommendations
        content.append("### General Recommendations")
        content.append("")
        content.append("1. **Establish Code Quality Gates**")
        content.append("   - Add automated duplicate detection to CI/CD")
        content.append("   - Set up regular unused code analysis")
        content.append("   - Implement code review checklists")
        content.append("")
        content.append("2. **Refactoring Best Practices**")
        content.append("   - Always add tests before refactoring")
        content.append("   - Make small, incremental changes")
        content.append("   - Review changes with team members")
        content.append("   - Document architectural decisions")
        content.append("")
        content.append("3. **Monitoring and Maintenance**")
        content.append("   - Run monthly code quality audits")
        content.append("   - Track metrics over time")
        content.append("   - Celebrate improvements and maintain standards")
        content.append("")

        return content

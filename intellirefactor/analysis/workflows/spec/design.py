"""
Design Document Generators

Generates sections for Design.md including architecture, components, strategy, and decisions.
"""

from typing import List
from pathlib import Path

from ..audit_models import AuditResult, AuditFindingType


class ArchitectureOverviewGenerator:
    """Generates architecture overview section."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate architecture overview section.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for architecture overview
        """
        content = []
        content.append("## Architecture Overview")
        content.append("")

        stats = audit_result.statistics

        # Current state analysis
        content.append("### Current State")
        content.append("")
        content.append(
            f"The analyzed codebase consists of {stats.files_analyzed} files with {stats.total_findings} identified issues."
        )
        content.append("")

        # Issue distribution
        duplicate_count = stats.findings_by_type.get("duplicate_block", 0)
        unused_count = stats.findings_by_type.get("unused_code", 0)
        quality_count = stats.findings_by_type.get("quality_issue", 0)

        if duplicate_count > 0:
            content.append(
                f"**Code Duplication:** {duplicate_count} duplicate blocks indicate opportunities for abstraction"
            )
        if unused_count > 0:
            content.append(
                f"**Dead Code:** {unused_count} unused elements suggest over-engineering or incomplete cleanup"
            )
        if quality_count > 0:
            content.append(
                f"**Quality Issues:** {quality_count} quality problems indicate architectural debt"
            )
        content.append("")

        # Architectural health assessment
        critical_count = stats.findings_by_severity.get("critical", 0)
        high_count = stats.findings_by_severity.get("high", 0)

        if critical_count == 0 and high_count == 0:
            content.append("### Architectural Health: 🟢 Good")
            content.append("The codebase shows good architectural health with no critical issues.")
        elif critical_count > 0:
            content.append("### Architectural Health: 🔴 Needs Attention")
            content.append(
                f"Critical issues ({critical_count}) require immediate architectural intervention."
            )
        elif high_count > 5:
            content.append("### Architectural Health: 🟡 Moderate Debt")
            content.append(
                f"High priority issues ({high_count}) indicate moderate architectural debt."
            )
        else:
            content.append("### Architectural Health: 🟡 Minor Issues")
            content.append("Minor architectural issues that can be addressed incrementally.")

        content.append("")
        return content


class ComponentAnalysisGenerator:
    """Generates component analysis section."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate component analysis section.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for component analysis
        """
        content = []
        content.append("## Component Analysis")
        content.append("")

        # Analyze findings by file to identify problematic components
        file_issues = {}
        for finding in audit_result.findings:
            file_path = finding.file_path
            if file_path not in file_issues:
                file_issues[file_path] = []
            file_issues[file_path].append(finding)

        # Sort files by issue count
        sorted_files = sorted(file_issues.items(), key=lambda x: len(x[1]), reverse=True)

        if sorted_files:
            content.append("### Components by Issue Density")
            content.append("")
            content.append("| Component | Issues | Severity | Primary Concerns |")
            content.append("|-----------|--------|----------|------------------|")

            for file_path, findings in sorted_files[:10]:  # Top 10 problematic files
                file_name = Path(file_path).name
                issue_count = len(findings)

                # Determine primary severity
                severities = [f.severity.value for f in findings]
                if "critical" in severities:
                    severity_label = "🚨 Critical"
                elif "high" in severities:
                    severity_label = "⚠️ High"
                elif "medium" in severities:
                    severity_label = "📋 Medium"
                else:
                    severity_label = "💡 Low"

                # Identify primary concerns
                types = [f.finding_type.value for f in findings]
                concerns = []
                if "duplicate_block" in types:
                    concerns.append("Duplication")
                if "unused_code" in types:
                    concerns.append("Dead Code")
                if "quality_issue" in types:
                    concerns.append("Quality")

                concerns_str = ", ".join(concerns) if concerns else "Various"

                content.append(
                    f"| `{file_name}` | {issue_count} | {severity_label} | {concerns_str} |"
                )

            content.append("")

        # Component recommendations
        content.append("### Component Refactoring Recommendations")
        content.append("")

        high_issue_files = [f for f, issues in file_issues.items() if len(issues) >= 5]
        if high_issue_files:
            content.append("**High Priority Components:**")
            for file_path in high_issue_files[:5]:
                file_name = Path(file_path).name
                issue_count = len(file_issues[file_path])
                content.append(
                    f"- `{file_name}` ({issue_count} issues) - Consider major refactoring or decomposition"
                )
            content.append("")

        duplicate_files = set()
        for finding in audit_result.get_findings_by_type(AuditFindingType.DUPLICATE_BLOCK):
            duplicate_files.add(finding.file_path)

        if duplicate_files:
            content.append("**Duplication Hotspots:**")
            for file_path in list(duplicate_files)[:5]:
                file_name = Path(file_path).name
                content.append(f"- `{file_name}` - Extract common functionality")
            content.append("")

        return content


class RefactoringStrategyGenerator:
    """Generates refactoring strategy section."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate refactoring strategy section.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for refactoring strategy
        """
        content = []
        content.append("## Refactoring Strategy")
        content.append("")

        stats = audit_result.statistics

        # Strategy based on findings
        content.append("### Strategic Approach")
        content.append("")

        critical_count = stats.findings_by_severity.get("critical", 0)
        duplicate_count = stats.findings_by_type.get("duplicate_block", 0)
        unused_count = stats.findings_by_type.get("unused_code", 0)

        if critical_count > 0:
            content.append("**1. Crisis Response Strategy**")
            content.append(f"- Address {critical_count} critical issues immediately")
            content.append("- Focus on system stability and functionality")
            content.append("- Minimal changes to reduce risk")
            content.append("")

        if duplicate_count > 0:
            content.append("**2. Consolidation Strategy**")
            content.append(f"- Extract {duplicate_count} duplicate code blocks")
            content.append("- Create reusable components and utilities")
            content.append("- Establish common patterns and abstractions")
            content.append("")

        if unused_count > 0:
            content.append("**3. Cleanup Strategy**")
            content.append(f"- Remove {unused_count} unused code elements")
            content.append("- Simplify codebase and reduce maintenance burden")
            content.append("- Improve code clarity and focus")
            content.append("")

        # Refactoring patterns
        content.append("### Recommended Refactoring Patterns")
        content.append("")

        if duplicate_count > 0:
            content.append("**Extract Method Pattern:**")
            content.append("- Identify common code blocks")
            content.append("- Extract into parameterized methods")
            content.append("- Replace duplicates with method calls")
            content.append("")

            content.append("**Template Method Pattern:**")
            content.append("- For similar but not identical code")
            content.append("- Define algorithm structure in base class")
            content.append("- Allow subclasses to override specific steps")
            content.append("")

        quality_findings = audit_result.get_findings_by_type(AuditFindingType.QUALITY_ISSUE)
        if quality_findings:
            content.append("**Single Responsibility Pattern:**")
            content.append("- Break down large classes and methods")
            content.append("- Separate concerns into focused components")
            content.append("- Improve testability and maintainability")
            content.append("")

        return content


class DependencyAnalysisGenerator:
    """Generates dependency analysis section."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate dependency analysis section.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for dependency analysis
        """
        content = []
        content.append("## Dependency Analysis")
        content.append("")

        # Analyze dependencies from index if available
        if audit_result.index_result:
            content.append("### Dependency Overview")
            content.append("")
            content.append(
                f"**Total Dependencies:** {audit_result.index_result.dependencies_found}"
            )
            content.append("")

            content.append("### Dependency Risks")
            content.append("")
            content.append("**Refactoring Impact Assessment:**")
            content.append("- Changes to shared components may affect multiple modules")
            content.append("- Duplicate code removal may alter dependency patterns")
            content.append("- Unused code removal may break dynamic dependencies")
            content.append("")

            content.append("**Mitigation Strategies:**")
            content.append("- Run comprehensive tests after each refactoring step")
            content.append("- Use dependency injection to reduce coupling")
            content.append("- Maintain backward compatibility during transitions")
            content.append("")
        else:
            content.append("### Dependency Analysis")
            content.append("")
            content.append(
                "*Dependency analysis requires index data. Run with --include-index for detailed dependency information.*"
            )
            content.append("")

        return content


class RiskAssessmentGenerator:
    """Generates risk assessment section."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate risk assessment section.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for risk assessment
        """
        content = []
        content.append("## Risk Assessment")
        content.append("")

        stats = audit_result.statistics

        # Calculate risk levels
        critical_count = stats.findings_by_severity.get("critical", 0)
        high_count = stats.findings_by_severity.get("high", 0)
        total_findings = stats.total_findings

        # Overall risk level
        if critical_count > 0:
            risk_level = "🔴 High Risk"
            risk_description = "Critical issues present significant risk to system stability"
        elif high_count > 10:
            risk_level = "🟡 Medium Risk"
            risk_description = "Multiple high-priority issues require careful planning"
        elif total_findings > 20:
            risk_level = "🟡 Medium Risk"
            risk_description = "Large number of issues increases refactoring complexity"
        else:
            risk_level = "🟢 Low Risk"
            risk_description = "Manageable number of issues with low complexity"

        content.append(f"### Overall Risk Level: {risk_level}")
        content.append("")
        content.append(risk_description)
        content.append("")

        # Specific risks
        content.append("### Specific Risks")
        content.append("")

        if critical_count > 0:
            content.append("**Critical Issue Risk:**")
            content.append(f"- {critical_count} critical issues may cause system failures")
            content.append("- Immediate attention required before other refactoring")
            content.append("- Risk of introducing bugs during fixes")
            content.append("")

        duplicate_count = stats.findings_by_type.get("duplicate_block", 0)
        if duplicate_count > 5:
            content.append("**Refactoring Complexity Risk:**")
            content.append(f"- {duplicate_count} duplicate blocks increase refactoring complexity")
            content.append("- Risk of breaking functionality during consolidation")
            content.append("- Potential for introducing new bugs")
            content.append("")

        unused_count = stats.findings_by_type.get("unused_code", 0)
        if unused_count > 0:
            content.append("**Dead Code Risk:**")
            content.append(f"- {unused_count} unused elements may have hidden dependencies")
            content.append("- Risk of removing code that's used dynamically")
            content.append("- Potential for breaking external integrations")
            content.append("")

        # Risk mitigation
        content.append("### Risk Mitigation")
        content.append("")
        content.append("**Testing Strategy:**")
        content.append("- Comprehensive test coverage before refactoring")
        content.append("- Automated regression testing after each change")
        content.append("- Manual testing for critical functionality")
        content.append("")

        content.append("**Incremental Approach:**")
        content.append("- Small, focused changes to minimize risk")
        content.append("- Frequent commits and rollback points")
        content.append("- Gradual deployment and monitoring")
        content.append("")

        return content


class DesignDecisionsGenerator:
    """Generates design decisions section."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate design decisions section.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for design decisions
        """
        content = []
        content.append("## Design Decisions")
        content.append("")

        stats = audit_result.statistics

        # Decision framework
        content.append("### Decision Framework")
        content.append("")
        content.append("Design decisions are based on:")
        content.append("- **Impact:** Effect on system functionality and performance")
        content.append("- **Risk:** Potential for introducing bugs or breaking changes")
        content.append("- **Effort:** Time and resources required for implementation")
        content.append("- **Confidence:** Certainty in the analysis and recommendations")
        content.append("")

        # Key decisions
        content.append("### Key Design Decisions")
        content.append("")

        duplicate_count = stats.findings_by_type.get("duplicate_block", 0)
        if duplicate_count > 0:
            content.append("**Decision 1: Code Consolidation Approach**")
            content.append("")
            content.append("*Problem:* Multiple duplicate code blocks reduce maintainability")
            content.append("")
            content.append("*Options Considered:*")
            content.append("- A) Leave duplicates as-is (low risk, high maintenance cost)")
            content.append("- B) Extract methods for exact duplicates (medium risk, high benefit)")
            content.append("- C) Create abstract base classes (high risk, high benefit)")
            content.append("")
            content.append("*Decision:* Option B - Extract methods for high-confidence duplicates")
            content.append("")
            content.append("*Rationale:*")
            content.append("- Balances risk and benefit effectively")
            content.append("- Preserves existing interfaces")
            content.append("- Enables incremental improvement")
            content.append("")

        unused_count = stats.findings_by_type.get("unused_code", 0)
        if unused_count > 0:
            content.append("**Decision 2: Unused Code Removal Strategy**")
            content.append("")
            content.append("*Problem:* Unused code increases maintenance burden")
            content.append("")
            content.append("*Options Considered:*")
            content.append("- A) Remove all unused code immediately (high risk)")
            content.append("- B) Remove only high-confidence unused code (medium risk)")
            content.append("- C) Mark unused code as deprecated first (low risk)")
            content.append("")
            content.append("*Decision:* Option B - Remove high-confidence unused code")
            content.append("")
            content.append("*Rationale:*")
            content.append("- Reduces codebase complexity")
            content.append("- Minimizes risk of breaking hidden dependencies")
            content.append("- Provides immediate maintenance benefits")
            content.append("")

        critical_count = stats.findings_by_severity.get("critical", 0)
        if critical_count > 0:
            content.append("**Decision 3: Critical Issue Resolution**")
            content.append("")
            content.append("*Problem:* Critical issues threaten system stability")
            content.append("")
            content.append("*Decision:* Address all critical issues before other refactoring")
            content.append("")
            content.append("*Rationale:*")
            content.append("- System stability is paramount")
            content.append("- Critical issues may mask other problems")
            content.append("- Provides stable foundation for further improvements")
            content.append("")

        return content

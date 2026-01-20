"""
Implementation Document Generators

Generates sections for Implementation.md including tasks, phases, testing, and rollback plans.
"""

from typing import List
from pathlib import Path

from ..audit_models import AuditResult, AuditFindingType


class TaskBreakdownGenerator:
    """Generates task breakdown section."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate task breakdown section.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for task breakdown
        """
        content = []
        content.append("## Task Breakdown")
        content.append("")

        stats = audit_result.statistics

        # Calculate task estimates
        critical_count = stats.findings_by_severity.get("critical", 0)
        high_count = stats.findings_by_severity.get("high", 0)
        medium_count = stats.findings_by_severity.get("medium", 0)
        duplicate_count = stats.findings_by_type.get("duplicate_block", 0)
        unused_count = stats.findings_by_type.get("unused_code", 0)

        content.append("### Task Summary")
        content.append("")
        content.append("| Task Category | Count | Estimated Effort | Priority |")
        content.append("|---------------|-------|------------------|----------|")

        if critical_count > 0:
            effort = f"{critical_count * 2} hours"
            content.append(f"| Critical Fixes | {critical_count} | {effort} | 🚨 Immediate |")

        if duplicate_count > 0:
            effort = f"{duplicate_count * 1} hours"
            content.append(f"| Duplicate Extraction | {duplicate_count} | {effort} | ⚠️ High |")

        if unused_count > 0:
            effort = f"{unused_count * 0.5:.1f} hours"
            content.append(f"| Unused Code Removal | {unused_count} | {effort} | 📋 Medium |")

        if high_count > 0:
            effort = f"{high_count * 1.5} hours"
            content.append(f"| Quality Improvements | {high_count} | {effort} | ⚠️ High |")

        if medium_count > 0:
            effort = f"{medium_count * 1} hours"
            content.append(f"| Minor Improvements | {medium_count} | {effort} | 💡 Low |")

        content.append("")

        # Total effort estimate
        total_hours = (
            critical_count * 2
            + duplicate_count * 1
            + unused_count * 0.5
            + high_count * 1.5
            + medium_count * 1
        )

        content.append(
            f"**Total Estimated Effort:** {total_hours:.1f} hours ({total_hours / 8:.1f} days)"
        )
        content.append("")

        return content


class PriorityPhasesGenerator:
    """Generates priority phases section."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate priority phases section.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for priority phases
        """
        content = []
        content.append("## Priority Phases")
        content.append("")

        stats = audit_result.statistics
        critical_count = stats.findings_by_severity.get("critical", 0)
        high_count = stats.findings_by_severity.get("high", 0)
        medium_count = stats.findings_by_severity.get("medium", 0)

        phase = 1

        if critical_count > 0:
            content.append(f"### Phase {phase}: Critical Issues Resolution")
            content.append("")
            content.append("**Duration:** 1-2 days")
            content.append("**Priority:** 🚨 Critical")
            content.append(f"**Tasks:** {critical_count} critical issues")
            content.append("")
            content.append("**Objectives:**")
            content.append("- Resolve all critical system issues")
            content.append("- Ensure system stability and functionality")
            content.append("- Establish foundation for further improvements")
            content.append("")
            content.append("**Success Criteria:**")
            content.append("- All critical issues resolved")
            content.append("- System passes all existing tests")
            content.append("- No regression in functionality")
            content.append("")
            phase += 1

        if high_count > 0:
            content.append(f"### Phase {phase}: High Priority Refactoring")
            content.append("")
            content.append("**Duration:** 1-2 weeks")
            content.append("**Priority:** ⚠️ High")
            content.append(f"**Tasks:** {high_count} high priority issues")
            content.append("")
            content.append("**Objectives:**")
            content.append("- Address major code quality issues")
            content.append("- Extract duplicate code blocks")
            content.append("- Remove high-confidence unused code")
            content.append("")
            content.append("**Success Criteria:**")
            content.append("- Significant reduction in code duplication")
            content.append("- Improved code maintainability metrics")
            content.append("- All tests continue to pass")
            content.append("")
            phase += 1

        if medium_count > 0:
            content.append(f"### Phase {phase}: Quality Improvements")
            content.append("")
            content.append("**Duration:** 2-4 weeks")
            content.append("**Priority:** 📋 Medium")
            content.append(f"**Tasks:** {medium_count} medium priority issues")
            content.append("")
            content.append("**Objectives:**")
            content.append("- Polish code quality and structure")
            content.append("- Address remaining duplication")
            content.append("- Improve documentation and clarity")
            content.append("")
            content.append("**Success Criteria:**")
            content.append("- Clean, well-structured codebase")
            content.append("- Comprehensive test coverage")
            content.append("- Improved developer experience")
            content.append("")

        return content


class RefactoringTasksGenerator:
    """Generates detailed refactoring tasks section."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate detailed refactoring tasks section.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for refactoring tasks
        """
        content = []
        content.append("## Detailed Refactoring Tasks")
        content.append("")

        # Group tasks by type
        duplicate_findings = audit_result.get_findings_by_type(AuditFindingType.DUPLICATE_BLOCK)
        unused_findings = audit_result.get_findings_by_type(AuditFindingType.UNUSED_CODE)
        quality_findings = audit_result.get_findings_by_type(AuditFindingType.QUALITY_ISSUE)

        if duplicate_findings:
            content.append("### Duplicate Code Extraction Tasks")
            content.append("")

            # Group by confidence level
            high_conf = [f for f in duplicate_findings if f.confidence >= 0.8]
            medium_conf = [f for f in duplicate_findings if 0.5 <= f.confidence < 0.8]

            if high_conf:
                content.append("**High Confidence Duplicates (Safe to Extract):**")
                content.append("")
                for i, finding in enumerate(high_conf, 1):
                    file_name = Path(finding.file_path).name
                    content.append(f"{i}. **{finding.title}**")
                    content.append(
                        f"   - File: `{file_name}:{finding.line_start}-{finding.line_end}`"
                    )
                    content.append(f"   - Confidence: {finding.confidence:.1%}")
                    content.append(
                        f"   - Action: {finding.recommendations[0] if finding.recommendations else 'Extract duplicate code'}"
                    )
                    content.append("   - Estimated time: 30-60 minutes")
                    content.append("")

            if medium_conf:
                content.append("**Medium Confidence Duplicates (Review Required):**")
                content.append("")
                for i, finding in enumerate(medium_conf, 1):
                    file_name = Path(finding.file_path).name
                    content.append(f"{i}. **{finding.title}**")
                    content.append(
                        f"   - File: `{file_name}:{finding.line_start}-{finding.line_end}`"
                    )
                    content.append(f"   - Confidence: {finding.confidence:.1%}")
                    content.append("   - Action: Review and potentially extract")
                    content.append("   - Estimated time: 60-90 minutes")
                    content.append("")

        if unused_findings:
            content.append("### Unused Code Removal Tasks")
            content.append("")

            # Group by confidence level
            high_conf = [f for f in unused_findings if f.confidence >= 0.8]
            medium_conf = [f for f in unused_findings if 0.5 <= f.confidence < 0.8]

            if high_conf:
                content.append("**Safe to Remove (High Confidence):**")
                content.append("")
                for i, finding in enumerate(high_conf, 1):
                    file_name = Path(finding.file_path).name
                    symbol_name = finding.metadata.get("symbol_name", "Unknown")
                    content.append(f"{i}. **{symbol_name}**")
                    content.append(f"   - File: `{file_name}:{finding.line_start}`")
                    content.append(f"   - Confidence: {finding.confidence:.1%}")
                    content.append("   - Action: Remove unused code")
                    content.append("   - Estimated time: 10-15 minutes")
                    content.append("")

            if medium_conf:
                content.append("**Review Before Removal (Medium Confidence):**")
                content.append("")
                for i, finding in enumerate(medium_conf, 1):
                    file_name = Path(finding.file_path).name
                    symbol_name = finding.metadata.get("symbol_name", "Unknown")
                    content.append(f"{i}. **{symbol_name}**")
                    content.append(f"   - File: `{file_name}:{finding.line_start}`")
                    content.append(f"   - Confidence: {finding.confidence:.1%}")
                    content.append("   - Action: Investigate usage and potentially remove")
                    content.append("   - Estimated time: 20-30 minutes")
                    content.append("")

        if quality_findings:
            content.append("### Quality Improvement Tasks")
            content.append("")

            for i, finding in enumerate(quality_findings, 1):
                file_name = Path(finding.file_path).name
                content.append(f"{i}. **{finding.title}**")
                content.append(f"   - File: `{file_name}:{finding.line_start}-{finding.line_end}`")
                content.append(f"   - Severity: {finding.severity.value.title()}")
                content.append(f"   - Description: {finding.description}")
                if finding.recommendations:
                    content.append(f"   - Action: {finding.recommendations[0]}")
                content.append("   - Estimated time: 45-90 minutes")
                content.append("")

        return content


class TestingStrategyGenerator:
    """Generates testing strategy section."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate testing strategy section.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for testing strategy
        """
        content = []
        content.append("## Testing Strategy")
        content.append("")

        content.append("### Pre-Refactoring Testing")
        content.append("")
        content.append("**Baseline Test Suite:**")
        content.append("- Run all existing unit tests")
        content.append("- Execute integration tests")
        content.append("- Perform manual smoke testing")
        content.append("- Document current test coverage")
        content.append("")

        content.append("**Test Enhancement:**")
        content.append("- Add tests for areas being refactored")
        content.append("- Increase coverage for critical components")
        content.append("- Create regression tests for known issues")
        content.append("")

        content.append("### During Refactoring Testing")
        content.append("")
        content.append("**Incremental Testing:**")
        content.append("- Run tests after each refactoring step")
        content.append("- Verify no functionality regression")
        content.append("- Test extracted methods independently")
        content.append("- Validate removed code doesn't break dependencies")
        content.append("")

        content.append("**Automated Testing:**")
        content.append("- Set up continuous integration")
        content.append("- Automated test execution on commits")
        content.append("- Code quality checks and metrics")
        content.append("")

        content.append("### Post-Refactoring Validation")
        content.append("")
        content.append("**Comprehensive Testing:**")
        content.append("- Full test suite execution")
        content.append("- Performance regression testing")
        content.append("- User acceptance testing")
        content.append("- Load testing for critical paths")
        content.append("")

        content.append("**Quality Metrics:**")
        content.append("- Code coverage analysis")
        content.append("- Cyclomatic complexity measurement")
        content.append("- Code duplication metrics")
        content.append("- Maintainability index calculation")
        content.append("")

        return content


class RollbackPlanGenerator:
    """Generates rollback plan section."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate rollback plan section.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for rollback plan
        """
        content = []
        content.append("## Rollback Plan")
        content.append("")

        content.append("### Version Control Strategy")
        content.append("")
        content.append("**Branching Strategy:**")
        content.append("- Create feature branch for refactoring work")
        content.append("- Make frequent, small commits")
        content.append("- Tag stable points for easy rollback")
        content.append("- Maintain clean commit history")
        content.append("")

        content.append("**Backup Strategy:**")
        content.append("- Full codebase backup before starting")
        content.append("- Database backups if applicable")
        content.append("- Configuration file backups")
        content.append("- Documentation of current state")
        content.append("")

        content.append("### Rollback Triggers")
        content.append("")
        content.append("**Automatic Rollback Conditions:**")
        content.append("- Test suite failure rate > 5%")
        content.append("- Performance degradation > 20%")
        content.append("- Critical functionality broken")
        content.append("- Build failures in CI/CD")
        content.append("")

        content.append("**Manual Rollback Conditions:**")
        content.append("- Unexpected behavior in production")
        content.append("- User-reported critical issues")
        content.append("- Team decision to abort refactoring")
        content.append("- Timeline constraints")
        content.append("")

        content.append("### Rollback Procedures")
        content.append("")
        content.append("**Immediate Rollback (< 1 hour):**")
        content.append("1. Revert to last known good commit")
        content.append("2. Run full test suite")
        content.append("3. Deploy to staging environment")
        content.append("4. Validate functionality")
        content.append("5. Deploy to production if needed")
        content.append("")

        content.append("**Partial Rollback:**")
        content.append("1. Identify problematic changes")
        content.append("2. Revert specific commits")
        content.append("3. Test affected functionality")
        content.append("4. Continue with remaining refactoring")
        content.append("")

        return content


class SuccessCriteriaGenerator:
    """Generates success criteria section."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate success criteria section.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for success criteria
        """
        content = []
        content.append("## Success Criteria")
        content.append("")

        stats = audit_result.statistics

        content.append("### Quantitative Success Metrics")
        content.append("")

        # Issue reduction targets
        critical_count = stats.findings_by_severity.get("critical", 0)
        high_count = stats.findings_by_severity.get("high", 0)
        duplicate_count = stats.findings_by_type.get("duplicate_block", 0)
        unused_count = stats.findings_by_type.get("unused_code", 0)

        content.append("**Issue Resolution Targets:**")
        if critical_count > 0:
            content.append(f"- ✅ Resolve 100% of critical issues ({critical_count} issues)")
        if high_count > 0:
            content.append(
                f"- ✅ Resolve 90% of high priority issues ({int(high_count * 0.9)} of {high_count} issues)"
            )
        if duplicate_count > 0:
            content.append(
                f"- ✅ Reduce code duplication by 80% ({int(duplicate_count * 0.8)} of {duplicate_count} blocks)"
            )
        if unused_count > 0:
            content.append(
                f"- ✅ Remove 70% of unused code ({int(unused_count * 0.7)} of {unused_count} elements)"
            )
        content.append("")

        content.append("**Code Quality Metrics:**")
        content.append("- ✅ Maintain or improve test coverage (target: >80%)")
        content.append("- ✅ Reduce average cyclomatic complexity by 15%")
        content.append("- ✅ Improve maintainability index by 20%")
        content.append("- ✅ Zero increase in technical debt")
        content.append("")

        content.append("**Performance Metrics:**")
        content.append("- ✅ No performance regression (< 5% slowdown)")
        content.append("- ✅ Maintain or improve memory usage")
        content.append("- ✅ Build time improvement (target: 10% faster)")
        content.append("")

        content.append("### Qualitative Success Criteria")
        content.append("")
        content.append("**Code Quality:**")
        content.append("- ✅ Improved code readability and clarity")
        content.append("- ✅ Better separation of concerns")
        content.append("- ✅ Reduced code complexity")
        content.append("- ✅ Enhanced maintainability")
        content.append("")

        content.append("**Developer Experience:**")
        content.append("- ✅ Easier to understand codebase")
        content.append("- ✅ Faster development cycles")
        content.append("- ✅ Reduced debugging time")
        content.append("- ✅ Improved team confidence in code changes")
        content.append("")

        content.append("**System Reliability:**")
        content.append("- ✅ No functional regressions")
        content.append("- ✅ Stable system behavior")
        content.append("- ✅ Maintained backward compatibility")
        content.append("- ✅ Improved error handling")
        content.append("")

        content.append("### Acceptance Criteria")
        content.append("")
        content.append("**Technical Acceptance:**")
        content.append("- All automated tests pass")
        content.append("- Code review approval from team leads")
        content.append("- Performance benchmarks meet targets")
        content.append("- Security scan shows no new vulnerabilities")
        content.append("")

        content.append("**Business Acceptance:**")
        content.append("- All user-facing functionality works correctly")
        content.append("- No customer-reported issues")
        content.append("- Stakeholder approval of changes")
        content.append("- Documentation updated and approved")
        content.append("")

        return content

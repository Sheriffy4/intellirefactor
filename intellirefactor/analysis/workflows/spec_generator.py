"""
Specification Generator for IntelliRefactor

This module generates specification documents from audit results,
including Requirements.md, Design.md, and Implementation.md with
findings, recommendations, and task breakdowns.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path

from .audit_models import AuditResult, AuditFinding, AuditFindingType
from intellirefactor.analysis.foundation.models import Severity
from .spec.sections import (
    RequirementsHeaderGenerator,
    DesignHeaderGenerator,
    ImplementationHeaderGenerator,
)
from .spec.statistics import (
    ExecutiveSummaryGenerator,
    StatisticsGenerator,
)
from .spec.findings import (
    CriticalFindingsGenerator,
    HighPriorityGenerator,
    DuplicateCodeGenerator,
    UnusedCodeGenerator,
    QualityPerformanceGenerator,
)
from .spec.design import (
    ArchitectureOverviewGenerator,
    ComponentAnalysisGenerator,
    RefactoringStrategyGenerator,
    DependencyAnalysisGenerator,
    RiskAssessmentGenerator,
    DesignDecisionsGenerator,
)
from .spec.implementation import (
    TaskBreakdownGenerator,
    PriorityPhasesGenerator,
    RefactoringTasksGenerator,
    TestingStrategyGenerator,
    RollbackPlanGenerator,
    SuccessCriteriaGenerator,
)
from .spec.extractors import (
    RefactoringPriorityExtractor,
    CleanupTaskExtractor,
)
from .spec.recommendations import (
    ImplementationRecommendationsGenerator,
)
from .spec.appendix import (
    RequirementsAppendixGenerator,
    DesignAppendixGenerator,
    ImplementationAppendixGenerator,
)


class SpecTemplate:
    """Template for specification generation."""

    def __init__(self, name: str, sections: List[str], format_type: str = "markdown"):
        self.name = name
        self.sections = sections
        self.format_type = format_type


class SpecGenerator:
    """
    Generates specification documents from audit results.

    Creates Requirements.md, Design.md, and Implementation.md documents
    with findings, evidence, and actionable recommendations for refactoring work.
    Supports customizable templates and multiple output formats.
    """

    def __init__(self, templates: Optional[Dict[str, SpecTemplate]] = None):
        """
        Initialize the specification generator.

        Args:
            templates: Custom templates for specification generation
        """
        self.templates = templates or self._get_default_templates()

    def _get_default_templates(self) -> Dict[str, SpecTemplate]:
        """Get default specification templates."""
        return {
            "requirements": SpecTemplate(
                name="Requirements",
                sections=[
                    "header",
                    "executive_summary",
                    "statistics",
                    "critical_findings",
                    "high_priority",
                    "duplicate_code",
                    "unused_code",
                    "quality_performance",
                    "implementation_recommendations",
                    "appendix",
                ],
            ),
            "design": SpecTemplate(
                name="Design",
                sections=[
                    "header",
                    "architecture_overview",
                    "component_analysis",
                    "refactoring_strategy",
                    "dependency_analysis",
                    "risk_assessment",
                    "design_decisions",
                    "appendix",
                ],
            ),
            "implementation": SpecTemplate(
                name="Implementation",
                sections=[
                    "header",
                    "task_breakdown",
                    "priority_phases",
                    "refactoring_tasks",
                    "testing_strategy",
                    "rollback_plan",
                    "success_criteria",
                    "appendix",
                ],
            ),
        }

    def generate_specification(
        self,
        spec_type: str,
        audit_result: AuditResult,
        custom_template: Optional[SpecTemplate] = None,
    ) -> str:
        """
        Generate a specification document of the given type.

        Args:
            spec_type: Type of specification ('requirements', 'design', 'implementation')
            audit_result: Complete audit results
            custom_template: Optional custom template to use

        Returns:
            Specification content as string

        Raises:
            ValueError: If spec_type is not supported
        """
        if spec_type == "requirements":
            return self.generate_requirements_from_audit(audit_result, custom_template)
        elif spec_type == "design":
            return self.generate_design_from_audit(audit_result, custom_template)
        elif spec_type == "implementation":
            return self.generate_implementation_from_audit(audit_result, custom_template)
        else:
            raise ValueError(f"Unsupported specification type: {spec_type}")

    def generate_all_specifications(self, audit_result: AuditResult) -> Dict[str, str]:
        """
        Generate all specification documents.

        Args:
            audit_result: Complete audit results

        Returns:
            Dictionary mapping spec type to content
        """
        return {
            "requirements": self.generate_requirements_from_audit(audit_result),
            "design": self.generate_design_from_audit(audit_result),
            "implementation": self.generate_implementation_from_audit(audit_result),
        }

    def generate_machine_readable_analysis(self, audit_result: AuditResult) -> Dict[str, Any]:
        """
        Generate machine-readable analysis artifacts.

        Args:
            audit_result: Complete audit results

        Returns:
            Dictionary with structured analysis data
        """
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "project_path": str(audit_result.project_path),
                "analysis_time": audit_result.statistics.analysis_time_seconds,
                "tool_version": "intellirefactor-0.1.0",
            },
            "statistics": {
                "total_findings": audit_result.statistics.total_findings,
                "files_analyzed": audit_result.statistics.files_analyzed,
                "findings_by_severity": audit_result.statistics.findings_by_severity,
                "findings_by_type": audit_result.statistics.findings_by_type,
                "confidence_distribution": audit_result.statistics.confidence_distribution,
            },
            "findings": [
                {
                    "id": finding.finding_id,
                    "type": finding.finding_type.value,
                    "severity": finding.severity.value,
                    "title": finding.title,
                    "description": finding.description,
                    "file_path": finding.file_path,
                    "line_start": finding.line_start,
                    "line_end": finding.line_end,
                    "confidence": finding.confidence,
                    "evidence": {
                        "description": finding.evidence.description,
                        "code_snippets": finding.evidence.code_snippets,
                        "confidence": finding.evidence.confidence,
                        "metadata": finding.evidence.metadata,
                    },
                    "recommendations": finding.recommendations,
                    "metadata": finding.metadata,
                }
                for finding in audit_result.findings
            ],
            "clone_groups": (
                [
                    {
                        "group_id": group.group_id,
                        "clone_type": group.clone_type.value,
                        "similarity_score": group.similarity_score,
                        "instances": [
                            {
                                "file_path": instance.file_path,
                                "line_start": instance.line_start,
                                "line_end": instance.line_end,
                                "fingerprint": instance.block_info.token_fingerprint,
                            }
                            for instance in group.instances
                        ],
                        "extraction_strategy": group.extraction_strategy,
                    }
                    for group in audit_result.clone_groups
                ]
                if audit_result.clone_groups
                else []
            ),
            "recommendations": {
                "critical_actions": [
                    finding.recommendations[0] if finding.recommendations else finding.title
                    for finding in audit_result.get_critical_findings()
                ],
                "refactoring_priorities": RefactoringPriorityExtractor.extract(audit_result),
                "cleanup_tasks": CleanupTaskExtractor.extract(audit_result),
            },
        }

    def generate_requirements_from_audit(
        self, audit_result: AuditResult, custom_template: Optional[SpecTemplate] = None
    ) -> str:
        """
        Generate a Requirements.md document from audit results.

        Args:
            audit_result: Complete audit results
            custom_template: Optional custom template to use

        Returns:
            Markdown content for Requirements.md
        """
        template = custom_template or self.templates["requirements"]
        content = []

        for section in template.sections:
            if section == "header":
                content.extend(RequirementsHeaderGenerator.generate(audit_result))
            elif section == "executive_summary":
                content.extend(ExecutiveSummaryGenerator.generate(audit_result))
            elif section == "statistics":
                content.extend(StatisticsGenerator.generate(audit_result))
            elif section == "critical_findings":
                critical_findings = audit_result.get_critical_findings()
                if critical_findings:
                    content.extend(CriticalFindingsGenerator.generate(critical_findings))
            elif section == "high_priority":
                high_priority = audit_result.get_findings_by_severity(Severity.HIGH)
                if high_priority:
                    content.extend(HighPriorityGenerator.generate(high_priority))
            elif section == "duplicate_code":
                duplicate_findings = audit_result.get_findings_by_type(
                    AuditFindingType.DUPLICATE_BLOCK
                )
                if duplicate_findings:
                    content.extend(DuplicateCodeGenerator.generate(duplicate_findings))
            elif section == "unused_code":
                unused_findings = audit_result.get_findings_by_type(AuditFindingType.UNUSED_CODE)
                if unused_findings:
                    content.extend(UnusedCodeGenerator.generate(unused_findings))
            elif section == "quality_performance":
                quality_findings = audit_result.get_findings_by_type(AuditFindingType.QUALITY_ISSUE)
                performance_findings = audit_result.get_findings_by_type(
                    AuditFindingType.PERFORMANCE_ISSUE
                )
                if quality_findings or performance_findings:
                    content.extend(
                        QualityPerformanceGenerator.generate(quality_findings, performance_findings)
                    )
            elif section == "implementation_recommendations":
                content.extend(ImplementationRecommendationsGenerator.generate(audit_result))
            elif section == "appendix":
                content.extend(RequirementsAppendixGenerator.generate(audit_result))

        return "\n".join(content)

    def generate_design_from_audit(
        self, audit_result: AuditResult, custom_template: Optional[SpecTemplate] = None
    ) -> str:
        """
        Generate a Design.md document from audit results.

        Args:
            audit_result: Complete audit results
            custom_template: Optional custom template to use

        Returns:
            Markdown content for Design.md
        """
        template = custom_template or self.templates["design"]
        content = []

        for section in template.sections:
            if section == "header":
                content.extend(DesignHeaderGenerator.generate(audit_result))
            elif section == "architecture_overview":
                content.extend(ArchitectureOverviewGenerator.generate(audit_result))
            elif section == "component_analysis":
                content.extend(ComponentAnalysisGenerator.generate(audit_result))
            elif section == "refactoring_strategy":
                content.extend(RefactoringStrategyGenerator.generate(audit_result))
            elif section == "dependency_analysis":
                content.extend(DependencyAnalysisGenerator.generate(audit_result))
            elif section == "risk_assessment":
                content.extend(RiskAssessmentGenerator.generate(audit_result))
            elif section == "design_decisions":
                content.extend(DesignDecisionsGenerator.generate(audit_result))
            elif section == "appendix":
                content.extend(DesignAppendixGenerator.generate(audit_result))

        return "\n".join(content)

    def generate_implementation_from_audit(
        self, audit_result: AuditResult, custom_template: Optional[SpecTemplate] = None
    ) -> str:
        """
        Generate an Implementation.md document from audit results.

        Args:
            audit_result: Complete audit results
            custom_template: Optional custom template to use

        Returns:
            Markdown content for Implementation.md
        """
        template = custom_template or self.templates["implementation"]
        content = []

        for section in template.sections:
            if section == "header":
                content.extend(ImplementationHeaderGenerator.generate(audit_result))
            elif section == "task_breakdown":
                content.extend(TaskBreakdownGenerator.generate(audit_result))
            elif section == "priority_phases":
                content.extend(PriorityPhasesGenerator.generate(audit_result))
            elif section == "refactoring_tasks":
                content.extend(RefactoringTasksGenerator.generate(audit_result))
            elif section == "testing_strategy":
                content.extend(TestingStrategyGenerator.generate(audit_result))
            elif section == "rollback_plan":
                content.extend(RollbackPlanGenerator.generate(audit_result))
            elif section == "success_criteria":
                content.extend(SuccessCriteriaGenerator.generate(audit_result))
            elif section == "appendix":
                content.extend(ImplementationAppendixGenerator.generate(audit_result))

        return "\n".join(content)

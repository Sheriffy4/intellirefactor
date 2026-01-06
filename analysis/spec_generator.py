"""
Specification Generator for IntelliRefactor

This module generates specification documents from audit results,
including Requirements.md, Design.md, and Implementation.md with
findings, recommendations, and task breakdowns.
"""

from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import json

from .audit_models import AuditResult, AuditFinding, AuditSeverity, AuditFindingType


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
            'requirements': SpecTemplate(
                name='Requirements',
                sections=[
                    'header', 'executive_summary', 'statistics', 
                    'critical_findings', 'high_priority', 'duplicate_code',
                    'unused_code', 'quality_performance', 'implementation_recommendations',
                    'appendix'
                ]
            ),
            'design': SpecTemplate(
                name='Design',
                sections=[
                    'header', 'architecture_overview', 'component_analysis',
                    'refactoring_strategy', 'dependency_analysis', 'risk_assessment',
                    'design_decisions', 'appendix'
                ]
            ),
            'implementation': SpecTemplate(
                name='Implementation',
                sections=[
                    'header', 'task_breakdown', 'priority_phases', 'refactoring_tasks',
                    'testing_strategy', 'rollback_plan', 'success_criteria', 'appendix'
                ]
            )
        }
    
    def generate_specification(self, spec_type: str, audit_result: AuditResult, 
                             custom_template: Optional[SpecTemplate] = None) -> str:
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
        if spec_type == 'requirements':
            return self.generate_requirements_from_audit(audit_result, custom_template)
        elif spec_type == 'design':
            return self.generate_design_from_audit(audit_result, custom_template)
        elif spec_type == 'implementation':
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
            'requirements': self.generate_requirements_from_audit(audit_result),
            'design': self.generate_design_from_audit(audit_result),
            'implementation': self.generate_implementation_from_audit(audit_result)
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
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'project_path': str(audit_result.project_path),
                'analysis_time': audit_result.statistics.analysis_time_seconds,
                'tool_version': 'intellirefactor-1.0.0'
            },
            'statistics': {
                'total_findings': audit_result.statistics.total_findings,
                'files_analyzed': audit_result.statistics.files_analyzed,
                'findings_by_severity': audit_result.statistics.findings_by_severity,
                'findings_by_type': audit_result.statistics.findings_by_type,
                'confidence_distribution': audit_result.statistics.confidence_distribution
            },
            'findings': [
                {
                    'id': finding.finding_id,
                    'type': finding.finding_type.value,
                    'severity': finding.severity.value,
                    'title': finding.title,
                    'description': finding.description,
                    'file_path': finding.file_path,
                    'line_start': finding.line_start,
                    'line_end': finding.line_end,
                    'confidence': finding.confidence,
                    'evidence': {
                        'description': finding.evidence.description,
                        'code_snippets': finding.evidence.code_snippets,
                        'confidence': finding.evidence.confidence,
                        'metadata': finding.evidence.metadata
                    },
                    'recommendations': finding.recommendations,
                    'metadata': finding.metadata
                }
                for finding in audit_result.findings
            ],
            'clone_groups': [
                {
                    'group_id': group.group_id,
                    'clone_type': group.clone_type.value,
                    'similarity_score': group.similarity_score,
                    'instances': [
                        {
                            'file_path': instance.file_path,
                            'line_start': instance.line_start,
                            'line_end': instance.line_end,
                            'fingerprint': instance.block_info.token_fingerprint
                        }
                        for instance in group.instances
                    ],
                    'extraction_strategy': group.extraction_strategy
                }
                for group in audit_result.clone_groups
            ] if audit_result.clone_groups else [],
            'recommendations': {
                'critical_actions': [
                    finding.recommendations[0] if finding.recommendations else finding.title
                    for finding in audit_result.get_critical_findings()
                ],
                'refactoring_priorities': self._extract_refactoring_priorities(audit_result),
                'cleanup_tasks': self._extract_cleanup_tasks(audit_result)
            }
        }
    
    def _extract_refactoring_priorities(self, audit_result: AuditResult) -> List[Dict[str, Any]]:
        """Extract refactoring priorities from audit results."""
        priorities = []
        
        # Group findings by type and severity
        duplicate_findings = audit_result.get_findings_by_type(AuditFindingType.DUPLICATE_BLOCK)
        unused_findings = audit_result.get_findings_by_type(AuditFindingType.UNUSED_CODE)
        quality_findings = audit_result.get_findings_by_type(AuditFindingType.QUALITY_ISSUE)
        
        if duplicate_findings:
            high_conf_duplicates = [f for f in duplicate_findings if f.confidence >= 0.8]
            priorities.append({
                'type': 'duplicate_code_refactoring',
                'priority': 'high' if len(high_conf_duplicates) > 5 else 'medium',
                'count': len(duplicate_findings),
                'high_confidence_count': len(high_conf_duplicates),
                'estimated_effort': 'medium',
                'description': 'Extract duplicate code blocks into reusable functions'
            })
        
        if unused_findings:
            safe_to_remove = [f for f in unused_findings if f.confidence >= 0.8]
            priorities.append({
                'type': 'unused_code_cleanup',
                'priority': 'medium',
                'count': len(unused_findings),
                'safe_to_remove_count': len(safe_to_remove),
                'estimated_effort': 'low',
                'description': 'Remove unused code to improve maintainability'
            })
        
        if quality_findings:
            critical_quality = [f for f in quality_findings if f.severity == AuditSeverity.CRITICAL]
            priorities.append({
                'type': 'quality_improvements',
                'priority': 'high' if critical_quality else 'medium',
                'count': len(quality_findings),
                'critical_count': len(critical_quality),
                'estimated_effort': 'high',
                'description': 'Address code quality and architectural issues'
            })
        
        return priorities
    
    def _extract_cleanup_tasks(self, audit_result: AuditResult) -> List[Dict[str, Any]]:
        """Extract cleanup tasks from audit results."""
        tasks = []
        
        unused_findings = audit_result.get_findings_by_type(AuditFindingType.UNUSED_CODE)
        if unused_findings:
            # Group by confidence level
            high_conf = [f for f in unused_findings if f.confidence >= 0.8]
            medium_conf = [f for f in unused_findings if 0.5 <= f.confidence < 0.8]
            
            if high_conf:
                tasks.append({
                    'type': 'safe_unused_removal',
                    'count': len(high_conf),
                    'confidence': 'high',
                    'estimated_time': f"{len(high_conf) * 5} minutes",
                    'description': 'Remove high-confidence unused code'
                })
            
            if medium_conf:
                tasks.append({
                    'type': 'review_unused_code',
                    'count': len(medium_conf),
                    'confidence': 'medium',
                    'estimated_time': f"{len(medium_conf) * 15} minutes",
                    'description': 'Review and potentially remove medium-confidence unused code'
                })
        
        return tasks
    
    def generate_requirements_from_audit(self, audit_result: AuditResult, 
                                        custom_template: Optional[SpecTemplate] = None) -> str:
        """
        Generate a Requirements.md document from audit results.
        
        Args:
            audit_result: Complete audit results
            custom_template: Optional custom template to use
            
        Returns:
            Markdown content for Requirements.md
        """
        template = custom_template or self.templates['requirements']
        content = []
        
        for section in template.sections:
            if section == 'header':
                content.extend(self._generate_requirements_header(audit_result))
            elif section == 'executive_summary':
                content.extend(self._generate_executive_summary(audit_result))
            elif section == 'statistics':
                content.extend(self._generate_statistics_section(audit_result))
            elif section == 'critical_findings':
                critical_findings = audit_result.get_critical_findings()
                if critical_findings:
                    content.extend(self._generate_critical_findings_section(critical_findings))
            elif section == 'high_priority':
                high_priority = audit_result.get_findings_by_severity(AuditSeverity.HIGH)
                if high_priority:
                    content.extend(self._generate_high_priority_section(high_priority))
            elif section == 'duplicate_code':
                duplicate_findings = audit_result.get_findings_by_type(AuditFindingType.DUPLICATE_BLOCK)
                if duplicate_findings:
                    content.extend(self._generate_duplicate_code_section(duplicate_findings))
            elif section == 'unused_code':
                unused_findings = audit_result.get_findings_by_type(AuditFindingType.UNUSED_CODE)
                if unused_findings:
                    content.extend(self._generate_unused_code_section(unused_findings))
            elif section == 'quality_performance':
                quality_findings = audit_result.get_findings_by_type(AuditFindingType.QUALITY_ISSUE)
                performance_findings = audit_result.get_findings_by_type(AuditFindingType.PERFORMANCE_ISSUE)
                if quality_findings or performance_findings:
                    content.extend(self._generate_quality_performance_section(quality_findings, performance_findings))
            elif section == 'implementation_recommendations':
                content.extend(self._generate_implementation_recommendations(audit_result))
            elif section == 'appendix':
                content.extend(self._generate_appendix(audit_result))
        
        return '\n'.join(content)
    
    def generate_design_from_audit(self, audit_result: AuditResult,
                                  custom_template: Optional[SpecTemplate] = None) -> str:
        """
        Generate a Design.md document from audit results.
        
        Args:
            audit_result: Complete audit results
            custom_template: Optional custom template to use
            
        Returns:
            Markdown content for Design.md
        """
        template = custom_template or self.templates['design']
        content = []
        
        for section in template.sections:
            if section == 'header':
                content.extend(self._generate_design_header(audit_result))
            elif section == 'architecture_overview':
                content.extend(self._generate_architecture_overview(audit_result))
            elif section == 'component_analysis':
                content.extend(self._generate_component_analysis(audit_result))
            elif section == 'refactoring_strategy':
                content.extend(self._generate_refactoring_strategy(audit_result))
            elif section == 'dependency_analysis':
                content.extend(self._generate_dependency_analysis(audit_result))
            elif section == 'risk_assessment':
                content.extend(self._generate_risk_assessment(audit_result))
            elif section == 'design_decisions':
                content.extend(self._generate_design_decisions(audit_result))
            elif section == 'appendix':
                content.extend(self._generate_design_appendix(audit_result))
        
        return '\n'.join(content)
    
    def generate_implementation_from_audit(self, audit_result: AuditResult,
                                         custom_template: Optional[SpecTemplate] = None) -> str:
        """
        Generate an Implementation.md document from audit results.
        
        Args:
            audit_result: Complete audit results
            custom_template: Optional custom template to use
            
        Returns:
            Markdown content for Implementation.md
        """
        template = custom_template or self.templates['implementation']
        content = []
        
        for section in template.sections:
            if section == 'header':
                content.extend(self._generate_implementation_header(audit_result))
            elif section == 'task_breakdown':
                content.extend(self._generate_task_breakdown(audit_result))
            elif section == 'priority_phases':
                content.extend(self._generate_priority_phases(audit_result))
            elif section == 'refactoring_tasks':
                content.extend(self._generate_refactoring_tasks(audit_result))
            elif section == 'testing_strategy':
                content.extend(self._generate_testing_strategy(audit_result))
            elif section == 'rollback_plan':
                content.extend(self._generate_rollback_plan(audit_result))
            elif section == 'success_criteria':
                content.extend(self._generate_success_criteria(audit_result))
            elif section == 'appendix':
                content.extend(self._generate_implementation_appendix(audit_result))
        
        return '\n'.join(content)
    
    def _generate_requirements_header(self, audit_result: AuditResult) -> List[str]:
        """Generate header for Requirements.md."""
        content = []
        content.append("# Project Refactoring Requirements")
        content.append("")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**Project:** {audit_result.project_path}")
        content.append(f"**Analysis Time:** {audit_result.statistics.analysis_time_seconds:.2f} seconds")
        content.append("")
        return content
    
    def _generate_design_header(self, audit_result: AuditResult) -> List[str]:
        """Generate header for Design.md."""
        content = []
        content.append("# Refactoring Design Document")
        content.append("")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**Project:** {audit_result.project_path}")
        content.append(f"**Total Findings:** {audit_result.statistics.total_findings}")
        content.append("")
        content.append("## Overview")
        content.append("")
        content.append("This document outlines the architectural design for refactoring the analyzed codebase.")
        content.append("It provides component analysis, refactoring strategies, and design decisions based on")
        content.append("the findings from the automated code analysis.")
        content.append("")
        return content
    
    def _generate_implementation_header(self, audit_result: AuditResult) -> List[str]:
        """Generate header for Implementation.md."""
        content = []
        content.append("# Refactoring Implementation Plan")
        content.append("")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**Project:** {audit_result.project_path}")
        content.append(f"**Total Tasks:** {audit_result.statistics.total_findings}")
        content.append("")
        content.append("## Overview")
        content.append("")
        content.append("This document provides a detailed implementation plan for refactoring the analyzed codebase.")
        content.append("It includes task breakdowns, priority phases, testing strategies, and success criteria.")
        content.append("")
        return content
    
    def _generate_executive_summary(self, audit_result: AuditResult) -> List[str]:
        """Generate executive summary section."""
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
            content.append(f"📊 **Analysis Results:** Found {stats.total_findings} issues across {stats.files_analyzed} files.")
            content.append("")
            
            # Priority breakdown
            critical_count = stats.findings_by_severity.get('critical', 0)
            high_count = stats.findings_by_severity.get('high', 0)
            medium_count = stats.findings_by_severity.get('medium', 0)
            low_count = stats.findings_by_severity.get('low', 0)
            
            if critical_count > 0:
                content.append(f"🚨 **{critical_count} Critical Issues** require immediate attention")
            if high_count > 0:
                content.append(f"⚠️ **{high_count} High Priority Issues** should be addressed soon")
            if medium_count > 0:
                content.append(f"📋 **{medium_count} Medium Priority Issues** for future improvement")
            if low_count > 0:
                content.append(f"💡 **{low_count} Low Priority Issues** for consideration")
            
            content.append("")
            
            # Key areas for improvement
            content.append("**Key Areas for Improvement:**")
            
            duplicate_count = stats.findings_by_type.get('duplicate_block', 0)
            unused_count = stats.findings_by_type.get('unused_code', 0)
            quality_count = stats.findings_by_type.get('quality_issue', 0)
            
            if duplicate_count > 0:
                content.append(f"- **Code Duplication:** {duplicate_count} duplicate code blocks found")
            if unused_count > 0:
                content.append(f"- **Unused Code:** {unused_count} unused code elements identified")
            if quality_count > 0:
                content.append(f"- **Code Quality:** {quality_count} quality issues detected")
            
            content.append("")
        
        return content
    
    def _generate_statistics_section(self, audit_result: AuditResult) -> List[str]:
        """Generate statistics section."""
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
        for severity in ['critical', 'high', 'medium', 'low', 'info']:
            count = stats.findings_by_severity.get(severity, 0)
            if count > 0:
                emoji = {'critical': '🚨', 'high': '⚠️', 'medium': '📋', 'low': '💡', 'info': 'ℹ️'}[severity]
                content.append(f"- {emoji} **{severity.title()}:** {count} findings")
        content.append("")
        
        # Type distribution
        content.append("### Findings by Type")
        content.append("")
        for finding_type, count in stats.findings_by_type.items():
            if count > 0:
                type_name = finding_type.replace('_', ' ').title()
                content.append(f"- **{type_name}:** {count} findings")
        content.append("")
        
        # Confidence distribution
        content.append("### Confidence Distribution")
        content.append("")
        high_conf = stats.confidence_distribution.get('high', 0)
        medium_conf = stats.confidence_distribution.get('medium', 0)
        low_conf = stats.confidence_distribution.get('low', 0)
        
        content.append(f"- **High Confidence (≥80%):** {high_conf} findings")
        content.append(f"- **Medium Confidence (50-79%):** {medium_conf} findings")
        content.append(f"- **Low Confidence (<50%):** {low_conf} findings")
        content.append("")
        
        return content
    
    def _generate_critical_findings_section(self, critical_findings: List[AuditFinding]) -> List[str]:
        """Generate critical findings section."""
        content = []
        content.append("## 🚨 Critical Issues")
        content.append("")
        content.append("These issues require immediate attention as they may impact system stability or functionality.")
        content.append("")
        
        for i, finding in enumerate(critical_findings, 1):
            content.append(f"### {i}. {finding.title}")
            content.append("")
            content.append(f"**File:** `{finding.file_path}:{finding.line_start}-{finding.line_end}`")
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
    
    def _generate_high_priority_section(self, high_priority: List[AuditFinding]) -> List[str]:
        """Generate high priority findings section."""
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
            type_name = finding_type.replace('_', ' ').title()
            content.append(f"### {type_name} ({len(findings)} issues)")
            content.append("")
            
            for finding in findings:
                content.append(f"- **{finding.title}** in `{Path(finding.file_path).name}:{finding.line_start}`")
                content.append(f"  - {finding.description}")
                content.append(f"  - Confidence: {finding.confidence:.1%}")
                if finding.recommendations:
                    content.append(f"  - Action: {finding.recommendations[0]}")
                content.append("")
        
        return content
    
    def _generate_duplicate_code_section(self, duplicate_findings: List[AuditFinding]) -> List[str]:
        """Generate duplicate code section."""
        content = []
        content.append("## 🔄 Code Duplication Analysis")
        content.append("")
        
        if not duplicate_findings:
            content.append("✅ No duplicate code blocks detected.")
            content.append("")
            return content
        
        content.append(f"Found {len(duplicate_findings)} duplicate code blocks that should be refactored.")
        content.append("")
        
        # Group by clone type
        by_clone_type = {}
        for finding in duplicate_findings:
            clone_type = finding.metadata.get('clone_type', 'unknown')
            if clone_type not in by_clone_type:
                by_clone_type[clone_type] = []
            by_clone_type[clone_type].append(finding)
        
        for clone_type, findings in by_clone_type.items():
            content.append(f"### {clone_type.title()} Clones ({len(findings)} groups)")
            content.append("")
            
            for finding in findings:
                instances = finding.evidence.metadata.get('all_instances', [])
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
        content.append("2. **Extract common methods** - Create shared functions for duplicate logic")
        content.append("3. **Use parameters** - Make extracted methods flexible with parameters")
        content.append("4. **Add tests** - Ensure behavior is preserved during refactoring")
        content.append("5. **Review structural clones** - May require more complex refactoring")
        content.append("")
        
        return content
    
    def _generate_unused_code_section(self, unused_findings: List[AuditFinding]) -> List[str]:
        """Generate unused code section."""
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
            unused_type = finding.metadata.get('unused_type', 'unknown')
            if unused_type not in by_unused_type:
                by_unused_type[unused_type] = []
            by_unused_type[unused_type].append(finding)
        
        for unused_type, findings in by_unused_type.items():
            type_name = unused_type.replace('_', ' ').title()
            content.append(f"### {type_name} ({len(findings)} items)")
            content.append("")
            
            # Group by confidence for better prioritization
            high_conf = [f for f in findings if f.confidence >= 0.8]
            medium_conf = [f for f in findings if 0.5 <= f.confidence < 0.8]
            low_conf = [f for f in findings if f.confidence < 0.5]
            
            if high_conf:
                content.append("**High Confidence (Safe to Remove):**")
                for finding in high_conf:
                    content.append(f"- `{finding.metadata.get('symbol_name', 'Unknown')}` in `{Path(finding.file_path).name}:{finding.line_start}`")
                content.append("")
            
            if medium_conf:
                content.append("**Medium Confidence (Review Before Removing):**")
                for finding in medium_conf:
                    content.append(f"- `{finding.metadata.get('symbol_name', 'Unknown')}` in `{Path(finding.file_path).name}:{finding.line_start}`")
                content.append("")
            
            if low_conf:
                content.append("**Low Confidence (Investigate Further):**")
                for finding in low_conf:
                    content.append(f"- `{finding.metadata.get('symbol_name', 'Unknown')}` in `{Path(finding.file_path).name}:{finding.line_start}`")
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
    
    def _generate_quality_performance_section(self, quality_findings: List[AuditFinding], 
                                            performance_findings: List[AuditFinding]) -> List[str]:
        """Generate quality and performance issues section."""
        content = []
        content.append("## 🔧 Quality & Performance Issues")
        content.append("")
        
        if quality_findings:
            content.append(f"### Quality Issues ({len(quality_findings)} items)")
            content.append("")
            for finding in quality_findings:
                content.append(f"- **{finding.title}**")
                content.append(f"  - {finding.description}")
                content.append(f"  - Location: `{Path(finding.file_path).name}:{finding.line_start}`")
                if finding.recommendations:
                    content.append(f"  - Action: {finding.recommendations[0]}")
                content.append("")
        
        if performance_findings:
            content.append(f"### Performance Issues ({len(performance_findings)} items)")
            content.append("")
            for finding in performance_findings:
                content.append(f"- **{finding.title}**")
                content.append(f"  - {finding.description}")
                content.append(f"  - Location: `{Path(finding.file_path).name}:{finding.line_start}`")
                if finding.recommendations:
                    content.append(f"  - Action: {finding.recommendations[0]}")
                content.append("")
        
        return content
    
    def _generate_implementation_recommendations(self, audit_result: AuditResult) -> List[str]:
        """Generate implementation recommendations section."""
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
        critical_count = stats.findings_by_severity.get('critical', 0)
        high_count = stats.findings_by_severity.get('high', 0)
        medium_count = stats.findings_by_severity.get('medium', 0)
        
        phase = 1
        
        if critical_count > 0:
            content.append(f"### Phase {phase}: Critical Issues (Immediate)")
            content.append("")
            content.append(f"**Timeline:** 1-2 days")
            content.append(f"**Priority:** 🚨 Critical")
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
            content.append(f"**Timeline:** 1-2 weeks")
            content.append(f"**Priority:** ⚠️ High")
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
            content.append(f"**Timeline:** 1-2 months")
            content.append(f"**Priority:** 📋 Medium")
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
    
    def _generate_appendix(self, audit_result: AuditResult) -> List[str]:
        """Generate appendix with technical details."""
        content = []
        content.append("## Appendix")
        content.append("")
        
        # Analysis configuration
        content.append("### Analysis Configuration")
        content.append("")
        metadata = audit_result.analysis_metadata
        content.append("| Setting | Value |")
        content.append("|---------|-------|")
        content.append(f"| Include Index | {metadata.get('include_index', 'N/A')} |")
        content.append(f"| Include Duplicates | {metadata.get('include_duplicates', 'N/A')} |")
        content.append(f"| Include Unused | {metadata.get('include_unused', 'N/A')} |")
        content.append(f"| Min Confidence | {metadata.get('min_confidence', 'N/A')} |")
        content.append(f"| Incremental Index | {metadata.get('incremental_index', 'N/A')} |")
        content.append("")
        
        # Index statistics
        if audit_result.index_result:
            content.append("### Index Statistics")
            content.append("")
            idx = audit_result.index_result
            content.append("| Metric | Value |")
            content.append("|--------|-------|")
            content.append(f"| Files Processed | {idx.files_processed} |")
            content.append(f"| Files Skipped | {idx.files_skipped} |")
            content.append(f"| Symbols Found | {idx.symbols_found} |")
            content.append(f"| Blocks Found | {idx.blocks_found} |")
            content.append(f"| Dependencies Found | {idx.dependencies_found} |")
            content.append(f"| Build Time | {idx.build_time_seconds:.2f}s |")
            content.append(f"| Incremental | {idx.incremental} |")
            content.append("")
        
        # Unused code statistics
        if audit_result.unused_result:
            content.append("### Unused Code Statistics")
            content.append("")
            unused_stats = audit_result.unused_result.statistics
            content.append("| Metric | Value |")
            content.append("|--------|-------|")
            content.append(f"| Total Findings | {unused_stats.get('total_findings', 0)} |")
            content.append(f"| Files Analyzed | {unused_stats.get('total_files_analyzed', 0)} |")
            content.append(f"| Entry Points | {len(audit_result.unused_result.entry_points)} |")
            content.append("")
            
            # Entry points
            if audit_result.unused_result.entry_points:
                content.append("**Entry Points:**")
                for entry_point in audit_result.unused_result.entry_points:
                    content.append(f"- `{entry_point}`")
                content.append("")
        
        # Clone detection statistics
        if audit_result.clone_groups:
            content.append("### Clone Detection Statistics")
            content.append("")
            content.append("| Metric | Value |")
            content.append("|--------|-------|")
            content.append(f"| Clone Groups | {len(audit_result.clone_groups)} |")
            
            total_instances = sum(len(group.instances) for group in audit_result.clone_groups)
            content.append(f"| Total Instances | {total_instances} |")
            
            if audit_result.clone_groups:
                avg_similarity = sum(group.similarity_score for group in audit_result.clone_groups) / len(audit_result.clone_groups)
                content.append(f"| Average Similarity | {avg_similarity:.3f} |")
            
            content.append("")
        
        # Tool information
        content.append("### Tool Information")
        content.append("")
        content.append("This report was generated by **IntelliRefactor** - Intelligent Project Analysis and Refactoring System.")
        content.append("")
        content.append("**Analysis Components:**")
        content.append("- Persistent Index Builder")
        content.append("- Block-level Clone Detector")
        content.append("- Three-level Unused Code Detector")
        content.append("- Evidence-based Analysis Engine")
        content.append("")
        content.append("For more information, visit: https://github.com/intellirefactor/intellirefactor")
        content.append("")
        
        return content
    
    def _generate_architecture_overview(self, audit_result: AuditResult) -> List[str]:
        """Generate architecture overview section for Design.md."""
        content = []
        content.append("## Architecture Overview")
        content.append("")
        
        stats = audit_result.statistics
        
        # Current state analysis
        content.append("### Current State")
        content.append("")
        content.append(f"The analyzed codebase consists of {stats.files_analyzed} files with {stats.total_findings} identified issues.")
        content.append("")
        
        # Issue distribution
        duplicate_count = stats.findings_by_type.get('duplicate_block', 0)
        unused_count = stats.findings_by_type.get('unused_code', 0)
        quality_count = stats.findings_by_type.get('quality_issue', 0)
        
        if duplicate_count > 0:
            content.append(f"**Code Duplication:** {duplicate_count} duplicate blocks indicate opportunities for abstraction")
        if unused_count > 0:
            content.append(f"**Dead Code:** {unused_count} unused elements suggest over-engineering or incomplete cleanup")
        if quality_count > 0:
            content.append(f"**Quality Issues:** {quality_count} quality problems indicate architectural debt")
        content.append("")
        
        # Architectural health assessment
        critical_count = stats.findings_by_severity.get('critical', 0)
        high_count = stats.findings_by_severity.get('high', 0)
        
        if critical_count == 0 and high_count == 0:
            content.append("### Architectural Health: 🟢 Good")
            content.append("The codebase shows good architectural health with no critical issues.")
        elif critical_count > 0:
            content.append("### Architectural Health: 🔴 Needs Attention")
            content.append(f"Critical issues ({critical_count}) require immediate architectural intervention.")
        elif high_count > 5:
            content.append("### Architectural Health: 🟡 Moderate Debt")
            content.append(f"High priority issues ({high_count}) indicate moderate architectural debt.")
        else:
            content.append("### Architectural Health: 🟡 Minor Issues")
            content.append("Minor architectural issues that can be addressed incrementally.")
        
        content.append("")
        return content
    
    def _generate_component_analysis(self, audit_result: AuditResult) -> List[str]:
        """Generate component analysis section for Design.md."""
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
                if 'critical' in severities:
                    severity = '🚨 Critical'
                elif 'high' in severities:
                    severity = '⚠️ High'
                elif 'medium' in severities:
                    severity = '📋 Medium'
                else:
                    severity = '💡 Low'
                
                # Identify primary concerns
                types = [f.finding_type.value for f in findings]
                concerns = []
                if 'duplicate_block' in types:
                    concerns.append('Duplication')
                if 'unused_code' in types:
                    concerns.append('Dead Code')
                if 'quality_issue' in types:
                    concerns.append('Quality')
                
                concerns_str = ', '.join(concerns) if concerns else 'Various'
                
                content.append(f"| `{file_name}` | {issue_count} | {severity} | {concerns_str} |")
            
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
                content.append(f"- `{file_name}` ({issue_count} issues) - Consider major refactoring or decomposition")
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
    
    def _generate_refactoring_strategy(self, audit_result: AuditResult) -> List[str]:
        """Generate refactoring strategy section for Design.md."""
        content = []
        content.append("## Refactoring Strategy")
        content.append("")
        
        stats = audit_result.statistics
        
        # Strategy based on findings
        content.append("### Strategic Approach")
        content.append("")
        
        critical_count = stats.findings_by_severity.get('critical', 0)
        duplicate_count = stats.findings_by_type.get('duplicate_block', 0)
        unused_count = stats.findings_by_type.get('unused_code', 0)
        
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
    
    def _generate_dependency_analysis(self, audit_result: AuditResult) -> List[str]:
        """Generate dependency analysis section for Design.md."""
        content = []
        content.append("## Dependency Analysis")
        content.append("")
        
        # Analyze dependencies from index if available
        if audit_result.index_result:
            content.append("### Dependency Overview")
            content.append("")
            content.append(f"**Total Dependencies:** {audit_result.index_result.dependencies_found}")
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
            content.append("*Dependency analysis requires index data. Run with --include-index for detailed dependency information.*")
            content.append("")
        
        return content
    
    def _generate_risk_assessment(self, audit_result: AuditResult) -> List[str]:
        """Generate risk assessment section for Design.md."""
        content = []
        content.append("## Risk Assessment")
        content.append("")
        
        stats = audit_result.statistics
        
        # Calculate risk levels
        critical_count = stats.findings_by_severity.get('critical', 0)
        high_count = stats.findings_by_severity.get('high', 0)
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
        
        duplicate_count = stats.findings_by_type.get('duplicate_block', 0)
        if duplicate_count > 5:
            content.append("**Refactoring Complexity Risk:**")
            content.append(f"- {duplicate_count} duplicate blocks increase refactoring complexity")
            content.append("- Risk of breaking functionality during consolidation")
            content.append("- Potential for introducing new bugs")
            content.append("")
        
        unused_count = stats.findings_by_type.get('unused_code', 0)
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
    
    def _generate_design_decisions(self, audit_result: AuditResult) -> List[str]:
        """Generate design decisions section for Design.md."""
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
        
        duplicate_count = stats.findings_by_type.get('duplicate_block', 0)
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
        
        unused_count = stats.findings_by_type.get('unused_code', 0)
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
        
        critical_count = stats.findings_by_severity.get('critical', 0)
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
    
    def _generate_design_appendix(self, audit_result: AuditResult) -> List[str]:
        """Generate appendix for Design.md."""
        content = []
        content.append("## Appendix")
        content.append("")
        
        # Design principles
        content.append("### Design Principles")
        content.append("")
        content.append("**SOLID Principles:**")
        content.append("- Single Responsibility: Each class should have one reason to change")
        content.append("- Open/Closed: Open for extension, closed for modification")
        content.append("- Liskov Substitution: Subtypes must be substitutable for base types")
        content.append("- Interface Segregation: Clients shouldn't depend on unused interfaces")
        content.append("- Dependency Inversion: Depend on abstractions, not concretions")
        content.append("")
        
        content.append("**DRY Principle:**")
        content.append("- Don't Repeat Yourself: Eliminate code duplication")
        content.append("- Single source of truth for each piece of knowledge")
        content.append("- Reduce maintenance burden and bug potential")
        content.append("")
        
        # Architecture patterns
        content.append("### Recommended Architecture Patterns")
        content.append("")
        content.append("**Layer Pattern:**")
        content.append("- Separate concerns into distinct layers")
        content.append("- Clear dependencies between layers")
        content.append("- Improved testability and maintainability")
        content.append("")
        
        content.append("**Repository Pattern:**")
        content.append("- Abstract data access logic")
        content.append("- Improve testability with mock repositories")
        content.append("- Centralize data access concerns")
        content.append("")
        
        return content
    
    def _generate_task_breakdown(self, audit_result: AuditResult) -> List[str]:
        """Generate task breakdown section for Implementation.md."""
        content = []
        content.append("## Task Breakdown")
        content.append("")
        
        stats = audit_result.statistics
        
        # Calculate task estimates
        critical_count = stats.findings_by_severity.get('critical', 0)
        high_count = stats.findings_by_severity.get('high', 0)
        medium_count = stats.findings_by_severity.get('medium', 0)
        duplicate_count = stats.findings_by_type.get('duplicate_block', 0)
        unused_count = stats.findings_by_type.get('unused_code', 0)
        
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
        total_hours = (critical_count * 2 + duplicate_count * 1 + unused_count * 0.5 + 
                      high_count * 1.5 + medium_count * 1)
        
        content.append(f"**Total Estimated Effort:** {total_hours:.1f} hours ({total_hours/8:.1f} days)")
        content.append("")
        
        return content
    
    def _generate_priority_phases(self, audit_result: AuditResult) -> List[str]:
        """Generate priority phases section for Implementation.md."""
        content = []
        content.append("## Priority Phases")
        content.append("")
        
        stats = audit_result.statistics
        critical_count = stats.findings_by_severity.get('critical', 0)
        high_count = stats.findings_by_severity.get('high', 0)
        medium_count = stats.findings_by_severity.get('medium', 0)
        
        phase = 1
        
        if critical_count > 0:
            content.append(f"### Phase {phase}: Critical Issues Resolution")
            content.append("")
            content.append(f"**Duration:** 1-2 days")
            content.append(f"**Priority:** 🚨 Critical")
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
            content.append(f"**Duration:** 1-2 weeks")
            content.append(f"**Priority:** ⚠️ High")
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
            content.append(f"**Duration:** 2-4 weeks")
            content.append(f"**Priority:** 📋 Medium")
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
    
    def _generate_refactoring_tasks(self, audit_result: AuditResult) -> List[str]:
        """Generate detailed refactoring tasks section for Implementation.md."""
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
                    content.append(f"   - File: `{file_name}:{finding.line_start}-{finding.line_end}`")
                    content.append(f"   - Confidence: {finding.confidence:.1%}")
                    content.append(f"   - Action: {finding.recommendations[0] if finding.recommendations else 'Extract duplicate code'}")
                    content.append(f"   - Estimated time: 30-60 minutes")
                    content.append("")
            
            if medium_conf:
                content.append("**Medium Confidence Duplicates (Review Required):**")
                content.append("")
                for i, finding in enumerate(medium_conf, 1):
                    file_name = Path(finding.file_path).name
                    content.append(f"{i}. **{finding.title}**")
                    content.append(f"   - File: `{file_name}:{finding.line_start}-{finding.line_end}`")
                    content.append(f"   - Confidence: {finding.confidence:.1%}")
                    content.append(f"   - Action: Review and potentially extract")
                    content.append(f"   - Estimated time: 60-90 minutes")
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
                    symbol_name = finding.metadata.get('symbol_name', 'Unknown')
                    content.append(f"{i}. **{symbol_name}**")
                    content.append(f"   - File: `{file_name}:{finding.line_start}`")
                    content.append(f"   - Confidence: {finding.confidence:.1%}")
                    content.append(f"   - Action: Remove unused code")
                    content.append(f"   - Estimated time: 10-15 minutes")
                    content.append("")
            
            if medium_conf:
                content.append("**Review Before Removal (Medium Confidence):**")
                content.append("")
                for i, finding in enumerate(medium_conf, 1):
                    file_name = Path(finding.file_path).name
                    symbol_name = finding.metadata.get('symbol_name', 'Unknown')
                    content.append(f"{i}. **{symbol_name}**")
                    content.append(f"   - File: `{file_name}:{finding.line_start}`")
                    content.append(f"   - Confidence: {finding.confidence:.1%}")
                    content.append(f"   - Action: Investigate usage and potentially remove")
                    content.append(f"   - Estimated time: 20-30 minutes")
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
                content.append(f"   - Estimated time: 45-90 minutes")
                content.append("")
        
        return content
    
    def _generate_testing_strategy(self, audit_result: AuditResult) -> List[str]:
        """Generate testing strategy section for Implementation.md."""
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
    
    def _generate_rollback_plan(self, audit_result: AuditResult) -> List[str]:
        """Generate rollback plan section for Implementation.md."""
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
    
    def _generate_success_criteria(self, audit_result: AuditResult) -> List[str]:
        """Generate success criteria section for Implementation.md."""
        content = []
        content.append("## Success Criteria")
        content.append("")
        
        stats = audit_result.statistics
        
        content.append("### Quantitative Success Metrics")
        content.append("")
        
        # Issue reduction targets
        critical_count = stats.findings_by_severity.get('critical', 0)
        high_count = stats.findings_by_severity.get('high', 0)
        duplicate_count = stats.findings_by_type.get('duplicate_block', 0)
        unused_count = stats.findings_by_type.get('unused_code', 0)
        
        content.append("**Issue Resolution Targets:**")
        if critical_count > 0:
            content.append(f"- ✅ Resolve 100% of critical issues ({critical_count} issues)")
        if high_count > 0:
            content.append(f"- ✅ Resolve 90% of high priority issues ({int(high_count * 0.9)} of {high_count} issues)")
        if duplicate_count > 0:
            content.append(f"- ✅ Reduce code duplication by 80% ({int(duplicate_count * 0.8)} of {duplicate_count} blocks)")
        if unused_count > 0:
            content.append(f"- ✅ Remove 70% of unused code ({int(unused_count * 0.7)} of {unused_count} elements)")
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
    
    def _generate_implementation_appendix(self, audit_result: AuditResult) -> List[str]:
        """Generate appendix for Implementation.md."""
        content = []
        content.append("## Appendix")
        content.append("")
        
        # Tools and resources
        content.append("### Tools and Resources")
        content.append("")
        content.append("**Development Tools:**")
        content.append("- IntelliRefactor for automated analysis")
        content.append("- Git for version control and rollback")
        content.append("- IDE with refactoring support")
        content.append("- Code coverage tools")
        content.append("- Performance profiling tools")
        content.append("")
        
        content.append("**Testing Tools:**")
        content.append("- Unit testing framework")
        content.append("- Integration testing tools")
        content.append("- Code quality analyzers")
        content.append("- Automated testing pipeline")
        content.append("")
        
        # Best practices
        content.append("### Refactoring Best Practices")
        content.append("")
        content.append("**General Principles:**")
        content.append("- Make small, incremental changes")
        content.append("- Test frequently and thoroughly")
        content.append("- Preserve existing behavior")
        content.append("- Improve design without changing functionality")
        content.append("")
        
        content.append("**Code Extraction:**")
        content.append("- Extract methods before extracting classes")
        content.append("- Use meaningful names for extracted elements")
        content.append("- Minimize parameter lists")
        content.append("- Maintain single responsibility principle")
        content.append("")
        
        content.append("**Cleanup Guidelines:**")
        content.append("- Remove unused imports and variables")
        content.append("- Eliminate dead code paths")
        content.append("- Simplify complex conditional logic")
        content.append("- Reduce nesting levels")
        content.append("")
        
        # Timeline template
        content.append("### Implementation Timeline Template")
        content.append("")
        
        total_findings = audit_result.statistics.total_findings
        if total_findings > 0:
            # Estimate timeline based on findings
            critical_count = audit_result.statistics.findings_by_severity.get('critical', 0)
            high_count = audit_result.statistics.findings_by_severity.get('high', 0)
            
            content.append("**Week 1:**")
            if critical_count > 0:
                content.append("- Days 1-2: Address critical issues")
                content.append("- Days 3-5: Begin high priority refactoring")
            else:
                content.append("- Days 1-5: High priority refactoring")
            content.append("")
            
            content.append("**Week 2:**")
            content.append("- Days 1-3: Continue refactoring work")
            content.append("- Days 4-5: Testing and validation")
            content.append("")
            
            if total_findings > 20:
                content.append("**Week 3:**")
                content.append("- Days 1-3: Medium priority improvements")
                content.append("- Days 4-5: Final testing and documentation")
                content.append("")
        else:
            content.append("**No implementation timeline needed - codebase is in excellent condition!**")
            content.append("")
        
        return content
"""
Data Extraction Utilities for Specification Generation

This module provides helper functions for extracting structured data
from audit results, including refactoring priorities and cleanup tasks.
"""

from typing import Dict, List, Any

from ..audit_models import AuditResult, AuditFindingType
from intellirefactor.analysis.foundation.models import Severity


class RefactoringPriorityExtractor:
    """Extracts refactoring priorities from audit results."""

    @staticmethod
    def extract(audit_result: AuditResult) -> List[Dict[str, Any]]:
        """
        Extract refactoring priorities from audit results.

        Args:
            audit_result: Complete audit results

        Returns:
            List of priority dictionaries with type, priority level, counts, and descriptions
        """
        priorities = []

        # Group findings by type and severity
        duplicate_findings = audit_result.get_findings_by_type(AuditFindingType.DUPLICATE_BLOCK)
        unused_findings = audit_result.get_findings_by_type(AuditFindingType.UNUSED_CODE)
        quality_findings = audit_result.get_findings_by_type(AuditFindingType.QUALITY_ISSUE)

        if duplicate_findings:
            high_conf_duplicates = [f for f in duplicate_findings if f.confidence >= 0.8]
            priorities.append(
                {
                    "type": "duplicate_code_refactoring",
                    "priority": "high" if len(high_conf_duplicates) > 5 else "medium",
                    "count": len(duplicate_findings),
                    "high_confidence_count": len(high_conf_duplicates),
                    "estimated_effort": "medium",
                    "description": "Extract duplicate code blocks into reusable functions",
                }
            )

        if unused_findings:
            safe_to_remove = [f for f in unused_findings if f.confidence >= 0.8]
            priorities.append(
                {
                    "type": "unused_code_cleanup",
                    "priority": "medium",
                    "count": len(unused_findings),
                    "safe_to_remove_count": len(safe_to_remove),
                    "estimated_effort": "low",
                    "description": "Remove unused code to improve maintainability",
                }
            )

        if quality_findings:
            critical_quality = [f for f in quality_findings if f.severity == Severity.CRITICAL]
            priorities.append(
                {
                    "type": "quality_improvements",
                    "priority": "high" if critical_quality else "medium",
                    "count": len(quality_findings),
                    "critical_count": len(critical_quality),
                    "estimated_effort": "high",
                    "description": "Address code quality and architectural issues",
                }
            )

        return priorities


class CleanupTaskExtractor:
    """Extracts cleanup tasks from audit results."""

    @staticmethod
    def extract(audit_result: AuditResult) -> List[Dict[str, Any]]:
        """
        Extract cleanup tasks from audit results.

        Args:
            audit_result: Complete audit results

        Returns:
            List of cleanup task dictionaries with type, counts, and time estimates
        """
        tasks = []

        unused_findings = audit_result.get_findings_by_type(AuditFindingType.UNUSED_CODE)
        if unused_findings:
            # Group by confidence level
            high_conf = [f for f in unused_findings if f.confidence >= 0.8]
            medium_conf = [f for f in unused_findings if 0.5 <= f.confidence < 0.8]

            if high_conf:
                tasks.append(
                    {
                        "type": "safe_unused_removal",
                        "count": len(high_conf),
                        "confidence": "high",
                        "estimated_time": f"{len(high_conf) * 5} minutes",
                        "description": "Remove high-confidence unused code",
                    }
                )

            if medium_conf:
                tasks.append(
                    {
                        "type": "review_unused_code",
                        "count": len(medium_conf),
                        "confidence": "medium",
                        "estimated_time": f"{len(medium_conf) * 15} minutes",
                        "description": "Review and potentially remove medium-confidence unused code",
                    }
                )

        return tasks

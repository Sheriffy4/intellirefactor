"""
Section Header Generators

Generates headers for Requirements, Design, and Implementation specification documents.
"""

from typing import List
from datetime import datetime

from ..audit_models import AuditResult


class RequirementsHeaderGenerator:
    """Generates header section for Requirements.md."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate header for Requirements.md.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for the header
        """
        content = []
        content.append("# Project Refactoring Requirements")
        content.append("")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**Project:** {audit_result.project_path}")
        content.append(
            f"**Analysis Time:** {audit_result.statistics.analysis_time_seconds:.2f} seconds"
        )
        content.append("")
        return content


class DesignHeaderGenerator:
    """Generates header section for Design.md."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate header for Design.md.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for the header
        """
        content = []
        content.append("# Refactoring Design Document")
        content.append("")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**Project:** {audit_result.project_path}")
        content.append(f"**Total Findings:** {audit_result.statistics.total_findings}")
        content.append("")
        content.append("## Overview")
        content.append("")
        content.append(
            "This document outlines the architectural design for refactoring the analyzed codebase."
        )
        content.append(
            "It provides component analysis, refactoring strategies, and design decisions based on"
        )
        content.append("the findings from the automated code analysis.")
        content.append("")
        return content


class ImplementationHeaderGenerator:
    """Generates header section for Implementation.md."""

    @staticmethod
    def generate(audit_result: AuditResult) -> List[str]:
        """
        Generate header for Implementation.md.

        Args:
            audit_result: Complete audit results

        Returns:
            List of markdown lines for the header
        """
        content = []
        content.append("# Refactoring Implementation Plan")
        content.append("")
        content.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        content.append(f"**Project:** {audit_result.project_path}")
        content.append(f"**Total Tasks:** {audit_result.statistics.total_findings}")
        content.append("")
        content.append("## Overview")
        content.append("")
        content.append(
            "This document provides a detailed implementation plan for refactoring the analyzed codebase."
        )
        content.append(
            "It includes task breakdowns, priority phases, testing strategies, and success criteria."
        )
        content.append("")
        return content

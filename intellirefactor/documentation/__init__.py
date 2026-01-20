"""
Documentation generation module for IntelliRefactor.

This module provides comprehensive documentation generation capabilities
including architecture diagrams, analysis flowcharts, call graphs,
and detailed reports.
"""

from .doc_generator import DocumentationGenerator
from .architecture_generator import ArchitectureGenerator
from .flowchart_generator import FlowchartGenerator
from .report_generator import ReportGenerator
from .registry_generator import RegistryGenerator

__all__ = [
    "DocumentationGenerator",
    "ArchitectureGenerator",
    "FlowchartGenerator",
    "ReportGenerator",
    "RegistryGenerator",
]

# NOTE:
# Shared AST helpers live in `documentation.ast_utils` (internal module).

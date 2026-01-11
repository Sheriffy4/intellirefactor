"""
Command-line interface for IntelliRefactor

Provides comprehensive CLI access to all IntelliRefactor functionality.
Supports all major operations through intuitive command-line commands with
rich terminal output, progress bars, and beautiful formatting.
"""

import argparse
import json
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add project root to sys.path when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intellirefactor.api import IntelliRefactor, AnalysisResult, RefactoringResult
from intellirefactor.config import IntelliRefactorConfig, SafetyLevel, load_config
from intellirefactor.templates import TemplateGenerator
from intellirefactor.analysis.index_builder import IndexBuilder, IndexBuildResult
from intellirefactor.analysis.index_store import IndexStore
from intellirefactor.analysis.index_query import IndexQuery
from intellirefactor.analysis.audit_engine import AuditEngine
from intellirefactor.analysis.spec_generator import SpecGenerator
from intellirefactor.analysis.responsibility_clusterer import (
    ResponsibilityClusterer,
    ClusteringConfig,
    ClusteringAlgorithm,
)
from intellirefactor.analysis.refactoring_decision_engine import (
    RefactoringDecisionEngine,
    DecisionCriteria,
    RefactoringPriority,
)
from intellirefactor.cli.rich_output import RichOutputManager, set_rich_enabled
from intellirefactor.cli.integration import CLIIntegrationManager


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser with comprehensive command support."""
    parser = argparse.ArgumentParser(
        prog="intellirefactor",
        description="IntelliRefactor - Intelligent Project Analysis and Refactoring System",
        epilog='Use "intellirefactor <command> --help" for detailed command help.',
    )

    # Global options
    parser.add_argument("--config", "-c", type=str, help="Path to configuration file")

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output and debug logging",
    )

    parser.add_argument(
        "--no-rich",
        action="store_true",
        help="Disable rich terminal output (use plain text)",
    )

    parser.add_argument("--no-color", action="store_true", help="Disable colored output")

    parser.add_argument(
        "--machine-readable",
        action="store_true",
        help="Output in machine-readable format (JSON)",
    )

    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze", help="Analyze project or file for refactoring opportunities"
    )
    analyze_parser.add_argument("path", help="Path to project directory or file to analyze")
    analyze_parser.add_argument("--output", "-o", help="Output file for analysis results")
    analyze_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )
    analyze_parser.add_argument(
        "--no-metrics", action="store_true", help="Skip detailed metrics calculation"
    )
    analyze_parser.add_argument(
        "--no-opportunities",
        action="store_true",
        help="Skip refactoring opportunity identification",
    )
    analyze_parser.add_argument(
        "--project-root",
        help="Project root directory for cross-module analysis (auto-detected if not specified)",
    )
    analyze_parser.add_argument(
        "--isolated",
        action="store_true",
        help="Perform isolated analysis (no cross-module checks)",
    )
    analyze_parser.add_argument(
        "--entry-point", help="Main script to trace execution flow (e.g., cli.py)"
    )
    analyze_parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate Mermaid flowcharts and diagrams",
    )

    # Opportunities command
    opportunities_parser = subparsers.add_parser(
        "opportunities", help="Identify and list refactoring opportunities"
    )
    opportunities_parser.add_argument("path", help="Path to project directory")
    opportunities_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=10,
        help="Maximum number of opportunities to show (default: 10)",
    )
    opportunities_parser.add_argument("--output", "-o", help="Output file for opportunities list")
    opportunities_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )

    # Refactor command
    refactor_parser = subparsers.add_parser("refactor", help="Perform refactoring operations")
    refactor_parser.add_argument("path", help="Path to project directory to refactor")
    refactor_parser.add_argument("--strategy", help="Refactoring strategy to use")
    refactor_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without applying changes",
    )
    refactor_parser.add_argument(
        "--max-operations",
        type=int,
        help="Maximum number of refactoring operations to perform",
    )
    refactor_parser.add_argument(
        "--safety-level",
        choices=["conservative", "moderate", "aggressive"],
        help="Safety level for refactoring operations",
    )
    refactor_parser.add_argument("--output", "-o", help="Output file for refactoring results")
    refactor_parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip validation of refactoring results",
    )

    # Apply command (for applying specific opportunities)
    apply_parser = subparsers.add_parser("apply", help="Apply a specific refactoring opportunity")
    apply_parser.add_argument(
        "opportunity_file",
        help="JSON file containing the refactoring opportunity to apply",
    )
    apply_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without applying changes",
    )
    apply_parser.add_argument(
        "--no-validation",
        action="store_true",
        help="Skip validation of refactoring results",
    )
    apply_parser.add_argument("--output", "-o", help="Output file for refactoring results")

    # Knowledge command
    knowledge_parser = subparsers.add_parser("knowledge", help="Query and manage knowledge base")
    knowledge_subparsers = knowledge_parser.add_subparsers(dest="knowledge_action")

    # Knowledge query
    query_parser = knowledge_subparsers.add_parser("query", help="Query knowledge base")
    query_parser.add_argument("query", help="Knowledge query string")
    query_parser.add_argument("--limit", type=int, default=10, help="Maximum number of results")
    query_parser.add_argument("--output", "-o", help="Output file for query results")

    # Knowledge add
    add_parser = knowledge_subparsers.add_parser("add", help="Add knowledge item")
    add_parser.add_argument("knowledge_file", help="JSON file containing knowledge item to add")

    # Knowledge status
    knowledge_subparsers.add_parser("status", help="Show knowledge base status")

    # Report command
    report_parser = subparsers.add_parser(
        "report", help="Generate comprehensive refactoring reports"
    )
    report_parser.add_argument("results_file", help="JSON file containing refactoring results")
    report_parser.add_argument(
        "--format",
        choices=["text", "json", "html"],
        default="text",
        help="Report format (default: text)",
    )
    report_parser.add_argument("--output", "-o", help="Output file for the report")

    # Status command
    status_parser = subparsers.add_parser(
        "status", help="Show system status and health information"
    )
    status_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )

    # Config command
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(dest="config_action")

    config_subparsers.add_parser("show", help="Show current configuration")

    init_parser = config_subparsers.add_parser("init", help="Initialize configuration file")
    init_parser.add_argument(
        "--path", default="intellirefactor.json", help="Configuration file path"
    )
    init_parser.add_argument("--template", help="Configuration template to use")
    init_parser.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="json",
        help="Configuration file format",
    )

    validate_parser = config_subparsers.add_parser("validate", help="Validate configuration file")
    validate_parser.add_argument("config_file", help="Configuration file to validate")

    # Template command
    template_parser = subparsers.add_parser("template", help="Configuration template management")
    template_subparsers = template_parser.add_subparsers(dest="template_action")

    # List templates
    template_subparsers.add_parser("list", help="List available configuration templates")

    # Generate template
    generate_parser = template_subparsers.add_parser(
        "generate", help="Generate configuration from template"
    )
    generate_parser.add_argument("template_name", help="Name of template to use")
    generate_parser.add_argument("output_path", help="Output path for generated configuration")
    generate_parser.add_argument(
        "--format", choices=["json", "yaml"], default="json", help="Output format"
    )
    generate_parser.add_argument("--project-name", help="Project name for customization")
    generate_parser.add_argument(
        "--safety-level",
        choices=["conservative", "moderate", "aggressive"],
        default="moderate",
        help="Refactoring safety level",
    )

    # Show template
    show_parser = template_subparsers.add_parser("show", help="Show template contents")
    show_parser.add_argument("template_name", help="Name of template to show")
    show_parser.add_argument(
        "--format", choices=["json", "yaml"], default="json", help="Display format"
    )

    # Expert analysis command
    expert_parser = subparsers.add_parser(
        "expert-analyze", help="Run expert refactoring analysis for safe code restructuring"
    )
    expert_parser.add_argument("project_path", help="Path to project root directory")
    expert_parser.add_argument("target_file", help="Path to target file to analyze")
    expert_parser.add_argument(
        "--output", "-o", help="Output directory for analysis results (default: ./expert_analysis)"
    )
    expert_parser.add_argument(
        "--format", 
        choices=["json", "markdown", "both"], 
        default="both", 
        help="Output format (default: both)"
    )
    expert_parser.add_argument(
        "--detailed", action="store_true", 
        help="Export detailed analysis data as requested by experts (includes full call graph, specific locations, executable tests)"
    )
    expert_parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    # Index command
    index_parser = subparsers.add_parser(
        "index", help="Index management for persistent analysis data"
    )
    index_subparsers = index_parser.add_subparsers(dest="index_action", help="Index operations")

    # Index build
    build_parser = index_subparsers.add_parser("build", help="Build or update the project index")
    build_parser.add_argument("project_path", help="Path to project directory to index")
    build_parser.add_argument(
        "--incremental",
        "-i",
        action="store_true",
        default=True,
        help="Perform incremental update (default: True)",
    )
    build_parser.add_argument(
        "--full", "-f", action="store_true", help="Force full rebuild of the index"
    )
    build_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=100,
        help="Batch size for processing files (default: 100)",
    )
    build_parser.add_argument("--output", "-o", help="Output file for build results")
    build_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )

    # Index status
    status_parser = index_subparsers.add_parser("status", help="Show index status and statistics")
    status_parser.add_argument(
        "project_path",
        nargs="?",
        help="Path to project directory (optional, uses current directory if not specified)",
    )
    status_parser.add_argument(
        "--detailed", "-d", action="store_true", help="Show detailed statistics"
    )
    status_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )

    # Index rebuild
    rebuild_parser = index_subparsers.add_parser(
        "rebuild", help="Completely rebuild the index from scratch"
    )
    rebuild_parser.add_argument(
        "project_path", help="Path to project directory to rebuild index for"
    )
    rebuild_parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=100,
        help="Batch size for processing files (default: 100)",
    )
    rebuild_parser.add_argument("--output", "-o", help="Output file for rebuild results")
    rebuild_parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="text",
        help="Output format (default: text)",
    )

    # Duplicates command group
    duplicates_parser = subparsers.add_parser(
        "duplicates", help="Code clone detection and analysis"
    )
    duplicates_subparsers = duplicates_parser.add_subparsers(
        dest="duplicates_action", help="Clone detection operations"
    )

    # Duplicates blocks
    blocks_parser = duplicates_subparsers.add_parser(
        "blocks", help="Detect block-level code clones"
    )
    blocks_parser.add_argument(
        "project_path", help="Path to project directory to analyze for block clones"
    )
    blocks_parser.add_argument(
        "--exact-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for exact clones (default: 0.95)",
    )
    blocks_parser.add_argument(
        "--structural-threshold",
        type=float,
        default=0.85,
        help="Similarity threshold for structural clones (default: 0.85)",
    )
    blocks_parser.add_argument(
        "--semantic-threshold",
        type=float,
        default=0.75,
        help="Similarity threshold for semantic clones (default: 0.75)",
    )
    blocks_parser.add_argument(
        "--min-clone-size",
        type=int,
        default=3,
        help="Minimum size (lines of code) for clone detection (default: 3)",
    )
    blocks_parser.add_argument(
        "--min-instances",
        type=int,
        default=2,
        help="Minimum number of instances to form a clone group (default: 2)",
    )
    blocks_parser.add_argument(
        "--include-patterns",
        nargs="*",
        default=["**/*.py"],
        help="File patterns to include (default: **/*.py)",
    )
    blocks_parser.add_argument(
        "--exclude-patterns",
        nargs="*",
        default=["**/test_*.py", "**/*_test.py", "**/tests/**"],
        help="File patterns to exclude (default: test files)",
    )
    blocks_parser.add_argument("--output", "-o", help="Output file for clone detection results")
    blocks_parser.add_argument(
        "--format",
        choices=["json", "text", "html"],
        default="text",
        help="Output format (default: text)",
    )
    blocks_parser.add_argument(
        "--show-code", action="store_true", help="Include code snippets in output"
    )
    blocks_parser.add_argument(
        "--group-by",
        choices=["type", "file", "similarity"],
        default="type",
        help="Group results by clone type, file, or similarity (default: type)",
    )

    # Duplicates methods
    methods_parser = duplicates_subparsers.add_parser(
        "methods", help="Detect method-level code clones"
    )
    methods_parser.add_argument(
        "project_path", help="Path to project directory to analyze for method clones"
    )
    methods_parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.8,
        help="Similarity threshold for method clones (default: 0.8)",
    )
    methods_parser.add_argument(
        "--min-method-size",
        type=int,
        default=5,
        help="Minimum method size (lines of code) for clone detection (default: 5)",
    )
    methods_parser.add_argument(
        "--include-patterns",
        nargs="*",
        default=["**/*.py"],
        help="File patterns to include (default: **/*.py)",
    )
    methods_parser.add_argument(
        "--exclude-patterns",
        nargs="*",
        default=["**/test_*.py", "**/*_test.py", "**/tests/**"],
        help="File patterns to exclude (default: test files)",
    )
    methods_parser.add_argument(
        "--output", "-o", help="Output file for method clone detection results"
    )
    methods_parser.add_argument(
        "--format",
        choices=["json", "text", "html"],
        default="text",
        help="Output format (default: text)",
    )
    methods_parser.add_argument(
        "--show-signatures",
        action="store_true",
        help="Include method signatures in output",
    )
    methods_parser.add_argument(
        "--extraction-recommendations",
        action="store_true",
        help="Include extraction strategy recommendations",
    )

    # Duplicates similar (semantic similarity)
    similar_parser = duplicates_subparsers.add_parser(
        "similar", help="Find semantically similar methods"
    )
    similar_parser.add_argument(
        "project_path",
        help="Path to project directory to analyze for semantic similarity",
    )
    similar_parser.add_argument(
        "--target-method",
        help="Specific method to find similarities for (format: module.ClassName.method_name)",
    )
    similar_parser.add_argument(
        "--similarity-types",
        nargs="*",
        choices=["structural", "functional", "behavioral", "hybrid"],
        default=["structural", "functional", "behavioral"],
        help="Types of similarity to search for (default: all types)",
    )
    similar_parser.add_argument(
        "--structural-threshold",
        type=float,
        default=0.8,
        help="Minimum similarity for structural matches (default: 0.8)",
    )
    similar_parser.add_argument(
        "--functional-threshold",
        type=float,
        default=0.7,
        help="Minimum similarity for functional matches (default: 0.7)",
    )
    similar_parser.add_argument(
        "--behavioral-threshold",
        type=float,
        default=0.6,
        help="Minimum similarity for behavioral matches (default: 0.6)",
    )
    similar_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold for matches (default: 0.5)",
    )
    similar_parser.add_argument(
        "--include-patterns",
        nargs="*",
        default=["**/*.py"],
        help="File patterns to include (default: **/*.py)",
    )
    similar_parser.add_argument(
        "--exclude-patterns",
        nargs="*",
        default=["**/test_*.py", "**/*_test.py", "**/tests/**"],
        help="File patterns to exclude (default: test files)",
    )
    similar_parser.add_argument("--output", "-o", help="Output file for similarity results")
    similar_parser.add_argument(
        "--format",
        choices=["json", "text", "html"],
        default="text",
        help="Output format (default: text)",
    )
    similar_parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Include evidence and code snippets in output",
    )
    similar_parser.add_argument(
        "--show-differences",
        action="store_true",
        help="Show differences between similar methods",
    )
    similar_parser.add_argument(
        "--merge-recommendations",
        action="store_true",
        help="Include merge strategy recommendations",
    )
    similar_parser.add_argument(
        "--max-results",
        type=int,
        default=20,
        help="Maximum number of similarity matches to show (default: 20)",
    )

    # Unused command group
    unused_parser = subparsers.add_parser("unused", help="Unused code detection and analysis")
    unused_subparsers = unused_parser.add_subparsers(
        dest="unused_action", help="Unused code detection operations"
    )

    # Unused detect
    detect_parser = unused_subparsers.add_parser(
        "detect", help="Detect unused code at multiple levels"
    )
    detect_parser.add_argument(
        "project_path", help="Path to project directory to analyze for unused code"
    )
    detect_parser.add_argument(
        "--entry-points",
        nargs="*",
        help="Entry point files/modules (auto-detected if not specified)",
    )
    detect_parser.add_argument(
        "--level",
        choices=["1", "2", "3", "all"],
        default="all",
        help="Analysis level: 1=modules, 2=symbols, 3=dynamic, all=all levels (default: all)",
    )
    detect_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (0.0-1.0, default: 0.5)",
    )
    detect_parser.add_argument(
        "--include-patterns",
        nargs="*",
        default=["**/*.py"],
        help="File patterns to include (default: **/*.py)",
    )
    detect_parser.add_argument(
        "--exclude-patterns",
        nargs="*",
        default=["**/__pycache__/**", "**/.*"],
        help="File patterns to exclude (default: cache and hidden files)",
    )
    detect_parser.add_argument(
        "--filter-type",
        choices=[
            "module_unreachable",
            "symbol_unused",
            "private_method_unused",
            "public_export_unused",
            "tests_only",
            "scripts_only",
            "uncertain_dynamic",
        ],
        help="Filter results by unused code type",
    )
    detect_parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Include detailed evidence for each finding",
    )
    detect_parser.add_argument(
        "--show-usage",
        action="store_true",
        help="Include usage references for each finding",
    )
    detect_parser.add_argument(
        "--output", "-o", help="Output file for unused code detection results"
    )
    detect_parser.add_argument(
        "--format",
        choices=["json", "text", "html"],
        default="text",
        help="Output format (default: text)",
    )
    detect_parser.add_argument(
        "--group-by",
        choices=["type", "file", "confidence"],
        default="type",
        help="Group results by unused type, file, or confidence (default: type)",
    )

    # Audit command group
    audit_parser = subparsers.add_parser(
        "audit", help="Comprehensive project audit combining multiple analysis types"
    )
    audit_parser.add_argument("project_path", help="Path to project directory to audit")
    audit_parser.add_argument(
        "--skip-index",
        action="store_true",
        help="Skip index building (use existing index)",
    )
    audit_parser.add_argument(
        "--skip-duplicates", action="store_true", help="Skip duplicate code detection"
    )
    audit_parser.add_argument(
        "--skip-unused", action="store_true", help="Skip unused code detection"
    )
    audit_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (0.0-1.0, default: 0.5)",
    )
    audit_parser.add_argument(
        "--include-patterns",
        nargs="*",
        default=["**/*.py"],
        help="File patterns to include (default: **/*.py)",
    )
    audit_parser.add_argument(
        "--exclude-patterns",
        nargs="*",
        default=["**/__pycache__/**", "**/.*", "**/test_*.py", "**/*_test.py"],
        help="File patterns to exclude (default: cache, hidden, and test files)",
    )
    audit_parser.add_argument("--output", "-o", help="Output file for audit results")
    audit_parser.add_argument(
        "--format",
        choices=["json", "text", "html"],
        default="text",
        help="Output format (default: text)",
    )
    audit_parser.add_argument(
        "--emit-spec",
        action="store_true",
        help="Generate Requirements.md specification from findings",
    )
    audit_parser.add_argument(
        "--spec-output",
        help="Output path for generated specification (default: Requirements.md)",
    )
    audit_parser.add_argument(
        "--emit-json",
        action="store_true",
        help="Generate machine-readable JSON analysis artifacts",
    )
    audit_parser.add_argument(
        "--json-output", help="Output path for JSON artifacts (default: analysis.json)"
    )

    # Smells command group
    smells_parser = subparsers.add_parser(
        "smells", help="Architectural smell detection and analysis"
    )
    smells_subparsers = smells_parser.add_subparsers(
        dest="smells_action", help="Smell detection operations"
    )

    # Smells detect
    smells_detect_parser = smells_subparsers.add_parser(
        "detect", help="Detect architectural smells in project"
    )
    smells_detect_parser.add_argument("project_path", help="Path to project directory to analyze")
    smells_detect_parser.add_argument(
        "--include-patterns",
        nargs="*",
        default=["**/*.py"],
        help="File patterns to include (default: **/*.py)",
    )
    smells_detect_parser.add_argument(
        "--exclude-patterns",
        nargs="*",
        default=["**/test_*.py", "**/*_test.py", "**/tests/**", "**/__pycache__/**"],
        help="File patterns to exclude (default: test files and cache)",
    )
    smells_detect_parser.add_argument(
        "--smell-types",
        nargs="*",
        choices=[
            "god_class",
            "long_method",
            "high_complexity",
            "srp_violation",
            "feature_envy",
            "inappropriate_intimacy",
        ],
        help="Specific smell types to detect (default: all)",
    )
    smells_detect_parser.add_argument(
        "--severity",
        choices=["critical", "high", "medium", "low"],
        help="Minimum severity level to report",
    )
    smells_detect_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (0.0-1.0, default: 0.5)",
    )
    smells_detect_parser.add_argument(
        "--output", "-o", help="Output file for smell detection results"
    )
    smells_detect_parser.add_argument(
        "--format",
        choices=["json", "text", "html"],
        default="text",
        help="Output format (default: text)",
    )
    smells_detect_parser.add_argument(
        "--show-evidence",
        action="store_true",
        help="Include detailed evidence in output",
    )
    smells_detect_parser.add_argument(
        "--show-recommendations",
        action="store_true",
        help="Include remediation recommendations in output",
    )
    smells_detect_parser.add_argument(
        "--group-by",
        choices=["file", "type", "severity"],
        default="file",
        help="Group results by file, smell type, or severity (default: file)",
    )

    # Threshold configuration options
    smells_detect_parser.add_argument(
        "--god-class-methods",
        type=int,
        default=15,
        help="God Class method count threshold (default: 15)",
    )
    smells_detect_parser.add_argument(
        "--god-class-responsibilities",
        type=int,
        default=3,
        help="God Class responsibility count threshold (default: 3)",
    )
    smells_detect_parser.add_argument(
        "--long-method-lines",
        type=int,
        default=30,
        help="Long Method line count threshold (default: 30)",
    )
    smells_detect_parser.add_argument(
        "--high-complexity-cyclomatic",
        type=int,
        default=10,
        help="High Complexity cyclomatic complexity threshold (default: 10)",
    )

    # Cluster command group
    cluster_parser = subparsers.add_parser(
        "cluster", help="Responsibility clustering for component extraction"
    )
    cluster_subparsers = cluster_parser.add_subparsers(dest="cluster_action")

    # Cluster responsibility
    responsibility_parser = cluster_subparsers.add_parser(
        "responsibility",
        help="Cluster methods by shared responsibilities for component extraction",
    )
    responsibility_parser.add_argument("project_path", help="Path to project directory to analyze")
    responsibility_parser.add_argument(
        "--class-name", help="Specific class name to analyze (optional)"
    )
    responsibility_parser.add_argument(
        "--file-path", help="Specific file path to analyze (optional)"
    )
    responsibility_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for clustering (default: 0.7)",
    )
    responsibility_parser.add_argument(
        "--min-cohesion",
        type=float,
        default=0.6,
        help="Minimum cohesion score for clusters (default: 0.6)",
    )
    responsibility_parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help="Minimum number of methods per cluster (default: 3)",
    )
    responsibility_parser.add_argument(
        "--max-cluster-size",
        type=int,
        default=8,
        help="Maximum number of methods per cluster (default: 8)",
    )
    responsibility_parser.add_argument(
        "--algorithm",
        choices=["community_detection", "agglomerative", "hybrid"],
        default="hybrid",
        help="Clustering algorithm to use (default: hybrid)",
    )
    responsibility_parser.add_argument(
        "--output",
        choices=["text", "json", "html"],
        default="text",
        help="Output format (default: text)",
    )
    responsibility_parser.add_argument("--output-file", help="Output file path (optional)")
    responsibility_parser.add_argument(
        "--show-suggestions",
        action="store_true",
        help="Show component extraction suggestions",
    )
    responsibility_parser.add_argument(
        "--show-interfaces",
        action="store_true",
        help="Show generated component interfaces",
    )
    responsibility_parser.add_argument(
        "--show-plans", action="store_true", help="Show detailed extraction plans"
    )
    responsibility_parser.add_argument(
        "--group-by",
        choices=["class", "confidence", "complexity"],
        default="class",
        help="Group results by class, confidence, or complexity (default: class)",
    )

    # Decision Engine command group
    decide_parser = subparsers.add_parser(
        "decide", help="Refactoring decision engine for intelligent recommendations"
    )
    decide_subparsers = decide_parser.add_subparsers(
        dest="decide_action", help="Decision engine operations"
    )

    # Decide analyze
    decide_analyze_parser = decide_subparsers.add_parser(
        "analyze",
        help="Comprehensive project analysis with prioritized refactoring decisions",
    )
    decide_analyze_parser.add_argument("project_path", help="Path to project directory to analyze")
    decide_analyze_parser.add_argument(
        "--criteria-config", help="Path to decision criteria configuration file"
    )
    decide_analyze_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.7,
        help="Minimum confidence threshold for decisions (default: 0.7)",
    )
    decide_analyze_parser.add_argument(
        "--min-impact",
        type=float,
        default=0.5,
        help="Minimum impact threshold for decisions (default: 0.5)",
    )
    decide_analyze_parser.add_argument(
        "--max-effort",
        type=float,
        default=0.8,
        help="Maximum effort threshold for decisions (default: 0.8)",
    )
    decide_analyze_parser.add_argument(
        "--output", "-o", help="Output file for decision analysis results"
    )
    decide_analyze_parser.add_argument(
        "--format",
        choices=["json", "yaml", "text"],
        default="text",
        help="Output format (default: text)",
    )
    decide_analyze_parser.add_argument(
        "--export-decisions",
        help="Export decisions to machine-readable file (JSON/YAML)",
    )

    # Decide recommend
    decide_recommend_parser = decide_subparsers.add_parser(
        "recommend", help="Get specific refactoring recommendations with filtering"
    )
    decide_recommend_parser.add_argument(
        "analysis_file", help="Path to decision analysis results file"
    )
    decide_recommend_parser.add_argument(
        "--priority",
        choices=["critical", "high", "medium", "low"],
        help="Filter by priority level",
    )
    decide_recommend_parser.add_argument(
        "--type",
        choices=[
            "extract_method",
            "extract_class",
            "remove_unused_code",
            "decompose_god_class",
            "reduce_method_complexity",
            "eliminate_duplicates",
            "improve_cohesion",
            "reduce_coupling",
            "parameterize_duplicates",
            "template_method_pattern",
        ],
        help="Filter by refactoring type",
    )
    decide_recommend_parser.add_argument(
        "--min-confidence", type=float, help="Minimum confidence threshold"
    )
    decide_recommend_parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=10,
        help="Maximum number of recommendations to show (default: 10)",
    )
    decide_recommend_parser.add_argument("--output", "-o", help="Output file for recommendations")
    decide_recommend_parser.add_argument(
        "--format",
        choices=["json", "yaml", "text"],
        default="text",
        help="Output format (default: text)",
    )
    decide_recommend_parser.add_argument(
        "--include-evidence",
        action="store_true",
        help="Include detailed evidence in output",
    )
    decide_recommend_parser.add_argument(
        "--include-plans",
        action="store_true",
        help="Include implementation plans in output",
    )

    # Decide plan
    decide_plan_parser = decide_subparsers.add_parser(
        "plan", help="Generate refactoring sequence and execution plan"
    )
    decide_plan_parser.add_argument("analysis_file", help="Path to decision analysis results file")
    decide_plan_parser.add_argument(
        "--priority-filter",
        choices=["critical", "high", "medium", "low"],
        action="append",
        help="Include only specified priority levels (can be used multiple times)",
    )
    decide_plan_parser.add_argument(
        "--max-decisions",
        type=int,
        default=20,
        help="Maximum number of decisions to include in plan (default: 20)",
    )
    decide_plan_parser.add_argument(
        "--sequence-by",
        choices=["priority", "dependency", "effort", "risk"],
        default="priority",
        help="Sequence decisions by priority, dependency, effort, or risk (default: priority)",
    )
    decide_plan_parser.add_argument("--output", "-o", help="Output file for execution plan")
    decide_plan_parser.add_argument(
        "--format",
        choices=["json", "yaml", "text", "markdown"],
        default="text",
        help="Output format (default: text)",
    )
    decide_plan_parser.add_argument(
        "--include-timeline",
        action="store_true",
        help="Include estimated timeline in plan",
    )
    decide_plan_parser.add_argument(
        "--include-resources",
        action="store_true",
        help="Include resource requirements in plan",
    )

    # Generate command group
    generate_parser = subparsers.add_parser(
        "generate", help="Generate specification documents from analysis results"
    )
    generate_subparsers = generate_parser.add_subparsers(
        dest="generate_action", help="Specification generation operations"
    )

    # Generate spec
    generate_spec_parser = generate_subparsers.add_parser(
        "spec",
        help="Generate specification documents (Requirements.md, Design.md, Implementation.md)",
    )
    generate_spec_parser.add_argument(
        "project_path",
        help="Path to project directory to analyze and generate specs for",
    )
    generate_spec_parser.add_argument(
        "--spec-type",
        choices=["requirements", "design", "implementation", "all"],
        default="all",
        help="Type of specification to generate (default: all)",
    )
    generate_spec_parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory for generated specifications (default: current directory)",
    )
    generate_spec_parser.add_argument(
        "--template", help="Custom template file for specification generation"
    )
    generate_spec_parser.add_argument(
        "--include-index",
        action="store_true",
        help="Include index building in analysis",
    )
    generate_spec_parser.add_argument(
        "--include-duplicates",
        action="store_true",
        help="Include duplicate code detection",
    )
    generate_spec_parser.add_argument(
        "--include-unused", action="store_true", help="Include unused code detection"
    )
    generate_spec_parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence threshold (0.0-1.0, default: 0.5)",
    )
    generate_spec_parser.add_argument(
        "--include-patterns",
        nargs="*",
        default=["**/*.py"],
        help="File patterns to include (default: **/*.py)",
    )
    generate_spec_parser.add_argument(
        "--exclude-patterns",
        nargs="*",
        default=["**/__pycache__/**", "**/.*", "**/test_*.py", "**/*_test.py"],
        help="File patterns to exclude (default: cache, hidden, and test files)",
    )
    generate_spec_parser.add_argument(
        "--emit-json",
        action="store_true",
        help="Also generate machine-readable JSON analysis artifacts",
    )
    generate_spec_parser.add_argument(
        "--json-output", help="Output path for JSON artifacts (default: analysis.json)"
    )

    # System command group (new enhanced commands)
    system_parser = subparsers.add_parser(
        "system", help="System management and integration commands"
    )
    system_subparsers = system_parser.add_subparsers(dest="system_action")

    # System status
    system_status_parser = system_subparsers.add_parser(
        "status", help="Show comprehensive system status with rich output"
    )
    system_status_parser.add_argument(
        "--check-compatibility",
        action="store_true",
        help="Include compatibility check with existing systems",
    )
    system_status_parser.add_argument(
        "--show-plugins", action="store_true", help="Show detailed plugin information"
    )

    # System init
    system_init_parser = system_subparsers.add_parser(
        "init", help="Initialize IntelliRefactor for a project with full integration"
    )
    system_init_parser.add_argument("project_path", help="Path to project directory to initialize")
    system_init_parser.add_argument(
        "--create-config", action="store_true", help="Create default configuration file"
    )
    system_init_parser.add_argument(
        "--setup-knowledge",
        action="store_true",
        help="Initialize knowledge base for project",
    )
    system_init_parser.add_argument(
        "--load-plugins", action="store_true", help="Load and initialize plugins"
    )

    # System migrate
    system_migrate_parser = system_subparsers.add_parser(
        "migrate", help="Migrate legacy data to modern format"
    )
    system_migrate_parser.add_argument(
        "legacy_data_path", help="Path to legacy data file or directory"
    )
    system_migrate_parser.add_argument("--output", "-o", help="Output path for migrated data")
    system_migrate_parser.add_argument(
        "--format",
        choices=["json", "yaml"],
        default="json",
        help="Output format for migrated data",
    )

    # Enhanced analyze command with rich output
    analyze_enhanced_parser = subparsers.add_parser(
        "analyze-enhanced",
        help="Enhanced project analysis with rich terminal output and full integration",
    )
    analyze_enhanced_parser.add_argument(
        "project_path", help="Path to project directory to analyze"
    )
    analyze_enhanced_parser.add_argument("--output", "-o", help="Output file for analysis results")
    analyze_enhanced_parser.add_argument(
        "--format",
        choices=["json", "markdown", "html"],
        default="markdown",
        help="Output format (default: markdown)",
    )
    analyze_enhanced_parser.add_argument(
        "--include-metrics",
        action="store_true",
        default=True,
        help="Include detailed metrics analysis",
    )
    analyze_enhanced_parser.add_argument(
        "--include-opportunities",
        action="store_true",
        default=True,
        help="Include refactoring opportunities",
    )
    analyze_enhanced_parser.add_argument(
        "--include-safety",
        action="store_true",
        default=True,
        help="Include safety analysis",
    )
    analyze_enhanced_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with prompts",
    )

    # Visualize command
    visualize_parser = subparsers.add_parser(
        "visualize", help="Generate visualizations for code analysis"
    )
    visualize_subparsers = visualize_parser.add_subparsers(
        dest="visualize_action", help="Visualization operations"
    )

    # Visualize method
    method_parser = visualize_subparsers.add_parser("method", help="Generate method flowchart")
    method_parser.add_argument("file_path", help="Path to the Python file containing the method")
    method_parser.add_argument("method_name", help="Name of the method to visualize")
    method_parser.add_argument("--output", "-o", help="Output file for the visualization")
    method_parser.add_argument(
        "--format",
        choices=["mermaid", "png", "svg"],
        default="mermaid",
        help="Output format (default: mermaid)",
    )

    # Visualize call graph
    call_graph_parser = visualize_subparsers.add_parser(
        "call-graph", help="Generate call dependency graph"
    )
    call_graph_parser.add_argument(
        "symbol_name", help="Name of the function/method to trace calls for"
    )
    call_graph_parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum depth for call graph (default: 3)",
    )
    call_graph_parser.add_argument("--output", "-o", help="Output file for the visualization")

    # Enhanced refactor command with orchestration
    refactor_enhanced_parser = subparsers.add_parser(
        "refactor-enhanced",
        help="Enhanced refactoring with full orchestration and safety checks",
    )
    refactor_enhanced_parser.add_argument(
        "project_path", help="Path to project directory to refactor"
    )
    refactor_enhanced_parser.add_argument("plan_file", help="Path to refactoring plan file (JSON)")
    refactor_enhanced_parser.add_argument(
        "--dry-run", action="store_true", help="Perform dry run without making changes"
    )
    refactor_enhanced_parser.add_argument(
        "--create-backup",
        action="store_true",
        default=True,
        help="Create backup before refactoring",
    )
    refactor_enhanced_parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode with confirmations",
    )
    refactor_enhanced_parser.add_argument(
        "--output", "-o", help="Output file for refactoring results"
    )

    # Documentation command group
    docs_parser = subparsers.add_parser(
        "docs", help="Generate comprehensive documentation for modules"
    )
    docs_subparsers = docs_parser.add_subparsers(
        dest="docs_action", help="Documentation operations"
    )

    # Generate full documentation suite
    full_docs_parser = docs_subparsers.add_parser(
        "generate", help="Generate complete documentation suite"
    )
    full_docs_parser.add_argument("target_file", help="Path to Python file to document")
    full_docs_parser.add_argument(
        "--output-dir",
        "-o",
        help="Output directory for documentation files (default: current directory)",
    )
    full_docs_parser.add_argument(
        "--include",
        nargs="+",
        choices=[
            "architecture",
            "flowchart",
            "call_graph",
            "report",
            "registry",
            "llm_context",
            "project_structure",
        ],
        help="Documentation types to include (default: all)",
    )
    full_docs_parser.add_argument(
        "--format",
        choices=["markdown", "html"],
        default="markdown",
        help="Output format (default: markdown)",
    )

    # Generate specific documentation type
    type_docs_parser = docs_subparsers.add_parser(
        "type", help="Generate specific documentation type"
    )
    type_docs_parser.add_argument("target_file", help="Path to Python file to document")
    type_docs_parser.add_argument(
        "doc_type",
        choices=[
            "architecture",
            "flowchart",
            "call_graph",
            "report",
            "registry",
            "llm_context",
            "project_structure",
        ],
        help="Type of documentation to generate",
    )
    type_docs_parser.add_argument("--output", "-o", help="Output file path (optional)")

    # List available documentation types
    docs_subparsers.add_parser("list", help="List available documentation types")

    return parser


def format_analysis_result(result: AnalysisResult, format_type: str) -> str:
    """Format analysis result for output."""
    if format_type == "json":
        return json.dumps(
            {
                "success": result.success,
                "data": result.data,
                "errors": result.errors,
                "warnings": result.warnings,
                "metadata": result.metadata,
            },
            indent=2,
        )
    else:
        output = []
        if result.success:
            output.append("Analysis completed successfully")
            if result.metadata.get("project_path"):
                output.append(f"Project: {result.metadata['project_path']}")
            elif result.metadata.get("file_path"):
                output.append(f"File: {result.metadata['file_path']}")

            if "metrics" in result.data:
                output.append("\nMetrics:")
                for key, value in result.data["metrics"].items():
                    output.append(f"  {key}: {value}")

            if "refactoring_opportunities" in result.data:
                opportunities = result.data["refactoring_opportunities"]
                output.append(f"\nRefactoring opportunities found: {len(opportunities)}")
                for i, opp in enumerate(opportunities[:5], 1):  # Show first 5
                    output.append(f"  {i}. {opp.get('description', 'Unknown opportunity')}")
                if len(opportunities) > 5:
                    output.append(f"  ... and {len(opportunities) - 5} more")
        else:
            output.append("Analysis failed")
            for error in result.errors:
                output.append(f"Error: {error}")

        if result.warnings:
            output.append("\nWarnings:")
            for warning in result.warnings:
                output.append(f"  {warning}")

        return "\n".join(output)


def format_refactoring_result(result: RefactoringResult, format_type: str) -> str:
    """Format refactoring result for output."""
    if format_type == "json":
        return json.dumps(
            {
                "success": result.success,
                "operations_applied": result.operations_applied,
                "changes_made": result.changes_made,
                "validation_results": result.validation_results,
                "errors": result.errors,
                "warnings": result.warnings,
                "metadata": result.metadata,
            },
            indent=2,
        )
    else:
        output = []
        if result.success:
            output.append("Refactoring completed successfully")
            output.append(f"Operations applied: {result.operations_applied}")
            output.append(f"Changes made: {len(result.changes_made)}")

            if result.validation_results:
                output.append(f"Validation: {result.validation_results.get('status', 'Unknown')}")
        else:
            output.append("Refactoring failed")
            for error in result.errors:
                output.append(f"Error: {error}")

        if result.warnings:
            output.append("\nWarnings:")
            for warning in result.warnings:
                output.append(f"  {warning}")

        return "\n".join(output)


def cmd_analyze(args, intellirefactor: IntelliRefactor) -> None:
    """Handle analyze command."""
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path does not exist: {path}", file=sys.stderr)
        sys.exit(1)

    include_metrics = not args.no_metrics
    include_opportunities = not args.no_opportunities

    if path.is_file():
        # Handle file analysis with project context
        project_root = Path(args.project_root) if args.project_root else None

        if args.isolated:
            # Force isolated analysis
            result = intellirefactor.analyze_file(path, project_root=None)
        else:
            # Use provided project root or auto-detect
            result = intellirefactor.analyze_file(path, project_root=project_root)
    else:
        result = intellirefactor.analyze_project(path, include_metrics, include_opportunities)

    output = format_analysis_result(result, args.format)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Analysis results written to {args.output}")
    else:
        print(output)

    # Handle visualization if requested
    if hasattr(args, "visualize") and args.visualize:
        from intellirefactor.orchestration.global_refactoring_orchestrator import (
            GlobalRefactoringOrchestrator,
        )

        print("\nGenerating visualizations...")

        # Create orchestrator
        orchestrator = GlobalRefactoringOrchestrator(
            project_root=path if path.is_dir() else path.parent
        )

        # Prepare visualization config
        viz_config = (
            {"entry_point": args.entry_point}
            if hasattr(args, "entry_point") and args.entry_point
            else {}
        )
        viz_config["name"] = "visualization"
        viz_config["description"] = "Visualization generation"

        # Run visualization stage directly
        try:
            stage_result = orchestrator._handle_visualization_stage(viz_config)
            if stage_result.success:
                print(
                    f"✅ Visualizations generated successfully in: {stage_result.details.get('output_dir', 'unknown')}"
                )
            else:
                print(f"⚠️  Visualization generation had issues: {stage_result.message}")
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}")

    if not result.success:
        sys.exit(1)


def cmd_opportunities(args, intellirefactor: IntelliRefactor) -> None:
    """Handle opportunities command."""
    opportunities = intellirefactor.identify_opportunities(args.path, args.limit)

    if args.format == "json":
        output = json.dumps(opportunities, indent=2)
    else:
        output = f"Refactoring opportunities for {args.path}:\n"
        if opportunities:
            for i, opp in enumerate(opportunities, 1):
                output += f"{i}. {opp.get('description', 'Unknown opportunity')}\n"
                if "priority" in opp:
                    output += f"   Priority: {opp['priority']}\n"
                if "estimated_impact" in opp:
                    output += f"   Impact: {opp['estimated_impact']}\n"
                output += "\n"
        else:
            output += "No refactoring opportunities found.\n"

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Opportunities written to {args.output}")
    else:
        print(output)


def cmd_refactor(args, intellirefactor: IntelliRefactor) -> None:
    """Handle refactor command."""
    # Update safety level if specified
    if args.safety_level:
        intellirefactor.config.refactoring_settings.safety_level = SafetyLevel(args.safety_level)

    result = intellirefactor.auto_refactor_project(
        args.path, args.strategy, args.max_operations, args.dry_run
    )

    output = format_refactoring_result(result, "text")

    if args.output:
        # Save detailed JSON results to file
        detailed_output = format_refactoring_result(result, "json")
        with open(args.output, "w") as f:
            f.write(detailed_output)
        print(f"Detailed refactoring results written to {args.output}")

    print(output)

    if not result.success:
        sys.exit(1)


def cmd_apply(args, intellirefactor: IntelliRefactor) -> None:
    """Handle apply command."""
    try:
        with open(args.opportunity_file, "r", encoding="utf-8") as f:
            opportunity = json.load(f)
    except Exception as e:
        print(f"Error: Failed to load opportunity file: {e}", file=sys.stderr)
        sys.exit(1)

    validate_results = not args.no_validation
    result = intellirefactor.apply_refactoring(opportunity, args.dry_run, validate_results)

    output = format_refactoring_result(result, "text")

    if args.output:
        detailed_output = format_refactoring_result(result, "json")
        with open(args.output, "w") as f:
            f.write(detailed_output)
        print(f"Detailed results written to {args.output}")

    print(output)

    if not result.success:
        sys.exit(1)


def cmd_knowledge(args, intellirefactor: IntelliRefactor) -> None:
    """Handle knowledge command."""
    if args.knowledge_action == "query":
        results = intellirefactor.query_knowledge(args.query, args.limit)

        output = f"Knowledge query results for '{args.query}':\n"
        if results:
            for i, result in enumerate(results, 1):
                output += f"{i}. {result.get('description', 'No description')}\n"
                if "confidence" in result:
                    output += f"   Confidence: {result['confidence']:.2f}\n"
                if "type" in result:
                    output += f"   Type: {result['type']}\n"
                output += "\n"
        else:
            output += "No results found.\n"

        if args.output:
            with open(args.output, "w") as f:
                f.write(json.dumps(results, indent=2))
            print(f"Query results written to {args.output}")
        else:
            print(output)

    elif args.knowledge_action == "add":
        try:
            with open(args.knowledge_file, "r", encoding="utf-8") as f:
                knowledge_item = json.load(f)

            success = intellirefactor.add_knowledge(knowledge_item)
            if success:
                print("Knowledge item added successfully")
            else:
                print("Failed to add knowledge item", file=sys.stderr)
                sys.exit(1)
        except Exception as e:
            print(f"Error: Failed to load knowledge file: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.knowledge_action == "status":
        # Get knowledge base status
        status = intellirefactor.get_system_status()
        knowledge_status = status.get("components", {}).get("knowledge_manager", "unknown")
        print(f"Knowledge base status: {knowledge_status}")
        print(
            f"Knowledge base path: {intellirefactor.config.knowledge_settings.knowledge_base_path}"
        )


def cmd_report(args, intellirefactor: IntelliRefactor) -> None:
    """Handle report command."""
    try:
        with open(args.results_file, "r", encoding="utf-8") as f:
            results_data = json.load(f)

        # Convert dict data back to RefactoringResult objects
        results = []
        if isinstance(results_data, list):
            for result_data in results_data:
                result = RefactoringResult(
                    success=result_data.get("success", False),
                    operations_applied=result_data.get("operations_applied", 0),
                    changes_made=result_data.get("changes_made", []),
                    validation_results=result_data.get("validation_results", {}),
                    errors=result_data.get("errors", []),
                    warnings=result_data.get("warnings", []),
                    metadata=result_data.get("metadata", {}),
                )
                results.append(result)

        report = intellirefactor.generate_report(results, args.format)

        if args.output:
            with open(args.output, "w") as f:
                f.write(report)
            print(f"Report written to {args.output}")
        else:
            print(report)

    except Exception as e:
        print(f"Error: Failed to generate report: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_visualize(args) -> None:
    """Handle visualize command."""
    from intellirefactor.visualization.diagram_generator import FlowchartGenerator
    from intellirefactor.analysis.index_store import IndexStore
    from pathlib import Path

    try:
        if args.visualize_action == "method":
            # Generate method flowchart
            generator = FlowchartGenerator(
                None
            )  # We'll create the generator without index for method visualization
            flowchart = generator.generate_method_flowchart(args.file_path, args.method_name)

            if args.output:
                output_path = Path(args.output)
            else:
                # Create default output path
                file_name = Path(args.file_path).stem
                output_path = (
                    Path("visualizations") / f"{file_name}_{args.method_name}_flowchart.md"
                )
                output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# Method Flowchart: {args.method_name}\n\n")
                f.write("```mermaid\n")
                f.write(flowchart)
                f.write("\n```\n")

            print(f"Method flowchart saved to: {output_path}")

        elif args.visualize_action == "call-graph":
            # Generate call graph - need index store for this
            db_path = Path("intellirefactor/.intellirefactor/index.db")
            if not db_path.exists():
                print(f"Error: Index database not found at {db_path}", file=sys.stderr)
                print(
                    "Please build the index first using: intellirefactor index build <project_path>",
                    file=sys.stderr,
                )
                sys.exit(1)

            store = IndexStore(db_path)
            generator = FlowchartGenerator(store)
            call_graph = generator.generate_call_graph(args.symbol_name, max_depth=args.max_depth)

            if args.output:
                output_path = Path(args.output)
            else:
                # Create default output path
                output_path = Path("visualizations") / f"{args.symbol_name}_call_graph.md"
                output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# Call Graph: {args.symbol_name}\n\n")
                f.write("```mermaid\n")
                f.write(call_graph)
                f.write("\n```\n")

            print(f"Call graph saved to: {output_path}")

        else:
            print(
                f"Error: Unknown visualize action: {args.visualize_action}",
                file=sys.stderr,
            )
            sys.exit(1)

    except Exception as e:
        print(f"Error in visualize command: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def cmd_docs(args) -> None:
    """Handle documentation generation command."""
    from intellirefactor.documentation.doc_generator import DocumentationGenerator
    from pathlib import Path
    import sys

    try:
        if args.docs_action == "list":
            # List available documentation types - no target file needed
            # Create a dummy generator to get type information
            doc_generator = DocumentationGenerator(Path.cwd())

            available_types = doc_generator.list_available_types()
            generator_info = doc_generator.get_generator_info()

            print("Available Documentation Types:")
            print("=" * 40)

            for doc_type in available_types:
                description = generator_info.get(doc_type, f"{doc_type.title()} generator")
                print(f"📋 {doc_type:<20} - {description}")

            print(f"\n📊 Total types available: {len(available_types)}")
            return

        # For other actions, we need a target file
        target_file = Path(args.target_file)

        if not target_file.exists():
            print(f"Error: Target file not found: {args.target_file}", file=sys.stderr)
            sys.exit(1)

        if not target_file.suffix == ".py":
            print("Error: Target file must be a Python file (.py)", file=sys.stderr)
            sys.exit(1)

        # Initialize documentation generator
        doc_generator = DocumentationGenerator(target_file.parent)

        if args.docs_action == "generate":
            # Generate full documentation suite
            output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
            include_types = args.include if args.include else None

            print(f"Generating comprehensive documentation for: {target_file}")
            print(f"Output directory: {output_dir}")

            generated_files = doc_generator.generate_full_documentation(
                target_file=target_file,
                output_dir=output_dir,
                include_types=include_types,
            )

            print("\n✅ Documentation generation completed successfully!")
            print("\nGenerated files:")
            for doc_type, file_path in generated_files.items():
                print(f"  📄 {doc_type.title()}: {file_path}")

            print(f"\n📊 Total files generated: {len(generated_files)}")

        elif args.docs_action == "type":
            # Generate specific documentation type
            output_file = args.output if args.output else None

            print(f"Generating {args.doc_type} documentation for: {target_file}")

            generated_file = doc_generator.generate_documentation_type(
                target_file=target_file, doc_type=args.doc_type, output_file=output_file
            )

            print(f"✅ {args.doc_type.title()} documentation saved to: {generated_file}")

        else:
            print(f"Error: Unknown docs action: {args.docs_action}", file=sys.stderr)
            sys.exit(1)

    except Exception as e:
        print(f"Error in documentation command: {e}", file=sys.stderr)
        if hasattr(args, "verbose") and args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def cmd_status(args, intellirefactor: IntelliRefactor) -> None:
    """Handle status command."""
    status = intellirefactor.get_system_status()

    if args.format == "json":
        print(json.dumps(status, indent=2))
    else:
        print("IntelliRefactor System Status:")
        print(f"Initialized: {status['initialized']}")
        print(f"Safety Level: {status['configuration']['safety_level']}")
        print(f"Auto Apply: {status['configuration']['auto_apply']}")
        print(f"Knowledge Path: {status['configuration']['knowledge_path']}")

        print("\nComponent Status:")
        for name, component_status in status["components"].items():
            print(f"  {name}: {component_status}")


def cmd_config(args) -> None:
    """Handle config command."""
    if args.config_action == "show":
        config = IntelliRefactorConfig.from_env()
        print("Current IntelliRefactor Configuration:")
        print(config.get_config_summary())

    elif args.config_action == "init":
        if args.template:
            # Use template
            try:
                TemplateGenerator.generate_config(args.template, args.path, args.format)
                print(f"Configuration file created from template '{args.template}' at {args.path}")
            except Exception as e:
                print(
                    f"Error: Failed to create configuration from template: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            # Use default configuration
            config = IntelliRefactorConfig.default()
            config.to_file(args.path, args.format)
            print(f"Default configuration file created at {args.path}")

        print("Edit the file to customize your IntelliRefactor settings.")

    elif args.config_action == "validate":
        try:
            config = IntelliRefactorConfig.load(args.config_file, validate=True)
            print(f"Configuration file {args.config_file} is valid")
        except Exception as e:
            print(f"Error: Configuration file is invalid: {e}", file=sys.stderr)
            sys.exit(1)


def cmd_template(args) -> None:
    """Handle template command."""
    if args.template_action == "list":
        templates = TemplateGenerator.list_templates()
        print("Available configuration templates:")
        print()
        for name, template_file in templates.items():
            description = TemplateGenerator.get_template_description(name)
            print(f"  {name}:")
            print(f"    {description}")
            print()

    elif args.template_action == "generate":
        try:
            customizations = {}
            if args.project_name:
                customizations["_description"] = f"Configuration for {args.project_name} project"
            if args.safety_level != "moderate":
                customizations["refactoring"] = {"safety_level": args.safety_level}

            TemplateGenerator.generate_config(
                args.template_name,
                args.output_path,
                args.format,
                customizations if customizations else None,
            )
            print(f"Generated {args.template_name} template at {args.output_path}")

            if args.project_name:
                print(f"Customized for project: {args.project_name}")
            if args.safety_level != "moderate":
                print(f"Safety level set to: {args.safety_level}")

        except Exception as e:
            print(f"Error: Failed to generate template: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.template_action == "show":
        try:
            template_data = TemplateGenerator.load_template(args.template_name, args.format)

            if args.format.lower() in ["yaml", "yml"]:
                import yaml

                print(yaml.dump(template_data, default_flow_style=False, indent=2))
            else:
                print(json.dumps(template_data, indent=2))

        except Exception as e:
            print(f"Error: Failed to show template: {e}", file=sys.stderr)
            sys.exit(1)


def get_index_db_path(project_path: Path) -> Path:
    """Get the index database path for a project."""
    return project_path / ".intellirefactor" / "index.db"


def format_index_build_result(result: IndexBuildResult, format_type: str) -> str:
    """Format index build result for output."""
    if format_type == "json":
        return json.dumps(
            {
                "success": result.success,
                "files_processed": result.files_processed,
                "files_skipped": result.files_skipped,
                "symbols_found": result.symbols_found,
                "blocks_found": result.blocks_found,
                "dependencies_found": result.dependencies_found,
                "errors": result.errors,
                "build_time_seconds": result.build_time_seconds,
                "incremental": result.incremental,
            },
            indent=2,
        )
    else:
        output = []
        if result.success:
            output.append(f"Index {'updated' if result.incremental else 'built'} successfully")
            output.append(f"Files processed: {result.files_processed}")
            if result.files_skipped > 0:
                output.append(f"Files skipped (unchanged): {result.files_skipped}")
            output.append(f"Symbols found: {result.symbols_found}")
            output.append(f"Blocks found: {result.blocks_found}")
            output.append(f"Dependencies found: {result.dependencies_found}")
            output.append(f"Build time: {result.build_time_seconds:.2f} seconds")

            if result.incremental:
                output.append("Mode: Incremental update")
            else:
                output.append("Mode: Full rebuild")
        else:
            output.append("Index build failed")
            for error in result.errors:
                output.append(f"Error: {error}")

        return "\n".join(output)


def format_index_status(stats: Dict[str, Any], format_type: str, detailed: bool = False) -> str:
    """Format index status for output."""
    if format_type == "json":
        return json.dumps(stats, indent=2)
    else:
        output = []
        output.append("Index Status:")
        output.append(f"Files indexed: {stats.get('files_count', 0)}")
        output.append(f"Symbols indexed: {stats.get('symbols_count', 0)}")
        output.append(f"Blocks indexed: {stats.get('blocks_count', 0)}")
        output.append(f"Dependencies indexed: {stats.get('dependencies_count', 0)}")
        output.append(f"Attribute accesses indexed: {stats.get('attribute_access_count', 0)}")

        if stats.get("database_size_bytes"):
            size_mb = stats["database_size_bytes"] / (1024 * 1024)
            output.append(f"Database size: {size_mb:.2f} MB")

        if stats.get("last_analysis"):
            output.append(f"Last analysis: {stats['last_analysis']}")

        if detailed and "detailed_stats" in stats:
            detailed_stats = stats["detailed_stats"]
            output.append("\nDetailed Statistics:")

            if "complexity_distribution" in detailed_stats:
                complexity = detailed_stats["complexity_distribution"]
                output.append(f"Average complexity: {complexity.get('avg_complexity', 0):.2f}")
                output.append(
                    f"High complexity symbols: {complexity.get('high_complexity_count', 0)}"
                )

            if "file_type_distribution" in detailed_stats:
                file_types = detailed_stats["file_type_distribution"]
                output.append(f"Test files: {file_types.get('test_files', 0)}")
                output.append(f"Source files: {file_types.get('source_files', 0)}")

            if "symbol_type_distribution" in detailed_stats:
                symbol_types = detailed_stats["symbol_type_distribution"]
                output.append("Symbol types:")
                for symbol_type, count in symbol_types.items():
                    output.append(f"  {symbol_type}: {count}")

        return "\n".join(output)


def format_clone_detection_results(
    clone_groups,
    statistics,
    format_type: str,
    show_code: bool = False,
    group_by: str = "type",
) -> str:
    """Format clone detection results for output."""
    from intellirefactor.analysis.block_clone_detector import ExtractionStrategy

    if format_type == "json":
        # Convert clone groups to JSON-serializable format
        json_groups = []
        for group in clone_groups:
            json_group = {
                "group_id": group.group_id,
                "clone_type": group.clone_type.value,
                "similarity_score": group.similarity_score,
                "extraction_strategy": (
                    group.extraction_strategy.value if group.extraction_strategy else None
                ),
                "extraction_confidence": group.extraction_confidence,
                "instance_count": len(group.instances),
                "instances": [],
            }

            for instance in group.instances:
                json_instance = {
                    "file_path": instance.file_path,
                    "line_start": instance.line_start,
                    "line_end": instance.line_end,
                    "lines_of_code": instance.block_info.lines_of_code,
                    "statement_count": instance.block_info.statement_count,
                    "similarity_score": instance.similarity_score,
                    "extraction_feasibility": instance.extraction_feasibility,
                }
                json_group["instances"].append(json_instance)

            json_groups.append(json_group)

        return json.dumps({"clone_groups": json_groups, "statistics": statistics}, indent=2)

    elif format_type == "html":
        # HTML format for rich display
        html_parts = []
        html_parts.append("<html><head><title>Clone Detection Results</title>")
        html_parts.append("<style>")
        html_parts.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html_parts.append(".clone-group { border: 1px solid #ccc; margin: 10px 0; padding: 10px; }")
        html_parts.append(".instance { background: #f5f5f5; margin: 5px 0; padding: 5px; }")
        html_parts.append(".code { background: #f0f0f0; padding: 10px; font-family: monospace; }")
        html_parts.append("</style></head><body>")

        html_parts.append("<h1>Clone Detection Results</h1>")
        html_parts.append(
            f"<p>Found {len(clone_groups)} clone groups with {statistics['total_instances']} total instances</p>"
        )

        for group in clone_groups:
            html_parts.append('<div class="clone-group">')
            html_parts.append(f"<h3>{group.clone_type.value.title()} Clone Group</h3>")
            html_parts.append(
                f"<p>Similarity: {group.similarity_score:.3f}, Instances: {len(group.instances)}</p>"
            )

            if (
                group.extraction_strategy
                and group.extraction_strategy != ExtractionStrategy.NO_EXTRACTION
            ):
                html_parts.append(
                    f"<p><strong>Recommended:</strong> {group.extraction_strategy.value.replace('_', ' ').title()} (confidence: {group.extraction_confidence:.2f})</p>"
                )

            for instance in group.instances:
                html_parts.append('<div class="instance">')
                html_parts.append(
                    f"<strong>{instance.file_path}:{instance.line_start}-{instance.line_end}</strong>"
                )
                html_parts.append(
                    f" ({instance.block_info.lines_of_code} LOC, {instance.block_info.statement_count} statements)"
                )
                html_parts.append("</div>")

            html_parts.append("</div>")

        html_parts.append("</body></html>")
        return "\n".join(html_parts)

    else:  # text format
        output = []
        output.append("Clone Detection Results")
        output.append("=" * 50)
        output.append(
            f"Found {len(clone_groups)} clone groups with {statistics['total_instances']} total instances"
        )
        output.append(f"Files affected: {statistics['files_affected']}")
        output.append(f"Average similarity: {statistics['average_similarity']}")
        output.append("")

        # Group results based on group_by parameter
        if group_by == "type":
            grouped = {}
            for group in clone_groups:
                clone_type = group.clone_type.value
                if clone_type not in grouped:
                    grouped[clone_type] = []
                grouped[clone_type].append(group)
        elif group_by == "file":
            grouped = {}
            for group in clone_groups:
                files = set(inst.file_path for inst in group.instances)
                key = f"{len(files)} files"
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(group)
        else:  # similarity
            grouped = {}
            for group in clone_groups:
                similarity_range = f"{int(group.similarity_score * 10) * 10}%-{int(group.similarity_score * 10) * 10 + 9}%"
                if similarity_range not in grouped:
                    grouped[similarity_range] = []
                grouped[similarity_range].append(group)

        for group_key, groups in grouped.items():
            output.append(f"{group_key.title()} ({len(groups)} groups):")
            output.append("-" * 40)

            for group in groups:
                output.append(f"  Clone Group: {group.clone_type.value.title()}")
                output.append(f"  Similarity: {group.similarity_score:.3f}")
                output.append(f"  Instances: {len(group.instances)}")

                if (
                    group.extraction_strategy
                    and group.extraction_strategy != ExtractionStrategy.NO_EXTRACTION
                ):
                    output.append(
                        f"  Recommended: {group.extraction_strategy.value.replace('_', ' ').title()} (confidence: {group.extraction_confidence:.2f})"
                    )

                output.append("  Locations:")
                for instance in group.instances:
                    output.append(
                        f"    {instance.file_path}:{instance.line_start}-{instance.line_end} ({instance.block_info.lines_of_code} LOC)"
                    )

                if show_code and group.instances:
                    # Show code snippet from first instance
                    first_instance = group.instances[0]
                    try:
                        with open(first_instance.file_path, "r", encoding="utf-8") as f:
                            lines = f.readlines()
                            start_idx = max(0, first_instance.line_start - 1)
                            end_idx = min(len(lines), first_instance.line_end)
                            code_lines = lines[start_idx:end_idx]

                            output.append("  Code snippet:")
                            for i, line in enumerate(code_lines, start=first_instance.line_start):
                                output.append(f"    {i:4d}: {line.rstrip()}")
                    except Exception:
                        output.append("  Code snippet: <unable to read file>")

                output.append("")

            output.append("")

        return "\n".join(output)


def format_similarity_results(
    results,
    format_type: str,
    show_evidence: bool = False,
    show_differences: bool = False,
    show_merge_recommendations: bool = False,
) -> str:
    """Format semantic similarity results for output."""
    if format_type == "json":
        import json

        return json.dumps(results, indent=2, default=str)

    if format_type == "html":
        output = ["<html><head><title>Semantic Similarity Results</title></head><body>"]
        output.append("<h1>Semantic Similarity Analysis Results</h1>")

        if not results:
            output.append("<p>No similar methods found.</p>")
        else:
            for result in results:
                output.append(f"<h2>Target Method: {result['target_method']['name']}</h2>")
                output.append(
                    f"<p><strong>File:</strong> {result['target_method']['file_path']}</p>"
                )
                output.append(
                    f"<p><strong>Lines:</strong> {result['target_method']['line_start']}-{result['target_method']['line_end']}</p>"
                )

                if result["similar_methods"]:
                    output.append("<h3>Similar Methods:</h3>")
                    output.append("<ul>")

                    for similar in result["similar_methods"]:
                        output.append("<li>")
                        output.append(f"<strong>{similar['method']['name']}</strong> ")
                        output.append(f"(Score: {similar['similarity_score']:.3f}, ")
                        output.append(f"Confidence: {similar['confidence']:.3f})<br>")
                        output.append(f"File: {similar['method']['file_path']}<br>")
                        output.append(
                            f"Lines: {similar['method']['line_start']}-{similar['method']['line_end']}<br>"
                        )
                        output.append(f"Type: {similar['similarity_type']}")

                        if show_evidence and similar.get("evidence"):
                            output.append("<br><strong>Evidence:</strong>")
                            output.append("<ul>")
                            for evidence in similar["evidence"]:
                                output.append(f"<li>{evidence}</li>")
                            output.append("</ul>")

                        if show_differences and similar.get("differences"):
                            output.append("<br><strong>Differences:</strong>")
                            output.append("<ul>")
                            for diff in similar["differences"]:
                                output.append(f"<li>{diff}</li>")
                            output.append("</ul>")

                        if show_merge_recommendations and similar.get("merge_recommendation"):
                            rec = similar["merge_recommendation"]
                            output.append(
                                f"<br><strong>Merge Strategy:</strong> {rec['strategy']}<br>"
                            )
                            output.append(f"<strong>Effort:</strong> {rec['effort']}<br>")
                            output.append(f"<strong>Description:</strong> {rec['description']}")

                        output.append("</li>")

                    output.append("</ul>")
                else:
                    output.append("<p>No similar methods found for this target.</p>")

                output.append("<hr>")

        output.append("</body></html>")
        return "\n".join(output)

    # Text format (default)
    output = []
    output.append("=== Semantic Similarity Analysis Results ===")
    output.append("")

    if not results:
        output.append("No similar methods found.")
        return "\n".join(output)

    for result in results:
        output.append(f"Target Method: {result['target_method']['name']}")
        output.append(f"  File: {result['target_method']['file_path']}")
        output.append(
            f"  Lines: {result['target_method']['line_start']}-{result['target_method']['line_end']}"
        )
        output.append("")

        if result["similar_methods"]:
            output.append("Similar Methods:")

            for i, similar in enumerate(result["similar_methods"], 1):
                output.append(f"  {i}. {similar['method']['name']}")
                output.append(f"     File: {similar['method']['file_path']}")
                output.append(
                    f"     Lines: {similar['method']['line_start']}-{similar['method']['line_end']}"
                )
                output.append(f"     Similarity Score: {similar['similarity_score']:.3f}")
                output.append(f"     Confidence: {similar['confidence']:.3f}")
                output.append(f"     Type: {similar['similarity_type']}")

                if show_evidence and similar.get("evidence"):
                    output.append("     Evidence:")
                    for evidence in similar["evidence"]:
                        output.append(f"       - {evidence}")

                if show_differences and similar.get("differences"):
                    output.append("     Differences:")
                    for diff in similar["differences"]:
                        output.append(f"       - {diff}")

                if show_merge_recommendations and similar.get("merge_recommendation"):
                    rec = similar["merge_recommendation"]
                    output.append(f"     Merge Strategy: {rec['strategy']}")
                    output.append(f"     Effort: {rec['effort']}")
                    output.append(f"     Description: {rec['description']}")

                output.append("")
        else:
            output.append("  No similar methods found for this target.")
            output.append("")

        output.append("-" * 60)
        output.append("")

    return "\n".join(output)


def cmd_index(args) -> None:
    """Handle index command."""
    if args.index_action == "build":
        project_path = Path(args.project_path)
        if not project_path.exists():
            print(f"Error: Project path does not exist: {project_path}", file=sys.stderr)
            sys.exit(1)

        if not project_path.is_dir():
            print(
                f"Error: Project path must be a directory: {project_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Create .intellirefactor directory if it doesn't exist
        intellirefactor_dir = project_path / ".intellirefactor"
        intellirefactor_dir.mkdir(exist_ok=True)

        db_path = get_index_db_path(project_path)

        try:
            # Progress callback for showing build progress
            def progress_callback(progress: float, files_processed: int, total_files: int):
                if total_files > 0:
                    print(
                        f"\rProgress: {progress:.1f}% ({files_processed}/{total_files} files)",
                        end="",
                        flush=True,
                    )

            with IndexBuilder(db_path, batch_size=args.batch_size) as builder:
                incremental = not args.full  # If --full is specified, don't do incremental
                result = builder.build_index(
                    project_path,
                    incremental=incremental,
                    progress_callback=progress_callback,
                )

            print()  # New line after progress

            output = format_index_build_result(result, args.format)

            if args.output:
                with open(args.output, "w") as f:
                    f.write(output)
                print(f"Build results written to {args.output}")
            else:
                print(output)

            if not result.success:
                sys.exit(1)

        except Exception as e:
            print(f"Error: Failed to build index: {e}", file=sys.stderr)
            if getattr(args, "verbose", False):
                import traceback

                traceback.print_exc()
            sys.exit(1)

    elif args.index_action == "status":
        project_path = Path(args.project_path) if args.project_path else Path.cwd()
        db_path = get_index_db_path(project_path)

        if not db_path.exists():
            print(f"No index found for project: {project_path}")
            print(f"Run 'intellirefactor index build {project_path}' to create an index.")
            return

        try:
            store = IndexStore(db_path)
            stats = store.get_statistics()

            if args.detailed:
                # Get additional detailed statistics
                query = IndexQuery(store)
                stats["detailed_stats"] = {
                    "complexity_distribution": query.get_complexity_distribution(),
                    "file_type_distribution": query.get_file_statistics(),
                }

                # Get symbol type distribution
                with store._get_connection() as conn:
                    cursor = conn.execute(
                        """
                        SELECT kind, COUNT(*) as count
                        FROM symbols
                        GROUP BY kind
                        ORDER BY count DESC
                    """
                    )
                    symbol_types = {row[0]: row[1] for row in cursor.fetchall()}
                    stats["detailed_stats"]["symbol_type_distribution"] = symbol_types

            output = format_index_status(stats, args.format, args.detailed)
            print(output)

        except Exception as e:
            print(f"Error: Failed to get index status: {e}", file=sys.stderr)
            if getattr(args, "verbose", False):
                import traceback

                traceback.print_exc()
            sys.exit(1)

    elif args.index_action == "rebuild":
        project_path = Path(args.project_path)
        if not project_path.exists():
            print(f"Error: Project path does not exist: {project_path}", file=sys.stderr)
            sys.exit(1)

        if not project_path.is_dir():
            print(
                f"Error: Project path must be a directory: {project_path}",
                file=sys.stderr,
            )
            sys.exit(1)

        # Create .intellirefactor directory if it doesn't exist
        intellirefactor_dir = project_path / ".intellirefactor"
        intellirefactor_dir.mkdir(exist_ok=True)

        db_path = get_index_db_path(project_path)

        try:
            # Progress callback for showing rebuild progress
            def progress_callback(progress: float, files_processed: int, total_files: int):
                if total_files > 0:
                    print(
                        f"\rProgress: {progress:.1f}% ({files_processed}/{total_files} files)",
                        end="",
                        flush=True,
                    )

            with IndexBuilder(db_path, batch_size=args.batch_size) as builder:
                result = builder.rebuild_index(project_path, progress_callback=progress_callback)

            print()  # New line after progress

            output = format_index_build_result(result, args.format)

            if args.output:
                with open(args.output, "w") as f:
                    f.write(output)
                print(f"Rebuild results written to {args.output}")
            else:
                print(output)

            if not result.success:
                sys.exit(1)

        except Exception as e:
            print(f"Error: Failed to rebuild index: {e}", file=sys.stderr)
            if getattr(args, "verbose", False):
                import traceback

                traceback.print_exc()
            sys.exit(1)


def cmd_duplicates(args) -> None:
    """Handle duplicates command."""
    from intellirefactor.analysis.block_extractor import BlockExtractor
    from intellirefactor.analysis.block_clone_detector import BlockCloneDetector
    import glob

    project_path = Path(args.project_path)
    if not project_path.exists():
        print(f"Error: Project path does not exist: {project_path}", file=sys.stderr)
        sys.exit(1)

    if not project_path.is_dir():
        print(f"Error: Project path must be a directory: {project_path}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.duplicates_action == "blocks":
            # Initialize components
            extractor = BlockExtractor()
            detector = BlockCloneDetector(
                exact_threshold=args.exact_threshold,
                structural_threshold=args.structural_threshold,
                semantic_threshold=args.semantic_threshold,
                min_clone_size=args.min_clone_size,
                min_instances=args.min_instances,
            )

            # Find Python files
            python_files = []
            for pattern in args.include_patterns:
                python_files.extend(glob.glob(str(project_path / pattern), recursive=True))

            # Filter out excluded patterns
            for exclude_pattern in args.exclude_patterns:
                excluded_files = set(glob.glob(str(project_path / exclude_pattern), recursive=True))
                python_files = [f for f in python_files if f not in excluded_files]

            python_files = [Path(f) for f in python_files if Path(f).is_file()]

            if not python_files:
                print("No Python files found matching the specified patterns.")
                return

            print(f"Analyzing {len(python_files)} Python files for block clones...")

            # Extract blocks from all files
            all_blocks = []
            for file_path in python_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source_code = f.read()

                    blocks = extractor.extract_blocks(source_code, str(file_path))
                    all_blocks.extend(blocks)

                except Exception as e:
                    print(f"Warning: Failed to analyze {file_path}: {e}", file=sys.stderr)
                    continue

            print(f"Extracted {len(all_blocks)} code blocks")

            # Detect clones
            clone_groups = detector.detect_clones(all_blocks)

            # Generate output
            output = format_clone_detection_results(
                clone_groups,
                detector.get_clone_statistics(clone_groups),
                args.format,
                args.show_code,
                args.group_by,
            )

            if args.output:
                with open(args.output, "w") as f:
                    f.write(output)
                print(f"Clone detection results written to {args.output}")
            else:
                print(output)

        elif args.duplicates_action == "methods":
            # Method-level clone detection
            print("Method-level clone detection is not yet implemented.")
            print("Use 'duplicates blocks' for block-level clone detection.")
            sys.exit(1)

        elif args.duplicates_action == "similar":
            # Semantic similarity matching
            from intellirefactor.analysis.semantic_similarity_matcher import (
                SemanticSimilarityMatcher,
                SimilarityType,
            )
            from intellirefactor.analysis.index_builder import IndexBuilder
            from intellirefactor.analysis.index_store import IndexStore

            project_path = Path(args.project_path)
            if not project_path.exists():
                print(
                    f"Error: Project path does not exist: {project_path}",
                    file=sys.stderr,
                )
                sys.exit(1)

            if not project_path.is_dir():
                print(
                    f"Error: Project path must be a directory: {project_path}",
                    file=sys.stderr,
                )
                sys.exit(1)

            print(f"Analyzing semantic similarity in project: {project_path}")

            # Build or load index to get method information
            index_db_path = project_path / ".intellirefactor" / "index.db"
            index_store = IndexStore(str(index_db_path))

            # Check if index exists and is recent
            if not index_db_path.exists():
                print("Building project index for semantic analysis...")
                index_builder = IndexBuilder(str(index_db_path))
                index_result = index_builder.build_index(project_path, incremental=False)
                print(f"Index built: {index_result.symbols_found} symbols found")

            # Get method information from index
            methods = index_store.get_all_deep_method_infos()

            if not methods:
                print("No methods found in the project index.")
                print("Try running with different include/exclude patterns.")
                return

            print(f"Found {len(methods)} methods for similarity analysis")

            # Initialize similarity matcher
            similarity_types = set()
            for sim_type in args.similarity_types:
                similarity_types.add(SimilarityType(sim_type.lower()))

            matcher = SemanticSimilarityMatcher(
                structural_threshold=args.structural_threshold,
                functional_threshold=args.functional_threshold,
                behavioral_threshold=args.behavioral_threshold,
                min_confidence=args.min_confidence,
            )

            # Find target method if specified
            target_method = None
            if args.target_method:
                target_method = next(
                    (m for m in methods if m.qualified_name == args.target_method), None
                )
                if not target_method:
                    print(f"Target method '{args.target_method}' not found.")
                    print("Available methods:")
                    for method in methods[:10]:  # Show first 10
                        print(f"  - {method.qualified_name}")
                    if len(methods) > 10:
                        print(f"  ... and {len(methods) - 10} more")
                    return

            # Find similar methods
            matches = matcher.find_similar_methods(
                methods, target_method=target_method, similarity_types=similarity_types
            )

            # Limit results
            if args.max_results and len(matches) > args.max_results:
                matches = matches[: args.max_results]

            # Convert SimilarityMatch objects to expected format
            formatted_results = []
            if target_method:
                # Single target method mode
                similar_methods = []
                for match in matches:
                    similar_method = (
                        match.method2 if match.method1 == target_method else match.method1
                    )
                    similar_methods.append(
                        {
                            "method": {
                                "name": similar_method.name,
                                "file_path": similar_method.file_reference.file_path,
                                "line_start": similar_method.file_reference.line_start,
                                "line_end": similar_method.file_reference.line_end,
                            },
                            "similarity_score": match.similarity_score,
                            "confidence": match.confidence,
                            "similarity_type": match.similarity_type.value,
                            "evidence": [match.evidence.description] if match.evidence else [],
                            "differences": match.differences,
                            "merge_recommendation": (
                                {
                                    "strategy": match.merge_strategy or "manual",
                                    "effort": "medium",
                                    "description": f"Consider merging similar {match.similarity_type.value} functionality",
                                }
                                if match.merge_strategy
                                else None
                            ),
                        }
                    )

                formatted_results.append(
                    {
                        "target_method": {
                            "name": target_method.name,
                            "file_path": target_method.file_reference.file_path,
                            "line_start": target_method.file_reference.line_start,
                            "line_end": target_method.file_reference.line_end,
                        },
                        "similar_methods": similar_methods,
                    }
                )
            else:
                # All pairs mode - group by first method
                from collections import defaultdict

                grouped_matches = defaultdict(list)

                for match in matches:
                    key = match.method1.qualified_name
                    grouped_matches[key].append(match)

                for method_name, method_matches in grouped_matches.items():
                    if not method_matches:
                        continue

                    target = method_matches[0].method1
                    similar_methods = []

                    for match in method_matches:
                        similar_method = match.method2
                        similar_methods.append(
                            {
                                "method": {
                                    "name": similar_method.name,
                                    "file_path": similar_method.file_reference.file_path,
                                    "line_start": similar_method.file_reference.line_start,
                                    "line_end": similar_method.file_reference.line_end,
                                },
                                "similarity_score": match.similarity_score,
                                "confidence": match.confidence,
                                "similarity_type": match.similarity_type.value,
                                "evidence": [match.evidence.description] if match.evidence else [],
                                "differences": match.differences,
                                "merge_recommendation": (
                                    {
                                        "strategy": match.merge_strategy or "manual",
                                        "effort": "medium",
                                        "description": f"Consider merging similar {match.similarity_type.value} functionality",
                                    }
                                    if match.merge_strategy
                                    else None
                                ),
                            }
                        )

                    formatted_results.append(
                        {
                            "target_method": {
                                "name": target.name,
                                "file_path": target.file_reference.file_path,
                                "line_start": target.file_reference.line_start,
                                "line_end": target.file_reference.line_end,
                            },
                            "similar_methods": similar_methods,
                        }
                    )

            # Generate output
            output = format_similarity_results(
                formatted_results,
                args.format,
                args.show_evidence,
                args.show_differences,
                args.merge_recommendations,
            )

            if args.output:
                with open(args.output, "w") as f:
                    f.write(output)
                print(f"Similarity results written to {args.output}")
            else:
                print(output)

    except Exception as e:
        print(f"Error: Failed to detect clones: {e}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)


def format_unused_code_results(
    analysis_result,
    format_type: str,
    show_evidence: bool = False,
    show_usage: bool = False,
    group_by: str = "type",
) -> str:
    """Format unused code detection results for output."""

    if format_type == "json":
        return json.dumps(analysis_result.to_dict(), indent=2)

    elif format_type == "html":
        # HTML format for rich display
        html_parts = []
        html_parts.append("<html><head><title>Unused Code Detection Results</title>")
        html_parts.append("<style>")
        html_parts.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html_parts.append(".finding { border: 1px solid #ccc; margin: 10px 0; padding: 10px; }")
        html_parts.append(".evidence { background: #f5f5f5; margin: 5px 0; padding: 5px; }")
        html_parts.append(".usage { background: #f0f0f0; padding: 5px; font-family: monospace; }")
        html_parts.append("</style></head><body>")

        html_parts.append("<h1>Unused Code Detection Results</h1>")
        html_parts.append(f"<p>Found {len(analysis_result.findings)} unused code findings</p>")
        html_parts.append(f"<p>Entry points: {', '.join(analysis_result.entry_points)}</p>")

        for finding in analysis_result.findings:
            html_parts.append('<div class="finding">')
            html_parts.append(
                f"<h3>{finding.unused_type.value.replace('_', ' ').title()}: {finding.symbol_name}</h3>"
            )
            html_parts.append(
                f"<p><strong>File:</strong> {finding.file_path}:{finding.line_start}-{finding.line_end}</p>"
            )
            html_parts.append(
                f"<p><strong>Confidence:</strong> {finding.confidence:.3f} ({finding.confidence_level.value})</p>"
            )
            html_parts.append(f"<p><strong>Severity:</strong> {finding.severity}</p>")

            if show_evidence:
                html_parts.append('<div class="evidence">')
                html_parts.append(f"<strong>Evidence:</strong> {finding.evidence.description}")
                html_parts.append("</div>")

            html_parts.append("</div>")

        html_parts.append("</body></html>")
        return "\n".join(html_parts)

    else:  # text format
        output = []
        output.append("Unused Code Detection Results")
        output.append("=" * 50)
        output.append(f"Found {len(analysis_result.findings)} unused code findings")
        output.append(f"Entry points: {', '.join(analysis_result.entry_points)}")
        output.append(
            f"Analysis confidence threshold: {analysis_result.analysis_metadata.get('confidence_threshold', 'N/A')}"
        )
        output.append("")

        # Statistics
        stats = analysis_result.statistics
        output.append("Statistics:")
        output.append(f"  Total findings: {stats['total_findings']}")
        output.append(f"  Files analyzed: {stats['total_files_analyzed']}")
        output.append(f"  High confidence: {stats['findings_by_confidence']['high']}")
        output.append(f"  Medium confidence: {stats['findings_by_confidence']['medium']}")
        output.append(f"  Low confidence: {stats['findings_by_confidence']['low']}")
        output.append("")

        # Group results based on group_by parameter
        if group_by == "type":
            grouped = {}
            for finding in analysis_result.findings:
                unused_type = finding.unused_type.value
                if unused_type not in grouped:
                    grouped[unused_type] = []
                grouped[unused_type].append(finding)
        elif group_by == "file":
            grouped = {}
            for finding in analysis_result.findings:
                file_path = finding.file_path
                if file_path not in grouped:
                    grouped[file_path] = []
                grouped[file_path].append(finding)
        else:  # confidence
            grouped = {"high": [], "medium": [], "low": []}
            for finding in analysis_result.findings:
                grouped[finding.confidence_level.value].append(finding)

        for group_key, findings in grouped.items():
            if not findings:
                continue

            output.append(f"{group_key.replace('_', ' ').title()} ({len(findings)} findings):")
            output.append("-" * 40)

            for finding in findings:
                output.append(f"  Symbol: {finding.symbol_name}")
                output.append(f"  Type: {finding.unused_type.value.replace('_', ' ')}")
                output.append(
                    f"  Location: {finding.file_path}:{finding.line_start}-{finding.line_end}"
                )
                output.append(
                    f"  Confidence: {finding.confidence:.3f} ({finding.confidence_level.value})"
                )
                output.append(f"  Severity: {finding.severity}")

                if show_evidence:
                    output.append(f"  Evidence: {finding.evidence.description}")
                    if finding.evidence.metadata:
                        for key, value in finding.evidence.metadata.items():
                            if key not in ["dynamic_patterns"]:  # Skip verbose metadata
                                output.append(f"    {key}: {value}")

                if show_usage and finding.usage_references:
                    output.append(f"  Usage references ({len(finding.usage_references)}):")
                    for ref in finding.usage_references[:3]:  # Show first 3
                        output.append(
                            f"    {ref.file_path}:{ref.line_number} ({ref.pattern.value})"
                        )
                    if len(finding.usage_references) > 3:
                        output.append(f"    ... and {len(finding.usage_references) - 3} more")

                if finding.dynamic_usage_indicators:
                    output.append(
                        f"  Dynamic usage indicators: {len(finding.dynamic_usage_indicators)}"
                    )

                output.append("")

            output.append("")

        return "\n".join(output)


def cmd_unused(args) -> None:
    """Handle unused command."""
    from intellirefactor.analysis.unused_code_detector import (
        UnusedCodeDetector,
        UnusedCodeType,
    )

    project_path = Path(args.project_path)
    if not project_path.exists():
        print(f"Error: Project path does not exist: {project_path}", file=sys.stderr)
        sys.exit(1)

    if not project_path.is_dir():
        print(f"Error: Project path must be a directory: {project_path}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.unused_action == "detect":
            print(f"Analyzing {project_path} for unused code...")

            # Initialize detector
            detector = UnusedCodeDetector(project_path)

            # Run analysis
            result = detector.detect_unused_code(
                entry_points=args.entry_points,
                include_patterns=args.include_patterns,
                exclude_patterns=args.exclude_patterns,
                min_confidence=args.min_confidence,
            )

            # Filter by level if specified
            if args.level != "all":
                level_filters = {
                    "1": [
                        UnusedCodeType.MODULE_UNREACHABLE,
                        UnusedCodeType.TESTS_ONLY,
                        UnusedCodeType.SCRIPTS_ONLY,
                    ],
                    "2": [
                        UnusedCodeType.SYMBOL_UNUSED,
                        UnusedCodeType.PRIVATE_METHOD_UNUSED,
                        UnusedCodeType.PUBLIC_EXPORT_UNUSED,
                    ],
                    "3": [UnusedCodeType.UNCERTAIN_DYNAMIC],
                }
                allowed_types = level_filters.get(args.level, [])
                result.findings = [f for f in result.findings if f.unused_type in allowed_types]
                # Update statistics
                result.statistics = detector._generate_statistics(result.findings, [])

            # Filter by type if specified
            if args.filter_type:
                filter_type = UnusedCodeType(args.filter_type)
                result.findings = [f for f in result.findings if f.unused_type == filter_type]
                # Update statistics
                result.statistics = detector._generate_statistics(result.findings, [])

            print(f"Found {len(result.findings)} unused code findings")

            # Generate output
            output = format_unused_code_results(
                result, args.format, args.show_evidence, args.show_usage, args.group_by
            )

            if args.output:
                with open(args.output, "w") as f:
                    f.write(output)
                print(f"Unused code detection results written to {args.output}")
            else:
                print(output)

    except Exception as e:
        print(f"Error: Failed to detect unused code: {e}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)


def format_audit_results(audit_result, format_type: str) -> str:
    """Format audit results for output."""
    from intellirefactor.analysis.audit_models import AuditSeverity

    if format_type == "json":
        return json.dumps(audit_result.to_dict(), indent=2)

    elif format_type == "html":
        # HTML format for rich display
        html_parts = []
        html_parts.append("<html><head><title>Project Audit Results</title>")
        html_parts.append("<style>")
        html_parts.append("body { font-family: Arial, sans-serif; margin: 20px; }")
        html_parts.append(".finding { border: 1px solid #ccc; margin: 10px 0; padding: 10px; }")
        html_parts.append(".critical { border-color: #d32f2f; background: #ffebee; }")
        html_parts.append(".high { border-color: #f57c00; background: #fff3e0; }")
        html_parts.append(".medium { border-color: #fbc02d; background: #fffde7; }")
        html_parts.append(".low { border-color: #388e3c; background: #e8f5e8; }")
        html_parts.append(".evidence { background: #f5f5f5; margin: 5px 0; padding: 5px; }")
        html_parts.append("</style></head><body>")

        html_parts.append("<h1>Project Audit Results</h1>")
        html_parts.append(f"<p><strong>Project:</strong> {audit_result.project_path}</p>")
        html_parts.append(f"<p><strong>Total Findings:</strong> {len(audit_result.findings)}</p>")
        html_parts.append(
            f"<p><strong>Files Analyzed:</strong> {audit_result.statistics.files_analyzed}</p>"
        )
        html_parts.append(
            f"<p><strong>Analysis Time:</strong> {audit_result.statistics.analysis_time_seconds:.2f}s</p>"
        )

        # Group findings by severity
        for severity in [
            AuditSeverity.CRITICAL,
            AuditSeverity.HIGH,
            AuditSeverity.MEDIUM,
            AuditSeverity.LOW,
        ]:
            severity_findings = audit_result.get_findings_by_severity(severity)
            if not severity_findings:
                continue

            html_parts.append(
                f"<h2>{severity.value.title()} Priority ({len(severity_findings)} findings)</h2>"
            )

            for finding in severity_findings:
                html_parts.append(f'<div class="finding {severity.value}">')
                html_parts.append(f"<h3>{finding.title}</h3>")
                html_parts.append(
                    f"<p><strong>Type:</strong> {finding.finding_type.value.replace('_', ' ').title()}</p>"
                )
                html_parts.append(
                    f"<p><strong>Location:</strong> {finding.file_path}:{finding.line_start}-{finding.line_end}</p>"
                )
                html_parts.append(f"<p><strong>Confidence:</strong> {finding.confidence:.3f}</p>")
                html_parts.append(f"<p><strong>Description:</strong> {finding.description}</p>")

                if finding.evidence.description:
                    html_parts.append('<div class="evidence">')
                    html_parts.append(f"<strong>Evidence:</strong> {finding.evidence.description}")
                    html_parts.append("</div>")

                if finding.recommendations:
                    html_parts.append("<p><strong>Recommendations:</strong></p>")
                    html_parts.append("<ul>")
                    for rec in finding.recommendations:
                        html_parts.append(f"<li>{rec}</li>")
                    html_parts.append("</ul>")

                html_parts.append("</div>")

        html_parts.append("</body></html>")
        return "\n".join(html_parts)

    else:  # text format
        output = []
        output.append("Project Audit Results")
        output.append("=" * 50)
        output.append(f"Project: {audit_result.project_path}")
        output.append(f"Total findings: {len(audit_result.findings)}")
        output.append(f"Files analyzed: {audit_result.statistics.files_analyzed}")
        output.append(f"Analysis time: {audit_result.statistics.analysis_time_seconds:.2f}s")
        output.append("")

        # Statistics
        stats = audit_result.statistics
        output.append("Statistics:")

        # Severity distribution
        output.append("  Findings by Severity:")
        for severity, count in stats.findings_by_severity.items():
            if count > 0:
                emoji = {
                    "critical": "[CRITICAL]",
                    "high": "[HIGH]",
                    "medium": "[MEDIUM]",
                    "low": "[LOW]",
                    "info": "[INFO]",
                }.get(severity, "")
                output.append(f"    {emoji} {severity.title()}: {count}")
        output.append("")

        # Type distribution
        output.append("  Findings by Type:")
        for finding_type, count in stats.findings_by_type.items():
            if count > 0:
                output.append(f"    {finding_type.replace('_', ' ').title()}: {count}")
        output.append("")

        # Confidence distribution
        output.append("  Confidence Distribution:")
        conf_dist = stats.confidence_distribution
        output.append(f"    High (>=80%): {conf_dist.get('high', 0)}")
        output.append(f"    Medium (50-79%): {conf_dist.get('medium', 0)}")
        output.append(f"    Low (<50%): {conf_dist.get('low', 0)}")
        output.append("")

        # Summary by analysis type
        if audit_result.index_result:
            output.append("Index Analysis:")
            idx = audit_result.index_result
            output.append(f"  Files processed: {idx.files_processed}")
            output.append(f"  Symbols found: {idx.symbols_found}")
            output.append(f"  Build time: {idx.build_time_seconds:.2f}s")
            output.append("")

        if audit_result.clone_groups:
            output.append("Duplicate Code Analysis:")
            output.append(f"  Clone groups found: {len(audit_result.clone_groups)}")
            total_instances = sum(len(group.instances) for group in audit_result.clone_groups)
            output.append(f"  Total instances: {total_instances}")
            output.append("")

        if audit_result.unused_result:
            output.append("Unused Code Analysis:")
            unused_stats = audit_result.unused_result.statistics
            output.append(f"  Unused items found: {unused_stats.get('total_findings', 0)}")
            output.append(f"  Entry points: {len(audit_result.unused_result.entry_points)}")
            output.append("")

        # Detailed findings by severity
        for severity in [
            AuditSeverity.CRITICAL,
            AuditSeverity.HIGH,
            AuditSeverity.MEDIUM,
            AuditSeverity.LOW,
        ]:
            severity_findings = audit_result.get_findings_by_severity(severity)
            if not severity_findings:
                continue

            emoji = {
                "critical": "[CRITICAL]",
                "high": "[HIGH]",
                "medium": "[MEDIUM]",
                "low": "[LOW]",
            }.get(severity.value, "")
            output.append(
                f"{emoji} {severity.value.title()} Priority Findings ({len(severity_findings)}):"
            )
            output.append("-" * 40)

            for finding in severity_findings:
                output.append(f"  {finding.title}")
                output.append(f"  Type: {finding.finding_type.value.replace('_', ' ')}")
                output.append(
                    f"  Location: {Path(finding.file_path).name}:{finding.line_start}-{finding.line_end}"
                )
                output.append(f"  Confidence: {finding.confidence:.1%}")
                output.append(f"  Description: {finding.description}")

                if finding.recommendations:
                    output.append("  Recommendations:")
                    for rec in finding.recommendations[:2]:  # Show first 2
                        output.append(f"    - {rec}")
                    if len(finding.recommendations) > 2:
                        output.append(f"    ... and {len(finding.recommendations) - 2} more")

                if finding.related_findings:
                    output.append(f"  Related: {', '.join(finding.related_findings)}")

                output.append("")

            output.append("")

        # Summary recommendations
        if len(audit_result.findings) > 0:
            output.append("Next Steps:")
            critical_count = stats.findings_by_severity.get("critical", 0)
            high_count = stats.findings_by_severity.get("high", 0)

            if critical_count > 0:
                output.append(
                    f"  1. [CRITICAL] Address {critical_count} critical issues immediately"
                )
            if high_count > 0:
                output.append(f"  2. [HIGH] Plan to fix {high_count} high priority issues")

            output.append("  3. [MEDIUM] Review medium priority issues for future sprints")
            output.append("  4. [LOW] Consider low priority improvements when time permits")
            output.append("")
            output.append("Use --emit-spec to generate detailed Requirements.md document")
        else:
            output.append("[SUCCESS] Excellent! No issues found in your codebase.")
            output.append("Your project appears to be well-maintained and follows good practices.")

        output.append("")

        return "\n".join(output)


def cmd_audit(args) -> None:
    """Handle audit command."""
    from intellirefactor.analysis.audit_engine import AuditEngine
    from intellirefactor.analysis.spec_generator import SpecGenerator

    project_path = Path(args.project_path)
    if not project_path.exists():
        print(f"Error: Project path does not exist: {project_path}", file=sys.stderr)
        sys.exit(1)

    if not project_path.is_dir():
        print(f"Error: Project path must be a directory: {project_path}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Starting comprehensive audit of {project_path}...")

        # Initialize audit engine
        audit_engine = AuditEngine(project_path)

        # Run comprehensive audit
        result = audit_engine.run_full_audit(
            include_index=not args.skip_index,
            include_duplicates=not args.skip_duplicates,
            include_unused=not args.skip_unused,
            generate_spec=args.emit_spec,
            min_confidence=args.min_confidence,
            include_patterns=args.include_patterns,
            exclude_patterns=args.exclude_patterns,
        )

        print(
            f"Audit completed: {len(result.findings)} findings in {result.statistics.analysis_time_seconds:.2f}s"
        )

        # Generate main output
        output = format_audit_results(result, args.format)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Audit results written to {args.output}")
        else:
            print(output)

        # Generate specification if requested (and not already generated)
        if args.emit_spec and not result.analysis_metadata.get("spec_generated"):
            spec_generator = SpecGenerator()

            # Determine output path
            if args.spec_output:
                spec_base_path = Path(args.spec_output)
                if spec_base_path.suffix == ".md":
                    # Single file specified - generate Requirements.md only
                    spec_content = spec_generator.generate_requirements_from_audit(result)
                    with open(spec_base_path, "w", encoding="utf-8") as f:
                        f.write(spec_content)
                    print(f"Requirements specification written to {spec_base_path}")
                else:
                    # Directory specified - generate all specifications
                    spec_base_path.mkdir(parents=True, exist_ok=True)

                    # Generate Requirements.md
                    requirements_content = spec_generator.generate_requirements_from_audit(result)
                    requirements_path = spec_base_path / "Requirements.md"
                    with open(requirements_path, "w", encoding="utf-8") as f:
                        f.write(requirements_content)
                    print(f"Requirements specification written to {requirements_path}")

                    # Generate Design.md
                    design_content = spec_generator.generate_design_from_audit(result)
                    design_path = spec_base_path / "Design.md"
                    with open(design_path, "w", encoding="utf-8") as f:
                        f.write(design_content)
                    print(f"Design specification written to {design_path}")

                    # Generate Implementation.md
                    implementation_content = spec_generator.generate_implementation_from_audit(
                        result
                    )
                    implementation_path = spec_base_path / "Implementation.md"
                    with open(implementation_path, "w", encoding="utf-8") as f:
                        f.write(implementation_content)
                    print(f"Implementation specification written to {implementation_path}")
            else:
                # Default: generate Requirements.md in project directory
                spec_path = project_path / "AUDIT_REQUIREMENTS.md"
                spec_content = spec_generator.generate_requirements_from_audit(result)
                with open(spec_path, "w", encoding="utf-8") as f:
                    f.write(spec_content)
                print(f"Requirements specification written to {spec_path}")

        # Generate JSON artifacts if requested
        if args.emit_json:
            spec_generator = SpecGenerator()
            json_path = (
                Path(args.json_output) if args.json_output else project_path / "analysis.json"
            )

            # Generate comprehensive machine-readable analysis
            json_data = spec_generator.generate_machine_readable_analysis(result)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, default=str)
            print(f"Analysis JSON written to {json_path}")

        # Exit with error code if critical issues found
        critical_findings = result.get_critical_findings()
        if critical_findings:
            print(
                f"\nWARNING: Found {len(critical_findings)} critical issues that require immediate attention!"
            )
            sys.exit(1)

    except Exception as e:
        print(f"Error: Failed to run audit: {e}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)


def cmd_smells(args) -> None:
    """Handle smells command."""
    from intellirefactor.analysis.architectural_smell_detector import (
        ArchitecturalSmellDetector,
        SmellThresholds,
        SmellType,
        SmellSeverity,
    )
    import glob
    import time

    if args.smells_action == "detect":
        print(f"Starting architectural smell detection for {args.project_path}...")
        start_time = time.time()

        # Create custom thresholds from command line arguments
        thresholds = SmellThresholds(
            god_class_methods=args.god_class_methods,
            god_class_responsibilities=args.god_class_responsibilities,
            long_method_lines=args.long_method_lines,
            high_complexity_cyclomatic=args.high_complexity_cyclomatic,
        )

        # Initialize detector
        detector = ArchitecturalSmellDetector(thresholds)

        # Find Python files
        project_path = Path(args.project_path)
        if not project_path.exists():
            print(f"Error: Project path '{project_path}' does not exist.")
            sys.exit(1)

        python_files = []
        for pattern in args.include_patterns:
            python_files.extend(glob.glob(str(project_path / pattern), recursive=True))

        # Filter out excluded patterns
        for exclude_pattern in args.exclude_patterns:
            excluded_files = set(glob.glob(str(project_path / exclude_pattern), recursive=True))
            python_files = [f for f in python_files if f not in excluded_files]

        python_files = [Path(f) for f in python_files if Path(f).is_file() and f.endswith(".py")]

        if not python_files:
            print("No Python files found matching the specified patterns.")
            return

        print(f"Analyzing {len(python_files)} Python files...")

        # Detect smells in all files
        all_smells = []
        processed_files = 0

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source_code = f.read()

                smells = detector.detect_smells(source_code, str(file_path))

                # Filter by smell types if specified
                if args.smell_types:
                    smell_type_filter = {SmellType(t) for t in args.smell_types}
                    smells = [s for s in smells if s.smell_type in smell_type_filter]

                # Filter by severity if specified
                if args.severity:
                    min_severity = SmellSeverity(args.severity)
                    severity_order = [
                        SmellSeverity.LOW,
                        SmellSeverity.MEDIUM,
                        SmellSeverity.HIGH,
                        SmellSeverity.CRITICAL,
                    ]
                    min_index = severity_order.index(min_severity)
                    smells = [s for s in smells if severity_order.index(s.severity) >= min_index]

                # Filter by confidence
                smells = [s for s in smells if s.confidence >= args.min_confidence]

                all_smells.extend(smells)
                processed_files += 1

            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
                continue

        analysis_time = time.time() - start_time

        print(f"Smell detection completed: {len(all_smells)} smells found in {analysis_time:.2f}s")

        # Format and output results
        if args.format == "json":
            output = format_smells_json(all_smells, args, analysis_time, processed_files)
        elif args.format == "html":
            output = format_smells_html(all_smells, args, analysis_time, processed_files)
        else:
            output = format_smells_text(all_smells, args, analysis_time, processed_files)

        # Write output
        if args.output:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Results written to {args.output}")
        else:
            print(output)

        # Exit with error code if critical smells found (but not for JSON output to avoid breaking parsing)
        critical_smells = [s for s in all_smells if s.severity == SmellSeverity.CRITICAL]
        if critical_smells and args.format != "json":
            print(
                f"\nWARNING: Found {len(critical_smells)} critical architectural smells that require immediate attention!"
            )
            sys.exit(1)
        elif critical_smells and args.format == "json":
            # For JSON output, exit with error code but don't print warning to avoid breaking JSON
            sys.exit(1)


def format_smells_text(smells, args, analysis_time, processed_files):
    """Format smell detection results as text."""
    output = []

    # Header
    output.append("Architectural Smell Detection Results")
    output.append("=" * 50)
    output.append(f"Project: {args.project_path}")
    output.append(f"Total smells: {len(smells)}")
    output.append(f"Files analyzed: {processed_files}")
    output.append(f"Analysis time: {analysis_time:.2f}s")
    output.append("")

    if not smells:
        output.append("[SUCCESS] No architectural smells detected!")
        output.append("Your code appears to follow good architectural practices.")
        return "\n".join(output)

    # Statistics
    output.append("Statistics:")

    # By severity
    severity_counts = {}
    for smell in smells:
        severity_counts[smell.severity.value] = severity_counts.get(smell.severity.value, 0) + 1

    output.append("  By Severity:")
    for severity in ["critical", "high", "medium", "low"]:
        count = severity_counts.get(severity, 0)
        if count > 0:
            icon = {
                "critical": "CRITICAL",
                "high": "HIGH",
                "medium": "MEDIUM",
                "low": "LOW",
            }[severity]
            output.append(f"    {icon}: {count}")

    # By type
    type_counts = {}
    for smell in smells:
        type_counts[smell.smell_type.value] = type_counts.get(smell.smell_type.value, 0) + 1

    output.append("  By Type:")
    for smell_type, count in type_counts.items():
        output.append(f"    {smell_type.replace('_', ' ').title()}: {count}")

    output.append("")

    # Group smells
    if args.group_by == "file":
        grouped_smells = {}
        for smell in smells:
            file_path = smell.file_path
            if file_path not in grouped_smells:
                grouped_smells[file_path] = []
            grouped_smells[file_path].append(smell)

        for file_path, file_smells in grouped_smells.items():
            output.append(f"File: {file_path}")
            output.append("-" * 40)
            for smell in file_smells:
                output.extend(_format_single_smell_text(smell, args))
            output.append("")

    elif args.group_by == "type":
        grouped_smells = {}
        for smell in smells:
            smell_type = smell.smell_type.value
            if smell_type not in grouped_smells:
                grouped_smells[smell_type] = []
            grouped_smells[smell_type].append(smell)

        for smell_type, type_smells in grouped_smells.items():
            output.append(f"Smell Type: {smell_type.replace('_', ' ').title()}")
            output.append("-" * 40)
            for smell in type_smells:
                output.extend(_format_single_smell_text(smell, args))
            output.append("")

    elif args.group_by == "severity":
        severity_order = ["critical", "high", "medium", "low"]
        for severity in severity_order:
            severity_smells = [s for s in smells if s.severity.value == severity]
            if severity_smells:
                output.append(f"Severity: {severity.upper()}")
                output.append("-" * 40)
                for smell in severity_smells:
                    output.extend(_format_single_smell_text(smell, args))
                output.append("")

    return "\n".join(output)


def _format_single_smell_text(smell, args):
    """Format a single smell for text output."""
    lines = []

    # Basic info
    lines.append(f"  {smell.smell_type.value.replace('_', ' ').title()}")
    lines.append(f"  Symbol: {smell.symbol_name}")
    lines.append(f"  Location: {smell.file_path}:{smell.line_start}-{smell.line_end}")
    lines.append(f"  Severity: {smell.severity.value.upper()}")
    lines.append(f"  Confidence: {smell.confidence:.1%}")

    # Metrics
    if smell.metrics:
        lines.append("  Metrics:")
        for key, value in smell.metrics.items():
            if isinstance(value, float):
                lines.append(f"    {key}: {value:.3f}")
            else:
                lines.append(f"    {key}: {value}")

    # Evidence
    if args.show_evidence and smell.evidence:
        lines.append("  Evidence:")
        lines.append(f"    {smell.evidence.description}")
        if smell.evidence.metadata:
            for key, value in smell.evidence.metadata.items():
                if key not in ["responsibilities"]:  # Skip complex metadata
                    lines.append(f"    {key}: {value}")

    # Recommendations
    if args.show_recommendations and smell.recommendations:
        lines.append("  Recommendations:")
        for rec in smell.recommendations:
            lines.append(f"    - {rec}")

    # Decomposition strategy for God Classes
    if smell.decomposition_strategy:
        lines.append(f"  Decomposition Strategy: {smell.decomposition_strategy}")

    lines.append("")
    return lines


def format_smells_json(smells, args, analysis_time, processed_files):
    """Format smell detection results as JSON."""
    result = {
        "project_path": args.project_path,
        "analysis_metadata": {
            "total_smells": len(smells),
            "files_analyzed": processed_files,
            "analysis_time_seconds": analysis_time,
            "min_confidence": args.min_confidence,
            "include_patterns": args.include_patterns,
            "exclude_patterns": args.exclude_patterns,
        },
        "statistics": {"by_severity": {}, "by_type": {}},
        "smells": [],
    }

    # Calculate statistics
    for smell in smells:
        severity = smell.severity.value
        smell_type = smell.smell_type.value

        result["statistics"]["by_severity"][severity] = (
            result["statistics"]["by_severity"].get(severity, 0) + 1
        )
        result["statistics"]["by_type"][smell_type] = (
            result["statistics"]["by_type"].get(smell_type, 0) + 1
        )

    # Convert smells to dictionaries
    for smell in smells:
        smell_dict = {
            "smell_type": smell.smell_type.value,
            "severity": smell.severity.value,
            "confidence": smell.confidence,
            "file_path": smell.file_path,
            "symbol_name": smell.symbol_name,
            "line_start": smell.line_start,
            "line_end": smell.line_end,
            "metrics": smell.metrics,
            "evidence": {
                "description": smell.evidence.description,
                "confidence": smell.evidence.confidence,
                "metadata": smell.evidence.metadata,
            },
            "recommendations": smell.recommendations,
        }

        if smell.decomposition_strategy:
            smell_dict["decomposition_strategy"] = smell.decomposition_strategy

        if smell.extraction_candidates:
            smell_dict["extraction_candidates"] = smell.extraction_candidates

        result["smells"].append(smell_dict)

    return json.dumps(result, indent=2)


def format_smells_html(smells, args, analysis_time, processed_files):
    """Format smell detection results as HTML."""
    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Architectural Smell Detection Results</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; }}
        .statistics {{ margin: 20px 0; }}
        .smell {{ border: 1px solid #ddd; margin: 10px 0; padding: 15px; border-radius: 5px; }}
        .critical {{ border-left: 5px solid #d32f2f; }}
        .high {{ border-left: 5px solid #f57c00; }}
        .medium {{ border-left: 5px solid #fbc02d; }}
        .low {{ border-left: 5px solid #388e3c; }}
        .metrics {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
        .recommendations {{ background-color: #e3f2fd; padding: 10px; margin: 10px 0; }}
        ul {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Architectural Smell Detection Results</h1>
        <p><strong>Project:</strong> {args.project_path}</p>
        <p><strong>Total smells:</strong> {len(smells)}</p>
        <p><strong>Files analyzed:</strong> {processed_files}</p>
        <p><strong>Analysis time:</strong> {analysis_time:.2f}s</p>
    </div>
"""

    if not smells:
        html += """
    <div class="success">
        <h2>Success!</h2>
        <p>No architectural smells detected. Your code appears to follow good architectural practices.</p>
    </div>
</body>
</html>
"""
        return html

    # Statistics
    html += '<div class="statistics"><h2>Statistics</h2>'

    # By severity
    severity_counts = {}
    for smell in smells:
        severity_counts[smell.severity.value] = severity_counts.get(smell.severity.value, 0) + 1

    html += "<h3>By Severity</h3><ul>"
    for severity in ["critical", "high", "medium", "low"]:
        count = severity_counts.get(severity, 0)
        if count > 0:
            html += f"<li><strong>{severity.upper()}:</strong> {count}</li>"
    html += "</ul>"

    # By type
    type_counts = {}
    for smell in smells:
        type_counts[smell.smell_type.value] = type_counts.get(smell.smell_type.value, 0) + 1

    html += "<h3>By Type</h3><ul>"
    for smell_type, count in type_counts.items():
        html += f"<li><strong>{smell_type.replace('_', ' ').title()}:</strong> {count}</li>"
    html += "</ul></div>"

    # Smells
    html += "<h2>Detected Smells</h2>"

    for smell in smells:
        html += f"""
    <div class="smell {smell.severity.value}">
        <h3>{smell.smell_type.value.replace("_", " ").title()}</h3>
        <p><strong>Symbol:</strong> {smell.symbol_name}</p>
        <p><strong>Location:</strong> {smell.file_path}:{smell.line_start}-{smell.line_end}</p>
        <p><strong>Severity:</strong> {smell.severity.value.upper()}</p>
        <p><strong>Confidence:</strong> {smell.confidence:.1%}</p>
        
        <div class="metrics">
            <h4>Metrics</h4>
            <ul>
"""

        for key, value in smell.metrics.items():
            if isinstance(value, float):
                html += f"<li><strong>{key}:</strong> {value:.3f}</li>"
            else:
                html += f"<li><strong>{key}:</strong> {value}</li>"

        html += "</ul></div>"

        if smell.recommendations:
            html += '<div class="recommendations"><h4>Recommendations</h4><ul>'
            for rec in smell.recommendations:
                html += f"<li>{rec}</li>"
            html += "</ul></div>"

        html += "</div>"

    html += "</body></html>"
    return html


def cmd_cluster(args) -> None:
    """Handle cluster command."""
    import time
    from pathlib import Path

    if args.cluster_action == "responsibility":
        start_time = time.time()

        project_path = Path(args.project_path)
        if not project_path.exists():
            print(f"Error: Project path '{project_path}' does not exist.")
            return

        # Create clustering configuration
        config = ClusteringConfig(
            min_confidence=args.min_confidence,
            min_cohesion=args.min_cohesion,
            min_cluster_size=args.min_cluster_size,
            max_cluster_size=args.max_cluster_size,
            algorithm=ClusteringAlgorithm(args.algorithm),
        )

        clusterer = ResponsibilityClusterer(config)

        # Find Python files to analyze
        if args.file_path:
            python_files = [Path(args.file_path)]
        else:
            python_files = list(project_path.rglob("*.py"))

        if not python_files:
            print("No Python files found.")
            return

        print(f"Analyzing {len(python_files)} Python files for responsibility clustering...")

        all_results = []
        all_suggestions = []
        processed_files = 0

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    source_code = f.read()

                # Parse AST to find classes
                import ast

                try:
                    tree = ast.parse(source_code)
                except SyntaxError:
                    continue

                # Find all classes in the file
                classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

                for class_node in classes:
                    # Skip if specific class name is requested and this isn't it
                    if args.class_name and class_node.name != args.class_name:
                        continue

                    # Perform clustering analysis
                    result = clusterer.cluster_class_methods(
                        source_code, class_node.name, str(file_path)
                    )

                    if result and result.clusters:
                        all_results.append(result)

                        # Generate component suggestions if requested
                        if args.show_suggestions:
                            suggestions = clusterer.generate_component_suggestions(result)
                            all_suggestions.extend(suggestions)

                processed_files += 1

            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
                continue

        analysis_time = time.time() - start_time

        print(
            f"Clustering analysis completed: {len(all_results)} classes analyzed in {analysis_time:.2f}s"
        )

        if not all_results:
            print("No clustering opportunities found.")
            return

        # Format and output results
        if args.output == "json":
            output = format_clustering_results_json(all_results, all_suggestions, args)
        elif args.output == "html":
            output = format_clustering_results_html(all_results, all_suggestions, args)
        else:
            output = format_clustering_results_text(all_results, all_suggestions, args)

        # Output to file or stdout
        if args.output_file:
            with open(args.output_file, "w", encoding="utf-8") as f:
                f.write(output)
            print(f"Results written to {args.output_file}")
        else:
            print(output)


def format_clustering_results_text(results, suggestions, args):
    """Format clustering results as text."""
    output = []

    # Group results
    if args.group_by == "confidence":
        results.sort(key=lambda r: r.confidence, reverse=True)
    elif args.group_by == "complexity":
        results.sort(key=lambda r: len(r.clusters), reverse=True)
    else:  # group by class (default)
        results.sort(key=lambda r: r.class_name)

    output.append("=" * 80)
    output.append("RESPONSIBILITY CLUSTERING ANALYSIS")
    output.append("=" * 80)
    output.append("")

    for result in results:
        output.append(f"Class: {result.class_name}")
        output.append(f"File: {result.file_path}")
        output.append(f"Total Methods: {result.total_methods}")
        output.append(f"Clusters Found: {len(result.clusters)}")
        output.append(
            f"Unclustered Methods: {len(result.unclustered_methods)} ({result.unclustered_ratio:.1%})"
        )
        output.append(f"Average Cohesion: {result.average_cohesion:.3f}")
        output.append(f"Silhouette Score: {result.silhouette_score:.3f}")
        output.append(f"Extraction Recommended: {'Yes' if result.extraction_recommended else 'No'}")
        output.append(f"Overall Confidence: {result.confidence:.3f}")
        output.append("")

        if result.clusters:
            output.append("Clusters:")
            for i, cluster in enumerate(result.clusters, 1):
                output.append(f"  {i}. {cluster.suggested_name} (Quality: {cluster.quality.value})")
                output.append(f"     Methods: {', '.join([m.name for m in cluster.methods])}")
                output.append(
                    f"     Responsibilities: {', '.join(cluster.dominant_responsibilities)}"
                )
                output.append(f"     Cohesion: {cluster.cohesion_score:.3f}")
                output.append(f"     Confidence: {cluster.confidence:.3f}")

                if cluster.shared_attributes:
                    output.append(f"     Shared Attributes: {', '.join(cluster.shared_attributes)}")

                if cluster.shared_dependencies:
                    output.append(
                        f"     Shared Dependencies: {', '.join(cluster.shared_dependencies)}"
                    )

                output.append("")

        if result.recommendations:
            output.append("Recommendations:")
            for rec in result.recommendations:
                output.append(f"  - {rec}")
            output.append("")

        output.append("-" * 80)
        output.append("")

    # Show component suggestions if requested
    if args.show_suggestions and suggestions:
        output.append("=" * 80)
        output.append("COMPONENT EXTRACTION SUGGESTIONS")
        output.append("=" * 80)
        output.append("")

        for suggestion in suggestions:
            output.append(f"Suggestion Type: {suggestion.suggestion_type.upper()}")
            output.append(f"Priority: {suggestion.priority.upper()}")
            output.append(f"Component: {suggestion.interface.component_name}")
            output.append(f"Confidence: {suggestion.confidence:.3f}")
            output.append("")
            output.append(f"Rationale: {suggestion.rationale}")
            output.append("")

            if suggestion.benefits:
                output.append("Benefits:")
                for benefit in suggestion.benefits:
                    output.append(f"  + {benefit}")
                output.append("")

            if suggestion.trade_offs:
                output.append("Trade-offs:")
                for trade_off in suggestion.trade_offs:
                    output.append(f"  - {trade_off}")
                output.append("")

            if args.show_interfaces:
                interface = suggestion.interface
                output.append("Interface Definition:")
                output.append(f"  Component Name: {interface.component_name}")
                output.append(f"  Public Methods: {', '.join(interface.public_methods)}")
                if interface.private_methods:
                    output.append(f"  Private Methods: {', '.join(interface.private_methods)}")
                output.append(f"  Required Attributes: {', '.join(interface.required_attributes)}")
                if interface.external_dependencies:
                    output.append(
                        f"  External Dependencies: {', '.join(interface.external_dependencies)}"
                    )
                output.append(f"  Complexity: {interface.complexity.value}")
                output.append("")

            if args.show_plans:
                plan = suggestion.extraction_plan
                output.append("Extraction Plan:")
                output.append(f"  Extraction Type: {plan.extraction_type}")
                output.append(f"  Estimated Effort: {plan.estimated_effort_hours} hours")
                output.append(f"  Complexity: {plan.complexity.value}")

                if plan.risk_factors:
                    output.append("  Risk Factors:")
                    for risk in plan.risk_factors:
                        output.append(f"    - {risk}")

                if plan.implementation_steps:
                    output.append("  Implementation Steps:")
                    for step in plan.implementation_steps:
                        output.append(f"    {step}")

                output.append("")

            output.append("-" * 80)
            output.append("")

    return "\n".join(output)


def format_clustering_results_json(results, suggestions, args):
    """Format clustering results as JSON."""
    import json

    def serialize_result(result):
        return {
            "class_name": result.class_name,
            "file_path": result.file_path,
            "total_methods": result.total_methods,
            "clusters": [serialize_cluster(c) for c in result.clusters],
            "unclustered_methods": [
                {"name": m.name, "line_start": m.line_start} for m in result.unclustered_methods
            ],
            "unclustered_ratio": result.unclustered_ratio,
            "average_cohesion": result.average_cohesion,
            "silhouette_score": result.silhouette_score,
            "extraction_recommended": result.extraction_recommended,
            "confidence": result.confidence,
            "recommendations": result.recommendations,
        }

    def serialize_cluster(cluster):
        return {
            "cluster_id": cluster.cluster_id,
            "suggested_name": cluster.suggested_name,
            "methods": [{"name": m.name, "line_start": m.line_start} for m in cluster.methods],
            "shared_attributes": list(cluster.shared_attributes),
            "shared_dependencies": list(cluster.shared_dependencies),
            "dominant_responsibilities": cluster.dominant_responsibilities,
            "cohesion_score": cluster.cohesion_score,
            "confidence": cluster.confidence,
            "quality": cluster.quality.value,
            "interface_methods": cluster.interface_methods,
            "private_methods": cluster.private_methods,
        }

    def serialize_suggestion(suggestion):
        return {
            "suggestion_type": suggestion.suggestion_type,
            "priority": suggestion.priority,
            "confidence": suggestion.confidence,
            "rationale": suggestion.rationale,
            "benefits": suggestion.benefits,
            "trade_offs": suggestion.trade_offs,
            "interface": {
                "component_name": suggestion.interface.component_name,
                "public_methods": suggestion.interface.public_methods,
                "private_methods": suggestion.interface.private_methods,
                "required_attributes": list(suggestion.interface.required_attributes),
                "external_dependencies": list(suggestion.interface.external_dependencies),
                "complexity": suggestion.interface.complexity.value,
                "cohesion_score": suggestion.interface.cohesion_score,
                "confidence": suggestion.interface.confidence,
            },
            "extraction_plan": {
                "source_class": suggestion.extraction_plan.source_class,
                "target_component": suggestion.extraction_plan.target_component,
                "extraction_type": suggestion.extraction_plan.extraction_type,
                "methods_to_extract": suggestion.extraction_plan.methods_to_extract,
                "attributes_to_extract": list(suggestion.extraction_plan.attributes_to_extract),
                "dependencies_to_inject": list(suggestion.extraction_plan.dependencies_to_inject),
                "complexity": suggestion.extraction_plan.complexity.value,
                "estimated_effort_hours": suggestion.extraction_plan.estimated_effort_hours,
                "risk_factors": suggestion.extraction_plan.risk_factors,
                "implementation_steps": suggestion.extraction_plan.implementation_steps,
                "testing_requirements": suggestion.extraction_plan.testing_requirements,
                "confidence": suggestion.extraction_plan.confidence,
                "expected_cohesion_improvement": suggestion.extraction_plan.expected_cohesion_improvement,
            },
        }

    output = {
        "clustering_results": [serialize_result(r) for r in results],
        "component_suggestions": (
            [serialize_suggestion(s) for s in suggestions] if args.show_suggestions else []
        ),
        "summary": {
            "total_classes_analyzed": len(results),
            "total_clusters_found": sum(len(r.clusters) for r in results),
            "total_suggestions": len(suggestions) if args.show_suggestions else 0,
            "extraction_recommended_count": sum(1 for r in results if r.extraction_recommended),
        },
    }

    return json.dumps(output, indent=2)


def format_clustering_results_html(results, suggestions, args):
    """Format clustering results as HTML."""
    html = """<!DOCTYPE html>
<html>
<head>
    <title>Responsibility Clustering Analysis</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 10px; border-radius: 5px; }
        .result { margin: 20px 0; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
        .cluster { margin: 10px 0; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
        .suggestion { margin: 20px 0; border: 2px solid #007acc; padding: 15px; border-radius: 5px; }
        .high-priority { border-color: #d32f2f; }
        .medium-priority { border-color: #f57c00; }
        .low-priority { border-color: #388e3c; }
        .metrics { display: flex; gap: 20px; margin: 10px 0; }
        .metric { background-color: #e3f2fd; padding: 5px 10px; border-radius: 3px; }
        ul { margin: 5px 0; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Responsibility Clustering Analysis</h1>
        <p>Analysis of method clustering for component extraction opportunities</p>
    </div>
"""

    for result in results:
        html += f"""
    <div class="result">
        <h2>{result.class_name}</h2>
        <p><strong>File:</strong> {result.file_path}</p>
        <div class="metrics">
            <div class="metric">Methods: {result.total_methods}</div>
            <div class="metric">Clusters: {len(result.clusters)}</div>
            <div class="metric">Unclustered: {len(result.unclustered_methods)} ({result.unclustered_ratio:.1%})</div>
            <div class="metric">Cohesion: {result.average_cohesion:.3f}</div>
            <div class="metric">Confidence: {result.confidence:.3f}</div>
        </div>
        <p><strong>Extraction Recommended:</strong> {"Yes" if result.extraction_recommended else "No"}</p>
"""

        if result.clusters:
            html += "<h3>Clusters</h3>"
            for cluster in result.clusters:
                html += f"""
        <div class="cluster">
            <h4>{cluster.suggested_name} ({cluster.quality.value} quality)</h4>
            <p><strong>Methods:</strong> {", ".join([m.name for m in cluster.methods])}</p>
            <p><strong>Responsibilities:</strong> {", ".join(cluster.dominant_responsibilities)}</p>
            <div class="metrics">
                <div class="metric">Cohesion: {cluster.cohesion_score:.3f}</div>
                <div class="metric">Confidence: {cluster.confidence:.3f}</div>
            </div>
"""
                if cluster.shared_attributes:
                    html += f"<p><strong>Shared Attributes:</strong> {', '.join(cluster.shared_attributes)}</p>"
                if cluster.shared_dependencies:
                    html += f"<p><strong>Shared Dependencies:</strong> {', '.join(cluster.shared_dependencies)}</p>"
                html += "</div>"

        if result.recommendations:
            html += "<h3>Recommendations</h3><ul>"
            for rec in result.recommendations:
                html += f"<li>{rec}</li>"
            html += "</ul>"

        html += "</div>"

    # Add suggestions if requested
    if args.show_suggestions and suggestions:
        html += '<div class="header"><h2>Component Extraction Suggestions</h2></div>'

        for suggestion in suggestions:
            priority_class = f"{suggestion.priority}-priority"
            html += f"""
    <div class="suggestion {priority_class}">
        <h3>{suggestion.interface.component_name} ({suggestion.suggestion_type.upper()})</h3>
        <div class="metrics">
            <div class="metric">Priority: {suggestion.priority.upper()}</div>
            <div class="metric">Confidence: {suggestion.confidence:.3f}</div>
            <div class="metric">Complexity: {suggestion.interface.complexity.value}</div>
        </div>
        <p><strong>Rationale:</strong> {suggestion.rationale}</p>
"""

            if suggestion.benefits:
                html += "<h4>Benefits</h4><ul>"
                for benefit in suggestion.benefits:
                    html += f"<li>{benefit}</li>"
                html += "</ul>"

            if suggestion.trade_offs:
                html += "<h4>Trade-offs</h4><ul>"
                for trade_off in suggestion.trade_offs:
                    html += f"<li>{trade_off}</li>"
                html += "</ul>"

            html += "</div>"

    html += "</body></html>"
    return html


def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Setup logging
    setup_logging(getattr(args, "verbose", False))

    # Setup rich output
    use_rich = not getattr(args, "no_rich", False) and not getattr(args, "machine_readable", False)
    set_rich_enabled(use_rich)

    # Handle config commands that don't need IntelliRefactor instance
    if args.command == "config":
        cmd_config(args)
        return

    # Handle template commands that don't need IntelliRefactor instance
    if args.command == "template":
        cmd_template(args)
        return

    # Handle index commands that don't need IntelliRefactor instance
    if args.command == "index":
        cmd_index(args)
        return

    # Handle duplicates commands that don't need IntelliRefactor instance
    if args.command == "duplicates":
        cmd_duplicates(args)
        return

    # Handle unused commands that don't need IntelliRefactor instance
    if args.command == "unused":
        cmd_unused(args)
        return

    # Handle system commands (new enhanced commands)
    if args.command == "system":
        cmd_system(args)
        return

    # Handle enhanced commands
    if args.command == "analyze-enhanced":
        cmd_analyze_enhanced(args)
        return

    if args.command == "refactor-enhanced":
        cmd_refactor_enhanced(args)
        return

    # Handle audit commands that don't need IntelliRefactor instance
    if args.command == "audit":
        cmd_audit(args)
        return
    elif args.command == "smells":
        cmd_smells(args)
        return
    elif args.command == "cluster":
        cmd_cluster(args)
        return

    # Handle decision engine commands that don't need IntelliRefactor instance
    if args.command == "decide":
        cmd_decide(args)
        return

    # Handle generate commands that don't need IntelliRefactor instance
    if args.command == "generate":
        cmd_generate(args)
        return

    # Handle expert analysis commands that don't need IntelliRefactor instance
    if args.command == "expert-analyze":
        cmd_expert_analyze(args)
        return

    try:
        # Load configuration and create IntelliRefactor instance
        config = load_config(getattr(args, "config", None))
        intellirefactor = IntelliRefactor(config)

        # Route to appropriate command handler
        if args.command == "analyze":
            cmd_analyze(args, intellirefactor)
        elif args.command == "opportunities":
            cmd_opportunities(args, intellirefactor)
        elif args.command == "refactor":
            cmd_refactor(args, intellirefactor)
        elif args.command == "apply":
            cmd_apply(args, intellirefactor)
        elif args.command == "knowledge":
            cmd_knowledge(args, intellirefactor)
        elif args.command == "report":
            cmd_report(args, intellirefactor)
        elif args.command == "status":
            cmd_status(args, intellirefactor)
        elif args.command == "visualize":
            cmd_visualize(args)
        elif args.command == "docs":
            cmd_docs(args)

    except KeyboardInterrupt:
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)


def cmd_decide(args) -> None:
    """Handle decision engine commands."""
    if args.decide_action == "analyze":
        cmd_decide_analyze(args)
    elif args.decide_action == "recommend":
        cmd_decide_recommend(args)
    elif args.decide_action == "plan":
        cmd_decide_plan(args)
    else:
        print("Error: No decision action specified. Use 'analyze', 'recommend', or 'plan'.")
        sys.exit(1)


def cmd_decide_analyze(args) -> None:
    """Perform comprehensive decision analysis."""
    try:
        project_path = Path(args.project_path)
        if not project_path.exists():
            print(f"Error: Project path does not exist: {project_path}")
            sys.exit(1)

        print(f"Performing decision analysis for: {project_path}")

        # Load decision criteria
        criteria = DecisionCriteria()
        if args.criteria_config:
            try:
                with open(args.criteria_config, "r", encoding="utf-8") as f:
                    criteria_data = json.load(f)
                    criteria = DecisionCriteria(**criteria_data)
            except Exception as e:
                print(f"Warning: Failed to load criteria config: {e}")
                print("Using default criteria.")

        # Override criteria with command line arguments
        if args.min_confidence:
            criteria.min_confidence_threshold = args.min_confidence
        if args.min_impact:
            criteria.min_impact_threshold = args.min_impact
        if args.max_effort:
            criteria.max_effort_threshold = args.max_effort

        # Run audit analysis
        print("Running comprehensive audit analysis...")
        audit_engine = AuditEngine(project_path)
        audit_result = audit_engine.run_full_audit(
            include_index=True,
            include_duplicates=True,
            include_unused=True,
            min_confidence=criteria.min_confidence_threshold,
            incremental_index=True,
        )

        # Run decision analysis
        print("Analyzing refactoring decisions...")
        decision_engine = RefactoringDecisionEngine(criteria)
        analysis_result = decision_engine.analyze_project(audit_result)

        # Format output
        if args.format == "json":
            output = json.dumps(analysis_result.to_dict(), indent=2, default=str)
        elif args.format == "yaml":
            import yaml

            output = yaml.dump(analysis_result.to_dict(), default_flow_style=False, sort_keys=False)
        else:
            output = format_decision_analysis_text(analysis_result)

        # Write output
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Decision analysis results written to: {args.output}")
        else:
            print(output)

        # Export decisions if requested
        if args.export_decisions:
            export_format = "json" if args.export_decisions.endswith(".json") else "yaml"
            exported = decision_engine.export_decisions(analysis_result, export_format)
            with open(args.export_decisions, "w", encoding="utf-8") as f:
                f.write(exported)
            print(f"Decisions exported to: {args.export_decisions}")

        print(f"\nAnalysis completed: {len(analysis_result.decisions)} decisions generated")

    except Exception as e:
        print(f"Error during decision analysis: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def cmd_decide_recommend(args) -> None:
    """Get filtered refactoring recommendations."""
    try:
        # Load analysis results
        with open(args.analysis_file, "r", encoding="utf-8") as f:
            if args.analysis_file.endswith(".yaml") or args.analysis_file.endswith(".yml"):
                import yaml

                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        # Filter decisions
        decisions = data.get("decisions", [])
        filtered_decisions = []

        for decision_data in decisions:
            # Filter by priority
            if args.priority and decision_data.get("priority") != args.priority:
                continue

            # Filter by type
            if args.type and decision_data.get("refactoring_type") != args.type:
                continue

            # Filter by confidence
            if args.min_confidence and decision_data.get("confidence", 0) < args.min_confidence:
                continue

            filtered_decisions.append(decision_data)

        # Limit results
        if args.limit:
            filtered_decisions = filtered_decisions[: args.limit]

        # Format output
        if args.format == "json":
            output_data = {
                "recommendations": filtered_decisions,
                "total_count": len(filtered_decisions),
                "filters_applied": {
                    "priority": args.priority,
                    "type": args.type,
                    "min_confidence": args.min_confidence,
                    "limit": args.limit,
                },
            }
            output = json.dumps(output_data, indent=2, default=str)
        elif args.format == "yaml":
            import yaml

            output_data = {
                "recommendations": filtered_decisions,
                "total_count": len(filtered_decisions),
                "filters_applied": {
                    "priority": args.priority,
                    "type": args.type,
                    "min_confidence": args.min_confidence,
                    "limit": args.limit,
                },
            }
            output = yaml.dump(output_data, default_flow_style=False, sort_keys=False)
        else:
            output = format_recommendations_text(
                filtered_decisions,
                include_evidence=args.include_evidence,
                include_plans=args.include_plans,
            )

        # Write output
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Recommendations written to: {args.output}")
        else:
            print(output)

        print(f"\nFound {len(filtered_decisions)} matching recommendations")

    except Exception as e:
        print(f"Error getting recommendations: {e}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def cmd_decide_plan(args) -> None:
    """Generate refactoring execution plan."""
    try:
        # Load analysis results
        with open(args.analysis_file, "r", encoding="utf-8") as f:
            if args.analysis_file.endswith(".yaml") or args.analysis_file.endswith(".yml"):
                import yaml

                data = yaml.safe_load(f)
            else:
                data = json.load(f)

        decisions = data.get("decisions", [])

        # Filter by priority if specified
        if args.priority_filter:
            decisions = [d for d in decisions if d.get("priority") in args.priority_filter]

        # Limit decisions
        if args.max_decisions:
            decisions = decisions[: args.max_decisions]

        # Sort decisions based on sequence criteria
        if args.sequence_by == "priority":
            priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            decisions.sort(key=lambda d: priority_order.get(d.get("priority", "low"), 3))
        elif args.sequence_by == "effort":
            decisions.sort(key=lambda d: d.get("feasibility", {}).get("effort_score", 0.5))
        elif args.sequence_by == "risk":
            decisions.sort(key=lambda d: d.get("feasibility", {}).get("risk_score", 0.5))
        # dependency sequencing would require more complex analysis

        # Generate execution plan
        plan_data = {
            "project_path": data.get("project_path"),
            "plan_created": datetime.now().isoformat(),
            "sequence_criteria": args.sequence_by,
            "total_decisions": len(decisions),
            "estimated_total_hours": sum(
                d.get("feasibility", {}).get("estimated_hours", 1.0) for d in decisions
            ),
            "decisions": decisions,
        }

        # Add timeline if requested
        if args.include_timeline:
            plan_data["timeline"] = generate_timeline(decisions)

        # Add resources if requested
        if args.include_resources:
            plan_data["resources"] = generate_resource_requirements(decisions)

        # Format output
        if args.format == "json":
            output = json.dumps(plan_data, indent=2, default=str)
        elif args.format == "yaml":
            import yaml

            output = yaml.dump(plan_data, default_flow_style=False, sort_keys=False)
        elif args.format == "markdown":
            output = format_plan_markdown(plan_data)
        else:
            output = format_plan_text(plan_data)

        # Write output
        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Execution plan written to: {args.output}")
        else:
            print(output)

        print(f"\nExecution plan generated for {len(decisions)} decisions")

    except Exception as e:
        print(f"Error generating execution plan: {e}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


def format_decision_analysis_text(analysis_result) -> str:
    """Format decision analysis results as text."""
    lines = []
    lines.append("=" * 80)
    lines.append("REFACTORING DECISION ANALYSIS RESULTS")
    lines.append("=" * 80)
    lines.append(f"Project: {analysis_result.project_path}")
    lines.append(f"Analysis ID: {analysis_result.analysis_run_id}")
    lines.append(f"Created: {analysis_result.created_at}")
    lines.append(f"Total Decisions: {len(analysis_result.decisions)}")
    lines.append("")

    # Summary by priority
    lines.append("DECISIONS BY PRIORITY:")
    lines.append("-" * 40)
    for priority in RefactoringPriority:
        count = len(analysis_result.get_decisions_by_priority(priority))
        if count > 0:
            lines.append(f"  {priority.value.upper()}: {count}")
    lines.append("")

    # Top decisions
    lines.append("TOP PRIORITY DECISIONS:")
    lines.append("-" * 40)

    top_decisions = sorted(
        analysis_result.decisions,
        key=lambda d: d.get_risk_adjusted_priority_score(),
        reverse=True,
    )[:10]

    for i, decision in enumerate(top_decisions, 1):
        lines.append(f"{i}. {decision.title}")
        lines.append(f"   Type: {decision.refactoring_type.value}")
        lines.append(f"   Priority: {decision.priority.value}")
        lines.append(f"   Confidence: {decision.confidence:.2f}")
        lines.append(f"   Files: {', '.join(decision.target_files[:3])}")
        if len(decision.target_files) > 3:
            lines.append(f"          ... and {len(decision.target_files) - 3} more")
        lines.append("")

    return "\n".join(lines)


def format_recommendations_text(
    decisions: List[Dict], include_evidence: bool = False, include_plans: bool = False
) -> str:
    """Format recommendations as text."""
    lines = []
    lines.append("=" * 80)
    lines.append("REFACTORING RECOMMENDATIONS")
    lines.append("=" * 80)
    lines.append(f"Total Recommendations: {len(decisions)}")
    lines.append("")

    for i, decision in enumerate(decisions, 1):
        lines.append(f"{i}. {decision.get('title', 'Untitled')}")
        lines.append(f"   Type: {decision.get('refactoring_type', 'unknown')}")
        lines.append(f"   Priority: {decision.get('priority', 'unknown')}")
        lines.append(f"   Confidence: {decision.get('confidence', 0):.2f}")
        lines.append(f"   Description: {decision.get('description', 'No description')}")

        if decision.get("target_files"):
            lines.append(f"   Files: {', '.join(decision['target_files'][:3])}")
            if len(decision["target_files"]) > 3:
                lines.append(f"          ... and {len(decision['target_files']) - 3} more")

        if include_evidence and decision.get("evidence"):
            lines.append("   Evidence:")
            for evidence in decision["evidence"][:2]:  # Show first 2 pieces of evidence
                lines.append(f"     - {evidence.get('description', 'No description')}")

        if include_plans and decision.get("implementation_plan"):
            lines.append("   Implementation Steps:")
            for step in decision["implementation_plan"][:3]:  # Show first 3 steps
                lines.append(
                    f"     {step.get('step_number', '?')}. {step.get('title', 'Untitled step')}"
                )

        lines.append("")

    return "\n".join(lines)


def format_plan_text(plan_data: Dict) -> str:
    """Format execution plan as text."""
    lines = []
    lines.append("=" * 80)
    lines.append("REFACTORING EXECUTION PLAN")
    lines.append("=" * 80)
    lines.append(f"Project: {plan_data.get('project_path', 'Unknown')}")
    lines.append(f"Created: {plan_data.get('plan_created', 'Unknown')}")
    lines.append(f"Sequence: {plan_data.get('sequence_criteria', 'priority')}")
    lines.append(f"Total Decisions: {plan_data.get('total_decisions', 0)}")
    lines.append(f"Estimated Hours: {plan_data.get('estimated_total_hours', 0):.1f}")
    lines.append("")

    lines.append("EXECUTION SEQUENCE:")
    lines.append("-" * 40)

    for i, decision in enumerate(plan_data.get("decisions", []), 1):
        lines.append(f"{i}. {decision.get('title', 'Untitled')}")
        lines.append(f"   Priority: {decision.get('priority', 'unknown')}")
        lines.append(
            f"   Effort: {decision.get('feasibility', {}).get('estimated_hours', 1.0):.1f} hours"
        )
        lines.append(f"   Risk: {decision.get('feasibility', {}).get('risk_score', 0.5):.2f}")
        lines.append("")

    return "\n".join(lines)


def format_plan_markdown(plan_data: Dict) -> str:
    """Format execution plan as markdown."""
    lines = []
    lines.append("# Refactoring Execution Plan")
    lines.append("")
    lines.append(f"**Project:** {plan_data.get('project_path', 'Unknown')}")
    lines.append(f"**Created:** {plan_data.get('plan_created', 'Unknown')}")
    lines.append(f"**Sequence:** {plan_data.get('sequence_criteria', 'priority')}")
    lines.append(f"**Total Decisions:** {plan_data.get('total_decisions', 0)}")
    lines.append(f"**Estimated Hours:** {plan_data.get('estimated_total_hours', 0):.1f}")
    lines.append("")

    lines.append("## Execution Sequence")
    lines.append("")

    for i, decision in enumerate(plan_data.get("decisions", []), 1):
        lines.append(f"### {i}. {decision.get('title', 'Untitled')}")
        lines.append("")
        lines.append(f"- **Priority:** {decision.get('priority', 'unknown')}")
        lines.append(f"- **Type:** {decision.get('refactoring_type', 'unknown')}")
        lines.append(
            f"- **Effort:** {decision.get('feasibility', {}).get('estimated_hours', 1.0):.1f} hours"
        )
        lines.append(f"- **Risk:** {decision.get('feasibility', {}).get('risk_score', 0.5):.2f}")
        lines.append(f"- **Description:** {decision.get('description', 'No description')}")
        lines.append("")

        if decision.get("implementation_plan"):
            lines.append("**Implementation Steps:**")
            for step in decision["implementation_plan"]:
                lines.append(
                    f"{step.get('step_number', '?')}. {step.get('title', 'Untitled step')}"
                )
            lines.append("")

    return "\n".join(lines)


def generate_timeline(decisions: List[Dict]) -> Dict[str, Any]:
    """Generate timeline for execution plan."""
    from datetime import datetime, timedelta

    start_date = datetime.now()
    current_date = start_date
    timeline = []

    for decision in decisions:
        estimated_hours = decision.get("feasibility", {}).get("estimated_hours", 1.0)
        estimated_days = max(1, int(estimated_hours / 8))  # Assume 8 hours per day

        end_date = current_date + timedelta(days=estimated_days)

        timeline.append(
            {
                "decision_id": decision.get("decision_id"),
                "title": decision.get("title"),
                "start_date": current_date.isoformat(),
                "end_date": end_date.isoformat(),
                "estimated_days": estimated_days,
                "estimated_hours": estimated_hours,
            }
        )

        current_date = end_date + timedelta(days=1)  # Add buffer day

    return {
        "start_date": start_date.isoformat(),
        "end_date": current_date.isoformat(),
        "total_days": (current_date - start_date).days,
        "milestones": timeline,
    }


def generate_resource_requirements(decisions: List[Dict]) -> Dict[str, Any]:
    """Generate resource requirements for execution plan."""
    total_hours = sum(d.get("feasibility", {}).get("estimated_hours", 1.0) for d in decisions)

    # Categorize by skill level required
    junior_hours = 0
    senior_hours = 0

    for decision in decisions:
        hours = decision.get("feasibility", {}).get("estimated_hours", 1.0)
        complexity = decision.get("feasibility", {}).get("complexity_score", 0.5)

        if complexity > 0.7:
            senior_hours += hours
        else:
            junior_hours += hours * 0.7  # Junior can do 70% of simple tasks
            senior_hours += hours * 0.3  # Senior oversight needed

    return {
        "total_hours": total_hours,
        "estimated_cost": {
            "junior_hours": junior_hours,
            "senior_hours": senior_hours,
            "total_hours": junior_hours + senior_hours,
        },
        "skill_requirements": [
            "Python refactoring experience",
            "Code analysis tools familiarity",
            "Testing and validation skills",
        ],
        "tools_needed": [
            "IDE with refactoring support",
            "Version control system",
            "Automated testing framework",
        ],
    }


def cmd_generate(args) -> None:
    """Handle specification generation commands."""
    if args.generate_action == "spec":
        cmd_generate_spec(args)
    else:
        print("Error: No generate action specified. Use 'spec'.")
        sys.exit(1)


def cmd_generate_spec(args) -> None:
    """Generate specification documents from analysis results."""
    try:
        project_path = Path(args.project_path)
        if not project_path.exists():
            print(f"Error: Project path does not exist: {project_path}")
            sys.exit(1)

        print(f"Generating specifications for project: {project_path}")

        # Run audit analysis first
        print("Running comprehensive project audit...")

        # Create audit engine
        audit_engine = AuditEngine(project_path)

        # Configure analysis options
        analysis_config = {
            "include_index": args.include_index,
            "include_duplicates": args.include_duplicates,
            "include_unused": args.include_unused,
            "min_confidence": args.min_confidence,
            "include_patterns": args.include_patterns,
            "exclude_patterns": args.exclude_patterns,
            "incremental_index": True,
        }

        # Run audit
        audit_result = audit_engine.run_full_audit(
            include_index=analysis_config["include_index"],
            include_duplicates=analysis_config["include_duplicates"],
            include_unused=analysis_config["include_unused"],
            min_confidence=analysis_config["min_confidence"],
            include_patterns=analysis_config["include_patterns"],
            exclude_patterns=analysis_config["exclude_patterns"],
            incremental_index=analysis_config["incremental_index"],
        )

        print(f"Analysis complete. Found {audit_result.statistics.total_findings} issues.")

        # Create spec generator
        spec_generator = SpecGenerator()

        # Load custom template if provided
        custom_template = None
        if args.template:
            template_path = Path(args.template)
            if template_path.exists():
                print(f"Loading custom template: {template_path}")
                # TODO: Implement custom template loading
                print("Warning: Custom templates not yet implemented, using default templates")
            else:
                print(f"Warning: Template file not found: {template_path}")

        # Determine output directory
        output_dir = Path(args.output_dir) if args.output_dir else Path.cwd()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Generate specifications
        if args.spec_type == "all":
            print("Generating all specification documents...")

            # Generate Requirements.md
            print("  - Generating Requirements.md...")
            requirements_content = spec_generator.generate_requirements_from_audit(
                audit_result, custom_template
            )
            requirements_path = output_dir / "Requirements.md"
            with open(requirements_path, "w", encoding="utf-8") as f:
                f.write(requirements_content)
            print(f"    ✓ Requirements.md written to: {requirements_path}")

            # Generate Design.md
            print("  - Generating Design.md...")
            design_content = spec_generator.generate_design_from_audit(
                audit_result, custom_template
            )
            design_path = output_dir / "Design.md"
            with open(design_path, "w", encoding="utf-8") as f:
                f.write(design_content)
            print(f"    ✓ Design.md written to: {design_path}")

            # Generate Implementation.md
            print("  - Generating Implementation.md...")
            implementation_content = spec_generator.generate_implementation_from_audit(
                audit_result, custom_template
            )
            implementation_path = output_dir / "Implementation.md"
            with open(implementation_path, "w", encoding="utf-8") as f:
                f.write(implementation_content)
            print(f"    ✓ Implementation.md written to: {implementation_path}")

        else:
            # Generate single specification type
            print(f"Generating {args.spec_type} specification...")

            content = spec_generator.generate_specification(
                args.spec_type, audit_result, custom_template
            )
            filename = f"{args.spec_type.title()}.md"
            output_path = output_dir / filename

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(content)
            print(f"✓ {filename} written to: {output_path}")

        # Generate machine-readable JSON if requested
        if args.emit_json:
            print("Generating machine-readable analysis artifacts...")

            json_data = spec_generator.generate_machine_readable_analysis(audit_result)
            json_path = Path(args.json_output) if args.json_output else output_dir / "analysis.json"

            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, default=str)
            print(f"✓ Analysis artifacts written to: {json_path}")

        # Summary
        print("\n" + "=" * 60)
        print("SPECIFICATION GENERATION COMPLETE")
        print("=" * 60)
        print(f"Project: {project_path}")
        print(f"Output Directory: {output_dir}")
        print(f"Analysis Results: {audit_result.statistics.total_findings} findings")
        print(f"Files Analyzed: {audit_result.statistics.files_analyzed}")
        print(f"Analysis Time: {audit_result.statistics.analysis_time_seconds:.2f}s")

        if audit_result.statistics.total_findings > 0:
            print("\nNext Steps:")
            print("1. Review the generated specifications")
            print("2. Prioritize refactoring tasks based on findings")
            print("3. Begin implementation following the generated plan")
        else:
            print("\n🎉 Excellent! No significant issues found in the codebase.")

    except Exception as e:
        print(f"Error generating specifications: {e}")
        if hasattr(args, "verbose") and args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


# Enhanced CLI command handlers with rich output and integration


def cmd_system(args) -> None:
    """Handle system management commands."""
    from intellirefactor.cli.rich_output import get_rich_output

    rich_output = get_rich_output()

    if args.system_action == "status":
        cmd_system_status(args, rich_output)
    elif args.system_action == "init":
        cmd_system_init(args, rich_output)
    elif args.system_action == "migrate":
        cmd_system_migrate(args, rich_output)
    else:
        rich_output.print_error("No system action specified. Use 'status', 'init', or 'migrate'.")
        sys.exit(1)


def cmd_system_status(args, rich_output: RichOutputManager) -> None:
    """Show comprehensive system status."""
    try:
        rich_output.print_header(
            "IntelliRefactor System Status",
            "Comprehensive system health and integration check",
        )

        # Load configuration
        config = load_config(getattr(args, "config", None))
        integration_manager = CLIIntegrationManager(config)

        with rich_output.status("Checking system status..."):
            # Get system status
            system_status = integration_manager.get_system_status()

            # Check compatibility if requested
            compatibility_report = None
            if getattr(args, "check_compatibility", False):
                compatibility_report = integration_manager.check_system_compatibility()

        # Display system status
        rich_output.print_section("Configuration")
        config_table = rich_output.create_table("Configuration Settings", ["Setting", "Value"])
        for key, value in system_status["config"].items():
            rich_output.add_table_row(config_table, key.replace("_", " ").title(), str(value))
        rich_output.print_table(config_table)

        # Display component status
        rich_output.print_section("Components")
        components_table = rich_output.create_table(
            "Component Status", ["Component", "Status", "Loaded"]
        )
        for component, loaded in system_status["components"].items():
            status = "✓ Ready" if loaded else "○ Not loaded"
            rich_output.add_table_row(
                components_table,
                component.replace("_", " ").title(),
                status,
                str(loaded),
            )
        rich_output.print_table(components_table)

        # Display plugin information
        if getattr(args, "show_plugins", False) or system_status["plugins"]["loaded"] > 0:
            rich_output.print_section("Plugins")
            plugins_table = rich_output.create_table("Plugin Information", ["Metric", "Count"])
            rich_output.add_table_row(
                plugins_table, "Loaded Plugins", system_status["plugins"]["loaded"]
            )
            rich_output.add_table_row(
                plugins_table,
                "Available Plugins",
                system_status["plugins"]["available"],
            )
            rich_output.print_table(plugins_table)

        # Display compatibility report if requested
        if compatibility_report:
            rich_output.print_section("Compatibility Check")

            if compatibility_report["overall_status"] == "compatible":
                rich_output.print_success("All components are compatible")
            elif compatibility_report["overall_status"] == "incompatible":
                rich_output.print_warning("Some compatibility issues found")
            else:
                rich_output.print_error("Compatibility check failed")

            if compatibility_report["issues"]:
                rich_output.print_info("Issues found:")
                for issue in compatibility_report["issues"]:
                    rich_output.print_warning(f"  • {issue}")

        # Output machine-readable format if requested
        if getattr(args, "machine_readable", False):
            output_data = {
                "system_status": system_status,
                "compatibility_report": compatibility_report,
            }
            rich_output.print_json(output_data, "System Status (JSON)")

        rich_output.print_success("System status check completed")

    except Exception as e:
        rich_output.print_error(f"Failed to get system status: {e}")
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)


def cmd_system_init(args, rich_output: RichOutputManager) -> None:
    """Initialize IntelliRefactor for a project."""
    try:
        project_path = Path(args.project_path)

        rich_output.print_header(
            "Project Initialization", f"Setting up IntelliRefactor for {project_path}"
        )

        # Load configuration
        config = load_config(getattr(args, "config", None))
        integration_manager = CLIIntegrationManager(config)

        # Initialize project
        with rich_output.status("Initializing project..."):
            init_result = integration_manager.initialize_project(project_path)

        rich_output.print_success("Project initialized successfully")

        # Display initialization results
        rich_output.print_section("Initialization Results")
        init_table = rich_output.create_table("Initialization Summary", ["Component", "Status"])
        rich_output.add_table_row(init_table, "Project Path", init_result["project_path"])
        rich_output.add_table_row(
            init_table,
            "Knowledge Base",
            "✓ Initialized" if init_result["knowledge_status"] else "○ Skipped",
        )
        rich_output.add_table_row(init_table, "Plugins Loaded", str(init_result["plugins_loaded"]))
        rich_output.add_table_row(
            init_table,
            "Safety Systems",
            "✓ Initialized" if init_result["safety_initialized"] else "○ Failed",
        )
        rich_output.print_table(init_table)

        # Create configuration file if requested
        if getattr(args, "create_config", False):
            config_path = project_path / "intellirefactor.json"
            with rich_output.status("Creating configuration file..."):
                config.save_to_file(str(config_path))
            rich_output.print_success(f"Configuration file created: {config_path}")

        # Show next steps
        rich_output.print_section("Next Steps")
        rich_output.print_info("You can now run analysis commands on this project:")
        rich_output.print_info("  • intellirefactor analyze-enhanced " + str(project_path))
        rich_output.print_info("  • intellirefactor index build " + str(project_path))
        rich_output.print_info("  • intellirefactor duplicates blocks " + str(project_path))

    except Exception as e:
        rich_output.print_error(f"Failed to initialize project: {e}")
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)


def cmd_system_migrate(args, rich_output: RichOutputManager) -> None:
    """Migrate legacy data to modern format."""
    try:
        legacy_path = Path(args.legacy_data_path)

        rich_output.print_header("Data Migration", f"Migrating legacy data from {legacy_path}")

        # Load configuration
        config = load_config(getattr(args, "config", None))
        integration_manager = CLIIntegrationManager(config)

        # Perform migration
        with rich_output.status("Migrating legacy data..."):
            migration_result = integration_manager.migrate_legacy_data(legacy_path)

        # Display migration results
        if migration_result["errors"]:
            rich_output.print_warning(
                f"Migration completed with {len(migration_result['errors'])} errors"
            )
            for error in migration_result["errors"]:
                rich_output.print_error(f"  • {error['error']}")
        else:
            rich_output.print_success("Migration completed successfully")

        # Display migration summary
        rich_output.print_section("Migration Summary")
        migration_table = rich_output.create_table("Migrated Items", ["Type", "Count"])
        for item in migration_result["migrated_items"]:
            rich_output.add_table_row(
                migration_table,
                item["type"].replace("_", " ").title(),
                str(item["count"]),
            )
        rich_output.print_table(migration_table)

        # Save migrated data if output path specified
        if getattr(args, "output", None):
            output_path = Path(args.output)
            output_format = getattr(args, "format", "json")

            with rich_output.status("Saving migrated data..."):
                if output_format == "json":
                    with open(output_path, "w", encoding="utf-8") as f:
                        json.dump(migration_result, f, indent=2, default=str)
                else:
                    # YAML format
                    import yaml

                    with open(output_path, "w", encoding="utf-8") as f:
                        yaml.dump(migration_result, f, default_flow_style=False)

            rich_output.print_success(f"Migrated data saved to: {output_path}")

    except Exception as e:
        rich_output.print_error(f"Failed to migrate data: {e}")
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)


def cmd_analyze_enhanced(args) -> None:
    """Enhanced project analysis with rich output and full integration."""
    from intellirefactor.cli.rich_output import get_rich_output
    from intellirefactor.cli.integration import CLIIntegrationManager

    rich_output = get_rich_output()

    try:
        project_path = Path(args.project_path)

        rich_output.print_header(
            "Enhanced Project Analysis", f"Comprehensive analysis of {project_path}"
        )

        # Load configuration
        config = load_config(getattr(args, "config", None))
        integration_manager = CLIIntegrationManager(config)

        # Interactive mode
        if getattr(args, "interactive", False):
            rich_output.print_info("Running in interactive mode")

            include_metrics = rich_output.confirm(
                "Include detailed metrics analysis?", default=True
            )
            include_opportunities = rich_output.confirm(
                "Include refactoring opportunities?", default=True
            )
            include_safety = rich_output.confirm("Include safety analysis?", default=True)
        else:
            include_metrics = getattr(args, "include_metrics", True)
            include_opportunities = getattr(args, "include_opportunities", True)
            include_safety = getattr(args, "include_safety", True)

        # Initialize project
        with rich_output.status("Initializing project analysis..."):
            init_result = integration_manager.initialize_project(project_path)

        # Run comprehensive analysis
        progress = rich_output.create_progress("Analyzing project...")

        with progress:
            analysis_task = rich_output.add_progress_task(
                progress, "Running comprehensive analysis", 100
            )

            analysis_result = integration_manager.run_comprehensive_analysis(
                project_path,
                include_metrics=include_metrics,
                include_opportunities=include_opportunities,
                include_safety_check=include_safety,
            )

            rich_output.update_progress(progress, analysis_task, 100)

        # Display results
        rich_output.print_section("Analysis Results")

        # Show metrics if included
        if include_metrics and "metrics" in analysis_result:
            rich_output.print_metrics_summary(analysis_result["metrics"])

        # Show opportunities if included
        if include_opportunities and "opportunities" in analysis_result:
            opportunities = analysis_result["opportunities"]
            if opportunities:
                rich_output.print_section("Refactoring Opportunities")
                opp_table = rich_output.create_table(
                    "Top Opportunities", ["Title", "Priority", "Effort", "Impact"]
                )
                for opp in opportunities[:10]:  # Show top 10
                    # Handle both GenericRefactoringOpportunity objects and dictionaries
                    if hasattr(opp, "description"):
                        # It's a GenericRefactoringOpportunity object
                        title = getattr(opp, "description", "Unknown")
                        priority = getattr(opp, "priority", "Medium")
                        effort = getattr(opp, "automation_confidence", "Unknown")
                        impact = str(getattr(opp, "estimated_impact", {}))
                    else:
                        # It's a dictionary (legacy format)
                        title = opp.get("title", "Unknown")
                        priority = opp.get("priority", "Medium")
                        effort = opp.get("effort_estimate", "Unknown")
                        impact = opp.get("impact_estimate", "Unknown")

                    rich_output.add_table_row(
                        opp_table, title, str(priority), str(effort), str(impact)
                    )
                rich_output.print_table(opp_table)

        # Show safety report if included
        if include_safety and "safety_report" in analysis_result:
            safety_report = analysis_result["safety_report"]
            rich_output.print_section("Safety Analysis")

            if safety_report.get("safe_to_refactor", True):
                rich_output.print_success("Project is safe for refactoring")
            else:
                rich_output.print_warning("Safety concerns found")
                for concern in safety_report.get("concerns", []):
                    rich_output.print_warning(f"  • {concern}")

        # Generate and save report
        if getattr(args, "output", None):
            output_path = Path(args.output)
            output_format = getattr(args, "format", "markdown")

            with rich_output.status("Generating comprehensive report..."):
                report_content = integration_manager.generate_comprehensive_report(
                    project_path, analysis_result, output_format
                )

                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(report_content)

            rich_output.print_success(f"Analysis report saved to: {output_path}")

        # Machine-readable output
        if getattr(args, "machine_readable", False):
            rich_output.print_json(analysis_result, "Analysis Results (JSON)")

        rich_output.print_success("Enhanced analysis completed successfully")

    except Exception as e:
        rich_output.print_error(f"Analysis failed: {e}")
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)


def cmd_expert_analyze(args) -> None:
    """Handle expert analysis command."""
    from intellirefactor.cli.rich_output import get_rich_output
    
    rich_output = get_rich_output()
    
    try:
        # Import the expert analyzer
        from intellirefactor.analysis.expert import ExpertRefactoringAnalyzer
        
        project_path = Path(args.project_path).resolve()
        target_file = Path(args.target_file).resolve()
        
        # Validate inputs
        if not project_path.exists():
            rich_output.error(f"Project path does not exist: {project_path}")
            sys.exit(1)
        
        if not target_file.exists():
            rich_output.error(f"Target file does not exist: {target_file}")
            sys.exit(1)
        
        if not target_file.suffix == '.py':
            rich_output.error(f"Target file must be a Python file: {target_file}")
            sys.exit(1)
        
        # Set up output directory
        output_dir = Path(args.output) if args.output else Path("./expert_analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        rich_output.info("Starting expert refactoring analysis...")
        rich_output.info(f"Project: {project_path}")
        rich_output.info(f"Target: {target_file}")
        rich_output.info(f"Output: {output_dir}")
        
        # Initialize analyzer
        analyzer = ExpertRefactoringAnalyzer(
            project_root=str(project_path),
            target_module=str(target_file),
            output_dir=str(output_dir)
        )
        
        # Run analysis
        with rich_output.progress("Running expert analysis..."):
            result = analyzer.analyze_for_expert_refactoring()
        
        # Export detailed data if requested
        detailed_data = None
        if args.detailed:
            with rich_output.progress("Exporting detailed expert data..."):
                detailed_data = analyzer.export_detailed_expert_data()
        
        # Generate reports based on format
        reports_generated = []
        
        if args.format in ["json", "both"]:
            json_path = output_dir / f"expert_analysis_{analyzer.timestamp}.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                import json
                json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
            reports_generated.append(str(json_path))
            
            # Export detailed data if requested
            if detailed_data:
                detailed_json_path = output_dir / f"expert_analysis_detailed_{analyzer.timestamp}.json"
                with open(detailed_json_path, 'w', encoding='utf-8') as f:
                    json.dump(detailed_data, f, indent=2, ensure_ascii=False)
                reports_generated.append(str(detailed_json_path))
                
                # Also create the characterization test file
                if 'characterization_tests' in detailed_data:
                    test_file_path = detailed_data['characterization_tests'].get('test_file_path', 'test_characterization.py')
                    test_code = detailed_data['characterization_tests'].get('executable_test_code', '')
                    if test_code:
                        test_path = output_dir / test_file_path
                        with open(test_path, 'w', encoding='utf-8') as f:
                            f.write(test_code)
                        reports_generated.append(str(test_path))
        
        if args.format in ["markdown", "both"]:
            md_path = analyzer.generate_expert_report(str(output_dir))
            reports_generated.append(md_path)
        
        # Display results
        rich_output.success("Expert analysis completed successfully!")
        rich_output.info(f"Quality Score: {result.analysis_quality_score:.1f}/100")
        rich_output.info(f"Risk Level: {result.risk_assessment.value.upper()}")
        
        if args.detailed:
            rich_output.success("✅ Detailed expert data exported - all expert requirements addressed!")
            if detailed_data and 'expert_recommendations' in detailed_data:
                expert_recs = detailed_data['expert_recommendations']
                if 'expert_1_requirements' in expert_recs:
                    rich_output.info("Expert 1 Requirements:")
                    for rec in expert_recs['expert_1_requirements']:
                        rich_output.info(f"  {rec}")
                if 'expert_2_requirements' in expert_recs:
                    rich_output.info("Expert 2 Requirements:")
                    for rec in expert_recs['expert_2_requirements']:
                        rich_output.info(f"  {rec}")
        
        if result.recommendations:
            rich_output.info("Key Recommendations:")
            for i, rec in enumerate(result.recommendations[:5], 1):
                rich_output.info(f"  {i}. {rec}")
        
        rich_output.info("Generated Reports:")
        for report in reports_generated:
            rich_output.info(f"  • {report}")
        
        # Show enhanced statistics for detailed mode
        if args.detailed and detailed_data:
            rich_output.info("Detailed Analysis Statistics:")
            
            # Call graph details
            if 'call_graph' in detailed_data:
                cg = detailed_data['call_graph']['call_graph']
                rich_output.info(f"  • Complete call graph: {cg.get('total_relationships', 0)} relationships")
            
            # External usage details
            if 'external_usage' in detailed_data:
                eu = detailed_data['external_usage']['files_summary']
                rich_output.info(f"  • External usage: {eu.get('total_files', 0)} files with specific locations")
            
            # Duplicate details
            if 'duplicates' in detailed_data:
                dup = detailed_data['duplicates']['summary']
                rich_output.info(f"  • Code duplicates: {dup.get('total_duplicates', 0)} fragments, {dup.get('total_savings', 0)} lines savings")
            
            # Test details
            if 'characterization_tests' in detailed_data:
                ct = detailed_data['characterization_tests']['summary']
                rich_output.info(f"  • Characterization tests: {ct.get('total_tests', 0)} executable test cases")
            
            # Missing test details
            if 'test_analysis' in detailed_data:
                ta = detailed_data['test_analysis']['missing_test_coverage']
                rich_output.info(f"  • Missing test coverage: {ta.get('total_missing', 0)} specific methods identified")
        else:
            # Show enhanced statistics for detailed mode
            stats = []
            if result.call_graph:
                stats.append(f"Methods: {len(result.call_graph.nodes)}")
                stats.append(f"Call relationships: {len(result.call_graph.edges)}")
                if result.call_graph.cycles:
                    stats.append(f"Cycles detected: {len(result.call_graph.cycles)}")
            
            if result.external_callers:
                stats.append(f"External callers: {len(result.external_callers)}")
            
            if result.characterization_tests:
                stats.append(f"Test cases generated: {len(result.characterization_tests)}")
            
            if result.duplicate_fragments:
                total_savings = sum(frag.estimated_savings for frag in result.duplicate_fragments)
                stats.append(f"Duplicate savings: {total_savings} lines")
            
            if stats:
                rich_output.info("Basic Analysis Statistics:")
                for stat in stats:
                    rich_output.info(f"  • {stat}")
        
    except ImportError as e:
        rich_output.error(f"Expert analyzer not available: {e}")
        rich_output.info("Make sure the expert analysis module is properly installed")
        sys.exit(1)
    except Exception as e:
        rich_output.error(f"Expert analysis failed: {e}")
        if args.verbose:
            import traceback
            rich_output.error(traceback.format_exc())
        sys.exit(1)


def cmd_refactor_enhanced(args) -> None:
    """Enhanced refactoring with full orchestration and safety checks."""
    from intellirefactor.cli.rich_output import get_rich_output
    from intellirefactor.cli.integration import CLIIntegrationManager

    rich_output = get_rich_output()

    try:
        project_path = Path(args.project_path)
        plan_file = Path(args.plan_file)

        rich_output.print_header(
            "Enhanced Refactoring", f"Executing refactoring plan for {project_path}"
        )

        # Load refactoring plan
        if not plan_file.exists():
            rich_output.print_error(f"Refactoring plan file not found: {plan_file}")
            sys.exit(1)

        with open(plan_file, "r", encoding="utf-8") as f:
            refactoring_plan = json.load(f)

        # Load configuration
        config = load_config(getattr(args, "config", None))
        integration_manager = CLIIntegrationManager(config)

        # Interactive confirmation
        if getattr(args, "interactive", False):
            rich_output.print_section("Refactoring Plan Summary")
            rich_output.print_info(
                f"Plan contains {len(refactoring_plan.get('steps', []))} refactoring steps"
            )

            if not rich_output.confirm("Proceed with refactoring?", default=False):
                rich_output.print_info("Refactoring cancelled by user")
                return

        # Execute refactoring
        dry_run = getattr(args, "dry_run", False)
        create_backup = getattr(args, "create_backup", True)

        if dry_run:
            rich_output.print_info("Running in dry-run mode (no changes will be made)")

        progress = rich_output.create_progress("Executing refactoring plan...")

        with progress:
            refactor_task = rich_output.add_progress_task(
                progress, "Processing refactoring steps", 100
            )

            execution_result = integration_manager.execute_refactoring_plan(
                project_path, refactoring_plan, dry_run, create_backup
            )

            rich_output.update_progress(progress, refactor_task, 100)

        # Display results
        rich_output.print_section("Refactoring Results")

        if execution_result["execution_result"].get("success", False):
            rich_output.print_success("Refactoring completed successfully")
        else:
            rich_output.print_error("Refactoring failed")
            for error in execution_result["execution_result"].get("errors", []):
                rich_output.print_error(f"  • {error}")

        # Show backup information
        if execution_result["backup_info"]:
            rich_output.print_info(
                f"Backup created: {execution_result['backup_info']['backup_path']}"
            )

        # Show validation results
        if "validation" in execution_result["execution_result"]:
            validation = execution_result["execution_result"]["validation"]
            if validation.get("valid", True):
                rich_output.print_success("Refactoring validation passed")
            else:
                rich_output.print_warning("Refactoring validation issues found")
                for issue in validation.get("issues", []):
                    rich_output.print_warning(f"  • {issue}")

        # Save results if output specified
        if getattr(args, "output", None):
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(execution_result, f, indent=2, default=str)
            rich_output.print_success(f"Refactoring results saved to: {output_path}")

        # Machine-readable output
        if getattr(args, "machine_readable", False):
            rich_output.print_json(execution_result, "Refactoring Results (JSON)")

    except Exception as e:
        rich_output.print_error(f"Refactoring failed: {e}")
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

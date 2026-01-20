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
from typing import Dict, Any, List, TextIO

# Add project root to sys.path when running as script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from intellirefactor.api import IntelliRefactor, AnalysisResult, RefactoringResult
from intellirefactor.config import IntelliRefactorConfig, SafetyLevel, load_config
from intellirefactor.templates import TemplateGenerator
from intellirefactor.analysis.index.builder import IndexBuilder, IndexBuildResult
from intellirefactor.analysis.index.store import IndexStore
from intellirefactor.analysis.index.query import IndexQuery
from intellirefactor.analysis.workflows.audit_engine import AuditEngine
from intellirefactor.analysis.workflows.spec_generator import SpecGenerator
from intellirefactor.analysis.decompose.responsibility_clusterer import (
    ResponsibilityClusterer,
    ClusteringConfig,
    ClusteringAlgorithm,
)
from intellirefactor.analysis.refactor.refactoring_decision_engine import (
    RefactoringDecisionEngine,
    DecisionCriteria,
    RefactoringPriority,
)
from intellirefactor.cli.rich_output import RichOutputManager, set_rich_enabled
from intellirefactor.cli.integration import CLIIntegrationManager

# Import extracted modules (Step 1 & 2 & 3 & 4 refactoring)
from intellirefactor.cli.formatters import (
    format_clone_detection_results,
    format_similarity_results,
    format_unused_code_results,
    format_audit_results,
    format_smells_text,
    format_smells_json,
    format_smells_html,
)
from intellirefactor.cli.commands.analysis import (
    cmd_analyze,
    cmd_visualize,
    cmd_docs,
    format_analysis_result,
)
from intellirefactor.cli.commands.config import (
    cmd_status,
    cmd_config,
    cmd_template,
)
from intellirefactor.cli.commands.refactoring import (
    cmd_opportunities,
    cmd_refactor,
    cmd_apply,
    format_refactoring_result,
)
from intellirefactor.cli.commands.knowledge import (
    cmd_knowledge,
    cmd_report,
)
from intellirefactor.cli.commands.deduplication import (
    cmd_index,
    cmd_duplicates,
    get_index_db_path,
    format_index_build_result,
    format_index_status,
)
from intellirefactor.cli.commands.quality import (
    cmd_unused,
    cmd_audit,
    cmd_smells,
    format_audit_results,
)


# =============================================================================

def _is_machine_readable(args: Any) -> bool:
    return bool(getattr(args, "machine_readable", False))



def _json_stdout(args: Any) -> TextIO:
    """
    When --machine-readable is enabled, main() redirects sys.stdout -> sys.stderr
    to prevent accidental non-JSON output. This function returns the original stdout.
    """
    return getattr(args, "_json_stdout", sys.__stdout__)



def _force_json_format_if_possible(args: Any) -> None:
    """
    Enforce JSON output for commands that support --format when --machine-readable is set.
    Also fixes the cluster command which uses args.output as format selector.
    """
    if not _is_machine_readable(args):
        return

    if hasattr(args, "format"):
        try:
            args.format = "json"
        except Exception:
            pass

    # cluster uses args.output as output-format selector (text/json/html)
    if getattr(args, "command", None) == "cluster" and hasattr(args, "output"):
        try:
            args.output = "json"
        except Exception:
            pass



def _print_json_to_stdout(args: Any, payload: Any) -> None:
    """
    Always print JSON to the original stdout in machine-readable mode.
    """
    s = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    print(s, file=_json_stdout(args))



def _maybe_print_output(args: Any, output_text_or_json: str) -> None:
    """
    Print final command output:
    - in normal mode: print to sys.stdout
    - in machine-readable mode: print to original stdout (args._json_stdout)
    """
    if _is_machine_readable(args):
        print(output_text_or_json, file=_json_stdout(args))
    else:
        print(output_text_or_json)



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
    # NOTE: keep --verbose/-v only as a global flag (defined on the root parser)

    # Index command
    index_parser = subparsers.add_parser(
        "index", help="Index management for persistent analysis data"
    )
    index_subparsers = index_parser.add_subparsers(dest="index_action", help="Index operations")

    # Index build
    build_parser = index_subparsers.add_parser("build", help="Build or update the project index")
    build_parser.add_argument("project_path", help="Path to project directory to index")
    # Correct semantics:
    # - default: incremental=True
    # - --full => incremental=False
    # - optional explicit --incremental keeps backward compat
    mode_group = build_parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--incremental",
        "-i",
        dest="incremental",
        action="store_true",
        help="Perform incremental update (default)",
    )
    mode_group.add_argument(
        "--full",
        "-f",
        dest="incremental",
        action="store_false",
        help="Force full rebuild of the index",
    )
    build_parser.set_defaults(incremental=True)
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

    # Collect command (new): export artifacts for GUI/LLM usage
    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect analysis artifacts (JSON+MD) for refactor/dedup/decompose into a run directory",
    )
    collect_parser.add_argument("project_path", help="Path to project root directory")
    collect_parser.add_argument(
        "--target-file",
        help="Optional target file for file-level/refactor-expert analyses",
    )
    collect_parser.add_argument(
        "--out",
        default="./intellirefactor_out",
        help="Output base directory (default: ./intellirefactor_out)",
    )
    collect_parser.add_argument("--run-id", help="Override run id (default: timestamp)")

    # Sections (if none selected explicitly -> run all)
    collect_parser.add_argument("--refactor", action="store_true", help="Run refactor section")
    collect_parser.add_argument("--dedup", action="store_true", help="Run dedup section")
    collect_parser.add_argument("--decompose", action="store_true", help="Run decompose section")

    # Full mode (extra/heavier artifacts)
    collect_parser.add_argument(
        "--full",
        action="store_true",
        help="Export full-size artifacts (extra instances/snippets; writes *full.json where supported)",
    )

    # Index controls
    collect_parser.add_argument("--no-index", action="store_true", help="Skip index build")
    collect_parser.add_argument("--index-full", action="store_true", help="Force full index rebuild")

    # Optional extras
    collect_parser.add_argument("--visualize", action="store_true", help="Generate visualization artifacts (requires index)")
    collect_parser.add_argument("--entry-point", help="Entry point file for execution-flow diagram")
    collect_parser.add_argument(
        "--dedup-max-results",
        type=int,
        default=200,
        help="Max semantic similarity matches to save (default: 200)",
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
    # Defaults: include everything. Provide explicit skip-flags (store_false).
    analyze_enhanced_parser.set_defaults(
        include_metrics=True,
        include_opportunities=True,
        include_safety=True,
    )
    analyze_enhanced_parser.add_argument(
        "--no-metrics",
        dest="include_metrics",
        action="store_false",
        help="Skip detailed metrics analysis",
    )
    analyze_enhanced_parser.add_argument(
        "--no-opportunities",
        dest="include_opportunities",
        action="store_false",
        help="Skip refactoring opportunities",
    )
    analyze_enhanced_parser.add_argument(
        "--no-safety",
        dest="include_safety",
        action="store_false",
        help="Skip safety analysis",
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
    # Default: create backup. Allow disabling explicitly.
    backup_group = refactor_enhanced_parser.add_mutually_exclusive_group()
    backup_group.add_argument(
        "--create-backup",
        dest="create_backup",
        action="store_true",
        help="Create backup before refactoring (default)",
    )
    backup_group.add_argument(
        "--no-backup",
        dest="create_backup",
        action="store_false",
        help="Do not create backup before refactoring",
    )
    refactor_enhanced_parser.set_defaults(create_backup=True)
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



def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # If machine-readable: force JSON output & ensure stdout is JSON-only.
    # We do this by redirecting sys.stdout -> sys.stderr, and explicitly printing JSON to args._json_stdout.
    if _is_machine_readable(args):
        args._json_stdout = sys.stdout
        _force_json_format_if_possible(args)
        sys.stdout = sys.stderr

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

    # Collect command (exports artifacts to run directory)
    if args.command == "collect":
        from intellirefactor.cli.commands.collect import run_collect

        payload = run_collect(args)
        if _is_machine_readable(args):
            _print_json_to_stdout(args, payload)
        else:
            # Human-friendly minimal output; artifacts include summary.md
            print(f"Collect completed. Run dir: {payload.get('run_dir')}", file=sys.stderr)
            print(f"Manifest: {payload.get('manifest')}", file=sys.stderr)
        if not payload.get("success", False):
            sys.exit(1)
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
        if _is_machine_readable(args):
            _print_json_to_stdout(args, {"success": False, "error": "cancelled_by_user"})
        print("\nOperation cancelled by user", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        if _is_machine_readable(args):
            _print_json_to_stdout(
                args,
                {"success": False, "error": str(e), "command": getattr(args, "command", None)},
            )
        print(f"Error: {e}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)



if __name__ == "__main__":
    main()

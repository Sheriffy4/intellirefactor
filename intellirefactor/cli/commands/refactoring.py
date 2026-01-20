"""
Refactoring commands for IntelliRefactor CLI.

This module contains command handlers for:
- Identifying refactoring opportunities
- Applying refactoring operations
- Managing refactoring workflows
"""

import json
import sys
from pathlib import Path

from intellirefactor.api import IntelliRefactor, RefactoringResult
from intellirefactor.config import SafetyLevel


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


def cmd_opportunities(args, intellirefactor: IntelliRefactor) -> None:
    """Handle opportunities command."""
    from intellirefactor.cli_entry import _is_machine_readable, _maybe_print_output
    
    opportunities = intellirefactor.identify_opportunities(args.path, args.limit)

    format_type = "json" if _is_machine_readable(args) else args.format

    if format_type == "json":
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
        print(f"Opportunities written to {args.output}", file=sys.stderr)

    # Always print final output (JSON in machine-readable) to stdout
    _maybe_print_output(args, output)


def cmd_refactor(args, intellirefactor: IntelliRefactor) -> None:
    """Handle refactor command."""
    from intellirefactor.cli_entry import _is_machine_readable, _maybe_print_output
    
    # Update safety level if specified
    if args.safety_level:
        intellirefactor.config.refactoring_settings.safety_level = SafetyLevel(args.safety_level)

    result = intellirefactor.auto_refactor_project(
        args.path, args.strategy, args.max_operations, args.dry_run
    )

    # In machine-readable mode: output JSON to stdout, everything else stays on stderr.
    if _is_machine_readable(args):
        output_json = format_refactoring_result(result, "json")
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_json)
            print(f"Detailed refactoring results written to {args.output}", file=sys.stderr)
        _maybe_print_output(args, output_json)
        if not result.success:
            sys.exit(1)
        return

    output = format_refactoring_result(result, "text")

    if args.output:
        # Save detailed JSON results to file
        detailed_output = format_refactoring_result(result, "json")
        with open(args.output, "w") as f:
            f.write(detailed_output)
        print(f"Detailed refactoring results written to {args.output}", file=sys.stderr)

    print(output)

    if not result.success:
        sys.exit(1)


def cmd_apply(args, intellirefactor: IntelliRefactor) -> None:
    """Handle apply command."""
    from intellirefactor.cli_entry import _is_machine_readable, _maybe_print_output
    
    try:
        with open(args.opportunity_file, "r", encoding="utf-8") as f:
            opportunity = json.load(f)
    except Exception as e:
        print(f"Error: Failed to load opportunity file: {e}", file=sys.stderr)
        sys.exit(1)

    validate_results = not args.no_validation
    result = intellirefactor.apply_refactoring(opportunity, args.dry_run, validate_results)

    # In machine-readable mode: output JSON to stdout, everything else stays on stderr.
    if _is_machine_readable(args):
        output_json = format_refactoring_result(result, "json")
        if args.output:
            with open(args.output, "w") as f:
                f.write(output_json)
            print(f"Detailed results written to {args.output}", file=sys.stderr)
        _maybe_print_output(args, output_json)
        if not result.success:
            sys.exit(1)
        return

    output = format_refactoring_result(result, "text")

    if args.output:
        detailed_output = format_refactoring_result(result, "json")
        with open(args.output, "w") as f:
            f.write(detailed_output)
        print(f"Detailed results written to {args.output}", file=sys.stderr)

    print(output)

    if not result.success:
        sys.exit(1)

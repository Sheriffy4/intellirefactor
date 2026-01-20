"""
Knowledge and report command handlers.

This module contains commands for:
- Knowledge base queries and management
- Report generation from refactoring results
"""

import json
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from intellirefactor.api import IntelliRefactor


def cmd_knowledge(args, intellirefactor: "IntelliRefactor") -> None:
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


def cmd_report(args, intellirefactor: "IntelliRefactor") -> None:
    """Handle report command."""
    from intellirefactor.refactoring.refactoring_result import RefactoringResult

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

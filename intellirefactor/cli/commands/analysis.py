"""
Analysis command handlers.

Handles analyze, visualize, and documentation generation commands.
"""

import json
import sys
from pathlib import Path
from typing import Any

from intellirefactor.api import IntelliRefactor, AnalysisResult
from intellirefactor.analysis.index.store import IndexStore
from intellirefactor.visualization.diagram_generator import FlowchartGenerator
from intellirefactor.documentation.doc_generator import DocumentationGenerator


def _is_machine_readable(args: Any) -> bool:
    """Check if machine-readable mode is enabled."""
    return bool(getattr(args, "machine_readable", False))


def _print_json_to_stdout(args: Any, payload: Any) -> None:
    """Print JSON to stdout in machine-readable mode."""
    s = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    json_stdout = getattr(args, "_json_stdout", sys.__stdout__)
    print(s, file=json_stdout)


def _maybe_print_output(args: Any, output_text_or_json: str) -> None:
    """Print output to appropriate stream."""
    if _is_machine_readable(args):
        json_stdout = getattr(args, "_json_stdout", sys.__stdout__)
        print(output_text_or_json, file=json_stdout)
    else:
        print(output_text_or_json)


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

    format_type = "json" if _is_machine_readable(args) else args.format
    output = format_analysis_result(result, format_type)

    if args.output:
        with open(args.output, "w") as f:
            f.write(output)
        print(f"Analysis results written to {args.output}", file=sys.stderr)

    # In --machine-readable mode we must ALWAYS emit JSON to stdout
    _maybe_print_output(args, output)

    # Handle visualization if requested
    if hasattr(args, "visualize") and args.visualize:
        from intellirefactor.orchestration.global_refactoring_orchestrator import (
            GlobalRefactoringOrchestrator,
        )

        print("\nGenerating visualizations...", file=sys.stderr)

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
                    f"✅ Visualizations generated successfully in: {stage_result.details.get('output_dir', 'unknown')}",
                    file=sys.stderr,
                )
            else:
                print(f"⚠️  Visualization generation had issues: {stage_result.message}", file=sys.stderr)
        except Exception as e:
            print(f"❌ Error generating visualizations: {e}", file=sys.stderr)

    if not result.success:
        sys.exit(1)


def cmd_visualize(args) -> None:
    """Handle visualize command."""
    from intellirefactor.cli.commands.deduplication import get_index_db_path
    
    try:
        if args.visualize_action == "method":
            # Generate method flowchart
            generator = FlowchartGenerator(None)
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

            if _is_machine_readable(args):
                _print_json_to_stdout(args, {"success": True, "output_path": str(output_path)})
            else:
                print(f"Method flowchart saved to: {output_path}")

        elif args.visualize_action == "call-graph":
            # Generate call graph - need index store for this
            project_root = Path.cwd()
            db_path = get_index_db_path(project_root)
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

            if _is_machine_readable(args):
                _print_json_to_stdout(args, {"success": True, "output_path": str(output_path)})
            else:
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
    try:
        if args.docs_action == "list":
            # List available documentation types - no target file needed
            doc_generator = DocumentationGenerator(Path.cwd())

            available_types = doc_generator.list_available_types()
            generator_info = doc_generator.get_generator_info()

            if _is_machine_readable(args):
                _print_json_to_stdout(
                    args,
                    {
                        "success": True,
                        "available_types": available_types,
                        "generator_info": generator_info,
                    },
                )
                return

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

            if _is_machine_readable(args):
                _print_json_to_stdout(args, {"success": True, "generated_file": str(generated_file)})
                return

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

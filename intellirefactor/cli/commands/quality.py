"""
Quality analysis command handlers.

This module contains commands for:
- Unused code detection
- Project auditing
- Architectural smell detection
"""

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def cmd_unused(args) -> None:
    """Handle unused command."""
    from intellirefactor.analysis.refactor.unused_code_detector import (
        UnusedCodeDetector,
        UnusedCodeType,
    )
    from intellirefactor.cli.formatters import format_unused_code_results
    
    # Import helpers locally to avoid circular dependency
    def _maybe_print_output(args, output_text_or_json: str) -> None:
        """Print output to appropriate stream."""
        machine_readable = bool(getattr(args, "machine_readable", False))
        if machine_readable:
            json_stdout = getattr(args, "_json_stdout", sys.__stdout__)
            print(output_text_or_json, file=json_stdout)
        else:
            print(output_text_or_json)
    
    def _is_machine_readable(args) -> bool:
        """Check if machine-readable mode is enabled."""
        return bool(getattr(args, "machine_readable", False))

    project_path = Path(args.project_path)
    if not project_path.exists():
        print(f"Error: Project path does not exist: {project_path}", file=sys.stderr)
        sys.exit(1)

    if not project_path.is_dir():
        print(f"Error: Project path must be a directory: {project_path}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.unused_action == "detect":
            print(f"Analyzing {project_path} for unused code...", file=sys.stderr)

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

            print(f"Found {len(result.findings)} unused code findings", file=sys.stderr)

            # Generate output
            output = format_unused_code_results(
                result, args.format, args.show_evidence, args.show_usage, args.group_by
            )

            if args.output:
                with open(args.output, "w") as f:
                    f.write(output)
                print(f"Unused code detection results written to {args.output}", file=sys.stderr)
                # In machine-readable mode we MUST still emit JSON to stdout
                if _is_machine_readable(args):
                    _maybe_print_output(args, output)
            else:
                _maybe_print_output(args, output)

    except Exception as e:
        print(f"Error: Failed to detect unused code: {e}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)



def format_audit_results(audit_result, format_type: str) -> str:
    """Format audit results for output."""
    from intellirefactor.analysis.foundation.models import Severity

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
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
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
        for severity_key, count in stats.findings_by_severity.items():
            if count > 0:
                emoji = {
                    "critical": "[CRITICAL]",
                    "high": "[HIGH]",
                    "medium": "[MEDIUM]",
                    "low": "[LOW]",
                    "info": "[INFO]",
                }.get(severity_key, "")
                output.append(f"    {emoji} {severity_key.title()}: {count}")
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
            Severity.CRITICAL,
            Severity.HIGH,
            Severity.MEDIUM,
            Severity.LOW,
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
    from intellirefactor.analysis.workflows.audit_engine import AuditEngine
    from intellirefactor.analysis.workflows.spec_generator import SpecGenerator
    
    # Import helpers locally to avoid circular dependency
    def _maybe_print_output(args, output_text_or_json: str) -> None:
        """Print output to appropriate stream."""
        machine_readable = bool(getattr(args, "machine_readable", False))
        if machine_readable:
            json_stdout = getattr(args, "_json_stdout", sys.__stdout__)
            print(output_text_or_json, file=json_stdout)
        else:
            print(output_text_or_json)
    
    def _is_machine_readable(args) -> bool:
        """Check if machine-readable mode is enabled."""
        return bool(getattr(args, "machine_readable", False))

    project_path = Path(args.project_path)
    if not project_path.exists():
        print(f"Error: Project path does not exist: {project_path}", file=sys.stderr)
        sys.exit(1)

    if not project_path.is_dir():
        print(f"Error: Project path must be a directory: {project_path}", file=sys.stderr)
        sys.exit(1)

    try:
        print(f"Starting comprehensive audit of {project_path}...", file=sys.stderr)

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
            f"Audit completed: {len(result.findings)} findings in {result.statistics.analysis_time_seconds:.2f}s",
            file=sys.stderr,
        )

        # Generate main output
        output = format_audit_results(result, args.format)

        if args.output:
            with open(args.output, "w") as f:
                f.write(output)
            print(f"Audit results written to {args.output}", file=sys.stderr)
            # In machine-readable mode we MUST still emit JSON to stdout
            if _is_machine_readable(args):
                _maybe_print_output(args, output)
        else:
            _maybe_print_output(args, output)

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
                    print(f"Requirements specification written to {spec_base_path}", file=sys.stderr)
                else:
                    # Directory specified - generate all specifications
                    spec_base_path.mkdir(parents=True, exist_ok=True)

                    # Generate Requirements.md
                    requirements_content = spec_generator.generate_requirements_from_audit(result)
                    requirements_path = spec_base_path / "Requirements.md"
                    with open(requirements_path, "w", encoding="utf-8") as f:
                        f.write(requirements_content)
                    print(f"Requirements specification written to {requirements_path}", file=sys.stderr)

                    # Generate Design.md
                    design_content = spec_generator.generate_design_from_audit(result)
                    design_path = spec_base_path / "Design.md"
                    with open(design_path, "w", encoding="utf-8") as f:
                        f.write(design_content)
                    print(f"Design specification written to {design_path}", file=sys.stderr)

                    # Generate Implementation.md
                    implementation_content = spec_generator.generate_implementation_from_audit(
                        result
                    )
                    implementation_path = spec_base_path / "Implementation.md"
                    with open(implementation_path, "w", encoding="utf-8") as f:
                        f.write(implementation_content)
                    print(f"Implementation specification written to {implementation_path}", file=sys.stderr)
            else:
                # Default: generate Requirements.md in project directory
                spec_path = project_path / "AUDIT_REQUIREMENTS.md"
                spec_content = spec_generator.generate_requirements_from_audit(result)
                with open(spec_path, "w", encoding="utf-8") as f:
                    f.write(spec_content)
                print(f"Requirements specification written to {spec_path}", file=sys.stderr)

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
            print(f"Analysis JSON written to {json_path}", file=sys.stderr)

        # Exit with error code if critical issues found
        critical_findings = result.get_critical_findings()
        if critical_findings:
            print(
                f"\nWARNING: Found {len(critical_findings)} critical issues that require immediate attention!",
                file=sys.stderr,
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
    from intellirefactor.analysis.decompose.architectural_smell_detector import (
        ArchitecturalSmellDetector,
        SmellThresholds,
        SmellType,
        SmellSeverity,
    )
    from intellirefactor.cli.formatters import format_smells_text, format_smells_json, format_smells_html
    import glob
    
    # Import helpers locally to avoid circular dependency
    def _maybe_print_output(args, output_text_or_json: str) -> None:
        """Print output to appropriate stream."""
        machine_readable = bool(getattr(args, "machine_readable", False))
        if machine_readable:
            json_stdout = getattr(args, "_json_stdout", sys.__stdout__)
            print(output_text_or_json, file=json_stdout)
        else:
            print(output_text_or_json)
    
    def _is_machine_readable(args) -> bool:
        """Check if machine-readable mode is enabled."""
        return bool(getattr(args, "machine_readable", False))

    if args.smells_action == "detect":
        print(f"Starting architectural smell detection for {args.project_path}...", file=sys.stderr)
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
            print(f"Error: Project path '{project_path}' does not exist.", file=sys.stderr)
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
            print("No Python files found matching the specified patterns.", file=sys.stderr)
            return

        print(f"Analyzing {len(python_files)} Python files...", file=sys.stderr)

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
                print(f"Warning: Could not analyze {file_path}: {e}", file=sys.stderr)
                continue

        analysis_time = time.time() - start_time

        print(
            f"Smell detection completed: {len(all_smells)} smells found in {analysis_time:.2f}s",
            file=sys.stderr,
        )

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
            print(f"Results written to {args.output}", file=sys.stderr)
            # In machine-readable mode we MUST still emit JSON to stdout
            if _is_machine_readable(args):
                _maybe_print_output(args, output)
        else:
            _maybe_print_output(args, output)

        # Exit with error code if critical smells found (but not for JSON output to avoid breaking parsing)
        critical_smells = [s for s in all_smells if s.severity == SmellSeverity.CRITICAL]
        if critical_smells and args.format != "json":
            print(
                f"\nWARNING: Found {len(critical_smells)} critical architectural smells that require immediate attention!",
                file=sys.stderr,
            )
            sys.exit(1)
        elif critical_smells and args.format == "json":
            # For JSON output, exit with error code but don't print warning to avoid breaking JSON
            sys.exit(1)

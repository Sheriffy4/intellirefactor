"""
Index and deduplication command handlers.

This module contains commands for:
- Index building, status, and management
- Block-level clone detection
- Semantic similarity analysis
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from intellirefactor.analysis.index.builder import IndexBuildResult


def get_index_db_path(project_path: Path) -> Path:
    """Get the index database path for a project."""
    return project_path / ".intellirefactor" / "index.db"


def format_index_build_result(result: "IndexBuildResult", format_type: str) -> str:
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


def cmd_index(args) -> None:
    """Handle index command."""
    from intellirefactor.analysis.index.builder import IndexBuilder
    from intellirefactor.analysis.index.store import IndexStore
    from intellirefactor.analysis.index.query import IndexQuery
    
    # Import machine-readable helpers from parent module
    from intellirefactor.cli_entry import _maybe_print_output

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
                        file=sys.stderr,
                    )

            with IndexBuilder(db_path, batch_size=args.batch_size) as builder:
                incremental = bool(getattr(args, "incremental", True))
                result = builder.build_index(
                    project_path,
                    incremental=incremental,
                    progress_callback=progress_callback,
                )

            print(file=sys.stderr)  # New line after progress

            output = format_index_build_result(result, args.format)

            if args.output:
                with open(args.output, "w") as f:
                    f.write(output)
                print(f"Build results written to {args.output}", file=sys.stderr)
            else:
                _maybe_print_output(args, output)

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
            print(f"No index found for project: {project_path}", file=sys.stderr)
            print(
                f"Run 'intellirefactor index build {project_path}' to create an index.",
                file=sys.stderr,
            )
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
            _maybe_print_output(args, output)

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
                        file=sys.stderr,
                    )

            with IndexBuilder(db_path, batch_size=args.batch_size) as builder:
                result = builder.rebuild_index(project_path, progress_callback=progress_callback)

            print(file=sys.stderr)  # New line after progress

            output = format_index_build_result(result, args.format)

            if args.output:
                with open(args.output, "w") as f:
                    f.write(output)
                print(f"Rebuild results written to {args.output}", file=sys.stderr)
            else:
                _maybe_print_output(args, output)

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
    from intellirefactor.analysis.dedup.block_extractor import BlockExtractor
    from intellirefactor.analysis.dedup.block_clone_detector import BlockCloneDetector
    from intellirefactor.cli.formatters import format_clone_detection_results, format_similarity_results
    from intellirefactor.cli_entry import _maybe_print_output, _is_machine_readable
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
                print("No Python files found matching the specified patterns.", file=sys.stderr)
                return

            print(
                f"Analyzing {len(python_files)} Python files for block clones...",
                file=sys.stderr,
            )

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

            print(f"Extracted {len(all_blocks)} code blocks", file=sys.stderr)

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
                print(f"Clone detection results written to {args.output}", file=sys.stderr)
                # In machine-readable mode we MUST still emit JSON to stdout
                if _is_machine_readable(args):
                    _maybe_print_output(args, output)
            else:
                _maybe_print_output(args, output)

        elif args.duplicates_action == "methods":
            # Method-level clone detection
            print("Method-level clone detection is not yet implemented.", file=sys.stderr)
            print("Use 'duplicates blocks' for block-level clone detection.", file=sys.stderr)
            sys.exit(1)

        elif args.duplicates_action == "similar":
            # Semantic similarity matching
            from intellirefactor.analysis.dedup.semantic_similarity_matcher import (
                SemanticSimilarityMatcher,
                SimilarityType,
            )
            from intellirefactor.analysis.index.builder import IndexBuilder
            from intellirefactor.analysis.index.store import IndexStore

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

            print(f"Analyzing semantic similarity in project: {project_path}", file=sys.stderr)

            # Build or load index to get method information
            index_db_path = project_path / ".intellirefactor" / "index.db"
            index_db_path.parent.mkdir(parents=True, exist_ok=True)

            if not index_db_path.exists():
                print("Building project index for semantic analysis...", file=sys.stderr)
                index_builder = IndexBuilder(index_db_path)
                index_result = index_builder.build_index(project_path, incremental=False)
                print(f"Index built: {index_result.symbols_found} symbols found", file=sys.stderr)

            index_store = IndexStore(index_db_path)

            # Get method information from index
            methods = index_store.get_all_deep_method_infos()

            if not methods:
                print("No methods found in the project index.", file=sys.stderr)
                print("Try running with different include/exclude patterns.", file=sys.stderr)
                return

            print(f"Found {len(methods)} methods for similarity analysis", file=sys.stderr)

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
                    print(f"Target method '{args.target_method}' not found.", file=sys.stderr)
                    print("Available methods:", file=sys.stderr)
                    for method in methods[:10]:  # Show first 10
                        print(f"  - {method.qualified_name}", file=sys.stderr)
                    if len(methods) > 10:
                        print(f"  ... and {len(methods) - 10} more", file=sys.stderr)
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
                print(f"Similarity results written to {args.output}", file=sys.stderr)
                # In machine-readable mode we MUST still emit JSON to stdout
                if _is_machine_readable(args):
                    _maybe_print_output(args, output)
            else:
                _maybe_print_output(args, output)

    except Exception as e:
        print(f"Error: Failed to detect clones: {e}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        sys.exit(1)

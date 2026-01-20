"""
Main entry point for IntelliRefactor CLI.

This module provides the main() function that serves as the entry point
for the command-line interface.
"""


def main():
    """Main CLI entry point."""
    import sys
    import importlib.util
    from pathlib import Path
    
    try:
        # Import main directly from cli_entry.py module
        cli_entry_path = Path(__file__).parent / "cli_entry.py"
        spec = importlib.util.spec_from_file_location("intellirefactor_cli_entry", cli_entry_path)
        cli_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli_module)
        cli_main = cli_module.main
    except Exception as e:
        print(f"Error: Could not import CLI: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        return 1

    try:
        rv = cli_main()
        return int(rv) if rv is not None else 0
    except SystemExit as e:
        # If CLI calls sys.exit(...)
        return int(getattr(e, "code", 0) or 0)
    except Exception as e:
        print(f"Error: Could not run CLI: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

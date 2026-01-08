"""
Main entry point for IntelliRefactor CLI.

This module provides the main() function that serves as the entry point
for the command-line interface.
"""


def main():
    """Main CLI entry point."""
    # Import CLI main function from cli.py module directly
    import importlib.util
    import sys
    from pathlib import Path

    # Get path to cli.py
    cli_path = Path(__file__).parent / "cli.py"

    try:
        # Load cli.py module directly
        spec = importlib.util.spec_from_file_location("cli_module", cli_path)
        cli_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli_module)

        # Call main function
        cli_module.main()
    except Exception as e:
        print(f"Error: Could not run CLI: {e}")
        print("Please ensure IntelliRefactor is properly installed.")
        sys.exit(1)


if __name__ == "__main__":
    main()

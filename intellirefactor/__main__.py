"""Entry point for running intellirefactor as a module."""

if __name__ == "__main__":
    import sys
    from pathlib import Path
    
    # Add project root to path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    try:
        from intellirefactor.main import main
        main()
    except ImportError:
        # Fallback to direct CLI import
        try:
            from intellirefactor.cli import main
            main()
        except ImportError as e:
            print(f"Error: Could not import CLI module: {e}")
            print("Please run from the project root directory or use the direct runner:")
            print("python run_expert_analysis.py <project_path> <target_file> --detailed")
            sys.exit(1)

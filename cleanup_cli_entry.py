"""
Script to clean up cli_entry.py by removing duplicate function definitions.

This script removes old function definitions that have been extracted to separate modules,
keeping only:
- Imports
- Helper functions (_is_machine_readable, _json_stdout, etc.)
- setup_logging
- create_parser
- main function
"""

from pathlib import Path
import re


def clean_cli_entry():
    """Clean up cli_entry.py by removing duplicate definitions."""
    cli_entry_path = Path("intellirefactor/cli_entry.py")
    
    if not cli_entry_path.exists():
        print(f"Error: {cli_entry_path} not found")
        return False
    
    content = cli_entry_path.read_text(encoding="utf-8")
    lines = content.splitlines()
    
    # Functions that have been extracted and should be removed
    extracted_functions = [
        "format_analysis_result",
        "format_refactoring_result",
        "cmd_analyze",
        "cmd_opportunities",
        "cmd_refactor",
        "cmd_apply",
        "cmd_knowledge",
        "cmd_report",
        "cmd_visualize",
        "cmd_docs",
        "cmd_status",
        "cmd_config",
        "cmd_template",
        "get_index_db_path",
        "format_index_build_result",
        "format_index_status",
        "cmd_index",
        "cmd_duplicates",
        "format_clone_detection_results",
        "format_similarity_results",
        "format_unused_code_results",
        "cmd_unused",
        "format_audit_results",
        "cmd_audit",
        "cmd_smells",
        "format_smells_text",
        "format_smells_json",
        "format_smells_html",
    ]
    
    # Functions to keep in cli_entry.py
    keep_functions = [
        "_is_machine_readable",
        "_json_stdout",
        "_force_json_format_if_possible",
        "_print_json_to_stdout",
        "_maybe_print_output",
        "setup_logging",
        "create_parser",
        "main",
    ]
    
    print(f"Original file: {len(lines)} lines")
    print(f"Functions to remove: {len(extracted_functions)}")
    print(f"Functions to keep: {len(keep_functions)}")
    
    # Find where imports end
    import_end_line = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("# ============="):
            import_end_line = i
            break
    
    print(f"Imports end at line: {import_end_line}")
    
    # Keep everything up to and including the separator
    new_lines = lines[:import_end_line + 1]
    
    # Add helper functions section
    new_lines.append("")
    
    # Find and keep helper functions
    i = import_end_line + 1
    while i < len(lines):
        line = lines[i]
        
        # Check if this is a function definition
        if line.strip().startswith("def "):
            func_match = re.match(r'\s*def\s+(\w+)\s*\(', line)
            if func_match:
                func_name = func_match.group(1)
                
                # Check if this function should be kept
                if func_name in keep_functions:
                    # Find the end of this function
                    func_lines = [line]
                    i += 1
                    indent_level = len(line) - len(line.lstrip())
                    
                    while i < len(lines):
                        next_line = lines[i]
                        # Check if we've reached the next function or end of file
                        if next_line.strip() and not next_line.strip().startswith("#"):
                            next_indent = len(next_line) - len(next_line.lstrip())
                            if next_indent <= indent_level and next_line.strip().startswith("def "):
                                break
                        func_lines.append(next_line)
                        i += 1
                    
                    # Add this function to new_lines
                    new_lines.extend(func_lines)
                    new_lines.append("")
                    continue
        
        i += 1
    
    # Write the cleaned content
    new_content = "\n".join(new_lines)
    
    # Save backup
    backup_path = cli_entry_path.with_suffix(".py.backup")
    cli_entry_path.rename(backup_path)
    print(f"Backup saved to: {backup_path}")
    
    # Write cleaned file
    cli_entry_path.write_text(new_content, encoding="utf-8")
    print(f"Cleaned file: {len(new_lines)} lines")
    print(f"Reduction: {len(lines) - len(new_lines)} lines removed")
    
    return True


if __name__ == "__main__":
    success = clean_cli_entry()
    if success:
        print("\n✓ cli_entry.py cleaned successfully!")
    else:
        print("\n✗ Failed to clean cli_entry.py")

#!/usr/bin/env python3
"""
Script to properly insert the new classes and functions into collect.py
"""

def insert_new_code():
    # Read the file
    with open('intellirefactor/cli/commands/collect.py', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find the insertion point (after _DEFAULT_EXCLUDE_GLOBS list)
    insertion_point = -1
    for i, line in enumerate(lines):
        if '_DEFAULT_EXCLUDE_GLOBS: List[str] = [' in line:
            # Find the closing bracket of the list
            bracket_count = 1
            j = i + 1
            while bracket_count > 0 and j < len(lines):
                for char in lines[j]:
                    if char == '[':
                        bracket_count += 1
                    elif char == ']':
                        bracket_count -= 1
                if bracket_count == 0:
                    insertion_point = j + 1
                    break
                j += 1
            break
    
    if insertion_point == -1:
        print("Could not find _DEFAULT_EXCLUDE_GLOBS list")
        return
    
    # Create the new code to insert
    new_code = [
        '',
        'class _SkipAnalysis(Exception):',
        '    """Internal control-flow: used to skip expensive/duplicate steps deterministically."""',
        '',
        '',
        'def _normpath(p: str) -> str:',
        '    return str(p or "").replace("\\\\", "/")',
        '',
        '',
        'def _same_path(a: str, b: str) -> bool:',
        '    """',
        '    Best-effort path comparison for cases where one side is absolute and the other is relative.',
        '    """',
        '    aa = _normpath(a)',
        '    bb = _normpath(b)',
        '    if not aa or not bb:',
        '        return False',
        '    return aa == bb or aa.endswith(bb) or bb.endswith(aa)',
        '',
        ''
    ]
    
    # Insert the new code
    for i, new_line in enumerate(new_code):
        lines.insert(insertion_point + i, new_line)
    
    # Write the file back
    with open('intellirefactor/cli/commands/collect.py', 'w', encoding='utf-8') as f:
        f.writelines(lines)
    
    print("Successfully inserted new code at line", insertion_point)

if __name__ == '__main__':
    insert_new_code()
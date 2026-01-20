#!/usr/bin/env python3
"""
Comprehensive script to apply all changes from ref.md to collect.py
"""

import re

def apply_all_changes():
    # Read the file
    with open('intellirefactor/cli/commands/collect.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. Add import os (this was already done)
    # Already present: import os
    
    # 2. Add the new classes and functions after _DEFAULT_EXCLUDE_GLOBS
    if 'class _SkipAnalysis(Exception):' not in content:
        # Find the position to insert
        insert_pos = content.find('_DEFAULT_EXCLUDE_GLOBS: List[str] = [')
        if insert_pos != -1:
            # Find the end of the list
            end_bracket_pos = content.find(']', insert_pos)
            if end_bracket_pos != -1:
                # Find the actual end (accounting for nested brackets)
                bracket_count = 1
                pos = end_bracket_pos + 1
                while bracket_count > 0 and pos < len(content):
                    if content[pos] == '[':
                        bracket_count += 1
                    elif content[pos] == ']':
                        bracket_count -= 1
                    pos += 1
                
                # Insert the new code
                new_code = '''

class _SkipAnalysis(Exception):
    """Internal control-flow: used to skip expensive/duplicate steps deterministically."""


def _normpath(p: str) -> str:
    return str(p or "").replace("\\\\", "/")


def _same_path(a: str, b: str) -> bool:
    """
    Best-effort path comparison for cases where one side is absolute and the other is relative.
    """
    aa = _normpath(a)
    bb = _normpath(b)
    if not aa or not bb:
        return False
    return aa == bb or aa.endswith(bb) or bb.endswith(aa)

'''
                content = content[:pos] + new_code + content[pos:]
    
    # 3. Update target_file assignment in _generate_refactoring_path_md
    content = content.replace(
        'target_file = str(hotspots[0].get("file_path"))',
        'target_file = str(hotspots[0].get("file_path") or "")'
    )
    
    # 4. Update file_findings filtering logic
    old_pattern = r'\[f for f in findings if isinstance\(f, dict\) and str\(f\.get\("file_path"\)\) == target_file\]'
    new_pattern = '''[
        f for f in findings if isinstance(f, dict) and _same_path(str(f.get("file_path") or ""), target_file)
    ]'''
    content = re.sub(old_pattern, new_pattern, content)
    
    # 5. Replace RuntimeError exceptions with _SkipAnalysis
    content = content.replace(
        'raise RuntimeError("__SKIP_UNUSED_IN_REFACTOR__")',
        'raise _SkipAnalysis("unused_via_audit")'
    )
    content = content.replace(
        'raise RuntimeError("__SKIP_DEDUP_SEMANTIC__")',
        'raise _SkipAnalysis("dedup_semantic_via_audit")'
    )
    content = content.replace(
        'raise RuntimeError("__SKIP_DEDUP_BLOCK_CLONES__")',
        'raise _SkipAnalysis("dedup_block_clones_via_audit")'
    )
    
    # 6. Update exception handling patterns
    # Update unused code exception handling
    content = re.sub(
        r'except Exception as e:\s*if str\(e\) != "__SKIP_UNUSED_IN_REFACTOR__":\s*writer\.add_warning\(f"Unused code analysis skipped/failed: {e}"\)',
        'except _SkipAnalysis:\n                pass\n            except Exception as e:\n                writer.add_warning(f"Unused code analysis skipped/failed: {e}")',
        content
    )
    
    # Update dedup semantic exception handling
    content = re.sub(
        r'except Exception as e:\s*if str\(e\) != "__SKIP_DEDUP_SEDUP_SEMANTIC__" and str\(e\) != "__SKIP_DEDUP_SEMANTIC__":\s*writer\.add_warning\(f"Dedup semantic similarity skipped/failed: {e}"\)',
        'except _SkipAnalysis:\n                pass\n            except Exception as e:\n                writer.add_warning(f"Dedup semantic similarity skipped/failed: {e}")',
        content
    )
    
    # Update dedup block clones exception handling
    content = re.sub(
        r'except Exception as e:\s*if str\(e\) != "__SKIP_DEDUP_BLOCK_CLONES__":\s*writer\.add_warning\(f"Dedup block clones skipped/failed: {e}"\)',
        'except _SkipAnalysis:\n                pass\n            except Exception as e:\n                writer.add_warning(f"Dedup block clones skipped/failed: {e}")',
        content
    )
    
    # Write the updated content
    with open('intellirefactor/cli/commands/collect.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("All changes from ref.md have been applied successfully!")

if __name__ == '__main__':
    apply_all_changes()
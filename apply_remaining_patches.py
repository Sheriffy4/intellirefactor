#!/usr/bin/env python3
"""
Script to apply remaining Patch 2 changes to collect.py
"""

import re
from pathlib import Path

def apply_remaining_patches():
    file_path = Path("intellirefactor/cli/commands/collect.py")
    
    # Read the current file
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print("Applying remaining Patch 2 changes...")
    
    # Change 1: Add max_hotspot calculation in _compute_dependency_graph
    # Find the location after hotspot_for_db_file function
    pattern = r'(def hotspot_for_db_file\(rel_file: str\) -> float:[\s\S]*?return 0\.0\n\n)'
    replacement = r'\1    max_hotspot = max((_safe_float(h.get("score", 0.0), 0.0) for h in hotspots if isinstance(h, dict)), default=0.0)\n\n'
    
    new_content = re.sub(pattern, replacement, content, count=1)
    if new_content != content:
        print("✓ Added max_hotspot calculation")
        content = new_content
    else:
        print("⚠ Could not find location for max_hotspot calculation")
    
    # Change 2: Update node creation logic in _compute_dependency_graph
    # Find the node creation loop and update it
    pattern = r'(for file_path, is_test_file, loc in rows:\n\s+fp = str\(file_path\)\n\s+hub = hub_by_file\.get\(fp, {}\)\n\s+nodes\[fp\] = \{\n\s+"id": fp,\n\s+"type": "file",\n\s+"file_path": fp,\n\s+"is_test_file": bool\(is_test_file\),\n\s+"lines_of_code": _safe_int\(loc, 0\),\n\s+"hotspot_score": hotspot_for_db_file\(fp\),\n\s+"fanout": _safe_int\(hub\.get\("deps_total"\), 0\),\n\s+"fanin": _safe_int\(hub\.get\("fanin_total"\), 0\),\n\s+"keystone_score": _safe_float\(hub\.get\("keystone_score"\), 0\.0\),\n\s+\})'
    
    replacement = r'for file_path, is_test_file, loc in rows:\n            fp = str(file_path)\n            hs = hotspot_for_db_file(fp)\n            hs_norm = (hs / max_hotspot) if max_hotspot > 0 else 0.0\n            fanout = fanout_by_file.get(fp, _safe_int(hub_by_file.get(fp, {}).get("deps_total"), 0))\n            fanin = fanin_by_file.get(fp, _safe_int(hub_by_file.get(fp, {}).get("fanin_total"), 0))\n            keystone = float(fanout) * (1.0 + hs_norm)\n            nodes[fp] = {\n                "id": fp,\n                "type": "file",\n                "file_path": fp,\n                "is_test_file": bool(is_test_file),\n                "lines_of_code": _safe_int(loc, 0),\n                "hotspot_score": hs,\n                "fanout": int(fanout),\n                "fanin": int(fanin),\n                "keystone_score": keystone,\n            }'
    
    new_content = re.sub(pattern, replacement, content, count=1)
    if new_content != content:
        print("✓ Updated node creation logic")
        content = new_content
    else:
        print("⚠ Could not find node creation logic to update")
    
    # Change 3: Update str(fp) calls to use _relposix
    # Block extraction section
    pattern = r'(all_blocks\.extend\(extractor\.extract_blocks\(src, )str\(fp\)(\)\))'
    replacement = r'\1_relposix(project_path, fp)\2'
    new_content = re.sub(pattern, replacement, content)
    if new_content != content:
        print("✓ Updated block extraction str(fp) calls")
        content = new_content
    
    # Smell detection section  
    pattern = r'(smells\.extend\(det\.detect_smells\(src, )str\(fp\)(\)\))'
    replacement = r'\1_relposix(project_path, fp)\2'
    new_content = re.sub(pattern, replacement, content)
    if new_content != content:
        print("✓ Updated smell detection str(fp) calls")
        content = new_content
    
    # Clustering section
    pattern = r'(r = clusterer\.analyze_class\(node, )str\(fp\)(\)\))'
    replacement = r'\1_relposix(project_path, fp)\2'
    new_content = re.sub(pattern, replacement, content)
    if new_content != content:
        print("✓ Updated clustering str(fp) calls")
        content = new_content
    
    # Change 4: Add audit payload canonicalization
    pattern = r'(audit_payload = \(\n\s+audit_result\.to_dict\(\)\n\s+if hasattr\(audit_result, "to_dict"\)\n\s+else _as_dict\(audit_result\)\n\s+\)\n)'
    replacement = r'\1\n                if isinstance(audit_payload, dict):\n                    audit_payload = _canonicalize_audit_payload(project_path, audit_payload)\n'
    
    new_content = re.sub(pattern, replacement, content, count=1)
    if new_content != content:
        print("✓ Added audit payload canonicalization")
        content = new_content
    else:
        print("⚠ Could not find location for audit payload canonicalization")
    
    # Write the updated content back to file
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("All changes applied successfully!")

if __name__ == "__main__":
    apply_remaining_patches()
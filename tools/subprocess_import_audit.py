"""
Subprocess-based import testing for maximum reliability in CI.

This runs each import in a separate subprocess to avoid:
- Import side effects affecting other tests
- Module caching issues
- Path/environment contamination
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
PKG_DIR = REPO_ROOT / "intellirefactor"


def iter_modules() -> List[str]:
    """Get all modules in the intellirefactor package."""
    modules = []
    for py in PKG_DIR.rglob("*.py"):
        if "__pycache__" in py.parts:
            continue
        rel = py.relative_to(REPO_ROOT)
        if py.name == "__init__.py":
            rel = py.parent.relative_to(REPO_ROOT)
        else:
            rel = rel.with_suffix("")
        modules.append(".".join(rel.parts))
    return sorted(set(modules))


def test_module_in_subprocess(module: str, timeout: float = 5.0) -> Tuple[bool, str]:
    """
    Test importing a single module in a subprocess.
    
    Args:
        module: Module name to import
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (success, error_message)
    """
    # Create a temporary script that just imports the module
    script_content = f"""
import sys
sys.path.insert(0, {repr(str(REPO_ROOT))})
try:
    __import__('{module}')
    print('SUCCESS')
except Exception as e:
    print('FAILURE')
    print(f'{{type(e).__name__}}: {{e}}')
    import traceback
    traceback.print_exc()
"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(script_content)
        temp_script = f.name
    
    try:
        # Run in subprocess
        result = subprocess.run(
            [sys.executable, temp_script],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(REPO_ROOT)
        )
        
        # Clean up temp file
        Path(temp_script).unlink()
        
        # Parse result
        output_lines = result.stdout.strip().split('\n')
        if output_lines and output_lines[0] == 'SUCCESS':
            return True, ""
        else:
            # Get error message (skip SUCCESS/FAILURE line)
            error_msg = '\n'.join(output_lines[1:]) if len(output_lines) > 1 else "Unknown error"
            return False, error_msg
            
    except subprocess.TimeoutExpired:
        return False, f"Timeout after {timeout}s"
    except Exception as e:
        return False, f"Subprocess error: {e}"
    finally:
        # Ensure cleanup
        if Path(temp_script).exists():
            Path(temp_script).unlink()


def main():
    """Main test runner."""
    modules = iter_modules()
    print(f"Testing {len(modules)} modules with subprocess isolation...")
    
    failures = []
    timeouts = []
    
    for i, module in enumerate(modules, 1):
        print(f"[{i}/{len(modules)}] Testing {module}...", end="", flush=True)
        
        success, error = test_module_in_subprocess(module, timeout=10.0)
        
        if success:
            print(" ✓")
        else:
            if "Timeout" in error:
                timeouts.append((module, error))
                print(" ⏱️ TIMEOUT")
            else:
                failures.append((module, error))
                print(" ❌ FAILED")
    
    # Summary
    print("\n=== RESULTS ===")
    print(f"Total modules: {len(modules)}")
    print(f"Successful: {len(modules) - len(failures) - len(timeouts)}")
    print(f"Failures: {len(failures)}")
    print(f"Timeouts: {len(timeouts)}")
    
    if failures:
        print("\n=== FAILURES ===")
        for module, error in failures[:10]:  # Show first 10
            print(f"\n--- {module} ---")
            print(error)
            print("-" * 50)
        if len(failures) > 10:
            print(f"... and {len(failures) - 10} more failures")
    
    if timeouts:
        print("\n=== TIMEOUTS ===")
        for module, error in timeouts:
            print(f"{module}: {error}")
    
    # Create JSON report
    report = {
        "total_modules": len(modules),
        "successful": len(modules) - len(failures) - len(timeouts),
        "failures": [{"module": m, "error": e} for m, e in failures],
        "timeouts": [{"module": m, "error": e} for m, e in timeouts]
    }
    
    report_file = REPO_ROOT / "subprocess_import_audit.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nDetailed report saved to: {report_file}")
    
    # Return exit code for CI
    return 0 if not failures and not timeouts else 1


if __name__ == "__main__":
    exit(main())
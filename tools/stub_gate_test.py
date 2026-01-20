#!/usr/bin/env python3
"""
Stub Gate Test for IntelliRefactor.

Ensures that CLI commands either:
1. Return valid AnalysisReport/JSON results, OR
2. Raise NotImplementedError with proper messaging

This prevents silent failures where commands return empty results without explanation.
"""

import subprocess
import sys
import json


def test_command(cmd_args: list, expect_not_implemented: bool = False) -> dict:
    """
    Test a CLI command and verify its behavior.
    
    Args:
        cmd_args: Command arguments to test
        expect_not_implemented: Whether we expect NotImplementedError
        
    Returns:
        Dictionary with test results
    """
    try:
        # Run command
        result = subprocess.run(
            [sys.executable, "-m", "intellirefactor"] + cmd_args,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Check if it's NotImplementedError
        is_not_implemented = (
            result.returncode != 0 and 
            ("NotImplementedError" in result.stderr or 
             "not implemented" in result.stderr.lower() or
             "not implemented" in result.stdout.lower())
        )
        
        if expect_not_implemented:
            if is_not_implemented:
                return {
                    "command": " ".join(cmd_args),
                    "status": "PASS",
                    "expected": "NotImplementedError",
                    "actual": "NotImplementedError",
                    "message": "Correctly raised NotImplementedError"
                }
            else:
                return {
                    "command": " ".join(cmd_args),
                    "status": "FAIL",
                    "expected": "NotImplementedError", 
                    "actual": "Success or other error",
                    "message": f"Expected NotImplementedError but got return code {result.returncode}"
                }
        else:
            # For implemented commands, check if they return valid output
            if result.returncode == 0:
                # Try to parse as JSON if requested
                if "--machine-readable" in cmd_args or "--format" in cmd_args:
                    try:
                        data = json.loads(result.stdout)
                        return {
                            "command": " ".join(cmd_args),
                            "status": "PASS",
                            "expected": "Valid JSON output",
                            "actual": "Valid JSON output",
                            "message": "Command returned valid JSON"
                        }
                    except json.JSONDecodeError:
                        return {
                            "command": " ".join(cmd_args),
                            "status": "WARN",
                            "expected": "Valid JSON output",
                            "actual": "Non-JSON output",
                            "message": "Command succeeded but didn't return valid JSON"
                        }
                else:
                    # Text output - just check it's not empty
                    if result.stdout.strip():
                        return {
                            "command": " ".join(cmd_args),
                            "status": "PASS",
                            "expected": "Non-empty output",
                            "actual": "Non-empty output", 
                            "message": "Command returned output"
                        }
                    else:
                        return {
                            "command": " ".join(cmd_args),
                            "status": "FAIL",
                            "expected": "Non-empty output",
                            "actual": "Empty output",
                            "message": "Command returned empty output (silent failure)"
                        }
            else:
                # Command failed
                if is_not_implemented:
                    return {
                        "command": " ".join(cmd_args),
                        "status": "INFO",
                        "expected": "Working implementation",
                        "actual": "NotImplementedError",
                        "message": "Command not implemented yet"
                    }
                else:
                    return {
                        "command": " ".join(cmd_args),
                        "status": "FAIL", 
                        "expected": "Working implementation",
                        "actual": f"Error (exit code {result.returncode})",
                        "message": f"Command failed: {result.stderr[:200]}"
                    }
                    
    except subprocess.TimeoutExpired:
        return {
            "command": " ".join(cmd_args),
            "status": "FAIL",
            "expected": "Completes in reasonable time",
            "actual": "Timeout",
            "message": "Command timed out"
        }
    except Exception as e:
        return {
            "command": " ".join(cmd_args),
            "status": "ERROR",
            "expected": "Runs successfully",
            "actual": f"Exception: {e}",
            "message": f"Failed to run command: {e}"
        }


def main():
    """Run stub gate tests."""
    print("🧪 IntelliRefactor Stub Gate Test")
    print("=" * 50)
    
    # Test commands that should work
    working_commands = [
        ["--version"],
        ["--help"],
        ["expert-analyze", ".", "intellirefactor/cli.py", "--format", "json"],
        ["audit", ".", "--format", "json"],
        ["smells", "detect", ".", "--format", "json"],
    ]
    
    # Test commands that might be stubs (we're not sure)
    # These are commands we want to verify behavior for
    unknown_commands = [
        ["index", "build", "."],
        ["index", "status"],
        ["duplicates", "blocks", "."],
        ["unused", "detect", "."],
        ["cluster", "responsibility", "."],
        ["decide", "analyze", "."],
    ]
    
    all_tests = []
    
    print("\n📋 Testing Working Commands:")
    print("-" * 30)
    
    for cmd in working_commands:
        result = test_command(cmd)
        all_tests.append(result)
        status_icon = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "⚠️"
        print(f"{status_icon} {result['command']}")
        if result["status"] != "PASS":
            print(f"    {result['message']}")
    
    print("\n📋 Testing Unknown/Potentially Stub Commands:")
    print("-" * 45)
    
    for cmd in unknown_commands:
        result = test_command(cmd)
        all_tests.append(result)
        status_icon = "✅" if result["status"] == "PASS" else "❌" if result["status"] == "FAIL" else "ℹ️"
        print(f"{status_icon} {result['command']}")
        if result["status"] != "PASS":
            print(f"    {result['message']}")
    
    # Summary
    print("\n📊 Test Summary:")
    print("-" * 20)
    
    passed = sum(1 for t in all_tests if t["status"] == "PASS")
    failed = sum(1 for t in all_tests if t["status"] == "FAIL")
    warnings = sum(1 for t in all_tests if t["status"] == "WARN")
    info = sum(1 for t in all_tests if t["status"] == "INFO")
    
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"⚠️  Warnings: {warnings}")
    print(f"ℹ️  Info: {info}")
    print(f"📋 Total: {len(all_tests)}")
    
    if failed > 0:
        print(f"\n🚨 {failed} tests failed - silent stubs detected!")
        return 1
    else:
        print("\n🎉 All tests passed - no silent stubs found!")
        return 0


if __name__ == "__main__":
    sys.exit(main())

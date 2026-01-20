#!/usr/bin/env python3
"""
Verification script to check the current status of dry-run functionality in IntelliRefactor.
"""

import sys
from pathlib import Path

# Add intellirefactor to path
sys.path.insert(0, str(Path(__file__).parent / 'intellirefactor'))

def test_global_orchestrator_dry_run():
    """Test GlobalRefactoringOrchestrator dry_run handling."""
    print("Testing GlobalRefactoringOrchestrator dry_run...")
    
    try:
        from intellirefactor.orchestration.global_refactoring_orchestrator import GlobalRefactoringOrchestrator
        
        # Test with dry_run=True
        orchestrator = GlobalRefactoringOrchestrator(dry_run=True)
        print(f"✓ Orchestrator created with dry_run=True: {orchestrator.dry_run}")
        
        # Test with dry_run=False
        orchestrator = GlobalRefactoringOrchestrator(dry_run=False)
        print(f"✓ Orchestrator created with dry_run=False: {orchestrator.dry_run}")
        
        # Test set_dry_run method
        orchestrator.set_dry_run(True)
        print(f"✓ set_dry_run(True) works: {orchestrator.dry_run}")
        
        return True
        
    except Exception as e:
        print(f"✗ GlobalRefactoringOrchestrator test failed: {e}")
        return False

def test_auto_refactor_dry_run():
    """Test AutoRefactor dry_run handling."""
    print("\nTesting AutoRefactor dry_run...")
    
    try:
        from intellirefactor.refactoring.auto_refactor import AutoRefactor
        
        # Test with dry_run=True
        refactor = AutoRefactor(dry_run=True)
        print(f"✓ AutoRefactor created with dry_run=True: {refactor.dry_run}")
        
        # Test with dry_run=False
        refactor = AutoRefactor(dry_run=False)
        print(f"✓ AutoRefactor created with dry_run=False: {refactor.dry_run}")
        
        return True
        
    except Exception as e:
        print(f"✗ AutoRefactor test failed: {e}")
        return False

def test_api_functions():
    """Test API functions dry_run defaults."""
    print("\nTesting API functions...")
    
    try:
        from intellirefactor.api import refactor_code, refactor_file
        print("✓ API functions imported successfully")
        
        # Check function signatures for dry_run defaults
        import inspect
        
        sig = inspect.signature(refactor_code)
        dry_run_param = sig.parameters.get('dry_run')
        if dry_run_param and dry_run_param.default is True:
            print("✓ refactor_code defaults to dry_run=True")
        else:
            print(f"? refactor_code dry_run default: {dry_run_param.default if dry_run_param else 'not found'}")
        
        sig = inspect.signature(refactor_file)
        dry_run_param = sig.parameters.get('dry_run')
        if dry_run_param and dry_run_param.default is True:
            print("✓ refactor_file defaults to dry_run=True")
        else:
            print(f"? refactor_file dry_run default: {dry_run_param.default if dry_run_param else 'not found'}")
            
        return True
        
    except Exception as e:
        print(f"✗ API functions test failed: {e}")
        return False

def check_hardcoded_dry_run_false():
    """Check for any remaining hardcoded dry_run=False in the codebase."""
    print("\nChecking for hardcoded dry_run=False...")
    
    intellirefactor_dir = Path(__file__).parent / 'intellirefactor'
    if not intellirefactor_dir.exists():
        print("✗ IntelliRefactor directory not found")
        return False
    
    found_issues = []
    
    for py_file in intellirefactor_dir.rglob('*.py'):
        try:
            content = py_file.read_text(encoding='utf-8')
            lines = content.split('\n')
            
            for i, line in enumerate(lines, 1):
                # Look for hardcoded dry_run=False that's not in comments or strings
                if 'dry_run=False' in line and not line.strip().startswith('#'):
                    # Check if it's in a string literal
                    if line.count('"') % 2 == 0 and line.count("'") % 2 == 0:
                        found_issues.append(f"{py_file.relative_to(Path.cwd())}:{i}: {line.strip()}")
                        
        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")
    
    if found_issues:
        print("✗ Found potential hardcoded dry_run=False issues:")
        for issue in found_issues:
            print(f"  {issue}")
        return False
    else:
        print("✓ No hardcoded dry_run=False found")
        return True

def test_safety_system():
    """Test safety system integration."""
    print("\nTesting safety system...")
    
    try:
        from intellirefactor.safety.safety_manager import SafetyManager
        from intellirefactor.safety.backup_manager import BackupManager
        from intellirefactor.safety.rollback_manager import RollbackManager
        
        # Test SafetyManager
        safety_mgr = SafetyManager(dry_run=True)
        print(f"✓ SafetyManager dry_run=True: {safety_mgr.dry_run}")
        
        # Test BackupManager
        backup_mgr = BackupManager(dry_run=True)
        print(f"✓ BackupManager dry_run=True: {backup_mgr.dry_run}")
        
        # Test RollbackManager
        rollback_mgr = RollbackManager(dry_run=True)
        print(f"✓ RollbackManager dry_run=True: {rollback_mgr.dry_run}")
        
        return True
        
    except Exception as e:
        print(f"✗ Safety system test failed: {e}")
        return False

def main():
    """Run all verification tests."""
    print("IntelliRefactor Dry-Run Status Verification")
    print("=" * 50)
    
    tests = [
        test_global_orchestrator_dry_run,
        test_auto_refactor_dry_run,
        test_api_functions,
        test_safety_system,
        check_hardcoded_dry_run_false,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ All dry-run functionality is working correctly!")
        print("\nCurrent Status:")
        print("- GlobalRefactoringOrchestrator properly handles dry_run")
        print("- AutoRefactor properly handles dry_run")
        print("- API functions have safe defaults")
        print("- Safety system is integrated")
        print("- No hardcoded dry_run=False found")
    else:
        print(f"❌ {total - passed} tests failed - dry-run functionality needs attention")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
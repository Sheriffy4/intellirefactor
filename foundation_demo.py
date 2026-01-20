#!/usr/bin/env python3
"""
Demonstration of IntelliRefactor Foundation Components.

This script shows how to use the standardized foundation models and error handling
to create consistent analysis tools.

Usage:
    python foundation_demo.py
"""

import tempfile
from pathlib import Path

# Import foundation components
from intellirefactor.analysis.foundation import (
    StandardErrorHandler,
    Severity,
    Location,
    Evidence,
    Finding,
    safe_read_file,
    safe_parse_ast
)


def demonstrate_error_handling():
    """Show how to use the standardized error handling."""
    print("=== Error Handling Demo ===")
    
    # Create error handler
    error_handler = StandardErrorHandler()
    
    # Simulate processing a file that doesn't exist
    fake_file = Path("/nonexistent/file.py")
    
    # Safe file reading
    content = safe_read_file(fake_file, error_handler=error_handler)
    
    print(f"File content: {content}")  # Will be None
    print(f"Errors collected: {error_handler.error_count()}")
    
    # Show error details
    for error in error_handler.get_errors():
        print(f"  - {error.stage.value}: {error.message}")
    
    print()


def demonstrate_model_usage():
    """Show how to create standardized findings."""
    print("=== Model Usage Demo ===")
    
    # Create a location
    location = Location(
        file_path="example.py",
        line_start=10,
        line_end=15
    )
    
    # Create evidence
    evidence = Evidence(
        description="External API call detected",
        confidence=0.85,
        locations=[location],
        code_snippets=["response = requests.get(url)"],
        metadata={"api_type": "REST"}
    )
    
    # Create a finding
    finding = Finding(
        id="EXT_API_001",
        type="external_api_call",
        severity=Severity.MEDIUM,
        confidence=0.85,
        title="External API Call Detected",
        description="Method makes external HTTP requests",
        location=location,
        evidence=evidence,
        recommendations=[
            "Consider adding timeout configuration",
            "Add retry logic for resilience"
        ]
    )
    
    # Convert to dictionary (for JSON serialization)
    finding_dict = finding.to_dict()
    print("Finding converted to dict:")
    print(f"  Type: {finding_dict['type']}")
    print(f"  Severity: {finding_dict['severity']}")
    print(f"  Confidence: {finding_dict['confidence']}")
    print()


def demonstrate_safe_execution():
    """Show how to use safe execution utilities."""
    print("=== Safe Execution Demo ===")
    
    error_handler = StandardErrorHandler()
    
    # Create a temporary file with invalid Python
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write("def broken_function(:\n    return invalid_syntax")
        temp_file = Path(f.name)
    
    try:
        # Try to parse the broken file
        source_code = safe_read_file(temp_file, error_handler=error_handler)
        if source_code:
            ast_tree = safe_parse_ast(source_code, temp_file, error_handler=error_handler)
            
            if ast_tree:
                print("AST parsed successfully")
            else:
                print("AST parsing failed")
        else:
            print("File reading failed")
        
        # Show collected errors
        print(f"\nCollected {error_handler.error_count()} errors:")
        for error in error_handler.get_errors():
            print(f"  {error.stage.value} error in {error.file_path}: {error.message}")
            
    finally:
        # Clean up
        temp_file.unlink()


def main():
    """Run all demonstrations."""
    print("IntelliRefactor Foundation Components Demo")
    print("=" * 50)
    print()
    
    demonstrate_error_handling()
    demonstrate_model_usage() 
    demonstrate_safe_execution()
    
    print("Demo completed!")


if __name__ == "__main__":
    main()

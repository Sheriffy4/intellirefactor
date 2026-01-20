#!/usr/bin/env python3
"""
Stub Audit Tool for IntelliRefactor.

Scans the codebase for NotImplementedErrors and generates a structured backlog
of planned features. This provides transparency about what's implemented vs planned.

Features:
- AST-based scanning for accurate detection
- Structured JSON output for backlog management
- Filtering by module/component
- Integration with roadmap steps
"""

import ast
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import argparse
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class StubVisitor(ast.NodeVisitor):
    """AST visitor to find NotImplementedErrors and stub usage."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.stubs_found: List[Dict[str, Any]] = []
        self.current_class: Optional[str] = None
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Track current class context."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions looking for stubs."""
        # Check for explicit raise NotImplementedError
        for stmt in node.body:
            if isinstance(stmt, ast.Raise):
                if self._is_not_implemented_error(stmt.exc):
                    self._record_stub(node, stmt.exc)
        
        # Check for calls to not_implemented() helper
        for stmt in node.body:
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                if self._is_not_implemented_call(stmt.value):
                    self._record_stub_call(node, stmt.value)
                    
        self.generic_visit(node)
        
    def _is_not_implemented_error(self, exc_node: Optional[ast.expr]) -> bool:
        """Check if exception is NotImplementedError."""
        if not exc_node:
            return False
            
        if isinstance(exc_node, ast.Call):
            if isinstance(exc_node.func, ast.Name):
                return exc_node.func.id == "NotImplementedError"
                
        return False
        
    def _is_not_implemented_call(self, call_node: ast.Call) -> bool:
        """Check if call is to not_implemented() helper."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id == "not_implemented"
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr == "not_implemented"
        return False
        
    def _record_stub(self, func_node: ast.FunctionDef, exc_node: ast.Call) -> None:
        """Record a NotImplementedError stub."""
        # Extract message if available
        message = ""
        if exc_node.args:
            arg = exc_node.args[0]
            if isinstance(arg, ast.Constant):
                message = str(arg.value)
                
        self.stubs_found.append({
            "file": str(self.file_path.relative_to(Path.cwd())),
            "line": exc_node.lineno,
            "module": self._get_module_name(),
            "class": self.current_class,
            "function": func_node.name,
            "type": "NotImplementedError",
            "message": message,
            "timestamp": ""
        })
        
    def _record_stub_call(self, func_node: ast.FunctionDef, call_node: ast.Call) -> None:
        """Record a not_implemented() helper call."""
        # Extract arguments
        args = {}
        for i, arg in enumerate(call_node.args):
            if isinstance(arg, ast.Constant):
                if i == 0:
                    args["feature"] = str(arg.value)
                    
        for keyword in call_node.keywords:
            if isinstance(keyword.value, ast.Constant):
                args[keyword.arg] = str(keyword.value)
                
        self.stubs_found.append({
            "file": str(self.file_path.relative_to(Path.cwd())),
            "line": call_node.lineno,
            "module": self._get_module_name(),
            "class": self.current_class,
            "function": func_node.name,
            "type": "not_implemented_call",
            "args": args,
            "timestamp": ""
        })
        
    def _get_module_name(self) -> str:
        """Get module name from file path."""
        rel_path = self.file_path.relative_to(Path.cwd())
        if rel_path.parts[0] == "intellirefactor":
            parts = rel_path.parts[1:]  # Remove 'intellirefactor' prefix
            if parts[-1] == "__init__.py":
                parts = parts[:-1]
            elif parts[-1].endswith(".py"):
                parts = parts[:-1] + (parts[-1][:-3],)
            return ".".join(parts)
        return str(rel_path.with_suffix(""))


def scan_directory(directory: Path, exclude_patterns: List[str] = None) -> List[Dict[str, Any]]:
    """
    Scan directory for Python files containing stubs.
    
    Args:
        directory: Directory to scan
        exclude_patterns: Patterns to exclude (e.g., ['test_*', '*_test.py'])
        
    Returns:
        List of stub entries
    """
    if exclude_patterns is None:
        exclude_patterns = [
            "test_*", "*_test.py", "__pycache__", "*.pyc", 
            "archive/", "docs/", "visualizations/"
        ]
        
    stubs = []
    
    for py_file in directory.rglob("*.py"):
        # Skip excluded patterns
        should_exclude = False
        for pattern in exclude_patterns:
            if py_file.match(pattern) or any(part in pattern for part in py_file.parts):
                should_exclude = True
                break
                
        if should_exclude:
            continue
            
        try:
            logger.debug(f"Scanning {py_file}")
            content = py_file.read_text(encoding='utf-8')
            tree = ast.parse(content, filename=str(py_file))
            
            visitor = StubVisitor(py_file)
            visitor.visit(tree)
            
            stubs.extend(visitor.stubs_found)
            
        except Exception as e:
            logger.warning(f"Failed to parse {py_file}: {e}")
            
    return stubs


def generate_backlog(stubs: List[Dict[str, Any]], output_file: Path) -> None:
    """
    Generate structured backlog JSON.
    
    Args:
        stubs: List of stub entries
        output_file: Output file path
    """
    # Group by planned step if mentioned in message
    backlog = {
        "generated_at": "",
        "total_stubs": len(stubs),
        "by_step": {},
        "by_component": {},
        "stubs": stubs
    }
    
    # Categorize stubs
    for stub in stubs:
        # Extract planned step from message
        message = stub.get("message", "")
        step = "unspecified"
        if "Step " in message:
            # Simple extraction - could be improved with regex
            for word in message.split():
                if word.startswith("Step"):
                    step = word.rstrip(".:,;")
                    break
                    
        if step not in backlog["by_step"]:
            backlog["by_step"][step] = []
        backlog["by_step"][step].append(stub)
        
        # Group by component (module)
        component = stub["module"]
        if component not in backlog["by_component"]:
            backlog["by_component"][component] = []
        backlog["by_component"][component].append(stub)
    
    # Write to file
    output_file.write_text(
        json.dumps(backlog, indent=2, ensure_ascii=False),
        encoding='utf-8'
    )
    
    logger.info(f"Generated backlog with {len(stubs)} stubs in {output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Audit NotImplemented stubs in codebase")
    parser.add_argument(
        "--directory", "-d",
        default=".",
        help="Directory to scan (default: current directory)"
    )
    parser.add_argument(
        "--output", "-o",
        default="stub_backlog.json",
        help="Output JSON file (default: stub_backlog.json)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        
    # Scan for stubs
    directory = Path(args.directory)
    logger.info(f"Scanning directory: {directory.absolute()}")
    
    stubs = scan_directory(directory)
    
    if not stubs:
        logger.info("No stubs found!")
        return
        
    # Generate backlog
    output_file = Path(args.output)
    generate_backlog(stubs, output_file)
    
    # Print summary
    print("\n🔍 Stub Audit Summary")
    print("====================")
    print(f"Total stubs found: {len(stubs)}")
    print(f"Output file: {output_file}")
    
    # Show breakdown by file
    files = {}
    for stub in stubs:
        file_key = stub["file"]
        if file_key not in files:
            files[file_key] = 0
        files[file_key] += 1
        
    print("\nTop files with stubs:")
    for file_path, count in sorted(files.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {file_path}: {count} stub(s)")


if __name__ == "__main__":
    main()

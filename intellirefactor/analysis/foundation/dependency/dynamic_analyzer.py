"""
analysis/dynamic_analyzer.py
Advanced dependency analysis tracking data flow for dynamic imports.
"""

import ast
from typing import Set, Optional, List


class DynamicImportTracker(ast.NodeVisitor):
    def __init__(self):
        self.imports = set()
        self.variable_values = {}  # Simple constant propagation
        self.dynamic_calls = []

    def visit_Assign(self, node: ast.Assign):
        """Track string assignments to variables."""
        if isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.variable_values[target.id] = node.value.value
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        """Detect importlib.import_module and __import__ calls."""
        func_name = self._get_func_name(node.func)

        if func_name in ["importlib.import_module", "__import__", "import_module"]:
            if node.args:
                arg = node.args[0]
                module_name = self._resolve_argument(arg)
                if module_name:
                    self.imports.add(module_name)
                    self.dynamic_calls.append(
                        {
                            "line": node.lineno,
                            "module": module_name,
                            "type": "dynamic_import",
                        }
                    )

        self.generic_visit(node)

    def _get_func_name(self, node) -> str:
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Attribute):
            return f"{self._get_func_name(node.value)}.{node.attr}"
        return ""

    def _resolve_argument(self, arg) -> Optional[str]:
        """Try to resolve the value of an argument."""
        # Direct string literal
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return arg.value

        # Variable reference (check our tracked values)
        if isinstance(arg, ast.Name) and arg.id in self.variable_values:
            return self.variable_values[arg.id]

        # f-string (simplified resolution)
        if isinstance(arg, ast.JoinedStr):
            # This is hard to resolve fully statically, but we can try
            pass

        return None


def analyze_dynamic_dependencies(file_path: str) -> Set[str]:
    with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
        tree = ast.parse(f.read(), filename=file_path)

    tracker = DynamicImportTracker()
    tracker.visit(tree)
    return tracker.imports


def analyze_dynamic_dependencies_json(file_path: str) -> List[str]:
    """JSON-friendly variant (sorted list instead of set)."""
    return sorted(analyze_dynamic_dependencies(file_path))

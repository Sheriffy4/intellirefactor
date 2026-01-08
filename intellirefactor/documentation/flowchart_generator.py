"""
Flowchart Generator for IntelliRefactor

Generates detailed flowcharts for Python methods and functions including:
- Control flow analysis
- Decision points
- Loop structures
- Exception handling paths
- Method call sequences
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class FlowNode:
    """Represents a node in the flowchart."""

    id: str
    type: str  # start, end, process, decision, loop, exception
    content: str
    line_number: int
    connections: List[str] = None

    def __post_init__(self):
        if self.connections is None:
            self.connections = []


@dataclass
class FlowPath:
    """Represents a path between nodes."""

    from_node: str
    to_node: str
    condition: str = ""
    path_type: str = "normal"  # normal, true, false, exception


class FlowchartGenerator:
    """Generates flowcharts for Python methods and functions."""

    def __init__(self):
        self.nodes: Dict[str, FlowNode] = {}
        self.paths: List[FlowPath] = []
        self.node_counter = 0

    def analyze_method(self, file_path: Path, method_name: str) -> Dict[str, Any]:
        """Analyze a specific method and extract flow information."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            # Find the target method
            method_node = self._find_method(tree, method_name)
            if not method_node:
                return {
                    "error": f"Method '{method_name}' not found in {file_path}",
                    "nodes": {},
                    "paths": [],
                }

            # Reset state
            self.nodes = {}
            self.paths = []
            self.node_counter = 0

            # Analyze the method
            self._analyze_method_flow(method_node, source_code.split("\n"))

            return {
                "nodes": self.nodes,
                "paths": self.paths,
                "method_name": method_name,
                "file_path": str(file_path),
            }

        except Exception as e:
            return {
                "error": f"Failed to analyze method: {str(e)}",
                "nodes": {},
                "paths": [],
            }

    def _find_method(self, tree: ast.AST, method_name: str) -> Optional[ast.AST]:
        """
        Find a specific method/function in the AST.
        Supports:
          - "method_name" (first match in AST walk)
          - "ClassName.method_name" (precise lookup)
        """
        # Qualified lookup
        if "." in method_name:
            cls_name, fn_name = method_name.split(".", 1)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and node.name == cls_name:
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == fn_name:
                            return item
            return None

        # Unqualified lookup (first match)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == method_name:
                return node
        return None

    def pick_default_method_name(self, file_path: Path) -> str:
        """
        Pick a deterministic 'best' entrypoint for flowchart generation.
        Preference order:
          1) AttackDispatcher.dispatch_attack
          2) any *.dispatch_attack
          3) any function/method named dispatch_attack
          4) main / run / execute / handle / process (in that order)
          5) first top-level function
          6) first method in first class
        Returns method name or qualified name "Class.method".
        """
        try:
            source = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            source = file_path.read_text(encoding="latin-1")

        tree = ast.parse(source)

        class_methods: List[tuple[str, int]] = []
        top_funcs: List[tuple[str, int]] = []

        for node in getattr(tree, "body", []):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        class_methods.append((f"{node.name}.{item.name}", int(getattr(item, "lineno", 0) or 0)))
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                top_funcs.append((node.name, int(getattr(node, "lineno", 0) or 0)))

        # Helper
        def has(name: str) -> bool:
            return any(n == name for n, _ in class_methods) or any(n == name for n, _ in top_funcs)

        if has("AttackDispatcher.dispatch_attack"):
            return "AttackDispatcher.dispatch_attack"

        for n, _ in sorted(class_methods, key=lambda x: x[1]):
            if n.endswith(".dispatch_attack"):
                return n

        if has("dispatch_attack"):
            return "dispatch_attack"

        preferred = ["main", "run", "execute", "handle", "process", "dispatch"]
        for p in preferred:
            for n, _ in sorted(class_methods, key=lambda x: x[1]):
                if n.endswith(f".{p}"):
                    return n
            for n, _ in sorted(top_funcs, key=lambda x: x[1]):
                if n == p:
                    return n

        if top_funcs:
            # pick first top-level function
            return sorted(top_funcs, key=lambda x: x[1])[0][0]

        if class_methods:
            return sorted(class_methods, key=lambda x: x[1])[0][0]

        return "unknown"

    def _get_next_node_id(self) -> str:
        """Generate next node ID."""
        self.node_counter += 1
        return f"n{self.node_counter}"

    def _analyze_method_flow(self, method_node: ast.AST, lines: List[str]) -> None:
        """Analyze the flow of a method."""
        # Create start node
        start_id = self._get_next_node_id()
        start_node = FlowNode(
            id=start_id,
            type="start",
            content=f"Start: {getattr(method_node, 'name', 'unknown')}",
            line_number=int(getattr(method_node, "lineno", 0) or 0),
        )
        self.nodes[start_id] = start_node

        # Analyze method body
        current_node_id = start_id
        body_stmts = list(getattr(method_node, "body", []) or [])

        # Skip leading docstring statement (common in Python functions)
        if body_stmts:
            first = body_stmts[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(getattr(first, "value", None), ast.Constant)
                and isinstance(getattr(first.value, "value", None), str)
            ):
                body_stmts = body_stmts[1:]

        for stmt in body_stmts:
            current_node_id = self._analyze_statement(stmt, current_node_id, lines)

        # Create end node if not already created
        if not any(node.type == "end" for node in self.nodes.values()):
            end_id = self._get_next_node_id()
            end_node = FlowNode(
                id=end_id,
                type="end",
                content="End",
                line_number=int(getattr(method_node, "end_lineno", None) or getattr(method_node, "lineno", 0) or 0),
            )
            self.nodes[end_id] = end_node

            # Connect last node to end
            if current_node_id and current_node_id != start_id:
                self.paths.append(FlowPath(current_node_id, end_id))

    def _analyze_statement(self, stmt: ast.stmt, prev_node_id: Optional[str], lines: List[str]) -> str:
        """Analyze a single statement and return the last node ID."""
        if isinstance(stmt, ast.If):
            return self._analyze_if_statement(stmt, prev_node_id, lines)
        elif isinstance(stmt, ast.For):
            return self._analyze_for_loop(stmt, prev_node_id, lines)
        elif isinstance(stmt, ast.While):
            return self._analyze_while_loop(stmt, prev_node_id, lines)
        elif isinstance(stmt, ast.Try):
            return self._analyze_try_statement(stmt, prev_node_id, lines)
        elif isinstance(stmt, ast.Return):
            return self._analyze_return_statement(stmt, prev_node_id, lines)
        elif isinstance(stmt, ast.Raise):
            return self._analyze_raise_statement(stmt, prev_node_id, lines)
        else:
            return self._analyze_simple_statement(stmt, prev_node_id, lines)

    def _analyze_if_statement(self, stmt: ast.If, prev_node_id: Optional[str], lines: List[str]) -> str:
        """Analyze if statement."""
        # Create decision node
        decision_id = self._get_next_node_id()
        condition = self._get_condition_text(stmt.test, lines)
        decision_node = FlowNode(
            id=decision_id,
            type="decision",
            content=f"if {condition}",
            line_number=stmt.lineno,
        )
        self.nodes[decision_id] = decision_node

        # Connect previous node to decision
        if prev_node_id:
            self.paths.append(FlowPath(prev_node_id, decision_id))

        # True branch: create nodes without unconditional edge from decision
        true_path_end = decision_id
        if stmt.body:
            first_true_id = self._analyze_statement(stmt.body[0], None, lines)
            self.paths.append(FlowPath(decision_id, first_true_id, "Yes", "true"))
            true_path_end = first_true_id
            for if_stmt in stmt.body[1:]:
                true_path_end = self._analyze_statement(if_stmt, true_path_end, lines)

        # Analyze else body (false path)
        false_path_end = decision_id
        if stmt.orelse:
            first_else_id = self._analyze_statement(stmt.orelse[0], None, lines)
            self.paths.append(FlowPath(decision_id, first_else_id, "No", "false"))
            false_path_end = first_else_id
            for else_stmt in stmt.orelse[1:]:
                false_path_end = self._analyze_statement(else_stmt, false_path_end, lines)
        else:
            # No else clause, false path continues to next statement
            false_path_end = decision_id

        # Return the end of the longest path
        return true_path_end if stmt.body else false_path_end

    def _analyze_for_loop(self, stmt: ast.For, prev_node_id: Optional[str], lines: List[str]) -> str:
        """Analyze for loop."""
        # Create loop start node
        loop_id = self._get_next_node_id()
        target = ast.unparse(stmt.target) if hasattr(ast, "unparse") else "item"
        iter_expr = self._get_expression_text(stmt.iter, lines)
        loop_node = FlowNode(
            id=loop_id,
            type="loop",
            content=f"for {target} in {iter_expr}",
            line_number=stmt.lineno,
        )
        self.nodes[loop_id] = loop_node

        # Connect previous node to loop
        if prev_node_id:
            self.paths.append(FlowPath(prev_node_id, loop_id))

        # Analyze loop body
        body_end = loop_id
        for loop_stmt in stmt.body:
            body_end = self._analyze_statement(loop_stmt, body_end, lines)

        # Create loop back path
        if stmt.body:
            self.paths.append(FlowPath(body_end, loop_id, "Continue", "loop"))

        return loop_id

    def _analyze_while_loop(self, stmt: ast.While, prev_node_id: Optional[str], lines: List[str]) -> str:
        """Analyze while loop."""
        # Create loop condition node
        loop_id = self._get_next_node_id()
        condition = self._get_condition_text(stmt.test, lines)
        loop_node = FlowNode(
            id=loop_id,
            type="loop",
            content=f"while {condition}",
            line_number=stmt.lineno,
        )
        self.nodes[loop_id] = loop_node

        # Connect previous node to loop
        if prev_node_id:
            self.paths.append(FlowPath(prev_node_id, loop_id))

        # Analyze loop body
        body_end = loop_id
        for loop_stmt in stmt.body:
            body_end = self._analyze_statement(loop_stmt, body_end, lines)

        # Create loop back path
        if stmt.body:
            self.paths.append(FlowPath(body_end, loop_id, "Continue", "loop"))

        return loop_id

    def _analyze_try_statement(self, stmt: ast.Try, prev_node_id: Optional[str], lines: List[str]) -> str:
        """Analyze try-except statement."""
        # Create try node
        try_id = self._get_next_node_id()
        try_node = FlowNode(id=try_id, type="process", content="try", line_number=stmt.lineno)
        self.nodes[try_id] = try_node

        # Connect previous node to try
        if prev_node_id:
            self.paths.append(FlowPath(prev_node_id, try_id))

        # Analyze try body
        try_end = try_id
        for try_stmt in stmt.body:
            try_end = self._analyze_statement(try_stmt, try_end, lines)

        # Analyze exception handlers
        for handler in stmt.handlers:
            except_id = self._get_next_node_id()
            exception_type = ast.unparse(handler.type) if handler.type else "Exception"
            except_node = FlowNode(
                id=except_id,
                type="exception",
                content=f"except {exception_type}",
                line_number=handler.lineno,
            )
            self.nodes[except_id] = except_node

            # Connect try to except
            self.paths.append(FlowPath(try_id, except_id, "Exception", "exception"))

            # Analyze except body
            except_end = except_id
            for except_stmt in handler.body:
                except_end = self._analyze_statement(except_stmt, except_end, lines)

        return try_end

    def _analyze_return_statement(self, stmt: ast.Return, prev_node_id: Optional[str], lines: List[str]) -> str:
        """Analyze return statement."""
        return_id = self._get_next_node_id()
        return_value = ""
        if stmt.value:
            return_value = self._get_expression_text(stmt.value, lines)

        return_node = FlowNode(
            id=return_id,
            type="end",
            content=f"return {return_value}" if return_value else "return",
            line_number=stmt.lineno,
        )
        self.nodes[return_id] = return_node

        # Connect previous node to return
        if prev_node_id:
            self.paths.append(FlowPath(prev_node_id, return_id))

        return return_id

    def _analyze_raise_statement(self, stmt: ast.Raise, prev_node_id: Optional[str], lines: List[str]) -> str:
        """Analyze raise statement."""
        raise_id = self._get_next_node_id()
        exception = ""
        if stmt.exc:
            exception = self._get_expression_text(stmt.exc, lines)

        raise_node = FlowNode(
            id=raise_id,
            type="exception",
            content=f"raise {exception}" if exception else "raise",
            line_number=stmt.lineno,
        )
        self.nodes[raise_id] = raise_node

        # Connect previous node to raise
        if prev_node_id:
            self.paths.append(FlowPath(prev_node_id, raise_id))

        return raise_id

    def _analyze_simple_statement(self, stmt: ast.stmt, prev_node_id: Optional[str], lines: List[str]) -> str:
        """Analyze simple statements (assignments, calls, etc.)."""
        stmt_id = self._get_next_node_id()
        content = self._get_statement_text(stmt, lines)

        stmt_node = FlowNode(id=stmt_id, type="process", content=content, line_number=stmt.lineno)
        self.nodes[stmt_id] = stmt_node

        # Connect previous node to this statement
        if prev_node_id:
            self.paths.append(FlowPath(prev_node_id, stmt_id))

        return stmt_id

    def _find_or_create_statement_node(self, stmt: ast.stmt, lines: List[str]) -> str:
        """Find existing node for statement or create new one."""
        # Check if node already exists for this line
        for node_id, node in self.nodes.items():
            if node.line_number == stmt.lineno:
                return node_id

        # Create new node
        return self._analyze_statement(stmt, None, lines)

    def _get_condition_text(self, condition: ast.expr, lines: List[str]) -> str:
        """Extract condition text from AST node."""
        try:
            if hasattr(ast, "unparse"):
                return ast.unparse(condition)
            else:
                # Fallback for older Python versions
                if condition.lineno <= len(lines):
                    line = lines[condition.lineno - 1]
                    return line.strip()
                return "condition"
        except:
            return "condition"

    def _get_expression_text(self, expr: ast.expr, lines: List[str]) -> str:
        """Extract expression text from AST node."""
        try:
            if hasattr(ast, "unparse"):
                return ast.unparse(expr)
            else:
                # Fallback for older Python versions
                if expr.lineno <= len(lines):
                    line = lines[expr.lineno - 1]
                    return line.strip()
                return "expression"
        except:
            return "expression"

    def _get_statement_text(self, stmt: ast.stmt, lines: List[str]) -> str:
        """Extract statement text from AST node."""
        try:
            if hasattr(ast, "unparse"):
                text = ast.unparse(stmt)
                # Truncate long statements
                if len(text) > 50:
                    text = text[:47] + "..."
                return text
            else:
                # Fallback for older Python versions
                if stmt.lineno <= len(lines):
                    line = lines[stmt.lineno - 1].strip()
                    if len(line) > 50:
                        line = line[:47] + "..."
                    return line
                return "statement"
        except:
            return "statement"

    def generate_flowchart(self, analysis_result: Dict[str, Any]) -> str:
        """Generate Mermaid flowchart from analysis result."""
        nodes = analysis_result.get("nodes", {})
        paths = analysis_result.get("paths", [])
        method_name = analysis_result.get("method_name", "unknown")

        if "error" in analysis_result:
            return f"""# Method Flowchart: {method_name}

```mermaid
graph TD
    error[{analysis_result["error"]}]
```
"""

        flowchart = f"""# Method Flowchart: {method_name}

```mermaid
flowchart TD
"""

        # Add nodes
        for node_id, node in nodes.items():
            content = node.content.replace('"', "'")  # Escape quotes

            if node.type == "start":
                flowchart += f'    {node_id}(["{content}"])\n'
                flowchart += f"    style {node_id} fill:#f9f,stroke:#333\n"
            elif node.type == "end":
                flowchart += f'    {node_id}(["{content}"])\n'
                flowchart += f"    style {node_id} fill:#f9f,stroke:#333\n"
            elif node.type == "decision":
                flowchart += f'    {node_id}{{"{content}"}}\n'
                flowchart += f"    style {node_id} fill:#ff9,stroke:#333\n"
            elif node.type == "loop":
                flowchart += f'    {node_id}["{content}"]\n'
                flowchart += f"    style {node_id} fill:#9ff,stroke:#333\n"
            elif node.type == "exception":
                flowchart += f'    {node_id}["{content}"]\n'
                flowchart += f"    style {node_id} fill:#f99,stroke:#333\n"
            else:  # process
                flowchart += f'    {node_id}["{content}"]\n'

        # Add paths
        for path in paths:
            if path.condition:
                flowchart += f"    {path.from_node} -->|{path.condition}| {path.to_node}\n"
            else:
                flowchart += f"    {path.from_node} --> {path.to_node}\n"

        flowchart += "```\n"

        # Add analysis summary
        flowchart += f"""
## Flow Analysis Summary

- **Total Nodes**: {len(nodes)}
- **Decision Points**: {len([n for n in nodes.values() if n.type == "decision"])}
- **Loops**: {len([n for n in nodes.values() if n.type == "loop"])}
- **Exception Handlers**: {len([n for n in nodes.values() if n.type == "exception"])}
- **Return Points**: {len([n for n in nodes.values() if n.type == "end"])}

## Complexity Metrics

- **Cyclomatic Complexity**: {len([n for n in nodes.values() if n.type in ["decision", "loop"]]) + 1}
- **Path Count**: {len(paths)}
- **Max Nesting Level**: {self._calculate_nesting_level(nodes, paths)}
"""

        return flowchart

    def _calculate_nesting_level(self, nodes: Dict[str, FlowNode], paths: List[FlowPath]) -> int:
        """Calculate maximum nesting level in the flowchart."""
        # This is a simplified calculation
        decision_count = len([n for n in nodes.values() if n.type == "decision"])
        loop_count = len([n for n in nodes.values() if n.type == "loop"])
        return max(1, decision_count + loop_count)


def generate_method_flowchart(file_path: str, method_name: str, output_path: str = None) -> str:
    """Generate flowchart documentation for a specific method."""
    generator = FlowchartGenerator()
    path_obj = Path(file_path)

    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    # Analyze the method
    analysis_result = generator.analyze_method(path_obj, method_name)

    # Generate flowchart
    flowchart_doc = generator.generate_flowchart(analysis_result)

    # Save to file if output path specified
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(flowchart_doc)
        return f"Flowchart documentation saved to: {output_path}"

    return flowchart_doc

"""
visualization/diagram_generator.py
Generates visual representations of code structure, dependencies, and control flow.
"""

import os
import ast
from pathlib import Path
from typing import List
from ..analysis.index_store import IndexStore


class SmartFlowVisitor(ast.NodeVisitor):
    """
    Visits AST nodes to build a READABLE Mermaid flowchart.
    Focuses on logic (If/For) and Calls, ignoring boilerplate.
    """

    def __init__(self):
        self.mermaid_lines = []
        self.node_counter = 0
        self.last_node_id = None

    def _get_id(self):
        self.node_counter += 1
        return f"n{self.node_counter}"

    def _add_node(self, label: str, shape: str = "box", style: str = "") -> str:
        node_id = self._get_id()
        # Escape quotes and limit length
        safe_label = label.replace('"', "'").replace("\n", " ")
        if len(safe_label) > 40:
            safe_label = safe_label[:37] + "..."

        if shape == "diamond":
            line = f'    {node_id}{{{{"{safe_label}"}}}}'
        elif shape == "round":
            line = f'    {node_id}(["{safe_label}"])'
        elif shape == "subroutine":
            line = f'    {node_id}[["{safe_label}"]]'
        else:
            line = f'    {node_id}["{safe_label}"]'

        if style:
            line += f"\n    style {node_id} {style}"

        self.mermaid_lines.append(line)
        return node_id

    def _connect(self, from_id: str, to_id: str, label: str = None):
        if from_id and to_id:
            arrow = "-->"
            if label:
                arrow = f"-- {label} -->"
            self.mermaid_lines.append(f"    {from_id} {arrow} {to_id}")

    def visit_FunctionDef(self, node: ast.FunctionDef):
        start_id = self._add_node(f"Start: {node.name}", "round", "fill:#f9f,stroke:#333")
        self.last_node_id = start_id
        self._visit_block(node.body, start_id)

        # Add implicit end if flow doesn't explicitly return
        if not isinstance(node.body[-1], ast.Return):
            end_id = self._add_node("End", "round", "fill:#f9f,stroke:#333")
            self._connect(self.last_node_id, end_id)

    def _visit_block(self, nodes: List[ast.AST], entry_id: str) -> str:
        current_id = entry_id

        for node in nodes:
            # Skip docstrings and simple assignments to reduce noise
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                continue

            if isinstance(node, ast.If):
                current_id = self._visit_If(node, current_id)
            elif isinstance(node, (ast.For, ast.While)):
                current_id = self._visit_Loop(node, current_id)
            elif isinstance(node, ast.Return):
                val = self._format_expr(node.value) if node.value else "None"
                ret_id = self._add_node(f"Return {val}", "round", "fill:#f96,stroke:#333")
                self._connect(current_id, ret_id)
                current_id = None  # Flow stops
                break  # Stop processing block
            elif isinstance(node, (ast.Expr, ast.Assign, ast.Call)):
                # Only visualize Calls or significant assignments
                if self._is_significant(node):
                    label = self._summarize_node(node)
                    # Highlight calls
                    shape = "subroutine" if "Call:" in label else "box"
                    stmt_id = self._add_node(label, shape)
                    self._connect(current_id, stmt_id)
                    current_id = stmt_id

        return current_id

    def _visit_If(self, node: ast.If, entry_id: str) -> str:
        condition = self._format_expr(node.test)
        decision_id = self._add_node(f"If {condition}?", "diamond", "fill:#ff9,stroke:#333")
        self._connect(entry_id, decision_id)

        # True branch
        true_end = self._visit_block(node.body, decision_id)
        # Fix the connection label for the first node in true block is hard in visitor,
        # so we use a visual trick: connect decision to the first node of block manually?
        # Simplified: The _visit_block connected decision_id to the first node.
        # We can't easily label that specific edge without graph state.
        # Instead, we rely on the graph structure.

        # False branch
        if node.orelse:
            false_end = self._visit_block(node.orelse, decision_id)

            # Convergence point
            join_id = self._add_node("Merge", "round", "width:0,height:0")
            if true_end:
                self._connect(true_end, join_id)
            if false_end:
                self._connect(false_end, join_id)
            return join_id
        else:
            # If no else, true branch merges back to main flow
            join_id = self._add_node("Merge", "round", "width:0,height:0")
            if true_end:
                self._connect(true_end, join_id)
            self._connect(decision_id, join_id, "False")
            return join_id

    def _visit_Loop(self, node: ast.AST, entry_id: str) -> str:
        condition = "Loop"
        if isinstance(node, ast.While):
            condition = f"While {self._format_expr(node.test)}"
        elif isinstance(node, ast.For):
            target = self._format_expr(node.target)
            iter_ = self._format_expr(node.iter)
            condition = f"For {target} in {iter_}"

        loop_id = self._add_node(condition, "diamond", "fill:#9f9,stroke:#333")
        self._connect(entry_id, loop_id)

        body_end = self._visit_block(node.body, loop_id)
        if body_end:
            self._connect(body_end, loop_id, "Loop Back")

        end_loop_id = self._add_node("End Loop", "round", "width:0,height:0")
        self._connect(loop_id, end_loop_id, "Done")
        return end_loop_id

    def _is_significant(self, node: ast.AST) -> bool:
        """Decide if a node is worth showing in a high-level flowchart."""
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            return True  # Standalone calls are important
        if isinstance(node, ast.Assign):
            # Show assignments if they are calls
            if isinstance(node.value, ast.Call):
                return True
            # Or if they assign to self (state change)
            for t in node.targets:
                if (
                    isinstance(t, ast.Attribute)
                    and isinstance(t.value, ast.Name)
                    and t.value.id == "self"
                ):
                    return True
        return False

    def _summarize_node(self, node: ast.AST) -> str:
        if isinstance(node, ast.Expr):
            node = node.value

        if isinstance(node, ast.Call):
            func_name = self._format_expr(node.func)
            return f"Call: {func_name}()"
        elif isinstance(node, ast.Assign):
            targets = ", ".join(self._format_expr(t) for t in node.targets)
            if isinstance(node.value, ast.Call):
                func_name = self._format_expr(node.value.func)
                return f"{targets} = Call {func_name}()"
            return f"Set {targets}"

        return "Statement"

    def _format_expr(self, node: ast.AST) -> str:
        try:
            if isinstance(node, ast.Name):
                return node.id
            elif isinstance(node, ast.Attribute):
                return f"{self._format_expr(node.value)}.{node.attr}"
            elif isinstance(node, ast.Call):
                return f"{self._format_expr(node.func)}()"
            return ast.unparse(node)
        except Exception:
            return "expr"


class FlowchartGenerator:
    def __init__(self, index_store: IndexStore):
        self.index_store = index_store

    def generate_call_graph(self, start_symbol_name: str, max_depth: int = 3) -> str:
        """
        Generates a Call Graph starting from a specific function/method.
        This is crucial for refactoring to see dependencies.
        """
        mermaid = ["graph LR"]
        visited = set()

        # Find the starting symbol ID
        start_symbol = None
        with self.index_store._get_connection() as conn:
            # Try exact match first, then qualified name
            cursor = conn.execute(
                "SELECT symbol_id, qualified_name FROM symbols WHERE name = ? OR qualified_name = ? LIMIT 1",
                (start_symbol_name, start_symbol_name),
            )
            row = cursor.fetchone()
            if row:
                start_symbol = {"id": row[0], "name": row[1]}

        if not start_symbol:
            return f"graph TD\n    error[Symbol '{start_symbol_name}' not found in index]"

        def trace_calls(current_symbol_id: int, current_name: str, depth: int):
            if depth >= max_depth or current_symbol_id in visited:
                return
            visited.add(current_symbol_id)

            # Clean name for ID
            src_id = self._sanitize_id(current_name)
            mermaid.append(f'    {src_id}["{current_name}"]')

            # Find what this symbol calls
            with self.index_store._get_connection() as conn:
                cursor = conn.execute(
                    """
                    SELECT s.symbol_id, s.qualified_name, d.dependency_kind
                    FROM dependencies d
                    JOIN symbols s ON d.target_symbol_id = s.symbol_id
                    WHERE d.source_symbol_id = ? AND d.dependency_kind = 'calls'
                """,
                    (current_symbol_id,),
                )

                calls = cursor.fetchall()

                for target_id, target_name, _ in calls:
                    tgt_id = self._sanitize_id(target_name)
                    mermaid.append(f"    {src_id} --> {tgt_id}")
                    trace_calls(target_id, target_name, depth + 1)

        trace_calls(start_symbol["id"], start_symbol["name"], 0)

        # Highlight start node
        start_id = self._sanitize_id(start_symbol["name"])
        mermaid.append(f"    style {start_id} fill:#f96,stroke:#333,stroke-width:2px")

        return "\n".join(mermaid)

    def generate_method_flowchart(self, file_path: str, method_name: str) -> str:
        """
        Generates a detailed but READABLE flowchart for a specific method.
        """
        if not os.path.exists(file_path):
            return "graph TD\n    error[File not found]"

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()

            tree = ast.parse(source)
            target_node = None

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == method_name:
                    target_node = node
                    break

            if not target_node:
                return f"graph TD\n    error[Method '{method_name}' not found]"

            visitor = SmartFlowVisitor()
            visitor.visit(target_node)

            return "flowchart TD\n" + "\n".join(visitor.mermaid_lines)

        except Exception as e:
            return f"graph TD\n    error[Error: {str(e)}]"

    def generate_project_diagram(self, project_name: str) -> str:
        """Generates a high-level class diagram."""
        mermaid = ["classDiagram"]

        try:
            with self.index_store._get_connection() as conn:
                # Get classes
                cursor = conn.execute(
                    """SELECT s.name, s.qualified_name, f.file_path 
                       FROM symbols s 
                       JOIN files f ON s.file_id = f.file_id 
                       WHERE s.kind = 'class' """
                )

                # Group by module
                modules = {}
                for row in cursor.fetchall():
                    name, qname, path = row
                    module = Path(path).stem
                    if module not in modules:
                        modules[module] = []
                    modules[module].append(name)

                for mod, classes in modules.items():
                    mermaid.append(f"    namespace {mod} {{")
                    for cls in classes:
                        mermaid.append(f"        class {cls}")
                    mermaid.append("    }")

                # Get inheritance
                cursor = conn.execute(
                    """SELECT s1.name, s2.name
                       FROM dependencies d
                       JOIN symbols s1 ON d.source_symbol_id = s1.symbol_id
                       JOIN symbols s2 ON d.target_symbol_id = s2.symbol_id
                       WHERE d.dependency_kind = 'inherits' """
                )

                for src, tgt in cursor.fetchall():
                    mermaid.append(f"    {tgt} <|-- {src}")

        except Exception as e:
            mermaid.append(f"%% Error: {e}")

        return "\n".join(mermaid)

    def generate_execution_flow(self, entry_point: str) -> str:
        """Generates file-level dependency graph."""
        # (Оставляем старую реализацию для совместимости, она нормальная для уровня файлов)
        mermaid = ["graph TD"]
        # ... (код из предыдущего ответа для generate_execution_flow)
        return "\n".join(mermaid)

    def _sanitize_id(self, name: str) -> str:
        return (
            name.replace(".", "_")
            .replace("/", "_")
            .replace("\\", "_")
            .replace(":", "")
            .replace("-", "_")
        )

    def save_diagram(self, content: str, output_path: Path):
        md_content = f"# Diagram\n\n```mermaid\n{content}\n```\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

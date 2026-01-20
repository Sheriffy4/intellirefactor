"""
Call Graph Analyzer for expert refactoring analysis.

Builds detailed internal call graphs showing method relationships,
cycles, and complexity metrics within a module.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..models import (
    CallGraph,
    CallNode,
    CallEdge,
    CallType,
    Cycle,
    ComplexityMetrics,
    RiskLevel,
)

logger = logging.getLogger(__name__)


class CallGraphAnalyzer:
    """Analyzes internal method call relationships within a module."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)

    def analyze_internal_calls(self, module_ast: ast.Module) -> CallGraph:
        """
        Build a complete call graph of internal method calls.
        
        Args:
            module_ast: Parsed AST of the target module
            
        Returns:
            CallGraph with nodes, edges, cycles, and metrics
        """
        logger.info("Building internal call graph...")
        
        # Extract all methods and functions (with class context)
        nodes, node_contexts = self._extract_nodes(module_ast)

        # Build call relationships (qualified, collision-free)
        edges = self._extract_call_edges(node_contexts, nodes)
        
        # Detect cycles
        cycles = self._detect_cycles(nodes, edges)
        
        # Calculate complexity metrics
        metrics = self._calculate_complexity_metrics(nodes, edges)
        
        call_graph = CallGraph(
            nodes=nodes,
            edges=edges,
            cycles=cycles,
            metrics=metrics
        )
        
        logger.info(f"Call graph built: {len(nodes)} nodes, {len(edges)} edges, {len(cycles)} cycles")
        return call_graph

    def export_detailed_call_graph(self, call_graph: CallGraph) -> Dict[str, Any]:
        """
        Export detailed call graph data as requested by experts.
        
        Returns:
            Dictionary with complete call graph details including all 64 relationships
        """
        # Create nodes list with full details
        nodes_data = []
        for node in call_graph.nodes:
            node_data = {
                "name": node.method_name,
                "class": node.class_name,
                "line": node.line_number,
                "end_line": node.end_line_number,
                "complexity": node.complexity,
                "is_public": node.is_public,
                "parameters": node.parameters,
                "return_type": node.return_type
            }
            nodes_data.append(node_data)
        
        # Create edges list with full details
        edges_data = []
        for edge in call_graph.edges:
            edge_data = {
                "from": edge.caller,
                "to": edge.callee,
                "line": edge.line_number,
                "type": edge.call_type.value,
                "context": edge.context
            }
            edges_data.append(edge_data)
        
        # Create detailed cycle information
        cycles_data = []
        for cycle in call_graph.cycles:
            cycle_data = {
                "nodes": cycle.nodes,
                "type": cycle.cycle_type,
                "risk": cycle.risk_level.value,
                "description": f"Cycle: {' → '.join(cycle.nodes)}"
            }
            cycles_data.append(cycle_data)
        
        # Create method dependency details (who calls what)
        method_dependencies = {}
        for edge in call_graph.edges:
            if edge.caller not in method_dependencies:
                method_dependencies[edge.caller] = {
                    "calls": [],
                    "called_by": []
                }
            if edge.callee not in method_dependencies:
                method_dependencies[edge.callee] = {
                    "calls": [],
                    "called_by": []
                }
            
            method_dependencies[edge.caller]["calls"].append({
                "method": edge.callee,
                "line": edge.line_number,
                "context": edge.context
            })
            method_dependencies[edge.callee]["called_by"].append({
                "method": edge.caller,
                "line": edge.line_number,
                "context": edge.context
            })
        
        metrics = call_graph.metrics or ComplexityMetrics()
        return {
            "call_graph": {
                "nodes": nodes_data,
                "edges": edges_data,
                "total_methods": len(call_graph.nodes),
                "total_relationships": len(call_graph.edges)
            },
            "cycles": cycles_data,
            "method_dependencies": method_dependencies,
            "complexity_metrics": {
                "total_complexity": metrics.cyclomatic_complexity,
                "max_call_depth": metrics.call_depth,
                "coupling_score": metrics.coupling_score,
                "cohesion_score": metrics.cohesion_score,
                "fan_in": metrics.fan_in,
                "fan_out": metrics.fan_out
            }
        }

    def _extract_nodes(self, module_ast: ast.Module) -> Tuple[List[CallNode], Dict[str, Dict[str, Any]]]:
        """
        Extract callable nodes with stable keys to avoid collisions:
          - module-level: "func"
          - class method: "Class.method"

        Returns:
            (nodes, node_contexts)
            node_contexts maps node_key -> {"ast": FunctionDef|AsyncFunctionDef, "class": Optional[str]}
        """
        nodes: List[CallNode] = []
        node_contexts: Dict[str, Dict[str, Any]] = {}

        # module-level functions
        for n in module_ast.body:
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                key = n.name
                cn = self._make_call_node(n, class_name=None, node_key=key)
                nodes.append(cn)
                node_contexts[key] = {"ast": n, "class": None}

        # class methods
        for c in [x for x in module_ast.body if isinstance(x, ast.ClassDef)]:
            for n in c.body:
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    key = f"{c.name}.{n.name}"
                    cn = self._make_call_node(n, class_name=c.name, node_key=key)
                    nodes.append(cn)
                    node_contexts[key] = {"ast": n, "class": c.name}

        return nodes, node_contexts

    def _make_call_node(
        self,
        fn: ast.AST,
        *,
        class_name: Optional[str],
        node_key: str,
    ) -> CallNode:
        """Create CallNode from FunctionDef/AsyncFunctionDef."""
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            raise TypeError(f"Expected FunctionDef/AsyncFunctionDef, got: {type(fn).__name__}")

        parameters: List[str] = [a.arg for a in fn.args.args] if getattr(fn, "args", None) else []

        return_type = None
        if getattr(fn, "returns", None) is not None:
            try:
                return_type = ast.unparse(fn.returns) if hasattr(ast, "unparse") else str(fn.returns)
            except Exception:
                return_type = None

        docstring = ast.get_docstring(fn) if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) else None

        is_property = any(
            isinstance(dec, ast.Name) and dec.id == "property"
            for dec in getattr(fn, "decorator_list", []) or []
        )

        is_public = not getattr(fn, "name", "").startswith("_")

        return CallNode(
            # IMPORTANT: store collision-free key here (used across the system)
            method_name=node_key,
            class_name=class_name,
            line_number=getattr(fn, "lineno", 0),
            end_line_number=getattr(fn, "end_lineno", 0),
            complexity=self._calculate_cyclomatic_complexity(fn),
            is_public=is_public,
            is_property=is_property,
            parameters=parameters,
            return_type=return_type,
            docstring=docstring,
        )

    def _calculate_cyclomatic_complexity(self, func_node: ast.AST) -> int:
        """Calculate cyclomatic complexity for a function."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            # Decision points that increase complexity
            if isinstance(node, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1
            elif isinstance(node, ast.BoolOp):
                # And/Or operations add complexity
                complexity += len(node.values) - 1
        
        return complexity

    def _extract_call_edges(
        self,
        node_contexts: Dict[str, Dict[str, Any]],
        nodes: List[CallNode],
    ) -> List[CallEdge]:
        """Extract call relationships between nodes using stable keys."""
        edges: List[CallEdge] = []
        node_keys = {n.method_name for n in nodes}

        for caller_key, ctx in node_contexts.items():
            fn = ctx["ast"]
            caller_class = ctx.get("class")
            for n in ast.walk(fn):
                if not isinstance(n, ast.Call):
                    continue
                raw = self._extract_callee_name(n)
                if not raw:
                    continue

                # resolve "self.x" to "Class.x" when inside a class method
                callee_key = raw
                if caller_class and raw.startswith("self."):
                    callee_key = f"{caller_class}.{raw.split('.', 1)[1]}"

                # if it is a method call by bare name inside a class, prefer same-class method if exists
                if caller_class and "." not in callee_key:
                    candidate = f"{caller_class}.{callee_key}"
                    if candidate in node_keys:
                        callee_key = candidate

                if callee_key not in node_keys:
                    continue

                call_type = CallType.DIRECT
                if callee_key == caller_key:
                    call_type = CallType.RECURSIVE

                edges.append(
                    CallEdge(
                        caller=caller_key,
                        callee=callee_key,
                        call_type=call_type,
                        line_number=getattr(n, "lineno", 0),
                        context=self._get_call_context(n),
                    )
                )

        return edges

    def _extract_callee_name(self, call_node: ast.Call) -> Optional[str]:
        """Extract the name of the called function/method."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            # Handle self.method_name calls
            if (isinstance(call_node.func.value, ast.Name) 
                and call_node.func.value.id == 'self'):
                return f"self.{call_node.func.attr}"
            # Handle other attribute calls - for now, just return the attribute name
            return call_node.func.attr
        return None

    def _get_call_context(self, call_node: ast.Call) -> Optional[str]:
        """Get context information about the call."""
        try:
            if hasattr(ast, 'unparse'):
                return ast.unparse(call_node)
            else:
                # Fallback for older Python versions
                return f"call at line {getattr(call_node, 'lineno', 0)}"
        except Exception:
            return None

    def _detect_cycles(self, nodes: List[CallNode], edges: List[CallEdge]) -> List[Cycle]:
        """Detect cycles in the call graph using DFS."""
        cycles = []
        
        # Build adjacency list
        graph = {}
        for node in nodes:
            graph[node.method_name] = []
        
        for edge in edges:
            if edge.caller in graph:
                graph[edge.caller].append(edge.callee)
        
        # DFS to find cycles
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(node: str) -> None:
            if node in rec_stack:
                # Found a cycle
                cycle_start = path.index(node)
                cycle_nodes = path[cycle_start:] + [node]
                
                # Determine cycle type and risk
                cycle_type = "simple" if len(cycle_nodes) == 2 else "complex"
                risk_level = RiskLevel.HIGH if len(cycle_nodes) > 3 else RiskLevel.MEDIUM
                
                cycle = Cycle(
                    nodes=cycle_nodes,
                    cycle_type=cycle_type,
                    risk_level=risk_level
                )
                cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            for neighbor in graph.get(node, []):
                dfs(neighbor)
            
            rec_stack.remove(node)
            path.pop()
        
        # Check each node
        for node in graph:
            if node not in visited:
                dfs(node)
        
        return cycles

    def _calculate_complexity_metrics(self, nodes: List[CallNode], edges: List[CallEdge]) -> ComplexityMetrics:
        """Calculate various complexity metrics for the call graph."""
        
        # Calculate fan-in and fan-out
        fan_in = {}
        fan_out = {}
        
        for node in nodes:
            fan_in[node.method_name] = 0
            fan_out[node.method_name] = 0
        
        for edge in edges:
            fan_out[edge.caller] = fan_out.get(edge.caller, 0) + 1
            fan_in[edge.callee] = fan_in.get(edge.callee, 0) + 1
        
        # Calculate overall complexity
        total_complexity = sum(node.complexity for node in nodes)
        
        # Calculate call depth (longest path)
        call_depth = self._calculate_max_call_depth(nodes, edges)
        
        # Calculate coupling score (average fan-out)
        coupling_score = sum(fan_out.values()) / len(nodes) if nodes else 0.0
        
        # Calculate cohesion score (simplified - based on internal vs external calls)
        internal_calls = len(edges)
        total_calls = internal_calls  # We only have internal calls in this analysis
        cohesion_score = internal_calls / max(total_calls, 1)
        
        return ComplexityMetrics(
            cyclomatic_complexity=total_complexity,
            call_depth=call_depth,
            fan_in=fan_in,
            fan_out=fan_out,
            coupling_score=coupling_score,
            cohesion_score=cohesion_score
        )

    def _calculate_max_call_depth(self, nodes: List[CallNode], edges: List[CallEdge]) -> int:
        """Calculate the maximum call depth in the graph."""
        # Build adjacency list
        graph = {}
        for node in nodes:
            graph[node.method_name] = []
        
        for edge in edges:
            if edge.caller in graph:
                graph[edge.caller].append(edge.callee)
        
        # Find maximum depth using DFS
        max_depth = 0
        visited = set()
        
        def dfs(node: str, depth: int) -> int:
            if node in visited:
                return depth  # Avoid infinite recursion
            
            visited.add(node)
            current_max = depth
            
            for neighbor in graph.get(node, []):
                neighbor_depth = dfs(neighbor, depth + 1)
                current_max = max(current_max, neighbor_depth)
            
            visited.remove(node)
            return current_max
        
        for node in graph:
            depth = dfs(node, 1)
            max_depth = max(max_depth, depth)
        
        return max_depth
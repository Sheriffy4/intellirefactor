"""
Call Graph Analyzer for expert refactoring analysis.

Builds detailed internal call graphs showing method relationships,
cycles, and complexity metrics within a module.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List, Optional

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
        
        # Extract all methods and functions
        nodes = self._extract_nodes(module_ast)
        
        # Build call relationships
        edges = self._extract_call_edges(module_ast, nodes)
        
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

    def export_detailed_call_graph(self, call_graph: CallGraph) -> Dict[str, any]:
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
                "total_complexity": call_graph.metrics.cyclomatic_complexity,
                "max_call_depth": call_graph.metrics.call_depth,
                "coupling_score": call_graph.metrics.coupling_score,
                "cohesion_score": call_graph.metrics.cohesion_score,
                "fan_in": call_graph.metrics.fan_in,
                "fan_out": call_graph.metrics.fan_out
            }
        }

    def _extract_nodes(self, module_ast: ast.Module) -> List[CallNode]:
        """Extract all callable nodes (functions and methods) from the AST."""
        nodes = []
        
        for node in ast.walk(module_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Determine if this is a method (inside a class)
                class_name = None
                for parent in ast.walk(module_ast):
                    if isinstance(parent, ast.ClassDef):
                        if node in ast.walk(parent):
                            class_name = parent.name
                            break
                
                # Extract parameters
                parameters = []
                if node.args:
                    parameters = [arg.arg for arg in node.args.args]
                
                # Extract return type annotation
                return_type = None
                if hasattr(node, 'returns') and node.returns:
                    return_type = ast.unparse(node.returns) if hasattr(ast, 'unparse') else str(node.returns)
                
                # Extract docstring
                docstring = None
                if (node.body and isinstance(node.body[0], ast.Expr) 
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                    docstring = node.body[0].value.value
                
                # Check if it's a property
                is_property = any(
                    isinstance(decorator, ast.Name) and decorator.id == 'property'
                    for decorator in node.decorator_list
                )
                
                # Determine if public (not starting with _)
                is_public = not node.name.startswith('_')
                
                call_node = CallNode(
                    method_name=node.name,
                    class_name=class_name,
                    line_number=getattr(node, 'lineno', 0),
                    end_line_number=getattr(node, 'end_lineno', 0),
                    complexity=self._calculate_cyclomatic_complexity(node),
                    is_public=is_public,
                    is_property=is_property,
                    parameters=parameters,
                    return_type=return_type,
                    docstring=docstring
                )
                nodes.append(call_node)
        
        return nodes

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

    def _extract_call_edges(self, module_ast: ast.Module, nodes: List[CallNode]) -> List[CallEdge]:
        """Extract call relationships between methods."""
        edges = []
        node_names = {node.method_name for node in nodes}
        
        # Create a mapping of method names to their containing functions
        method_contexts = {}
        
        for node in ast.walk(module_ast):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_contexts[node.name] = node
        
        # Find calls within each method
        for method_name, method_node in method_contexts.items():
            caller = method_name
            
            for node in ast.walk(method_node):
                if isinstance(node, ast.Call):
                    callee = self._extract_callee_name(node)
                    if callee and callee in node_names and callee != caller:
                        # Determine call type
                        call_type = CallType.DIRECT
                        if callee == caller:
                            call_type = CallType.RECURSIVE
                        
                        edge = CallEdge(
                            caller=caller,
                            callee=callee,
                            call_type=call_type,
                            line_number=getattr(node, 'lineno', 0),
                            context=self._get_call_context(node)
                        )
                        edges.append(edge)
        
        return edges

    def _extract_callee_name(self, call_node: ast.Call) -> Optional[str]:
        """Extract the name of the called function/method."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            # Handle self.method_name calls
            if (isinstance(call_node.func.value, ast.Name) 
                and call_node.func.value.id == 'self'):
                return call_node.func.attr
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
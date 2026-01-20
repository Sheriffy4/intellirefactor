"""
Data Schema Analyzer for expert refactoring analysis.

Detects dictionary-key usage patterns and builds best-effort "schema" summaries
to reduce refactoring risk (KeyError, inconsistent access patterns, etc.).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DataSchemaAnalyzer:
    """Analyzes data schemas and key usage patterns."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)

    def analyze_data_schemas(self, module_ast: ast.Module) -> Dict[str, Any]:
        """
        Analyze data schemas used in the module.
        
        Args:
            module_ast: Parsed AST of the target module
            
        Returns:
            Dictionary with data schema analysis
        """
        logger.info("Analyzing data schemas...")
        
        # Analyze dictionary key usage patterns
        key_usage = self._analyze_dictionary_keys(module_ast)
        
        # Analyze parameter schemas
        parameter_schemas = self._analyze_parameter_schemas(module_ast, key_usage)
        
        # Analyze data flow patterns
        data_flow = self._analyze_data_flow_patterns(module_ast)
        
        # Generate schema recommendations
        schema_recommendations = self._generate_schema_recommendations(key_usage, parameter_schemas)
        
        key_usage_serializable = self._make_key_usage_json_safe(key_usage)

        return {
            "key_usage_map": key_usage_serializable,
            "parameter_schemas": parameter_schemas,
            "data_flow_patterns": data_flow,
            "schema_recommendations": schema_recommendations,
            "refactoring_risks": self._assess_schema_refactoring_risks(key_usage_serializable, parameter_schemas)
        }

    def _make_key_usage_json_safe(self, key_usage: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        Ensure key_usage_map is JSON-serializable:
          - convert sets to lists
        """
        out: Dict[str, Dict[str, Any]] = {}
        for key, dicts in (key_usage or {}).items():
            out[key] = {}
            for dict_name, usage_info in (dicts or {}).items():
                if not isinstance(usage_info, dict):
                    out[key][dict_name] = usage_info
                    continue
                ui = dict(usage_info)
                m = ui.get("methods")
                if isinstance(m, set):
                    ui["methods"] = sorted(list(m))
                out[key][dict_name] = ui
        return out

    def _analyze_dictionary_keys(self, module_ast: ast.Module) -> Dict[str, Dict[str, Any]]:
        """Analyze all dictionary key usage in the module."""
        key_usage: Dict[str, Dict[str, Any]] = {}

        class _Visitor(ast.NodeVisitor):
            def __init__(self, outer: "DataSchemaAnalyzer"):
                self.outer = outer
                self.stack: List[str] = []

            def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
                self.stack.append(node.name)
                self.generic_visit(node)
                self.stack.pop()

            def _current_method(self) -> str:
                return self.stack[-1] if self.stack else "module"

            def visit_Subscript(self, node: ast.Subscript) -> Any:
                key_info = self.outer._extract_subscript_key(node)
                if key_info:
                    self.outer._record_key_usage(
                        key_usage, key_info, "subscript", method=self._current_method()
                    )
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> Any:
                if isinstance(node.func, ast.Attribute) and node.func.attr == "get":
                    key_info = self.outer._extract_get_key(node)
                    if key_info:
                        self.outer._record_key_usage(
                            key_usage, key_info, "get", method=self._current_method()
                        )
                self.generic_visit(node)

            def visit_Compare(self, node: ast.Compare) -> Any:
                key_info = self.outer._extract_in_key(node)
                if key_info:
                    self.outer._record_key_usage(
                        key_usage, key_info, "in_check", method=self._current_method()
                    )
                self.generic_visit(node)

        _Visitor(self).visit(module_ast)
        return key_usage

    def _extract_subscript_key(self, subscript_node: ast.Subscript) -> Optional[Dict[str, Any]]:
        """Extract key information from dict[key] access."""
        if isinstance(subscript_node.slice, ast.Constant):
            if isinstance(subscript_node.slice.value, str):
                return {
                    "key": subscript_node.slice.value,
                    "dict_name": self._extract_dict_name(subscript_node.value),
                    "line": getattr(subscript_node, 'lineno', 0)
                }
        return None

    def _extract_get_key(self, call_node: ast.Call) -> Optional[Dict[str, Any]]:
        """Extract key information from dict.get(key) call."""
        if call_node.args and isinstance(call_node.args[0], ast.Constant):
            if isinstance(call_node.args[0].value, str):
                return {
                    "key": call_node.args[0].value,
                    "dict_name": self._extract_dict_name(call_node.func.value),
                    "line": getattr(call_node, 'lineno', 0),
                    "has_default": len(call_node.args) > 1
                }
        return None

    def _extract_in_key(self, compare_node: ast.Compare) -> Optional[Dict[str, Any]]:
        """Extract key information from 'key' in dict check."""
        if (len(compare_node.ops) == 1 and 
            isinstance(compare_node.ops[0], ast.In) and
            isinstance(compare_node.left, ast.Constant) and
            isinstance(compare_node.left.value, str)):
            
            return {
                "key": compare_node.left.value,
                "dict_name": self._extract_dict_name(compare_node.comparators[0]),
                "line": getattr(compare_node, 'lineno', 0)
            }
        return None

    def _extract_dict_name(self, dict_node: ast.AST) -> str:
        """Extract the name of the dictionary being accessed."""
        if isinstance(dict_node, ast.Name):
            return dict_node.id
        elif isinstance(dict_node, ast.Attribute):
            return dict_node.attr
        else:
            return "unknown_dict"

    def _record_key_usage(self, key_usage: Dict[str, Dict[str, Any]], key_info: Dict[str, Any], access_type: str, *, method: str):
        """Record a key usage instance."""
        key = key_info["key"]
        dict_name = key_info["dict_name"]
        
        # Create nested structure: key_usage[key][dict_name][access_type]
        if key not in key_usage:
            key_usage[key] = {}
        
        if dict_name not in key_usage[key]:
            key_usage[key][dict_name] = {
                "subscript": [],
                "get": [],
                "in_check": [],
                "methods": set(),
                "total_usage": 0
            }
        
        # Record this usage
        usage_info: Dict[str, Any] = {
            "line": key_info["line"],
            "method": method,
            "has_default": key_info.get("has_default", False)
        }
        
        key_usage[key][dict_name][access_type].append(usage_info)
        key_usage[key][dict_name]["methods"].add(usage_info["method"])
        key_usage[key][dict_name]["total_usage"] += 1

    # _find_containing_method removed: we now track method via visitor stack

    def _analyze_parameter_schemas(self, module_ast: ast.Module, key_usage: Dict) -> Dict[str, Dict[str, Any]]:
        """Analyze parameter schemas based on key usage."""
        schemas = {}
        
        # Group keys by likely parameter names
        param_candidates = ["params", "packet_info", "config", "options", "kwargs"]
        
        for param_name in param_candidates:
            if any(param_name in key_usage[key] for key in key_usage):
                schema = self._build_parameter_schema(param_name, key_usage)
                if schema["keys"]:
                    schemas[param_name] = schema
        
        return schemas

    def _build_parameter_schema(self, param_name: str, key_usage: Dict) -> Dict[str, Any]:
        """Build a schema for a specific parameter."""
        schema = {
            "parameter_name": param_name,
            "keys": {},
            "total_keys": 0,
            "usage_patterns": {}
        }
        
        # Collect all keys used with this parameter
        for key, dict_usage in key_usage.items():
            if param_name in dict_usage:
                usage_info = dict_usage[param_name]
                
                schema["keys"][key] = {
                    "total_usage": usage_info["total_usage"],
                    "access_patterns": {
                        "subscript": len(usage_info["subscript"]),
                        "get": len(usage_info["get"]),
                        "in_check": len(usage_info["in_check"])
                    },
                    "methods_using": list(usage_info["methods"]),
                    "has_safe_access": len(usage_info["get"]) > 0 or len(usage_info["in_check"]) > 0,
                    "has_unsafe_access": len(usage_info["subscript"]) > 0
                }
        
        schema["total_keys"] = len(schema["keys"])
        
        # Analyze usage patterns
        safe_keys = sum(1 for key_info in schema["keys"].values() if key_info["has_safe_access"])
        unsafe_keys = sum(1 for key_info in schema["keys"].values() if key_info["has_unsafe_access"])
        
        schema["usage_patterns"] = {
            "safe_access_keys": safe_keys,
            "unsafe_access_keys": unsafe_keys,
            "safety_ratio": safe_keys / max(safe_keys + unsafe_keys, 1)
        }
        
        return schema

    def _analyze_data_flow_patterns(self, module_ast: ast.Module) -> Dict[str, Any]:
        """Analyze how data flows through the module."""
        data_flow = {
            "parameter_passing": [],
            "data_transformations": [],
            "return_patterns": []
        }
        
        # Analyze function calls and parameter passing
        for node in ast.walk(module_ast):
            if isinstance(node, ast.Call):
                # Analyze how dictionaries are passed to functions
                for arg in node.args:
                    if isinstance(arg, ast.Name):
                        data_flow["parameter_passing"].append({
                            "line": getattr(node, 'lineno', 0),
                            "function": self._extract_function_name(node.func),
                            "parameter": arg.id,
                            "type": "direct_pass"
                        })
            
            elif isinstance(node, ast.Dict):
                # Analyze dictionary construction
                if node.keys and node.values:
                    dict_info = {
                        "line": getattr(node, 'lineno', 0),
                        "keys": [],
                        "construction_type": "literal"
                    }
                    
                    for key in node.keys:
                        if isinstance(key, ast.Constant) and isinstance(key.value, str):
                            dict_info["keys"].append(key.value)
                    
                    data_flow["data_transformations"].append(dict_info)
        
        return data_flow

    def _extract_function_name(self, func_node: ast.AST) -> str:
        """Extract function name from a call."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            return func_node.attr
        else:
            return "unknown_function"

    def _generate_schema_recommendations(self, key_usage: Dict, parameter_schemas: Dict) -> List[str]:
        """Generate recommendations for schema improvements."""
        recommendations = []
        
        # Analyze key usage patterns
        for key, dict_usage in key_usage.items():
            total_usage = sum(info["total_usage"] for info in dict_usage.values())
            
            if total_usage > 10:
                recommendations.append(f"Key '{key}' is heavily used ({total_usage} times) - consider creating a typed parameter class")
        
        # Analyze parameter schemas
        for param_name, schema in parameter_schemas.items():
            if schema["usage_patterns"]["safety_ratio"] < 0.5:
                recommendations.append(f"Parameter '{param_name}' has unsafe access patterns - add validation or use .get() method")
            
            if schema["total_keys"] > 15:
                recommendations.append(f"Parameter '{param_name}' has many keys ({schema['total_keys']}) - consider breaking into smaller, focused parameters")
        
        # Check for common key patterns
        common_keys = {}
        for key in key_usage:
            if len(key_usage[key]) > 1:  # Used in multiple dictionaries
                common_keys[key] = len(key_usage[key])
        
        if common_keys:
            most_common = max(common_keys.items(), key=lambda x: x[1])
            recommendations.append(f"Key '{most_common[0]}' is used across {most_common[1]} different dictionaries - consider standardizing")
        
        return recommendations

    def _assess_schema_refactoring_risks(self, key_usage: Dict, parameter_schemas: Dict) -> List[str]:
        """Assess risks related to data schemas during refactoring."""
        risks = []
        
        # Check for unsafe access patterns
        unsafe_keys = []
        for key, dict_usage in key_usage.items():
            for dict_name, usage_info in dict_usage.items():
                if len(usage_info["subscript"]) > 0 and len(usage_info["get"]) == 0:
                    unsafe_keys.append(f"{dict_name}['{key}']")
        
        if unsafe_keys:
            risks.append(f"Unsafe dictionary access found: {', '.join(unsafe_keys[:5])} - may cause KeyError during refactoring")
        
        # Check for inconsistent key usage
        for param_name, schema in parameter_schemas.items():
            inconsistent_keys = []
            for key, key_info in schema["keys"].items():
                if key_info["has_safe_access"] and key_info["has_unsafe_access"]:
                    inconsistent_keys.append(key)
            
            if inconsistent_keys:
                risks.append(f"Parameter '{param_name}' has inconsistent access patterns for keys: {', '.join(inconsistent_keys)}")
        
        # Check for high-usage keys without validation
        for key, dict_usage in key_usage.items():
            total_usage = sum(info["total_usage"] for info in dict_usage.values())
            has_validation = any(len(info["in_check"]) > 0 for info in dict_usage.values())
            
            if total_usage > 5 and not has_validation:
                risks.append(f"High-usage key '{key}' ({total_usage} uses) lacks validation checks - refactoring may break assumptions")
        
        return risks

    def export_detailed_schema_analysis(self, schema_data: Dict[str, Any]) -> Dict[str, Any]:
        """Export detailed schema analysis for expert review."""
        
        # Create key usage summary
        key_summary = {}
        for key, dict_usage in schema_data["key_usage_map"].items():
            key_summary[key] = {
                "dictionaries_used_in": list(dict_usage.keys()),
                "total_usage_count": sum(info["total_usage"] for info in dict_usage.values()),
                "access_patterns": {},
                "safety_assessment": "safe" if all(
                    len(info["get"]) > 0 or len(info["in_check"]) > 0 
                    for info in dict_usage.values()
                ) else "unsafe"
            }
            
            # Aggregate access patterns
            for dict_name, usage_info in dict_usage.items():
                for pattern_type in ["subscript", "get", "in_check"]:
                    if pattern_type not in key_summary[key]["access_patterns"]:
                        key_summary[key]["access_patterns"][pattern_type] = 0
                    key_summary[key]["access_patterns"][pattern_type] += len(usage_info[pattern_type])
        
        # Create parameter schema summary
        schema_summary = {}
        for param_name, schema in schema_data["parameter_schemas"].items():
            schema_summary[param_name] = {
                "total_keys": schema["total_keys"],
                "key_list": list(schema["keys"].keys()),
                "safety_ratio": schema["usage_patterns"]["safety_ratio"],
                "most_used_keys": sorted(
                    schema["keys"].items(), 
                    key=lambda x: x[1]["total_usage"], 
                    reverse=True
                )[:10],  # Top 10 most used keys
                "risk_level": "high" if schema["usage_patterns"]["safety_ratio"] < 0.3 else 
                             "medium" if schema["usage_patterns"]["safety_ratio"] < 0.7 else "low"
            }
        
        return {
            "key_usage_summary": key_summary,
            "parameter_schemas": schema_summary,
            "data_flow_patterns": schema_data["data_flow_patterns"],
            "recommendations": schema_data["schema_recommendations"],
            "refactoring_risks": schema_data["refactoring_risks"],
            "expert_insights": {
                "most_critical_keys": self._identify_critical_keys(key_summary),
                "refactoring_priorities": self._generate_refactoring_priorities(schema_summary),
                "validation_gaps": self._identify_validation_gaps(key_summary)
            }
        }

    def _identify_critical_keys(self, key_summary: Dict) -> List[Dict[str, Any]]:
        """Identify the most critical keys for refactoring safety."""
        critical_keys = []
        
        for key, info in key_summary.items():
            if info["total_usage_count"] > 5 and info["safety_assessment"] == "unsafe":
                critical_keys.append({
                    "key": key,
                    "usage_count": info["total_usage_count"],
                    "risk_reason": "high usage with unsafe access patterns"
                })
        
        return sorted(critical_keys, key=lambda x: x["usage_count"], reverse=True)

    def _generate_refactoring_priorities(self, schema_summary: Dict) -> List[str]:
        """Generate prioritized refactoring recommendations."""
        priorities = []
        
        for param_name, schema in schema_summary.items():
            if schema["risk_level"] == "high":
                priorities.append(f"HIGH: Refactor {param_name} parameter - {schema['total_keys']} keys with {schema['safety_ratio']:.1%} safety ratio")
            elif schema["total_keys"] > 20:
                priorities.append(f"MEDIUM: Consider breaking down {param_name} parameter - {schema['total_keys']} keys is complex")
        
        return priorities

    def _identify_validation_gaps(self, key_summary: Dict) -> List[str]:
        """Identify keys that need validation."""
        gaps = []
        
        for key, info in key_summary.items():
            if info["access_patterns"].get("subscript", 0) > 0 and info["access_patterns"].get("in_check", 0) == 0:
                gaps.append(f"Key '{key}' needs validation - used {info['access_patterns']['subscript']} times without checks")
        
        return gaps

    def export_detailed_data_schemas(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export detailed data schema analysis as requested by experts.
        
        Returns:
            Dictionary with detailed schema information
        """
        return self.export_detailed_schema_analysis(analysis)
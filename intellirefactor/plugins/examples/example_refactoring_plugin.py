"""
Example refactoring plugin for IntelliRefactor

Demonstrates how to create custom refactoring rules using the hook system.
This plugin shows how to implement automated refactoring transformations.
"""

import ast
import re
from typing import Dict, List, Any

from ..plugin_interface import RefactoringPlugin, PluginMetadata, PluginType
from ..hook_system import HookSystem, HookType, HookPriority


class ExampleRefactoringPlugin(RefactoringPlugin):
    """
    Example plugin demonstrating custom refactoring rules.

    This plugin implements several refactoring transformations:
    1. Extract constants from magic numbers
    2. Replace string concatenation with f-strings
    3. Simplify boolean expressions
    4. Extract common code patterns into functions
    """

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="example_refactoring",
            version="1.0.0",
            description="Example plugin demonstrating custom refactoring transformations",
            author="IntelliRefactor Team",
            plugin_type=PluginType.REFACTORING,
            dependencies=[],
            config_schema={
                "min_string_length": {
                    "type": "integer",
                    "default": 10,
                    "description": "Minimum string length for constant extraction",
                },
                "enable_fstring_conversion": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable f-string conversion",
                },
                "enable_boolean_simplification": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable boolean expression simplification",
                },
                "min_duplicate_lines": {
                    "type": "integer",
                    "default": 3,
                    "description": "Minimum lines for duplicate code detection",
                },
            },
        )

    def initialize(self) -> bool:
        """Initialize the plugin and register hooks."""
        try:
            # Get hook system instance
            self.hook_system = getattr(self, "_hook_system", None)
            if not self.hook_system:
                self.logger.warning("Hook system not available, creating local instance")
                self.hook_system = HookSystem()

            # Register our refactoring hooks
            self._register_hooks()

            self.logger.info("Example refactoring plugin initialized successfully")
            return True

        except Exception as e:
            self.logger.error(f"Failed to initialize refactoring plugin: {e}")
            return False

    def _register_hooks(self) -> None:
        """Register all refactoring hooks."""
        # Pre-refactoring hook to prepare context
        self.hook_system.register_hook(
            hook_type=HookType.PRE_REFACTORING,
            callback=self._pre_refactoring_hook,
            name="example_pre_refactoring",
            priority=HookPriority.HIGH,
            plugin_name=self.metadata.name,
            description="Prepare refactoring context",
        )

        # Post-refactoring hook to validate results
        self.hook_system.register_hook(
            hook_type=HookType.POST_REFACTORING,
            callback=self._post_refactoring_hook,
            name="example_post_refactoring",
            priority=HookPriority.NORMAL,
            plugin_name=self.metadata.name,
            description="Validate refactoring results",
        )

        # Custom hooks for specific refactoring types
        self.hook_system.register_hook(
            hook_type=HookType.CUSTOM,
            callback=self._extract_constants,
            name="extract_constants",
            priority=HookPriority.NORMAL,
            plugin_name=self.metadata.name,
            description="Extract magic numbers as constants",
            custom_key="extract_constants",
        )

        self.hook_system.register_hook(
            hook_type=HookType.CUSTOM,
            callback=self._convert_to_fstrings,
            name="convert_fstrings",
            priority=HookPriority.NORMAL,
            plugin_name=self.metadata.name,
            description="Convert string concatenation to f-strings",
            custom_key="convert_fstrings",
        )

        self.hook_system.register_hook(
            hook_type=HookType.CUSTOM,
            callback=self._simplify_booleans,
            name="simplify_booleans",
            priority=HookPriority.NORMAL,
            plugin_name=self.metadata.name,
            description="Simplify boolean expressions",
            custom_key="simplify_booleans",
        )

    def _pre_refactoring_hook(self, opportunity: Dict[str, Any], context: Dict[str, Any]) -> None:
        """Pre-refactoring hook to prepare context."""
        self.logger.debug(f"Pre-refactoring hook for {opportunity.get('type', 'unknown')}")

        # Set up refactoring context
        context["refactoring"] = {
            "opportunity": opportunity,
            "transformations": [],
            "backup_created": False,
            "validation_results": {},
        }

    def _post_refactoring_hook(
        self,
        opportunity: Dict[str, Any],
        context: Dict[str, Any],
        result: Dict[str, Any],
    ) -> None:
        """Post-refactoring hook to validate results."""
        self.logger.debug(f"Post-refactoring hook for {opportunity.get('type', 'unknown')}")

        if "refactoring" in context:
            # Add our validation results
            result["custom_validation"] = context["refactoring"]["validation_results"]
            result["transformations_applied"] = len(context["refactoring"]["transformations"])

    def _extract_constants(
        self, file_path: str, ast_tree: ast.AST, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract magic numbers as named constants."""
        min_string_length = self.config.get("min_string_length", 10)
        constants_to_extract = []
        transformations = []

        class ConstantExtractor(ast.NodeVisitor):
            def __init__(self):
                self.constants = {}
                self.string_literals = {}

            def visit_Num(self, node):
                # Extract numeric constants (Python < 3.8)
                if hasattr(node, "n") and isinstance(node.n, (int, float)):
                    value = node.n
                    if abs(value) > 1 and value not in [0, 1, -1]:  # Skip common values
                        key = f"CONSTANT_{abs(int(value))}"
                        self.constants[key] = value
                        constants_to_extract.append(
                            {
                                "type": "numeric_constant",
                                "value": value,
                                "suggested_name": key,
                                "line": node.lineno,
                                "column": node.col_offset,
                            }
                        )
                self.generic_visit(node)

            def visit_Constant(self, node):
                # Extract constants (Python >= 3.8)
                if isinstance(node.value, (int, float)):
                    value = node.value
                    if abs(value) > 1 and value not in [0, 1, -1]:
                        key = f"CONSTANT_{abs(int(value))}"
                        self.constants[key] = value
                        constants_to_extract.append(
                            {
                                "type": "numeric_constant",
                                "value": value,
                                "suggested_name": key,
                                "line": node.lineno,
                                "column": node.col_offset,
                            }
                        )
                elif isinstance(node.value, str) and len(node.value) >= min_string_length:
                    value = node.value
                    # Create a reasonable constant name from the string
                    name_part = re.sub(r"[^a-zA-Z0-9]", "_", value[:20]).upper()
                    key = f"STRING_{name_part}"
                    self.string_literals[key] = value
                    constants_to_extract.append(
                        {
                            "type": "string_constant",
                            "value": value,
                            "suggested_name": key,
                            "line": node.lineno,
                            "column": node.col_offset,
                        }
                    )
                self.generic_visit(node)

        extractor = ConstantExtractor()
        extractor.visit(ast_tree)

        # Generate transformation suggestions
        if constants_to_extract:
            transformations.append(
                {
                    "type": "extract_constants",
                    "description": f"Extract {len(constants_to_extract)} constants",
                    "constants": constants_to_extract,
                    "suggested_location": "module_top",
                }
            )

        return {
            "constants_to_extract": constants_to_extract,
            "transformations": transformations,
        }

    def _convert_to_fstrings(
        self, file_path: str, ast_tree: ast.AST, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Convert string concatenation and formatting to f-strings."""
        if not self.config.get("enable_fstring_conversion", True):
            return {"conversions": []}

        conversions = []
        transformations = []

        class FStringConverter(ast.NodeVisitor):
            def visit_BinOp(self, node):
                # Look for string concatenation with %
                if isinstance(node.op, ast.Mod) and self._is_string_format(node):
                    conversions.append(
                        {
                            "type": "percent_format_to_fstring",
                            "line": node.lineno,
                            "column": node.col_offset,
                            "original": (
                                ast.unparse(node)
                                if hasattr(ast, "unparse")
                                else "<complex expression>"
                            ),
                            "suggested": self._convert_percent_to_fstring(node),
                        }
                    )

                # Look for string concatenation with +
                elif isinstance(node.op, ast.Add) and self._is_string_concatenation(node):
                    conversions.append(
                        {
                            "type": "concatenation_to_fstring",
                            "line": node.lineno,
                            "column": node.col_offset,
                            "original": (
                                ast.unparse(node)
                                if hasattr(ast, "unparse")
                                else "<complex expression>"
                            ),
                            "suggested": self._convert_concat_to_fstring(node),
                        }
                    )

                self.generic_visit(node)

            def visit_Call(self, node):
                # Look for .format() calls
                if (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "format"
                    and isinstance(node.func.value, ast.Constant)
                    and isinstance(node.func.value.value, str)
                ):
                    conversions.append(
                        {
                            "type": "format_to_fstring",
                            "line": node.lineno,
                            "column": node.col_offset,
                            "original": (
                                ast.unparse(node)
                                if hasattr(ast, "unparse")
                                else "<complex expression>"
                            ),
                            "suggested": self._convert_format_to_fstring(node),
                        }
                    )

                self.generic_visit(node)

            def _is_string_format(self, node):
                # Simple heuristic to detect string formatting
                return (
                    isinstance(node.left, ast.Constant)
                    and isinstance(node.left.value, str)
                    and "%" in node.left.value
                )

            def _is_string_concatenation(self, node):
                # Check if this is string concatenation
                def is_string_like(n):
                    return (
                        (isinstance(n, ast.Constant) and isinstance(n.value, str))
                        or (isinstance(n, ast.Name))
                        or (isinstance(n, ast.Call))
                    )

                return is_string_like(node.left) or is_string_like(node.right)

            def _convert_percent_to_fstring(self, node):
                # Simplified conversion - in practice, this would be more sophisticated
                return "f'<converted from % formatting>'"

            def _convert_concat_to_fstring(self, node):
                # Simplified conversion
                return "f'<converted from concatenation>'"

            def _convert_format_to_fstring(self, node):
                # Simplified conversion
                return "f'<converted from .format()>'"

        converter = FStringConverter()
        converter.visit(ast_tree)

        if conversions:
            transformations.append(
                {
                    "type": "convert_to_fstrings",
                    "description": f"Convert {len(conversions)} string operations to f-strings",
                    "conversions": conversions,
                }
            )

        return {"conversions": conversions, "transformations": transformations}

    def _simplify_booleans(
        self, file_path: str, ast_tree: ast.AST, context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simplify boolean expressions."""
        if not self.config.get("enable_boolean_simplification", True):
            return {"simplifications": []}

        simplifications = []
        transformations = []

        class BooleanSimplifier(ast.NodeVisitor):
            def visit_Compare(self, node):
                # Look for comparisons that can be simplified
                if len(node.ops) == 1 and len(node.comparators) == 1:
                    op = node.ops[0]
                    left = node.left
                    right = node.comparators[0]

                    # x == True -> x
                    if (
                        isinstance(op, ast.Eq)
                        and isinstance(right, ast.Constant)
                        and right.value is True
                    ):
                        simplifications.append(
                            {
                                "type": "remove_explicit_true_comparison",
                                "line": node.lineno,
                                "column": node.col_offset,
                                "original": (
                                    ast.unparse(node) if hasattr(ast, "unparse") else "<comparison>"
                                ),
                                "suggested": (
                                    ast.unparse(left) if hasattr(ast, "unparse") else "<simplified>"
                                ),
                            }
                        )

                    # x == False -> not x
                    elif (
                        isinstance(op, ast.Eq)
                        and isinstance(right, ast.Constant)
                        and right.value is False
                    ):
                        simplifications.append(
                            {
                                "type": "remove_explicit_false_comparison",
                                "line": node.lineno,
                                "column": node.col_offset,
                                "original": (
                                    ast.unparse(node) if hasattr(ast, "unparse") else "<comparison>"
                                ),
                                "suggested": f"not {ast.unparse(left) if hasattr(ast, 'unparse') else '<expr>'}",
                            }
                        )

                self.generic_visit(node)

            def visit_UnaryOp(self, node):
                # Look for double negations: not not x -> x
                if (
                    isinstance(node.op, ast.Not)
                    and isinstance(node.operand, ast.UnaryOp)
                    and isinstance(node.operand.op, ast.Not)
                ):
                    simplifications.append(
                        {
                            "type": "remove_double_negation",
                            "line": node.lineno,
                            "column": node.col_offset,
                            "original": (
                                ast.unparse(node)
                                if hasattr(ast, "unparse")
                                else "<double negation>"
                            ),
                            "suggested": (
                                ast.unparse(node.operand.operand)
                                if hasattr(ast, "unparse")
                                else "<simplified>"
                            ),
                        }
                    )

                self.generic_visit(node)

        simplifier = BooleanSimplifier()
        simplifier.visit(ast_tree)

        if simplifications:
            transformations.append(
                {
                    "type": "simplify_booleans",
                    "description": f"Simplify {len(simplifications)} boolean expressions",
                    "simplifications": simplifications,
                }
            )

        return {"simplifications": simplifications, "transformations": transformations}

    def identify_opportunities(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify refactoring opportunities from analysis results."""
        opportunities = []

        # Look for files that could benefit from our refactoring rules
        for file_path, file_results in analysis_results.get("files", {}).items():
            if file_path.endswith(".py"):
                try:
                    # Read the file to analyze
                    with open(file_path, "r", encoding="utf-8") as f:
                        content = f.read()

                    ast_tree = ast.parse(content, filename=file_path)
                    context = {"file_path": file_path}

                    # Run our custom analysis hooks to find opportunities
                    constant_results = self.hook_system.execute_custom_hooks(
                        "extract_constants", file_path, ast_tree, context
                    )
                    fstring_results = self.hook_system.execute_custom_hooks(
                        "convert_fstrings", file_path, ast_tree, context
                    )
                    boolean_results = self.hook_system.execute_custom_hooks(
                        "simplify_booleans", file_path, ast_tree, context
                    )

                    # Convert results to opportunities
                    for result_list in [
                        constant_results,
                        fstring_results,
                        boolean_results,
                    ]:
                        for result in result_list:
                            if result and "transformations" in result:
                                for transformation in result["transformations"]:
                                    opportunities.append(
                                        {
                                            "id": f"{transformation['type']}_{file_path}_{len(opportunities)}",
                                            "type": transformation["type"],
                                            "priority": self._get_priority(transformation["type"]),
                                            "description": transformation["description"],
                                            "file_path": file_path,
                                            "transformation": transformation,
                                            "estimated_impact": self._estimate_impact(
                                                transformation
                                            ),
                                        }
                                    )

                except Exception as e:
                    self.logger.error(f"Error analyzing {file_path} for opportunities: {e}")

        return opportunities

    def _get_priority(self, transformation_type: str) -> str:
        """Get priority for a transformation type."""
        priority_map = {
            "extract_constants": "medium",
            "convert_to_fstrings": "low",
            "simplify_booleans": "low",
        }
        return priority_map.get(transformation_type, "low")

    def _estimate_impact(self, transformation: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate the impact of a transformation."""
        transformation_type = transformation["type"]

        if transformation_type == "extract_constants":
            count = len(transformation.get("constants", []))
            return {
                "maintainability": "high" if count > 5 else "medium",
                "readability": "high",
                "performance": "none",
                "risk": "low",
            }
        elif transformation_type == "convert_to_fstrings":
            count = len(transformation.get("conversions", []))
            return {
                "maintainability": "medium",
                "readability": "high",
                "performance": "medium",
                "risk": "low",
            }
        elif transformation_type == "simplify_booleans":
            count = len(transformation.get("simplifications", []))
            return {
                "maintainability": "medium",
                "readability": "high",
                "performance": "low",
                "risk": "very_low",
            }

        return {
            "maintainability": "unknown",
            "readability": "unknown",
            "performance": "unknown",
            "risk": "unknown",
        }

    def apply_refactoring(
        self, opportunity: Dict[str, Any], context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply a refactoring transformation."""
        try:
            # Execute pre-refactoring hooks
            self.hook_system.execute_hooks(HookType.PRE_REFACTORING, opportunity, context)

            transformation = opportunity["transformation"]
            file_path = opportunity["file_path"]

            # Read the original file
            with open(file_path, "r", encoding="utf-8") as f:
                original_content = f.read()

            # Apply the transformation (simplified implementation)
            modified_content = self._apply_transformation(original_content, transformation)

            result = {
                "success": True,
                "file_path": file_path,
                "transformation_type": transformation["type"],
                "original_content": original_content,
                "modified_content": modified_content,
                "changes_made": len(transformation.get("constants", []))
                + len(transformation.get("conversions", []))
                + len(transformation.get("simplifications", [])),
            }

            # Execute post-refactoring hooks
            self.hook_system.execute_hooks(HookType.POST_REFACTORING, opportunity, context, result)

            return result

        except Exception as e:
            self.logger.error(f"Error applying refactoring: {e}")
            return {
                "success": False,
                "error": str(e),
                "file_path": opportunity.get("file_path", "unknown"),
            }

    def _apply_transformation(self, content: str, transformation: Dict[str, Any]) -> str:
        """Apply a specific transformation to content."""
        # This is a simplified implementation
        # In practice, you would use AST manipulation or other code transformation tools

        transformation_type = transformation["type"]

        if transformation_type == "extract_constants":
            # Add constants at the top of the file
            constants = transformation.get("constants", [])
            constant_definitions = []

            for constant in constants:
                if constant["type"] == "numeric_constant":
                    constant_definitions.append(
                        f"{constant['suggested_name']} = {constant['value']}"
                    )
                elif constant["type"] == "string_constant":
                    constant_definitions.append(
                        f"{constant['suggested_name']} = {repr(constant['value'])}"
                    )

            if constant_definitions:
                # Insert constants after imports (simplified)
                lines = content.split("\n")
                insert_index = 0

                # Find where to insert (after imports)
                for i, line in enumerate(lines):
                    if line.strip() and not line.startswith(("import ", "from ", "#")):
                        insert_index = i
                        break

                # Insert constants
                for const_def in reversed(constant_definitions):
                    lines.insert(insert_index, const_def)

                content = "\n".join(lines)

        # For other transformation types, we would implement similar logic
        # This is just a demonstration of the concept

        return content

    def get_supported_refactoring_types(self) -> List[str]:
        """Get list of refactoring types this plugin supports."""
        return ["extract_constants", "convert_to_fstrings", "simplify_booleans"]

    def cleanup(self) -> None:
        """Clean up plugin resources."""
        if hasattr(self, "hook_system"):
            # Unregister our hooks
            hooks_to_remove = [
                "example_pre_refactoring",
                "example_post_refactoring",
                "extract_constants",
                "convert_fstrings",
                "simplify_booleans",
            ]

            for hook_name in hooks_to_remove:
                self.hook_system.unregister_hook(hook_name)

        self.logger.info("Example refactoring plugin cleaned up")

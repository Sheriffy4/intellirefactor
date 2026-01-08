"""
LLM Context Generator for IntelliRefactor

Generates LLM-optimized context documentation including:
- Key abstractions and patterns
- Usage examples
- Integration points
- Common scenarios
- Best practices
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..analysis.file_analyzer import FileAnalyzer


@dataclass
class DesignPattern:
    """Represents a design pattern found in the code."""

    name: str
    description: str
    components: List[str]
    example_code: str


@dataclass
class UsagePattern:
    """Represents a common usage pattern."""

    name: str
    description: str
    code_example: str
    use_cases: List[str]


class LLMContextGenerator:
    """Generates LLM-optimized context documentation."""

    def __init__(self):
        self.file_analyzer = FileAnalyzer()

    def generate_llm_context(self, file_path: Path) -> str:
        """Generate comprehensive LLM context documentation."""
        try:
            # Analyze the file
            analysis_result = self.file_analyzer.analyze_file(file_path)

            if not analysis_result.success:
                return f"""# LLM Context: {file_path.name}

## Error
Failed to analyze file: {analysis_result.metadata.get("error", "Unknown error")}
"""

            # Parse source code for detailed analysis
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            # Extract patterns and abstractions
            design_patterns = self._identify_design_patterns(tree, source_code)
            usage_patterns = self._extract_usage_patterns(tree, source_code)
            key_abstractions = self._identify_key_abstractions(tree, source_code)

            # Generate context documentation
            context = self._generate_context_header(file_path, analysis_result.data)
            context += self._generate_key_components_section(tree, source_code)
            context += self._generate_design_patterns_section(design_patterns)
            context += self._generate_usage_patterns_section(usage_patterns)
            context += self._generate_abstractions_section(key_abstractions)
            context += self._generate_integration_points_section(tree, source_code)
            context += self._generate_common_scenarios_section(tree, source_code)

            return context

        except Exception as e:
            return f"""# LLM Context: {file_path.name}

## Error
Failed to generate LLM context: {str(e)}
"""

    def _generate_context_header(self, file_path: Path, analysis_data: Dict[str, Any]) -> str:
        """Generate context header."""
        module_name = file_path.stem

        return f"""# LLM Context: {module_name.replace("_", " ").title()} Module

## Module Purpose and Scope

The `{file_path.name}` module serves as {self._infer_module_purpose(analysis_data)}. This context provides LLMs with essential understanding needed to work effectively with this module.

## Key Components for LLM Understanding

"""

    def _generate_key_components_section(self, tree: ast.AST, source_code: str) -> str:
        """Generate key components section."""
        classes = self._extract_main_classes(tree, source_code)

        section = "### 1. Core Architecture\n\n"

        if classes:
            main_class = classes[0]  # Assume first class is main
            section += f"""```python
# Main class that orchestrates module functionality
class {main_class["name"]}:
    \"\"\"
    {main_class["purpose"]}
    
    Key responsibilities:
"""

            for method in main_class["methods"][:5]:  # Top 5 methods
                section += f"    - {method['name']}: {method['purpose']}\n"

            section += '    """\n```\n\n'

        return section

    def _generate_design_patterns_section(self, patterns: List[DesignPattern]) -> str:
        """Generate design patterns section."""
        if not patterns:
            return "### 2. Design Patterns Used\n\nNo specific design patterns identified.\n\n"

        section = "### 2. Design Patterns Used\n\n"

        for pattern in patterns:
            section += f"#### {pattern.name}\n"
            section += f"{pattern.description}\n\n"

            if pattern.example_code:
                section += f"```python\n{pattern.example_code}\n```\n\n"

        return section

    def _generate_usage_patterns_section(self, patterns: List[UsagePattern]) -> str:
        """Generate usage patterns section."""
        section = "## Common Usage Patterns for LLM\n\n"

        if not patterns:
            section += "### Basic Usage\n"
            section += "```python\n# Basic module usage pattern\n# (Specific patterns to be identified)\n```\n\n"
            return section

        for i, pattern in enumerate(patterns, 1):
            section += f"### {i}. {pattern.name}\n"
            section += f"{pattern.description}\n\n"
            section += f"```python\n{pattern.code_example}\n```\n\n"

            if pattern.use_cases:
                section += "**Use Cases:**\n"
                for use_case in pattern.use_cases:
                    section += f"- {use_case}\n"
                section += "\n"

        return section

    def _generate_abstractions_section(self, abstractions: List[Dict[str, Any]]) -> str:
        """Generate key abstractions section."""
        section = "## Key Abstractions for LLM Understanding\n\n"

        if not abstractions:
            section += "### Core Concepts\n"
            section += "- Module provides specific functionality (details to be analyzed)\n"
            section += "- Uses standard Python patterns and practices\n\n"
            return section

        for i, abstraction in enumerate(abstractions, 1):
            section += f"### {i}. **{abstraction['name']}**\n"
            section += f"{abstraction['description']}\n\n"

            if abstraction.get("example"):
                section += f"```python\n{abstraction['example']}\n```\n\n"

        return section

    def _generate_integration_points_section(self, tree: ast.AST, source_code: str) -> str:
        """Generate integration points section."""
        imports = self._extract_imports(tree)
        external_calls = self._extract_external_calls(tree)

        section = "## Integration Points\n\n"

        if imports:
            section += "### 1. **External Dependencies**\n"
            for imp in imports[:5]:  # Top 5 imports
                section += f"```python\n# {imp['description']}\n{imp['code']}\n```\n\n"

        if external_calls:
            section += "### 2. **External Integrations**\n"
            for call in external_calls[:3]:  # Top 3 external calls
                section += f"```python\n# {call['description']}\n{call['example']}\n```\n\n"

        return section

    def _generate_common_scenarios_section(self, tree: ast.AST, source_code: str) -> str:
        """Generate common scenarios section."""
        scenarios = self._identify_common_scenarios(tree, source_code)

        section = "## Common Scenarios and Examples\n\n"

        if not scenarios:
            section += "### Basic Operations\n"
            section += "```python\n# Common usage scenarios to be documented\n```\n\n"
            return section

        for i, scenario in enumerate(scenarios, 1):
            section += f"### {i}. {scenario['name']}\n"
            section += f"{scenario['description']}\n\n"
            section += f"```python\n{scenario['example']}\n```\n\n"

        return section

    def _identify_design_patterns(self, tree: ast.AST, source_code: str) -> List[DesignPattern]:
        """Identify design patterns in the code."""
        patterns = []

        # Strategy Pattern Detection
        if self._has_strategy_pattern(tree):
            patterns.append(
                DesignPattern(
                    name="Strategy Pattern",
                    description="Multiple implementations of similar functionality with interchangeable algorithms.",
                    components=self._get_strategy_components(tree),
                    example_code=self._get_strategy_example(tree, source_code),
                )
            )

        # Factory Pattern Detection
        if self._has_factory_pattern(tree):
            patterns.append(
                DesignPattern(
                    name="Factory Pattern",
                    description="Object creation through factory methods or classes.",
                    components=self._get_factory_components(tree),
                    example_code=self._get_factory_example(tree, source_code),
                )
            )

        # Observer Pattern Detection
        if self._has_observer_pattern(tree):
            patterns.append(
                DesignPattern(
                    name="Observer Pattern",
                    description="Event notification and callback mechanisms.",
                    components=self._get_observer_components(tree),
                    example_code=self._get_observer_example(tree, source_code),
                )
            )

        return patterns

    def _extract_usage_patterns(self, tree: ast.AST, source_code: str) -> List[UsagePattern]:
        """Extract common usage patterns."""
        patterns = []

        # Find initialization patterns
        init_pattern = self._find_initialization_pattern(tree, source_code)
        if init_pattern:
            patterns.append(init_pattern)

        # Find processing patterns
        processing_pattern = self._find_processing_pattern(tree, source_code)
        if processing_pattern:
            patterns.append(processing_pattern)

        # Find error handling patterns
        error_pattern = self._find_error_handling_pattern(tree, source_code)
        if error_pattern:
            patterns.append(error_pattern)

        return patterns

    def _identify_key_abstractions(self, tree: ast.AST, source_code: str) -> List[Dict[str, Any]]:
        """Identify key abstractions in the module."""
        abstractions = []

        # Find main classes and their abstractions
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                abstraction = self._analyze_class_abstraction(node, source_code)
                if abstraction:
                    abstractions.append(abstraction)

        return abstractions[:5]  # Top 5 abstractions

    def _extract_main_classes(self, tree: ast.AST, source_code: str) -> List[Dict[str, Any]]:
        """Extract main classes with their details."""
        classes = []
        lines = source_code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    "name": node.name,
                    "purpose": self._extract_docstring(node) or "Main class functionality",
                    "methods": [],
                }

                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        method_info = {
                            "name": item.name,
                            "purpose": self._extract_docstring(item)
                            or f"{item.name} functionality",
                        }
                        class_info["methods"].append(method_info)

                classes.append(class_info)

        return classes

    def _extract_imports(self, tree: ast.AST) -> List[Dict[str, str]]:
        """Extract import information."""
        imports = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(
                        {
                            "code": f"import {alias.name}",
                            "description": f"Imports {alias.name} for {self._guess_import_purpose(alias.name)}",
                        }
                    )
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(
                        {
                            "code": f"from {node.module} import {', '.join(alias.name for alias in node.names)}",
                            "description": f"Imports from {node.module} for {self._guess_import_purpose(node.module)}",
                        }
                    )

        return imports

    def _extract_external_calls(self, tree: ast.AST) -> List[Dict[str, str]]:
        """Extract external API calls."""
        calls = []

        # This is a simplified implementation
        # In practice, you'd want more sophisticated analysis

        return calls

    def _identify_common_scenarios(self, tree: ast.AST, source_code: str) -> List[Dict[str, str]]:
        """Identify common usage scenarios."""
        scenarios = []

        # Find main methods that represent common scenarios
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if not node.name.startswith("_") and node.name not in [
                    "__init__",
                    "__str__",
                    "__repr__",
                ]:
                    scenario = {
                        "name": node.name.replace("_", " ").title(),
                        "description": self._extract_docstring(node)
                        or f"Common {node.name} operation",
                        "example": f"# Example usage of {node.name}\n# (Implementation details to be added)",
                    }
                    scenarios.append(scenario)

        return scenarios[:5]  # Top 5 scenarios

    # Pattern detection helper methods
    def _has_strategy_pattern(self, tree: ast.AST) -> bool:
        """Check if strategy pattern is present."""
        # Look for multiple classes with similar method signatures
        method_signatures = {}

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if item.name not in method_signatures:
                            method_signatures[item.name] = []
                        method_signatures[item.name].append(node.name)

        # Check if any method is implemented by multiple classes
        for method, classes in method_signatures.items():
            if len(classes) >= 3 and method not in ["__init__", "__str__", "__repr__"]:
                return True

        return False

    def _has_factory_pattern(self, tree: ast.AST) -> bool:
        """Check if factory pattern is present."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if (
                    "create" in node.name.lower()
                    or "make" in node.name.lower()
                    or "build" in node.name.lower()
                    or "factory" in node.name.lower()
                ):
                    return True
        return False

    def _has_observer_pattern(self, tree: ast.AST) -> bool:
        """Check if observer pattern is present."""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if (
                    "notify" in node.name.lower()
                    or "update" in node.name.lower()
                    or "callback" in node.name.lower()
                    or "listener" in node.name.lower()
                ):
                    return True
        return False

    def _get_strategy_components(self, tree: ast.AST) -> List[str]:
        """Get strategy pattern components."""
        components = []
        # Implementation would identify strategy classes
        return components

    def _get_strategy_example(self, tree: ast.AST, source_code: str) -> str:
        """Get strategy pattern example."""
        return "# Strategy pattern implementation example\n# (To be extracted from actual code)"

    def _get_factory_components(self, tree: ast.AST) -> List[str]:
        """Get factory pattern components."""
        components = []
        # Implementation would identify factory methods/classes
        return components

    def _get_factory_example(self, tree: ast.AST, source_code: str) -> str:
        """Get factory pattern example."""
        return "# Factory pattern implementation example\n# (To be extracted from actual code)"

    def _get_observer_components(self, tree: ast.AST) -> List[str]:
        """Get observer pattern components."""
        components = []
        # Implementation would identify observer classes
        return components

    def _get_observer_example(self, tree: ast.AST, source_code: str) -> str:
        """Get observer pattern example."""
        return "# Observer pattern implementation example\n# (To be extracted from actual code)"

    def _find_initialization_pattern(
        self, tree: ast.AST, source_code: str
    ) -> Optional[UsagePattern]:
        """Find initialization usage pattern."""
        # Look for __init__ methods and factory functions
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                return UsagePattern(
                    name="Initialization Pattern",
                    description="Standard way to initialize the main class",
                    code_example="# Initialization example\n# (To be extracted from __init__ method)",
                    use_cases=[
                        "Basic setup",
                        "Configuration loading",
                        "Resource initialization",
                    ],
                )
        return None

    def _find_processing_pattern(self, tree: ast.AST, source_code: str) -> Optional[UsagePattern]:
        """Find processing usage pattern."""
        # Look for main processing methods
        processing_methods = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if (
                    "process" in node.name.lower()
                    or "run" in node.name.lower()
                    or "execute" in node.name.lower()
                    or "handle" in node.name.lower()
                ):
                    processing_methods.append(node.name)

        if processing_methods:
            return UsagePattern(
                name="Processing Pattern",
                description="Main processing workflow",
                code_example=f"# Processing example using {processing_methods[0]}\n# (Implementation details to be added)",
                use_cases=["Data processing", "Request handling", "Workflow execution"],
            )

        return None

    def _find_error_handling_pattern(
        self, tree: ast.AST, source_code: str
    ) -> Optional[UsagePattern]:
        """Find error handling usage pattern."""
        # Look for try-except blocks and custom exceptions
        has_try_except = False
        custom_exceptions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                has_try_except = True
            elif isinstance(node, ast.ClassDef):
                # Check if it's an exception class
                for base in node.bases:
                    if isinstance(base, ast.Name) and "Error" in base.id:
                        custom_exceptions.append(node.name)

        if has_try_except or custom_exceptions:
            return UsagePattern(
                name="Error Handling Pattern",
                description="Standard error handling approach",
                code_example="# Error handling example\n# (To be extracted from try-except blocks)",
                use_cases=[
                    "Exception handling",
                    "Graceful degradation",
                    "Error recovery",
                ],
            )

        return None

    def _analyze_class_abstraction(
        self, node: ast.ClassDef, source_code: str
    ) -> Optional[Dict[str, Any]]:
        """Analyze a class to identify its key abstraction."""
        if node.name.startswith("_"):  # Skip private classes
            return None

        docstring = self._extract_docstring(node)
        if not docstring:
            return None

        return {
            "name": node.name,
            "description": docstring,
            "example": f"# Example usage of {node.name}\n# (Implementation details to be added)",
        }

    def _extract_docstring(self, node: ast.AST) -> Optional[str]:
        """Extract docstring from a node."""
        if (
            hasattr(node, "body")
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        ):
            return node.body[0].value.value.strip()
        return None

    def _infer_module_purpose(self, analysis_data: Dict[str, Any]) -> str:
        """Infer the module's purpose from analysis data."""
        # This could be enhanced with more sophisticated analysis
        return "a core component providing essential functionality"

    def _guess_import_purpose(self, import_name: str) -> str:
        """Guess the purpose of an import."""
        purposes = {
            "asyncio": "asynchronous operations",
            "json": "JSON data handling",
            "logging": "logging and debugging",
            "pathlib": "file path operations",
            "typing": "type annotations",
            "dataclasses": "data structure definitions",
            "enum": "enumeration definitions",
            "abc": "abstract base classes",
            "collections": "specialized data structures",
            "datetime": "date and time operations",
            "os": "operating system interface",
            "sys": "system-specific parameters",
            "subprocess": "subprocess management",
            "time": "time-related functions",
            "traceback": "exception traceback handling",
            "re": "regular expressions",
            "inspect": "runtime inspection",
            "ast": "abstract syntax tree operations",
        }

        return purposes.get(import_name, "specialized functionality")

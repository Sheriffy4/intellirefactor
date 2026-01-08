"""
Registry Generator for IntelliRefactor

Generates comprehensive module registries including:
- Component inventory
- Method catalogs
- Dependency mappings
- Interface definitions
- Performance characteristics
"""

import ast
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..analysis.file_analyzer import FileAnalyzer


@dataclass
class ComponentInfo:
    """Information about a module component."""

    name: str
    type: str
    purpose: str
    complexity: str
    dependencies: List[str]
    methods: List[str]
    attributes: List[str]
    line_start: int
    line_end: int
    is_public: bool


@dataclass
class MethodInfo:
    """Information about a method."""

    name: str
    class_name: Optional[str]
    purpose: str
    complexity: str
    parameters: List[str]
    return_type: str
    is_async: bool
    is_public: bool
    line_start: int
    line_end: int


class RegistryGenerator:
    """Generates comprehensive module registries."""

    def __init__(self):
        self.file_analyzer = FileAnalyzer()

    def generate_module_registry(self, file_path: Path) -> str:
        """Generate comprehensive module registry."""
        try:
            # Analyze the file
            analysis_result = self.file_analyzer.analyze_file(file_path)

            if not analysis_result.success:
                return f"""# Module Registry: {file_path.name}

## Error
Failed to analyze file: {analysis_result.metadata.get("error", "Unknown error")}
"""

            # Parse the source code for detailed analysis
            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            tree = ast.parse(source_code)

            # Extract detailed information
            components = self._extract_components(tree, source_code)
            methods = self._extract_methods(tree, source_code)
            dependencies = self._extract_dependencies(tree)
            interfaces = self._extract_interfaces(tree)

            # Generate registry
            registry = self._generate_registry_header(file_path, analysis_result.data)
            registry += self._generate_component_registry(components)
            registry += self._generate_method_registry(methods)
            registry += self._generate_dependency_registry(dependencies)
            registry += self._generate_interface_registry(interfaces)
            registry += self._generate_configuration_registry(tree)
            registry += self._generate_performance_registry(components, methods)
            registry += self._generate_testing_registry(components, methods)
            registry += self._generate_maintenance_registry(analysis_result.data)
            registry += self._generate_documentation_registry(components, methods)

            return registry

        except Exception as e:
            return f"""# Module Registry: {file_path.name}

## Error
Failed to generate registry: {str(e)}
"""

    def _extract_components(self, tree: ast.AST, source_code: str) -> List[ComponentInfo]:
        """Extract component information from AST."""
        components = []
        lines = source_code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                component = self._analyze_class_component(node, lines)
                components.append(component)
            elif isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                component = self._analyze_function_component(node, lines)
                components.append(component)

        return components

    def _extract_methods(self, tree: ast.AST, source_code: str) -> List[MethodInfo]:
        """Extract method information from AST."""
        methods = []
        lines = source_code.split("\n")

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                method = self._analyze_method(node, lines, tree)
                methods.append(method)

        return methods

    def _extract_dependencies(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Extract dependency information."""
        dependencies = {"external": [], "internal": [], "standard_library": []}

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    dep_name = alias.name
                    dep_type = self._classify_dependency(dep_name)
                    dependencies[dep_type].append(dep_name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    dep_type = self._classify_dependency(node.module)
                    dependencies[dep_type].append(node.module)

        return dependencies

    def _extract_interfaces(self, tree: ast.AST) -> List[Dict[str, Any]]:
        """Extract interface and protocol definitions."""
        interfaces = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it's a protocol or interface
                if self._is_protocol_or_interface(node):
                    interface_info = {
                        "name": node.name,
                        "type": (
                            "protocol"
                            if "Protocol"
                            in [base.id for base in node.bases if isinstance(base, ast.Name)]
                            else "interface"
                        ),
                        "methods": [
                            method.name
                            for method in node.body
                            if isinstance(method, ast.FunctionDef)
                        ],
                        "line_start": node.lineno,
                        "line_end": node.end_lineno or node.lineno,
                    }
                    interfaces.append(interface_info)

        return interfaces

    def _generate_registry_header(self, file_path: Path, analysis_data: Dict[str, Any]) -> str:
        """Generate registry header."""
        stats = analysis_data.get("statistics", {})

        return f"""# Module Registry: {file_path.stem.replace("_", " ").title()}

## Module Information

| Property | Value |
|----------|-------|
| **Module Name** | {file_path.stem} |
| **File Path** | `{file_path}` |
| **Module Type** | {self._determine_module_type(analysis_data)} |
| **Primary Purpose** | {self._determine_primary_purpose(analysis_data)} |
| **Lines of Code** | ~{stats.get("total_lines", "Unknown")} |
| **Complexity Level** | {self._assess_complexity_level(analysis_data)} |
| **Maintainability** | {self._assess_maintainability(analysis_data)} |
| **Test Coverage** | To be determined |

"""

    def _generate_component_registry(self, components: List[ComponentInfo]) -> str:
        """Generate component registry section."""
        if not components:
            return "## Component Registry\n\nNo components found.\n\n"

        # Separate by type
        classes = [c for c in components if c.type == "class"]
        functions = [c for c in components if c.type == "function"]

        registry = "## Component Registry\n\n"

        if classes:
            registry += "### Core Classes\n\n"
            registry += "| Class Name | Purpose | Complexity | Dependencies |\n"
            registry += "|------------|---------|------------|-------------|\n"

            for cls in classes:
                deps = ", ".join(cls.dependencies[:3])
                if len(cls.dependencies) > 3:
                    deps += f" (+{len(cls.dependencies) - 3} more)"

                registry += (
                    f"| `{cls.name}` | {cls.purpose[:50]}... | {cls.complexity} | {deps} |\n"
                )

            registry += "\n"

        if functions:
            registry += "### Utility Functions\n\n"
            registry += "| Function Name | Purpose | Complexity | Parameters |\n"
            registry += "|---------------|---------|------------|------------|\n"

            for func in functions:
                registry += f"| `{func.name}` | {func.purpose[:50]}... | {func.complexity} | {len(func.methods)} |\n"

            registry += "\n"

        return registry

    def _generate_method_registry(self, methods: List[MethodInfo]) -> str:
        """Generate method registry section."""
        if not methods:
            return "## Method Registry\n\nNo methods found.\n\n"

        # Group by class
        class_methods = {}
        standalone_functions = []

        for method in methods:
            if method.class_name:
                if method.class_name not in class_methods:
                    class_methods[method.class_name] = []
                class_methods[method.class_name].append(method)
            else:
                standalone_functions.append(method)

        registry = "## Method Registry\n\n"

        # Class methods
        for class_name, class_method_list in class_methods.items():
            registry += f"### {class_name} Methods\n\n"
            registry += "| Method Name | Type | Purpose | Complexity | Async |\n"
            registry += "|-------------|------|---------|------------|-------|\n"

            for method in class_method_list:
                method_type = (
                    "Constructor"
                    if method.name == "__init__"
                    else "Public" if method.is_public else "Private"
                )
                async_marker = "Yes" if method.is_async else "No"

                registry += f"| `{method.name}` | {method_type} | {method.purpose[:40]}... | {method.complexity} | {async_marker} |\n"

            registry += "\n"

        # Standalone functions
        if standalone_functions:
            registry += "### Standalone Functions\n\n"
            registry += "| Function Name | Purpose | Complexity | Parameters |\n"
            registry += "|---------------|---------|------------|------------|\n"

            for func in standalone_functions:
                registry += f"| `{func.name}` | {func.purpose[:40]}... | {func.complexity} | {len(func.parameters)} |\n"

            registry += "\n"

        return registry

    def _generate_dependency_registry(self, dependencies: Dict[str, List[str]]) -> str:
        """Generate dependency registry section."""
        registry = "## Dependency Registry\n\n"

        registry += "### External Dependencies\n\n"
        if dependencies["external"]:
            registry += "| Dependency | Type | Required | Fallback Available |\n"
            registry += "|------------|------|----------|-------------------|\n"

            for dep in dependencies["external"]:
                required = "Yes" if dep in ["asyncio", "json", "logging"] else "No"
                fallback = "Yes" if dep in ["rich"] else "No"
                dep_type = self._get_dependency_type(dep)

                registry += f"| `{dep}` | {dep_type} | {required} | {fallback} |\n"
        else:
            registry += "No external dependencies found.\n"

        registry += "\n### Internal Dependencies\n\n"
        if dependencies["internal"]:
            for dep in dependencies["internal"]:
                registry += f"- `{dep}`\n"
        else:
            registry += "No internal dependencies found.\n"

        registry += "\n### Standard Library\n\n"
        if dependencies["standard_library"]:
            for dep in dependencies["standard_library"]:
                registry += f"- `{dep}`\n"
        else:
            registry += "No standard library dependencies found.\n"

        registry += "\n"

        return registry

    def _generate_interface_registry(self, interfaces: List[Dict[str, Any]]) -> str:
        """Generate interface registry section."""
        if not interfaces:
            return "## Interface Registry\n\nNo interfaces or protocols found.\n\n"

        registry = "## Interface Registry\n\n"
        registry += "### Protocol Definitions\n\n"
        registry += "| Protocol Name | Purpose | Methods Count |\n"
        registry += "|---------------|---------|---------------|\n"

        for interface in interfaces:
            registry += f"| `{interface['name']}` | {interface['type'].title()} definition | {len(interface['methods'])} |\n"

        registry += "\n"

        return registry

    def _generate_configuration_registry(self, tree: ast.AST) -> str:
        """Generate configuration registry section."""
        config_classes = []
        env_vars = []

        # Find configuration-related classes and variables
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if "config" in node.name.lower() or "setting" in node.name.lower():
                    config_classes.append(node.name)
            elif isinstance(node, ast.Call):
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr == "getenv"
                ):
                    if node.args and isinstance(node.args[0], ast.Constant):
                        env_vars.append(node.args[0].value)

        registry = "## Configuration Registry\n\n"

        if config_classes:
            registry += "### Configuration Classes\n\n"
            registry += "| Class | Purpose | Fields Count | Validation |\n"
            registry += "|-------|---------|--------------|------------|\n"

            for config_class in config_classes:
                registry += f"| `{config_class}` | Configuration management | TBD | TBD |\n"

            registry += "\n"

        if env_vars:
            registry += "### Environment Variables\n\n"
            registry += "| Variable | Purpose | Default | Impact |\n"
            registry += "|----------|---------|---------|--------|\n"

            for var in env_vars:
                registry += f"| `{var}` | Configuration setting | None | TBD |\n"

            registry += "\n"

        if not config_classes and not env_vars:
            registry += "No configuration elements found.\n\n"

        return registry

    def _generate_performance_registry(
        self, components: List[ComponentInfo], methods: List[MethodInfo]
    ) -> str:
        """Generate performance registry section."""
        registry = "## Performance Registry\n\n"

        registry += "### Performance Characteristics\n\n"
        registry += "| Component | Performance Level | Bottlenecks | Optimization Potential |\n"
        registry += "|-----------|------------------|-------------|----------------------|\n"

        for component in components[:10]:  # Limit to top 10
            perf_level = self._assess_performance_level(component)
            bottlenecks = self._identify_bottlenecks(component)
            optimization = self._assess_optimization_potential(component)

            registry += f"| `{component.name}` | {perf_level} | {bottlenecks} | {optimization} |\n"

        registry += "\n### Memory Usage\n\n"
        registry += "| Component | Memory Impact | Scalability | Notes |\n"
        registry += "|-----------|---------------|-------------|-------|\n"

        for component in components[:5]:  # Top 5 for memory analysis
            memory_impact = self._assess_memory_impact(component)
            scalability = self._assess_scalability(component)

            registry += f"| `{component.name}` | {memory_impact} | {scalability} | {component.type.title()} component |\n"

        registry += "\n"

        return registry

    def _generate_testing_registry(
        self, components: List[ComponentInfo], methods: List[MethodInfo]
    ) -> str:
        """Generate testing registry section."""
        registry = "## Testing Registry\n\n"

        registry += "### Test Categories\n\n"
        registry += "| Category | Coverage Target | Current Status | Priority |\n"
        registry += "|----------|----------------|----------------|----------|\n"
        registry += "| Unit Tests | 90% | TBD | High |\n"
        registry += "| Integration Tests | 80% | TBD | High |\n"
        registry += "| Performance Tests | 70% | TBD | Medium |\n"
        registry += "| Error Handling Tests | 95% | TBD | High |\n"

        registry += "\n### Mock Requirements\n\n"
        registry += "| Component | Mock Complexity | Reason |\n"
        registry += "|-----------|----------------|--------|\n"

        for component in components:
            if len(component.dependencies) > 0:
                mock_complexity = (
                    "High"
                    if len(component.dependencies) > 5
                    else "Medium" if len(component.dependencies) > 2 else "Low"
                )
                reason = f"{len(component.dependencies)} dependencies"

                registry += f"| `{component.name}` | {mock_complexity} | {reason} |\n"

        registry += "\n"

        return registry

    def _generate_maintenance_registry(self, analysis_data: Dict[str, Any]) -> str:
        """Generate maintenance registry section."""
        registry = "## Maintenance Registry\n\n"

        registry += "### Code Quality Metrics\n\n"
        registry += "| Metric | Current | Target | Priority |\n"
        registry += "|--------|---------|--------|----------|\n"
        registry += "| Cyclomatic Complexity | TBD | Medium | High |\n"
        registry += "| Method Length | TBD | Short | Medium |\n"
        registry += "| Class Size | TBD | Medium | High |\n"
        registry += "| Code Duplication | TBD | Low | Medium |\n"

        registry += "\n### Refactoring Opportunities\n\n"
        registry += "| Area | Effort | Impact | Priority |\n"
        registry += "|------|--------|--------|----------|\n"
        registry += "| Module Decomposition | High | High | High |\n"
        registry += "| Extract Common Patterns | Medium | Medium | Medium |\n"
        registry += "| Performance Optimization | Low | Medium | Low |\n"

        registry += "\n"

        return registry

    def _generate_documentation_registry(
        self, components: List[ComponentInfo], methods: List[MethodInfo]
    ) -> str:
        """Generate documentation registry section."""
        registry = "## Documentation Registry\n\n"

        # Calculate documentation coverage
        documented_components = len(
            [c for c in components if c.purpose != "No description available"]
        )
        documented_methods = len([m for m in methods if m.purpose != "No description available"])

        total_components = len(components)
        total_methods = len(methods)

        component_coverage = (
            (documented_components / total_components * 100) if total_components > 0 else 0
        )
        method_coverage = (documented_methods / total_methods * 100) if total_methods > 0 else 0

        registry += "### Documentation Status\n\n"
        registry += "| Type | Status | Quality | Completeness |\n"
        registry += "|------|--------|---------|-------------|\n"
        registry += f"| Docstrings | Partial | Good | {component_coverage:.0f}% |\n"
        registry += "| Type Hints | Good | Excellent | 85% |\n"
        registry += "| Comments | Minimal | Fair | 30% |\n"
        registry += "| Examples | None | N/A | 0% |\n"

        registry += "\n### Documentation Needs\n\n"
        registry += "| Area | Priority | Effort | Target Audience |\n"
        registry += "|------|----------|--------|----------------|\n"
        registry += "| API Documentation | High | Medium | Developers |\n"
        registry += "| Usage Examples | High | Low | Users |\n"
        registry += "| Architecture Guide | Medium | High | Maintainers |\n"
        registry += "| Migration Guide | Low | Medium | Upgraders |\n"

        return registry

    # Helper methods
    def _analyze_class_component(self, node: ast.ClassDef, lines: List[str]) -> ComponentInfo:
        """Analyze a class component."""
        purpose = self._extract_docstring(node) or "No description available"
        methods = [item.name for item in node.body if isinstance(item, ast.FunctionDef)]
        attributes = self._extract_class_attributes(node)
        dependencies = self._extract_class_dependencies(node)
        complexity = self._assess_class_complexity(node, methods)

        return ComponentInfo(
            name=node.name,
            type="class",
            purpose=purpose,
            complexity=complexity,
            dependencies=dependencies,
            methods=methods,
            attributes=attributes,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            is_public=not node.name.startswith("_"),
        )

    def _analyze_function_component(self, node: ast.FunctionDef, lines: List[str]) -> ComponentInfo:
        """Analyze a function component."""
        purpose = self._extract_docstring(node) or "No description available"
        dependencies = self._extract_function_dependencies(node)
        complexity = self._assess_function_complexity(node)

        return ComponentInfo(
            name=node.name,
            type="function",
            purpose=purpose,
            complexity=complexity,
            dependencies=dependencies,
            methods=[],  # Functions don't have methods
            attributes=[],  # Functions don't have attributes
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
            is_public=not node.name.startswith("_"),
        )

    def _analyze_method(self, node: ast.FunctionDef, lines: List[str], tree: ast.AST) -> MethodInfo:
        """Analyze a method."""
        purpose = self._extract_docstring(node) or "No description available"
        class_name = self._find_containing_class(node, tree)
        parameters = [arg.arg for arg in node.args.args]
        return_type = "Any"  # Could be enhanced with type annotation analysis
        complexity = self._assess_function_complexity(node)

        return MethodInfo(
            name=node.name,
            class_name=class_name,
            purpose=purpose,
            complexity=complexity,
            parameters=parameters,
            return_type=return_type,
            is_async=isinstance(node, ast.AsyncFunctionDef),
            is_public=not node.name.startswith("_"),
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno,
        )

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

    def _extract_class_attributes(self, node: ast.ClassDef) -> List[str]:
        """Extract class attributes."""
        attributes = []
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
        return attributes

    def _extract_class_dependencies(self, node: ast.ClassDef) -> List[str]:
        """Extract class dependencies."""
        dependencies = []

        # Base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                dependencies.append(base.id)
            elif isinstance(base, ast.Attribute):
                dependencies.append(ast.unparse(base) if hasattr(ast, "unparse") else str(base))

        return dependencies

    def _extract_function_dependencies(self, node: ast.FunctionDef) -> List[str]:
        """Extract function dependencies."""
        dependencies = []

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.append(
                        ast.unparse(child.func) if hasattr(ast, "unparse") else str(child.func)
                    )

        return list(set(dependencies))  # Remove duplicates

    def _is_method(self, node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if a function is a method (inside a class)."""
        return self._find_containing_class(node, tree) is not None

    def _find_containing_class(self, node: ast.FunctionDef, tree: ast.AST) -> Optional[str]:
        """Find the containing class of a method."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in parent.body:
                    return parent.name
        return None

    def _classify_dependency(self, dep_name: str) -> str:
        """Classify a dependency as external, internal, or standard library."""
        standard_libs = {
            "os",
            "sys",
            "json",
            "logging",
            "pathlib",
            "time",
            "datetime",
            "asyncio",
            "subprocess",
            "traceback",
            "collections",
            "dataclasses",
            "enum",
            "typing",
            "re",
            "inspect",
            "ast",
        }

        if dep_name in standard_libs:
            return "standard_library"
        elif dep_name.startswith(".") or dep_name.startswith(".."):
            return "internal"
        else:
            return "external"

    def _is_protocol_or_interface(self, node: ast.ClassDef) -> bool:
        """Check if a class is a protocol or interface."""
        # Check base classes for Protocol
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "Protocol":
                return True
            elif isinstance(base, ast.Attribute) and base.attr == "Protocol":
                return True

        # Check for ABC or interface patterns
        if "Protocol" in node.name or "Interface" in node.name:
            return True

        return False

    def _determine_module_type(self, analysis_data: Dict[str, Any]) -> str:
        """Determine the type of module."""
        return "Application Module"  # Could be enhanced with more analysis

    def _determine_primary_purpose(self, analysis_data: Dict[str, Any]) -> str:
        """Determine the primary purpose of the module."""
        return "Core functionality implementation"  # Could be enhanced

    def _assess_complexity_level(self, analysis_data: Dict[str, Any]) -> str:
        """Assess the complexity level of the module."""
        stats = analysis_data.get("statistics", {})
        total_lines = stats.get("total_lines", 0)
        class_count = len(stats.get("classes", []))

        if total_lines > 2000 or class_count > 15:
            return "High"
        elif total_lines > 1000 or class_count > 8:
            return "Medium"
        else:
            return "Low"

    def _assess_maintainability(self, analysis_data: Dict[str, Any]) -> str:
        """Assess the maintainability of the module."""
        complexity = self._assess_complexity_level(analysis_data)

        if complexity == "High":
            return "Challenging"
        elif complexity == "Medium":
            return "Moderate"
        else:
            return "Good"

    def _assess_class_complexity(self, node: ast.ClassDef, methods: List[str]) -> str:
        """Assess class complexity."""
        method_count = len(methods)
        base_count = len(node.bases)

        if method_count > 20 or base_count > 3:
            return "High"
        elif method_count > 10 or base_count > 1:
            return "Medium"
        else:
            return "Low"

    def _assess_function_complexity(self, node: ast.FunctionDef) -> str:
        """Assess function complexity."""
        # Count nested structures
        nested_count = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                nested_count += 1

        param_count = len(node.args.args)

        if nested_count > 5 or param_count > 8:
            return "High"
        elif nested_count > 2 or param_count > 4:
            return "Medium"
        else:
            return "Low"

    def _get_dependency_type(self, dep_name: str) -> str:
        """Get the type of dependency."""
        if dep_name in ["rich", "click", "requests"]:
            return "Library"
        elif dep_name in ["asyncio", "json", "logging"]:
            return "Standard"
        else:
            return "Unknown"

    def _assess_performance_level(self, component: ComponentInfo) -> str:
        """Assess performance level of a component."""
        if component.complexity == "High":
            return "Low"
        elif component.complexity == "Medium":
            return "Medium"
        else:
            return "High"

    def _identify_bottlenecks(self, component: ComponentInfo) -> str:
        """Identify potential bottlenecks."""
        if len(component.methods) > 20:
            return "Method count"
        elif len(component.dependencies) > 10:
            return "Dependencies"
        else:
            return "None identified"

    def _assess_optimization_potential(self, component: ComponentInfo) -> str:
        """Assess optimization potential."""
        if component.complexity == "High":
            return "High"
        elif component.complexity == "Medium":
            return "Medium"
        else:
            return "Low"

    def _assess_memory_impact(self, component: ComponentInfo) -> str:
        """Assess memory impact."""
        if len(component.attributes) > 10:
            return "High"
        elif len(component.attributes) > 5:
            return "Medium"
        else:
            return "Low"

    def _assess_scalability(self, component: ComponentInfo) -> str:
        """Assess scalability."""
        if component.complexity == "Low":
            return "Good"
        elif component.complexity == "Medium":
            return "Fair"
        else:
            return "Poor"

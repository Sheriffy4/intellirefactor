"""
Architecture Diagram Generator for IntelliRefactor

Generates comprehensive architecture diagrams for Python modules including:
- Component relationships
- Data flow diagrams
- Dependency graphs
- Design pattern identification
"""

import ast
import inspect
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
import re

from ..analysis.project_analyzer import ProjectAnalyzer
from ..analysis.file_analyzer import FileAnalyzer


@dataclass
class Component:
    """Represents a component in the architecture."""
    name: str
    type: str  # class, function, module, etc.
    purpose: str
    complexity: str
    dependencies: List[str]
    methods: List[str] = None
    attributes: List[str] = None
    line_start: int = 0
    line_end: int = 0


@dataclass
class Relationship:
    """Represents a relationship between components."""
    source: str
    target: str
    relationship_type: str  # inheritance, composition, dependency, etc.
    strength: str  # high, medium, low


class ArchitectureGenerator:
    """Generates architecture diagrams for Python modules."""
    
    def __init__(self):
        self.components: Dict[str, Component] = {}
        self.relationships: List[Relationship] = []
        self.design_patterns: List[str] = []
        self.file_analyzer = FileAnalyzer()
        
    def analyze_module(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a Python module and extract architecture information."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            # Extract components
            self._extract_components(tree, source_code)
            
            # Analyze relationships
            self._analyze_relationships(tree)
            
            # Identify design patterns
            self._identify_design_patterns()
            
            return {
                'components': self.components,
                'relationships': self.relationships,
                'design_patterns': self.design_patterns,
                'file_path': str(file_path)
            }
            
        except Exception as e:
            return {
                'error': f"Failed to analyze module: {str(e)}",
                'components': {},
                'relationships': [],
                'design_patterns': []
            }
    
    def _extract_components(self, tree: ast.AST, source_code: str) -> None:
        """Extract components (classes, functions) from AST."""
        lines = source_code.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._extract_class_component(node, lines)
            elif isinstance(node, ast.FunctionDef):
                self._extract_function_component(node, lines)
    
    def _extract_class_component(self, node: ast.ClassDef, lines: List[str]) -> None:
        """Extract class component information."""
        # Get docstring for purpose
        purpose = "No description available"
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            purpose = node.body[0].value.value.strip()
        
        # Extract methods
        methods = []
        attributes = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                methods.append(item.name)
            elif isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Name):
                        attributes.append(target.id)
        
        # Determine complexity based on number of methods and inheritance
        complexity = self._calculate_class_complexity(node, methods)
        
        # Extract dependencies (base classes, imports used)
        dependencies = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                dependencies.append(base.id)
            elif isinstance(base, ast.Attribute):
                dependencies.append(ast.unparse(base))
        
        component = Component(
            name=node.name,
            type="class",
            purpose=purpose,
            complexity=complexity,
            dependencies=dependencies,
            methods=methods,
            attributes=attributes,
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno
        )
        
        self.components[node.name] = component
    
    def _extract_function_component(self, node: ast.FunctionDef, lines: List[str]) -> None:
        """Extract function component information."""
        # Skip methods (they're handled in class extraction)
        if any(isinstance(parent, ast.ClassDef) for parent in ast.walk(ast.Module(body=[node]))):
            return
        
        # Get docstring for purpose
        purpose = "No description available"
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            purpose = node.body[0].value.value.strip()
        
        # Determine complexity based on function length and structure
        complexity = self._calculate_function_complexity(node)
        
        # Extract dependencies (function calls, imports)
        dependencies = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    dependencies.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    dependencies.append(ast.unparse(child.func))
        
        component = Component(
            name=node.name,
            type="function",
            purpose=purpose,
            complexity=complexity,
            dependencies=list(set(dependencies)),  # Remove duplicates
            line_start=node.lineno,
            line_end=node.end_lineno or node.lineno
        )
        
        self.components[node.name] = component
    
    def _calculate_class_complexity(self, node: ast.ClassDef, methods: List[str]) -> str:
        """Calculate class complexity based on various metrics."""
        method_count = len(methods)
        base_count = len(node.bases)
        
        if method_count > 20 or base_count > 3:
            return "High"
        elif method_count > 10 or base_count > 1:
            return "Medium"
        else:
            return "Low"
    
    def _calculate_function_complexity(self, node: ast.FunctionDef) -> str:
        """Calculate function complexity based on various metrics."""
        # Count nested structures
        nested_count = 0
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                nested_count += 1
        
        # Count parameters
        param_count = len(node.args.args)
        
        if nested_count > 5 or param_count > 8:
            return "High"
        elif nested_count > 2 or param_count > 4:
            return "Medium"
        else:
            return "Low"
    
    def _analyze_relationships(self, tree: ast.AST) -> None:
        """Analyze relationships between components."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                self._analyze_class_relationships(node)
    
    def _analyze_class_relationships(self, node: ast.ClassDef) -> None:
        """Analyze relationships for a specific class."""
        class_name = node.name
        
        # Inheritance relationships
        for base in node.bases:
            base_name = None
            if isinstance(base, ast.Name):
                base_name = base.id
            elif isinstance(base, ast.Attribute):
                base_name = ast.unparse(base)
            
            if base_name and base_name in self.components:
                relationship = Relationship(
                    source=class_name,
                    target=base_name,
                    relationship_type="inheritance",
                    strength="high"
                )
                self.relationships.append(relationship)
        
        # Composition/Aggregation relationships (instance variables)
        for item in node.body:
            if isinstance(item, ast.Assign):
                for target in item.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == "self":
                            # Check if the assigned value references another component
                            if isinstance(item.value, ast.Call):
                                if isinstance(item.value.func, ast.Name):
                                    target_name = item.value.func.id
                                    if target_name in self.components:
                                        relationship = Relationship(
                                            source=class_name,
                                            target=target_name,
                                            relationship_type="composition",
                                            strength="medium"
                                        )
                                        self.relationships.append(relationship)
    
    def _identify_design_patterns(self) -> None:
        """Identify common design patterns in the architecture."""
        self._identify_strategy_pattern()
        self._identify_factory_pattern()
        self._identify_facade_pattern()
        self._identify_observer_pattern()
        self._identify_singleton_pattern()
    
    def _identify_strategy_pattern(self) -> None:
        """Identify Strategy pattern usage."""
        # Look for classes with similar method signatures
        method_signatures = {}
        
        for name, component in self.components.items():
            if component.type == "class" and component.methods:
                for method in component.methods:
                    if method not in method_signatures:
                        method_signatures[method] = []
                    method_signatures[method].append(name)
        
        # Find methods implemented by multiple classes
        for method, classes in method_signatures.items():
            if len(classes) >= 3 and method not in ['__init__', '__str__', '__repr__']:
                self.design_patterns.append(f"Strategy Pattern: {method} method implemented by {', '.join(classes)}")
    
    def _identify_factory_pattern(self) -> None:
        """Identify Factory pattern usage."""
        for name, component in self.components.items():
            if component.type == "function":
                if ("create" in name.lower() or "make" in name.lower() or 
                    "build" in name.lower() or "factory" in name.lower()):
                    self.design_patterns.append(f"Factory Pattern: {name} function")
    
    def _identify_facade_pattern(self) -> None:
        """Identify Facade pattern usage."""
        for name, component in self.components.items():
            if component.type == "class":
                # Look for classes that aggregate many dependencies
                if len(component.dependencies) >= 5:
                    self.design_patterns.append(f"Facade Pattern: {name} class coordinates multiple subsystems")
    
    def _identify_observer_pattern(self) -> None:
        """Identify Observer pattern usage."""
        for name, component in self.components.items():
            if component.type == "class" and component.methods:
                observer_methods = [m for m in component.methods if 
                                 'notify' in m.lower() or 'update' in m.lower() or 
                                 'callback' in m.lower() or 'listener' in m.lower()]
                if observer_methods:
                    self.design_patterns.append(f"Observer Pattern: {name} class with {', '.join(observer_methods)}")
    
    def _identify_singleton_pattern(self) -> None:
        """Identify Singleton pattern usage."""
        for name, component in self.components.items():
            if component.type == "class" and component.methods:
                if 'instance' in component.methods or '__new__' in component.methods:
                    self.design_patterns.append(f"Singleton Pattern: {name} class")
    
    def generate_architecture_diagram(self, analysis_result: Dict[str, Any], 
                                    module_name: str) -> str:
        """Generate Mermaid architecture diagram."""
        components = analysis_result.get('components', {})
        relationships = analysis_result.get('relationships', [])
        design_patterns = analysis_result.get('design_patterns', [])
        
        diagram = f"""# Architecture Diagram: {module_name}

## Module Overview
Comprehensive architecture analysis of the {module_name} module showing components, relationships, and design patterns.

```mermaid
graph TB
    subgraph "{module_name} Architecture"
"""
        
        # Add components grouped by type
        class_components = {name: comp for name, comp in components.items() if comp.type == "class"}
        function_components = {name: comp for name, comp in components.items() if comp.type == "function"}
        
        if class_components:
            diagram += f"""
        subgraph "Core Classes"
"""
            for name, component in class_components.items():
                safe_name = name.replace('-', '_').replace('.', '_')
                diagram += f"""            {safe_name}[{name}]
"""
            diagram += "        end\n"
        
        if function_components:
            diagram += f"""
        subgraph "Utility Functions"
"""
            for name, component in function_components.items():
                safe_name = name.replace('-', '_').replace('.', '_')
                diagram += f"""            {safe_name}[{name}]
"""
            diagram += "        end\n"
        
        diagram += "    end\n"
        
        # Add relationships
        if relationships:
            diagram += "\n    %% Relationships\n"
            for rel in relationships:
                source_safe = rel.source.replace('-', '_').replace('.', '_')
                target_safe = rel.target.replace('-', '_').replace('.', '_')
                
                if rel.relationship_type == "inheritance":
                    diagram += f"    {source_safe} -.-> {target_safe}\n"
                elif rel.relationship_type == "composition":
                    diagram += f"    {source_safe} --> {target_safe}\n"
                else:
                    diagram += f"    {source_safe} --> {target_safe}\n"
        
        # Add styling
        diagram += """
    %% Styling
    classDef class fill:#e1f5fe,stroke:#01579b,stroke-width:2px
    classDef function fill:#f3e5f5,stroke:#4a148c,stroke-width:2px
    classDef high_complexity fill:#ffebee,stroke:#b71c1c,stroke-width:2px
    classDef medium_complexity fill:#fff3e0,stroke:#e65100,stroke-width:2px
    classDef low_complexity fill:#e8f5e8,stroke:#1b5e20,stroke-width:2px
"""
        
        # Apply styling based on component types and complexity
        for name, component in components.items():
            safe_name = name.replace('-', '_').replace('.', '_')
            if component.type == "class":
                diagram += f"    class {safe_name} class\n"
            elif component.type == "function":
                diagram += f"    class {safe_name} function\n"
            
            # Apply complexity styling
            if component.complexity == "High":
                diagram += f"    class {safe_name} high_complexity\n"
            elif component.complexity == "Medium":
                diagram += f"    class {safe_name} medium_complexity\n"
            else:
                diagram += f"    class {safe_name} low_complexity\n"
        
        diagram += "```\n\n"
        
        # Add component details
        diagram += "## Component Details\n\n"
        
        if class_components:
            diagram += "### Classes\n\n"
            for name, component in class_components.items():
                diagram += f"#### {name}\n"
                diagram += f"- **Purpose**: {component.purpose[:100]}...\n" if len(component.purpose) > 100 else f"- **Purpose**: {component.purpose}\n"
                diagram += f"- **Complexity**: {component.complexity}\n"
                diagram += f"- **Methods**: {len(component.methods or [])}\n"
                diagram += f"- **Dependencies**: {', '.join(component.dependencies[:5])}\n"
                if len(component.dependencies) > 5:
                    diagram += f"  (and {len(component.dependencies) - 5} more)\n"
                diagram += f"- **Lines**: {component.line_start}-{component.line_end}\n\n"
        
        if function_components:
            diagram += "### Functions\n\n"
            for name, component in function_components.items():
                diagram += f"#### {name}\n"
                diagram += f"- **Purpose**: {component.purpose[:100]}...\n" if len(component.purpose) > 100 else f"- **Purpose**: {component.purpose}\n"
                diagram += f"- **Complexity**: {component.complexity}\n"
                diagram += f"- **Dependencies**: {', '.join(component.dependencies[:3])}\n"
                if len(component.dependencies) > 3:
                    diagram += f"  (and {len(component.dependencies) - 3} more)\n"
                diagram += f"- **Lines**: {component.line_start}-{component.line_end}\n\n"
        
        # Add design patterns
        if design_patterns:
            diagram += "## Design Patterns Identified\n\n"
            for i, pattern in enumerate(design_patterns, 1):
                diagram += f"{i}. {pattern}\n"
            diagram += "\n"
        
        # Add relationships details
        if relationships:
            diagram += "## Relationships\n\n"
            inheritance_rels = [r for r in relationships if r.relationship_type == "inheritance"]
            composition_rels = [r for r in relationships if r.relationship_type == "composition"]
            
            if inheritance_rels:
                diagram += "### Inheritance\n"
                for rel in inheritance_rels:
                    diagram += f"- {rel.source} inherits from {rel.target}\n"
                diagram += "\n"
            
            if composition_rels:
                diagram += "### Composition\n"
                for rel in composition_rels:
                    diagram += f"- {rel.source} uses {rel.target}\n"
                diagram += "\n"
        
        # Add metrics summary
        diagram += "## Architecture Metrics\n\n"
        total_components = len(components)
        class_count = len(class_components)
        function_count = len(function_components)
        high_complexity = len([c for c in components.values() if c.complexity == "High"])
        
        diagram += f"- **Total Components**: {total_components}\n"
        diagram += f"- **Classes**: {class_count}\n"
        diagram += f"- **Functions**: {function_count}\n"
        diagram += f"- **High Complexity Components**: {high_complexity}\n"
        diagram += f"- **Design Patterns**: {len(design_patterns)}\n"
        diagram += f"- **Relationships**: {len(relationships)}\n"
        
        return diagram
    
    def generate_data_flow_diagram(self, analysis_result: Dict[str, Any], 
                                 module_name: str) -> str:
        """Generate data flow diagram."""
        components = analysis_result.get('components', {})
        
        diagram = f"""
## Data Flow Diagram

```mermaid
sequenceDiagram
    participant User
"""
        
        # Add main components as participants
        main_classes = [name for name, comp in components.items() 
                       if comp.type == "class" and len(comp.methods or []) > 3]
        
        for class_name in main_classes[:5]:  # Limit to 5 main classes
            diagram += f"    participant {class_name}\n"
        
        diagram += "\n"
        
        # Add typical flow (this is a simplified example)
        if main_classes:
            main_class = main_classes[0]
            diagram += f"    User->>+{main_class}: initialize\n"
            
            for other_class in main_classes[1:3]:  # Show interaction with 2 other classes
                diagram += f"    {main_class}->>+{other_class}: process_data\n"
                diagram += f"    {other_class}-->>-{main_class}: result\n"
            
            diagram += f"    {main_class}-->>-User: final_result\n"
        
        diagram += "```\n"
        
        return diagram


def generate_architecture_documentation(file_path: str, output_path: str = None) -> str:
    """Generate comprehensive architecture documentation for a module."""
    generator = ArchitectureGenerator()
    path_obj = Path(file_path)
    
    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Analyze the module
    analysis_result = generator.analyze_module(path_obj)
    
    if 'error' in analysis_result:
        return f"Error: {analysis_result['error']}"
    
    # Generate documentation
    module_name = path_obj.stem
    architecture_doc = generator.generate_architecture_diagram(analysis_result, module_name)
    data_flow_doc = generator.generate_data_flow_diagram(analysis_result, module_name)
    
    full_doc = architecture_doc + data_flow_doc
    
    # Save to file if output path specified
    if output_path:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(full_doc)
        return f"Architecture documentation saved to: {output_path}"
    
    return full_doc
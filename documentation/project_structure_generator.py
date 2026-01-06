"""
Project Structure Generator for IntelliRefactor

Generates comprehensive project structure documentation including:
- Current module organization
- Recommended refactored structure
- Migration strategies
- Dependency mappings
- Backward compatibility plans
"""

import ast
import os
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Tuple
from dataclasses import dataclass
import re

from ..analysis.file_analyzer import FileAnalyzer


@dataclass
class ModuleComponent:
    """Represents a component that should be extracted to its own module."""
    name: str
    type: str  # class, function, constant, etc.
    current_location: str
    suggested_location: str
    dependencies: List[str]
    dependents: List[str]
    extraction_complexity: str  # low, medium, high


@dataclass
class RefactoringPhase:
    """Represents a phase in the refactoring process."""
    phase_number: int
    name: str
    duration: str
    deliverables: List[str]
    risk_level: str
    dependencies: List[str]


class ProjectStructureGenerator:
    """Generates comprehensive project structure documentation."""
    
    def __init__(self):
        self.file_analyzer = FileAnalyzer()
        
    def generate_project_structure(self, file_path: Path) -> str:
        """Generate comprehensive project structure documentation."""
        try:
            # Analyze the file
            analysis_result = self.file_analyzer.analyze_file(file_path)
            
            if not analysis_result.success:
                return f"""# Project Structure: {file_path.name}

## Error
Failed to analyze file: {analysis_result.metadata.get('error', 'Unknown error')}
"""
            
            # Parse source code for detailed analysis
            with open(file_path, 'r', encoding='utf-8') as f:
                source_code = f.read()
            
            tree = ast.parse(source_code)
            
            # Analyze current structure
            current_structure = self._analyze_current_structure(tree, source_code, file_path)
            
            # Generate refactoring recommendations
            components = self._identify_extractable_components(tree, source_code)
            recommended_structure = self._design_recommended_structure(components, file_path)
            
            # Create migration plan
            migration_phases = self._create_migration_phases(components)
            
            # Generate documentation
            structure_doc = self._generate_structure_header(file_path, analysis_result.data)
            structure_doc += self._generate_current_structure_section(current_structure)
            structure_doc += self._generate_recommended_structure_section(recommended_structure)
            structure_doc += self._generate_migration_strategy_section(migration_phases)
            structure_doc += self._generate_benefits_section()
            structure_doc += self._generate_compatibility_section()
            structure_doc += self._generate_timeline_section(migration_phases)
            
            return structure_doc
            
        except Exception as e:
            return f"""# Project Structure: {file_path.name}

## Error
Failed to generate project structure: {str(e)}
"""
    
    def _analyze_current_structure(self, tree: ast.AST, source_code: str, file_path: Path) -> Dict[str, Any]:
        """Analyze the current structure of the module."""
        lines = source_code.split('\n')
        
        structure = {
            'file_path': str(file_path),
            'total_lines': len(lines),
            'classes': [],
            'functions': [],
            'constants': [],
            'imports': [],
            'complexity_assessment': 'high'  # Will be calculated
        }
        
        # Extract components
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_info = {
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno or node.lineno,
                    'methods': [item.name for item in node.body if isinstance(item, ast.FunctionDef)],
                    'complexity': self._assess_class_complexity(node)
                }
                structure['classes'].append(class_info)
            
            elif isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                func_info = {
                    'name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno or node.lineno,
                    'complexity': self._assess_function_complexity(node)
                }
                structure['functions'].append(func_info)
            
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                import_info = self._extract_import_info(node)
                structure['imports'].extend(import_info)
        
        return structure
    
    def _identify_extractable_components(self, tree: ast.AST, source_code: str) -> List[ModuleComponent]:
        """Identify components that can be extracted to separate modules."""
        components = []
        
        # Analyze classes for extraction
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                component = self._analyze_class_for_extraction(node, tree, source_code)
                if component:
                    components.append(component)
        
        # Analyze functions for extraction
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                component = self._analyze_function_for_extraction(node, tree, source_code)
                if component:
                    components.append(component)
        
        # Analyze constants and enums
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                component = self._analyze_constant_for_extraction(node, tree, source_code)
                if component:
                    components.append(component)
        
        return components
    
    def _design_recommended_structure(self, components: List[ModuleComponent], 
                                    original_file: Path) -> Dict[str, Any]:
        """Design the recommended project structure."""
        base_name = original_file.stem
        
        # Group components by suggested location
        structure_map = {}
        for component in components:
            location = component.suggested_location
            if location not in structure_map:
                structure_map[location] = []
            structure_map[location].append(component)
        
        # Create recommended structure
        recommended = {
            'base_directory': f"{base_name}/",
            'modules': {},
            'benefits': [],
            'challenges': []
        }
        
        # Core module
        recommended['modules']['__init__.py'] = {
            'purpose': 'Public API exports',
            'components': ['Public interface definitions'],
            'size_estimate': 'Small (< 100 lines)'
        }
        
        # Group components into logical modules
        for location, location_components in structure_map.items():
            recommended['modules'][location] = {
                'purpose': self._infer_module_purpose(location),
                'components': [comp.name for comp in location_components],
                'size_estimate': self._estimate_module_size(location_components)
            }
        
        return recommended
    
    def _create_migration_phases(self, components: List[ModuleComponent]) -> List[RefactoringPhase]:
        """Create migration phases based on component dependencies."""
        phases = []
        
        # Phase 1: Foundation (low-risk extractions)
        foundation_components = [c for c in components if c.extraction_complexity == 'low']
        if foundation_components:
            phases.append(RefactoringPhase(
                phase_number=1,
                name="Foundation",
                duration="1-2 weeks",
                deliverables=[
                    "Extract configuration classes",
                    "Extract exception hierarchy", 
                    "Create base module structure"
                ],
                risk_level="Low",
                dependencies=[]
            ))
        
        # Phase 2: Core Components (medium-risk extractions)
        core_components = [c for c in components if c.extraction_complexity == 'medium']
        if core_components:
            phases.append(RefactoringPhase(
                phase_number=2,
                name="Core Component Extraction",
                duration="2-3 weeks",
                deliverables=[
                    "Extract main functional components",
                    "Create component interfaces",
                    "Implement dependency injection"
                ],
                risk_level="Medium",
                dependencies=["Phase 1"]
            ))
        
        # Phase 3: Integration (high-risk changes)
        integration_components = [c for c in components if c.extraction_complexity == 'high']
        if integration_components:
            phases.append(RefactoringPhase(
                phase_number=3,
                name="Integration and Testing",
                duration="2-3 weeks",
                deliverables=[
                    "Integrate extracted components",
                    "Create comprehensive test suite",
                    "Performance optimization"
                ],
                risk_level="Medium",
                dependencies=["Phase 2"]
            ))
        
        # Phase 4: Documentation and Cleanup
        phases.append(RefactoringPhase(
            phase_number=len(phases) + 1,
            name="Documentation and Cleanup",
            duration="1 week",
            deliverables=[
                "Complete API documentation",
                "Create usage examples",
                "Final code cleanup"
            ],
            risk_level="Low",
            dependencies=[f"Phase {len(phases)}"] if phases else []
        ))
        
        return phases
    
    def _generate_structure_header(self, file_path: Path, analysis_data: Dict[str, Any]) -> str:
        """Generate structure documentation header."""
        return f"""# Project Structure: {file_path.stem.replace('_', ' ').title()}

## Current Structure

```
{file_path.parent.name}/
└── {file_path.name}    # Monolithic module ({analysis_data.get('statistics', {}).get('total_lines', 'Unknown')} lines)
"""
    
    def _generate_current_structure_section(self, current_structure: Dict[str, Any]) -> str:
        """Generate current structure analysis section."""
        classes = current_structure.get('classes', [])
        functions = current_structure.get('functions', [])
        
        section = f"""    ├── Classes: {len(classes)}
    ├── Functions: {len(functions)}
    ├── Imports: {len(current_structure.get('imports', []))}
    └── Total Lines: {current_structure.get('total_lines', 0)}
```

### Current Organization

"""
        
        if classes:
            section += "#### Classes\n"
            for cls in classes:
                section += f"- **{cls['name']}** (lines {cls['line_start']}-{cls['line_end']}, {len(cls['methods'])} methods)\n"
            section += "\n"
        
        if functions:
            section += "#### Functions\n"
            for func in functions:
                section += f"- **{func['name']}** (lines {func['line_start']}-{func['line_end']})\n"
            section += "\n"
        
        section += """### Issues with Current Structure

1. **Monolithic Design**: All functionality in a single large file
2. **High Complexity**: Difficult to navigate and understand
3. **Testing Challenges**: Large classes are hard to unit test
4. **Maintenance Burden**: Changes require understanding entire module
5. **Limited Reusability**: Components are tightly coupled

"""
        
        return section
    
    def _generate_recommended_structure_section(self, recommended: Dict[str, Any]) -> str:
        """Generate recommended structure section."""
        section = f"""## Recommended Refactored Structure

```
{recommended['base_directory']}
├── __init__.py                 # Public API exports
├── py.typed                    # Type checking marker
├── README.md                   # Module documentation
├── CHANGELOG.md               # Version history
│
"""
        
        # Generate module structure
        modules = recommended.get('modules', {})
        for module_path, module_info in modules.items():
            if module_path == '__init__.py':
                continue
                
            # Determine if it's a directory or file
            if '/' in module_path:
                # It's a directory structure
                parts = module_path.split('/')
                section += f"├── {parts[0]}/                      # {module_info['purpose']}\n"
                section += f"│   ├── __init__.py\n"
                for component in module_info['components'][:3]:  # Show first 3 components
                    section += f"│   ├── {component.lower().replace(' ', '_')}.py\n"
                if len(module_info['components']) > 3:
                    section += f"│   └── ... ({len(module_info['components']) - 3} more files)\n"
                section += "│\n"
            else:
                # It's a single file
                section += f"├── {module_path}             # {module_info['purpose']}\n"
        
        section += """├── tests/                     # Test suite
│   ├── __init__.py
│   ├── conftest.py            # Pytest configuration
│   ├── test_core/             # Core functionality tests
│   ├── test_components/       # Component tests
│   └── test_integration/      # Integration tests
│
├── docs/                      # Documentation
│   ├── api/                   # API documentation
│   ├── examples/              # Usage examples
│   └── guides/                # User guides
│
└── scripts/                   # Development scripts
    ├── build.py               # Build automation
    └── test.py                # Test runner
```

### Module Breakdown

"""
        
        for module_path, module_info in modules.items():
            if module_path == '__init__.py':
                continue
                
            section += f"#### `{module_path}`\n"
            section += f"- **Purpose**: {module_info['purpose']}\n"
            section += f"- **Size Estimate**: {module_info['size_estimate']}\n"
            section += f"- **Components**: {', '.join(module_info['components'][:5])}\n"
            if len(module_info['components']) > 5:
                section += f"  (and {len(module_info['components']) - 5} more)\n"
            section += "\n"
        
        return section
    
    def _generate_migration_strategy_section(self, phases: List[RefactoringPhase]) -> str:
        """Generate migration strategy section."""
        section = "## Migration Strategy\n\n"
        
        for phase in phases:
            section += f"### Phase {phase.phase_number}: {phase.name} ({phase.duration})\n\n"
            
            section += "**Deliverables:**\n"
            for deliverable in phase.deliverables:
                section += f"- {deliverable}\n"
            section += "\n"
            
            section += f"**Risk Level:** {phase.risk_level}\n\n"
            
            if phase.dependencies:
                section += f"**Dependencies:** {', '.join(phase.dependencies)}\n\n"
        
        return section
    
    def _generate_benefits_section(self) -> str:
        """Generate benefits section."""
        return """## Benefits of New Structure

### 1. **Improved Maintainability**
- Smaller, focused modules easier to understand and modify
- Clear separation of concerns
- Reduced cognitive load for developers

### 2. **Better Testability**
- Individual components can be tested in isolation
- Easier to mock dependencies
- More focused test cases

### 3. **Enhanced Extensibility**
- New components can be added easily
- Clear interfaces for extension points
- Pluggable architecture

### 4. **Better Performance**
- Lazy loading of optional components
- Reduced memory footprint
- More efficient imports

### 5. **Improved Documentation**
- Each module has focused documentation
- Clear API boundaries
- Better examples and guides

"""
    
    def _generate_compatibility_section(self) -> str:
        """Generate backward compatibility section."""
        return """## Backward Compatibility

### Import Compatibility
```python
# Old import (still works)
from original_module import MainClass, Config

# New imports (recommended)
from refactored_module.core import MainClass
from refactored_module.config import Config
```

### API Compatibility
- All public APIs remain unchanged
- Factory functions provide same interface
- Configuration objects maintain same structure
- Error types remain the same

### Migration Timeline
- **Immediate**: New structure available alongside old
- **3 months**: Deprecation warnings for old imports
- **6 months**: Old structure removed
- **Documentation**: Updated immediately with both approaches

"""
    
    def _generate_timeline_section(self, phases: List[RefactoringPhase]) -> str:
        """Generate implementation timeline section."""
        section = "## Implementation Timeline\n\n"
        
        section += "| Phase | Duration | Deliverables | Risk Level |\n"
        section += "|-------|----------|--------------|------------|\n"
        
        total_duration = 0
        for phase in phases:
            # Extract numeric duration (simplified)
            duration_weeks = self._extract_duration_weeks(phase.duration)
            total_duration += duration_weeks
            
            deliverables_str = "; ".join(phase.deliverables[:2])
            if len(phase.deliverables) > 2:
                deliverables_str += f" (+{len(phase.deliverables) - 2} more)"
            
            section += f"| **Phase {phase.phase_number}: {phase.name}** | {phase.duration} | {deliverables_str} | {phase.risk_level} |\n"
        
        section += f"| **Total** | **~{total_duration} weeks** | **Fully refactored module** | **Medium** |\n\n"
        
        section += "### Resource Requirements\n\n"
        section += f"- **Development Time**: {total_duration} weeks (1 developer)\n"
        section += "- **Testing Time**: 2 weeks (overlapping with development)\n"
        section += "- **Code Review**: 1 week (distributed across phases)\n"
        section += "- **Documentation**: 1 week (final phase)\n\n"
        
        return section
    
    # Helper methods
    def _is_method(self, node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if a function is a method (inside a class)."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef):
                if node in parent.body:
                    return True
        return False
    
    def _assess_class_complexity(self, node: ast.ClassDef) -> str:
        """Assess the complexity of a class."""
        method_count = len([item for item in node.body if isinstance(item, ast.FunctionDef)])
        base_count = len(node.bases)
        
        if method_count > 20 or base_count > 3:
            return "High"
        elif method_count > 10 or base_count > 1:
            return "Medium"
        else:
            return "Low"
    
    def _assess_function_complexity(self, node: ast.FunctionDef) -> str:
        """Assess the complexity of a function."""
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
    
    def _extract_import_info(self, node: ast.stmt) -> List[Dict[str, str]]:
        """Extract import information from import nodes."""
        imports = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    'name': alias.name,
                    'type': 'import',
                    'module': alias.name
                })
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for alias in node.names:
                    imports.append({
                        'name': alias.name,
                        'type': 'from_import',
                        'module': node.module
                    })
        
        return imports
    
    def _analyze_class_for_extraction(self, node: ast.ClassDef, tree: ast.AST, 
                                    source_code: str) -> Optional[ModuleComponent]:
        """Analyze a class to determine if it should be extracted."""
        # Determine suggested location based on class characteristics
        class_name = node.name
        
        # Configuration classes
        if 'config' in class_name.lower() or 'setting' in class_name.lower():
            suggested_location = 'core/config.py'
        # Exception classes
        elif 'error' in class_name.lower() or 'exception' in class_name.lower():
            suggested_location = 'core/exceptions.py'
        # Strategy classes
        elif 'strategy' in class_name.lower() or 'handler' in class_name.lower():
            suggested_location = 'components/strategies.py'
        # Model classes
        elif 'model' in class_name.lower() or 'data' in class_name.lower():
            suggested_location = 'models/data_models.py'
        # Main classes
        elif len([item for item in node.body if isinstance(item, ast.FunctionDef)]) > 10:
            suggested_location = 'core/main_class.py'
        else:
            suggested_location = 'components/utilities.py'
        
        # Assess extraction complexity
        method_count = len([item for item in node.body if isinstance(item, ast.FunctionDef)])
        base_count = len(node.bases)
        
        if method_count > 15 or base_count > 2:
            complexity = 'high'
        elif method_count > 8 or base_count > 0:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        return ModuleComponent(
            name=class_name,
            type='class',
            current_location='monolithic_module.py',
            suggested_location=suggested_location,
            dependencies=self._extract_class_dependencies(node),
            dependents=[],  # Would need more analysis
            extraction_complexity=complexity
        )
    
    def _analyze_function_for_extraction(self, node: ast.FunctionDef, tree: ast.AST, 
                                       source_code: str) -> Optional[ModuleComponent]:
        """Analyze a function to determine if it should be extracted."""
        func_name = node.name
        
        # Skip private functions for now
        if func_name.startswith('_'):
            return None
        
        # Determine suggested location
        if 'create' in func_name.lower() or 'make' in func_name.lower():
            suggested_location = 'utils/factories.py'
        elif 'validate' in func_name.lower():
            suggested_location = 'validation/validators.py'
        elif 'parse' in func_name.lower() or 'process' in func_name.lower():
            suggested_location = 'processing/processors.py'
        else:
            suggested_location = 'utils/helpers.py'
        
        # Assess complexity
        param_count = len(node.args.args)
        nested_count = len([child for child in ast.walk(node) 
                           if isinstance(child, (ast.If, ast.For, ast.While, ast.Try))])
        
        if param_count > 6 or nested_count > 4:
            complexity = 'high'
        elif param_count > 3 or nested_count > 2:
            complexity = 'medium'
        else:
            complexity = 'low'
        
        return ModuleComponent(
            name=func_name,
            type='function',
            current_location='monolithic_module.py',
            suggested_location=suggested_location,
            dependencies=[],  # Would need more analysis
            dependents=[],
            extraction_complexity=complexity
        )
    
    def _analyze_constant_for_extraction(self, node: ast.Assign, tree: ast.AST, 
                                       source_code: str) -> Optional[ModuleComponent]:
        """Analyze constants for extraction."""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated analysis
        return None
    
    def _extract_class_dependencies(self, node: ast.ClassDef) -> List[str]:
        """Extract dependencies for a class."""
        dependencies = []
        
        # Base classes
        for base in node.bases:
            if isinstance(base, ast.Name):
                dependencies.append(base.id)
        
        # Method calls and attribute access (simplified)
        for child in ast.walk(node):
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name):
                dependencies.append(child.func.id)
        
        return list(set(dependencies))  # Remove duplicates
    
    def _infer_module_purpose(self, location: str) -> str:
        """Infer the purpose of a module based on its location."""
        purposes = {
            'core/config.py': 'Configuration management',
            'core/exceptions.py': 'Exception hierarchy',
            'core/main_class.py': 'Main functionality',
            'components/strategies.py': 'Strategy implementations',
            'components/utilities.py': 'Utility components',
            'models/data_models.py': 'Data structures',
            'utils/factories.py': 'Factory functions',
            'utils/helpers.py': 'Helper utilities',
            'validation/validators.py': 'Input validation',
            'processing/processors.py': 'Data processing'
        }
        
        return purposes.get(location, 'Specialized functionality')
    
    def _estimate_module_size(self, components: List[ModuleComponent]) -> str:
        """Estimate the size of a module based on its components."""
        total_components = len(components)
        
        if total_components > 5:
            return 'Large (> 500 lines)'
        elif total_components > 2:
            return 'Medium (200-500 lines)'
        else:
            return 'Small (< 200 lines)'
    
    def _extract_duration_weeks(self, duration_str: str) -> int:
        """Extract duration in weeks from duration string."""
        # Simple extraction - could be more sophisticated
        if '1-2' in duration_str:
            return 2
        elif '2-3' in duration_str:
            return 3
        elif '1' in duration_str:
            return 1
        else:
            return 2  # Default
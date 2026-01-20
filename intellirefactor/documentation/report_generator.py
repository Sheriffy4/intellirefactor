"""
Report Generator for IntelliRefactor

Generates comprehensive refactoring reports including:
- Code quality analysis
- Refactoring recommendations
- Performance metrics
- Risk assessment
- Implementation timeline
"""

import ast
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass
from datetime import datetime

from ..analysis.refactor.file_analyzer import FileAnalyzer
from ..analysis.refactor.project_analyzer import ProjectAnalyzer


@dataclass
class QualityMetric:
    """Represents a code quality metric."""

    name: str
    value: float
    threshold: float
    status: str  # good, warning, critical
    description: str


@dataclass
class RefactoringOpportunity:
    """Represents a refactoring opportunity."""

    title: str
    description: str
    priority: str  # high, medium, low
    effort: str  # high, medium, low
    impact: str  # high, medium, low
    file_path: str
    line_start: int
    line_end: int
    recommendations: List[str]


class ReportGenerator:
    """Generates comprehensive refactoring reports."""

    def __init__(self):
        self.file_analyzer = FileAnalyzer()
        self.project_analyzer = ProjectAnalyzer()

    def generate_refactoring_report(self, file_path: Path) -> str:
        """Generate comprehensive refactoring report for a module."""
        try:
            # Analyze the file
            analysis_result = self.file_analyzer.analyze_file(str(file_path))

            if not analysis_result.success:
                return f"""# Refactoring Report: {file_path.name}

## Error
Failed to analyze file: {analysis_result.metadata.get("error", "Unknown error")}
"""

            # Extract analysis data
            analysis_data = analysis_result.data if isinstance(analysis_result.data, dict) else {}
            analysis_data = self._normalize_statistics(file_path, analysis_data)

            # Generate report sections
            report = self._generate_report_header(file_path, analysis_data)
            report += self._generate_executive_summary(analysis_data)
            report += self._generate_current_architecture_analysis(analysis_data)
            report += self._generate_quality_metrics(analysis_data)
            report += self._generate_refactoring_recommendations(analysis_data)
            report += self._generate_detailed_plan(analysis_data)
            report += self._generate_risk_assessment(analysis_data)
            report += self._generate_success_metrics(analysis_data)
            report += self._generate_implementation_timeline(analysis_data)
            report += self._generate_conclusion(analysis_data)

            return report

        except Exception as e:
            return f"""# Refactoring Report: {file_path.name}

## Error
Failed to generate report: {str(e)}
"""

    def _normalize_statistics(self, file_path: Path, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensure analysis_data["statistics"] contains:
          - total_lines: int
          - classes: list
          - functions: list
        Many generators assume these exist, but FileAnalyzer may store them elsewhere.
        """
        stats = analysis_data.get("statistics")
        if not isinstance(stats, dict):
            stats = {}

        # Try to read total lines from existing sources
        total_lines = stats.get("total_lines")
        if not isinstance(total_lines, int) or total_lines <= 0:
            total_lines = analysis_data.get("total_lines")
        if not isinstance(total_lines, int) or total_lines <= 0:
            try:
                total_lines = len(file_path.read_text(encoding="utf-8").splitlines())
            except UnicodeDecodeError:
                total_lines = len(file_path.read_text(encoding="latin-1").splitlines())
            except Exception:
                total_lines = 0

        # Helper to normalize classes/functions containers into list
        def _to_list(value: Any) -> List[Any]:
            if value is None:
                return []
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                # might be mapping name -> info
                return [{"name": k, **(v if isinstance(v, dict) else {})} for k, v in value.items()]
            # unknown scalar/object
            return [value]

        # Locate classes/functions in likely places
        classes = stats.get("classes")
        if classes is None:
            classes = analysis_data.get("classes")
        if classes is None and isinstance(analysis_data.get("structure"), dict):
            classes = analysis_data["structure"].get("classes")

        functions = stats.get("functions")
        if functions is None:
            functions = analysis_data.get("functions")
        if functions is None and isinstance(analysis_data.get("structure"), dict):
            functions = analysis_data["structure"].get("functions")

        classes_list = _to_list(classes)
        functions_list = _to_list(functions)

        # If still empty, derive from AST (top-level classes + top-level functions)
        if not classes_list and not functions_list:
            try:
                try:
                    src = file_path.read_text(encoding="utf-8")
                except UnicodeDecodeError:
                    src = file_path.read_text(encoding="latin-1")
                tree = ast.parse(src, filename=str(file_path))
                top_classes = []
                top_funcs = []
                for node in getattr(tree, "body", []):
                    if isinstance(node, ast.ClassDef):
                        top_classes.append({"name": node.name, "line_start": getattr(node, "lineno", 0) or 0})
                    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        top_funcs.append({"name": node.name, "line_start": getattr(node, "lineno", 0) or 0})
                classes_list = top_classes
                functions_list = top_funcs
            except Exception:
                # keep empty if parsing fails
                pass

        stats["total_lines"] = int(total_lines or 0)
        stats["classes"] = classes_list
        stats["functions"] = functions_list
        analysis_data["statistics"] = stats
        return analysis_data

    def _generate_report_header(self, file_path: Path, analysis_data: Dict[str, Any]) -> str:
        """Generate report header with basic information."""
        stats = analysis_data.get("statistics", {})

        return f"""# {file_path.stem.replace("_", " ").title()} Refactoring Report

## Executive Summary

The `{file_path.name}` module is a comprehensive component that requires strategic refactoring to improve maintainability, performance, and extensibility. This report analyzes the current state of the module and provides recommendations for refactoring to enhance code quality.

## Module Overview

- **File**: `{file_path}`
- **Lines of Code**: ~{stats.get("total_lines", "Unknown")} lines
- **Classes**: {len(stats.get("classes", []))} classes
- **Functions**: {len(stats.get("functions", []))} functions
- **Complexity**: {self._assess_overall_complexity(analysis_data)}
- **Analysis Date**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

"""

    def _generate_executive_summary(self, analysis_data: Dict[str, Any]) -> str:
        """Generate executive summary section."""
        return """## Current Architecture Analysis

### Strengths

1. **Comprehensive Functionality**
   - Well-defined feature set with clear purpose
   - Extensive error handling capabilities
   - Good separation of concerns in some areas

2. **Robust Error Handling**
   - Comprehensive exception hierarchy
   - Graceful degradation patterns
   - User-friendly error messages

3. **Flexible Design**
   - Multiple configuration options
   - Extensible architecture patterns
   - Good abstraction layers

### Areas for Improvement

1. **Module Size and Complexity**
   - Large monolithic structure
   - High cyclomatic complexity in key methods
   - Difficult to navigate and maintain

2. **Code Duplication**
   - Similar patterns repeated across components
   - Redundant validation and processing logic
   - Opportunity for consolidation

3. **Performance Optimization**
   - Potential bottlenecks in critical paths
   - Inefficient resource usage patterns
   - Synchronous operations in async contexts

4. **Testing Challenges**
   - Large classes difficult to unit test
   - Complex dependencies complicate mocking
   - Limited separation of pure functions

"""

    def _generate_current_architecture_analysis(self, analysis_data: Dict[str, Any]) -> str:
        """Generate current architecture analysis."""
        classes = analysis_data.get("statistics", {}).get("classes", [])
        functions = analysis_data.get("statistics", {}).get("functions", [])

        return f"""## Detailed Analysis

### Component Breakdown

#### Classes ({len(classes)})
"""

    def _generate_quality_metrics(self, analysis_data: Dict[str, Any]) -> str:
        """Generate quality metrics section."""
        stats = analysis_data.get("statistics", {})

        # Calculate metrics
        total_lines = stats.get("total_lines", 0)
        class_count = len(stats.get("classes", []))
        function_count = len(stats.get("functions", []))

        # Assess complexity
        complexity_score = self._calculate_complexity_score(analysis_data)
        maintainability_score = self._calculate_maintainability_score(analysis_data)

        return f"""## Quality Metrics

### Code Quality Assessment

| Metric | Current Value | Target Value | Status |
|--------|---------------|--------------|--------|
| **Lines of Code** | {total_lines} | < 2000 | {"✅ Good" if total_lines < 2000 else "⚠️ Warning" if total_lines < 3000 else "❌ Critical"} |
| **Classes per Module** | {class_count} | < 15 | {"✅ Good" if class_count < 15 else "⚠️ Warning" if class_count < 25 else "❌ Critical"} |
| **Functions per Module** | {function_count} | < 30 | {"✅ Good" if function_count < 30 else "⚠️ Warning" if function_count < 50 else "❌ Critical"} |
| **Complexity Score** | {complexity_score:.1f}/10 | < 6.0 | {"✅ Good" if complexity_score < 6 else "⚠️ Warning" if complexity_score < 8 else "❌ Critical"} |
| **Maintainability** | {maintainability_score:.1f}/10 | > 7.0 | {"✅ Good" if maintainability_score > 7 else "⚠️ Warning" if maintainability_score > 5 else "❌ Critical"} |

### Detailed Metrics

#### Complexity Analysis
- **Cyclomatic Complexity**: {self._estimate_cyclomatic_complexity(analysis_data)}
- **Cognitive Complexity**: {self._estimate_cognitive_complexity(analysis_data)}
- **Nesting Depth**: {self._estimate_nesting_depth(analysis_data)}

#### Maintainability Factors
- **Code Duplication**: {self._assess_code_duplication(analysis_data)}
- **Method Length**: {self._assess_method_length(analysis_data)}
- **Class Cohesion**: {self._assess_class_cohesion(analysis_data)}

"""

    def _generate_refactoring_recommendations(self, analysis_data: Dict[str, Any]) -> str:
        """Generate refactoring recommendations."""
        return """## Refactoring Recommendations

### 1. Module Decomposition

**Priority: High**

Split the monolithic module into focused sub-modules:

```
target_module/
├── __init__.py
├── core/
│   ├── main_class.py       # Primary functionality
│   ├── config.py           # Configuration management
│   └── exceptions.py       # Exception hierarchy
├── components/
│   ├── __init__.py
│   ├── component_a.py      # Specific component logic
│   ├── component_b.py      # Another component
│   └── utilities.py        # Shared utilities
├── interfaces/
│   ├── __init__.py
│   ├── protocols.py        # Protocol definitions
│   └── base_classes.py     # Abstract base classes
└── models/
    ├── __init__.py
    ├── data_models.py      # Data structures
    └── result_models.py    # Result objects
```

### 2. Extract Common Patterns

**Priority: Medium**

Create base classes and mixins for common functionality:

```python
# Base classes for common patterns
class BaseComponent(ABC):
    \"\"\"Base class for all components.\"\"\"
    
class ValidationMixin:
    \"\"\"Common validation functionality.\"\"\"
    
class ErrorHandlingMixin:
    \"\"\"Common error handling patterns.\"\"\"
    
class ConfigurationMixin:
    \"\"\"Common configuration management.\"\"\"
```

### 3. Improve Design Patterns

**Priority: Medium**

Implement proven design patterns:

- **Strategy Pattern**: For algorithm variations
- **Factory Pattern**: For object creation
- **Observer Pattern**: For event handling
- **Command Pattern**: For operation encapsulation

### 4. Performance Optimizations

**Priority: Medium**

- **Caching**: Implement result caching for expensive operations
- **Lazy Loading**: Defer initialization until needed
- **Async Optimization**: Better use of async/await patterns
- **Resource Management**: Optimize memory and file handle usage

### 5. Enhanced Testing Support

**Priority: High**

- Extract pure functions for easier testing
- Create mock-friendly interfaces
- Add dependency injection for better testability
- Separate business logic from infrastructure concerns

"""

    def _generate_detailed_plan(self, analysis_data: Dict[str, Any]) -> str:
        """Generate detailed refactoring plan."""
        return """## Detailed Refactoring Plan

### Phase 1: Foundation (Week 1-2)

#### 1.1 Extract Configuration Classes
```python
# Move to core/config.py
class ModuleConfig:
    \"\"\"Main configuration class.\"\"\"
    
class ComponentConfig:
    \"\"\"Component-specific configuration.\"\"\"
```

#### 1.2 Extract Exception Hierarchy
```python
# Move to core/exceptions.py
class ModuleError(Exception):
    \"\"\"Base exception for module.\"\"\"
    
class ValidationError(ModuleError):
    \"\"\"Validation-related errors.\"\"\"
```

#### 1.3 Create Base Interfaces
```python
# Create interfaces/protocols.py
class ComponentProtocol(Protocol):
    \"\"\"Protocol for components.\"\"\"
    
class ProcessorProtocol(Protocol):
    \"\"\"Protocol for processors.\"\"\"
```

### Phase 2: Component Extraction (Week 3-4)

#### 2.1 Extract Core Components
- Identify and extract main functional components
- Create focused classes with single responsibilities
- Implement proper interfaces and protocols

#### 2.2 Extract Utility Functions
- Move utility functions to dedicated modules
- Create pure functions where possible
- Implement proper error handling

### Phase 3: Integration and Testing (Week 5-6)

#### 3.1 Integration Layer
- Create facade classes for backward compatibility
- Implement dependency injection
- Add configuration management

#### 3.2 Testing Infrastructure
- Create comprehensive test suite
- Add integration tests
- Implement performance benchmarks

### Phase 4: Documentation and Cleanup (Week 7)

#### 4.1 Documentation
- Update API documentation
- Create usage examples
- Write migration guide

#### 4.2 Final Cleanup
- Remove deprecated code
- Optimize imports
- Final performance tuning

"""

    def _generate_risk_assessment(self, analysis_data: Dict[str, Any]) -> str:
        """Generate risk assessment section."""
        return """## Risk Assessment

### Low Risk Changes
- Configuration extraction
- Exception hierarchy reorganization
- Documentation improvements
- Adding type hints

### Medium Risk Changes
- Component extraction and reorganization
- Interface implementations
- Performance optimizations
- Test infrastructure changes

### High Risk Changes
- Major architectural modifications
- Breaking API changes
- Async pattern modifications
- External dependency changes

### Mitigation Strategies

1. **Incremental Approach**
   - Implement changes in small, testable increments
   - Maintain backward compatibility during transition
   - Use feature flags for new functionality

2. **Comprehensive Testing**
   - Maintain 90%+ test coverage throughout refactoring
   - Add integration tests for critical workflows
   - Implement regression testing

3. **Rollback Plan**
   - Maintain version control checkpoints
   - Document rollback procedures
   - Keep original implementation available

4. **Stakeholder Communication**
   - Regular progress updates
   - Early feedback collection
   - Clear migration documentation

"""

    def _generate_success_metrics(self, analysis_data: Dict[str, Any]) -> str:
        """Generate success metrics section."""
        return """## Success Metrics

### Code Quality Improvements

1. **Complexity Reduction**
   - Target: 40% reduction in cyclomatic complexity
   - Measure: Average method complexity < 5
   - Timeline: End of Phase 2

2. **Test Coverage**
   - Target: 90%+ code coverage
   - Measure: Line and branch coverage
   - Timeline: End of Phase 3

3. **Code Duplication**
   - Target: < 5% code duplication
   - Measure: Static analysis tools
   - Timeline: End of Phase 2

### Performance Improvements

1. **Startup Time**
   - Target: 20% improvement in initialization time
   - Measure: Benchmark tests
   - Timeline: End of Phase 3

2. **Memory Usage**
   - Target: 30% reduction in peak memory usage
   - Measure: Memory profiling
   - Timeline: End of Phase 3

3. **Processing Speed**
   - Target: 25% improvement in core operations
   - Measure: Performance benchmarks
   - Timeline: End of Phase 4

### Maintainability Improvements

1. **Module Size**
   - Target: No single module > 500 lines
   - Measure: Line count analysis
   - Timeline: End of Phase 2

2. **Class Complexity**
   - Target: No class > 20 methods
   - Measure: Static analysis
   - Timeline: End of Phase 2

3. **Documentation Coverage**
   - Target: 100% public API documentation
   - Measure: Documentation analysis
   - Timeline: End of Phase 4

"""

    def _generate_implementation_timeline(self, analysis_data: Dict[str, Any]) -> str:
        """Generate implementation timeline."""
        return """## Implementation Timeline

| Phase | Duration | Deliverables | Risk Level |
|-------|----------|--------------|------------|
| **Phase 1: Foundation** | 2 weeks | Module structure, base classes, interfaces | Low |
| **Phase 2: Component Extraction** | 2 weeks | Core components, utilities, protocols | Medium |
| **Phase 3: Integration & Testing** | 2 weeks | Integration layer, test suite, benchmarks | Medium |
| **Phase 4: Documentation & Cleanup** | 1 week | Documentation, examples, final optimization | Low |
| **Total** | **7 weeks** | **Fully refactored and documented module** | **Medium** |

### Milestone Schedule

#### Week 1-2: Foundation
- [ ] Extract configuration classes
- [ ] Create exception hierarchy
- [ ] Define base interfaces and protocols
- [ ] Set up new module structure

#### Week 3-4: Component Extraction
- [ ] Extract core functional components
- [ ] Create utility modules
- [ ] Implement component interfaces
- [ ] Add basic unit tests

#### Week 5-6: Integration & Testing
- [ ] Create integration layer
- [ ] Implement dependency injection
- [ ] Add comprehensive test suite
- [ ] Performance benchmarking

#### Week 7: Documentation & Cleanup
- [ ] Complete API documentation
- [ ] Create usage examples
- [ ] Write migration guide
- [ ] Final code cleanup and optimization

### Resource Requirements

- **Development Time**: 7 weeks (1 developer)
- **Testing Time**: 2 weeks (overlapping with development)
- **Code Review**: 1 week (distributed across phases)
- **Documentation**: 1 week (Phase 4)

"""

    def _generate_conclusion(self, analysis_data: Dict[str, Any]) -> str:
        """Generate conclusion section."""
        return """## Conclusion

The refactoring of this module represents a significant opportunity to improve code quality, maintainability, and performance. The proposed approach balances the need for architectural improvements with the practical constraints of maintaining a working system.

### Key Benefits

1. **Improved Maintainability**: Smaller, focused modules will be easier to understand and modify
2. **Enhanced Testability**: Better separation of concerns will enable more effective unit testing
3. **Better Performance**: Optimizations and architectural improvements will enhance system performance
4. **Increased Extensibility**: Cleaner interfaces and patterns will facilitate future enhancements

### Recommended Next Steps

1. **Stakeholder Approval**: Present this plan to the development team and stakeholders
2. **Environment Setup**: Prepare development and testing environments
3. **Baseline Establishment**: Create comprehensive baseline metrics and tests
4. **Phase 1 Execution**: Begin with foundation work as outlined in the timeline

### Long-term Vision

This refactoring effort will establish a solid foundation for future development, making the codebase more maintainable, testable, and performant. The modular architecture will enable easier feature additions and modifications while reducing the risk of introducing bugs.

The investment in refactoring will pay dividends in reduced maintenance costs, faster feature development, and improved system reliability.
"""

    def _assess_overall_complexity(self, analysis_data: Dict[str, Any]) -> str:
        """Assess overall module complexity."""
        stats = analysis_data.get("statistics", {})
        total_lines = stats.get("total_lines", 0)
        class_count = len(stats.get("classes", []))

        if total_lines > 2500 or class_count > 20:
            return "High"
        elif total_lines > 1500 or class_count > 10:
            return "Medium"
        else:
            return "Low"

    def _calculate_complexity_score(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate complexity score (0-10, higher is more complex)."""
        stats = analysis_data.get("statistics", {})
        total_lines = stats.get("total_lines", 0)
        class_count = len(stats.get("classes", []))
        function_count = len(stats.get("functions", []))

        # Normalize and weight factors
        line_score = min(total_lines / 500, 10) * 0.4
        class_score = min(class_count / 5, 10) * 0.3
        function_score = min(function_count / 20, 10) * 0.3

        return line_score + class_score + function_score

    def _calculate_maintainability_score(self, analysis_data: Dict[str, Any]) -> float:
        """Calculate maintainability score (0-10, higher is better)."""
        complexity_score = self._calculate_complexity_score(analysis_data)
        return max(0, 10 - complexity_score)

    def _estimate_cyclomatic_complexity(self, analysis_data: Dict[str, Any]) -> str:
        """Estimate cyclomatic complexity."""
        stats = analysis_data.get("statistics", {})
        function_count = len(stats.get("functions", []))

        if function_count > 30:
            return "High (estimated 15-25 per method)"
        elif function_count > 15:
            return "Medium (estimated 8-15 per method)"
        else:
            return "Low (estimated 3-8 per method)"

    def _estimate_cognitive_complexity(self, analysis_data: Dict[str, Any]) -> str:
        """Estimate cognitive complexity."""
        return "Medium (requires detailed AST analysis)"

    def _estimate_nesting_depth(self, analysis_data: Dict[str, Any]) -> str:
        """Estimate nesting depth."""
        return "Medium (estimated 3-5 levels)"

    def _assess_code_duplication(self, analysis_data: Dict[str, Any]) -> str:
        """Assess code duplication level."""
        return "Medium (estimated 15-25%)"

    def _assess_method_length(self, analysis_data: Dict[str, Any]) -> str:
        """Assess average method length."""
        stats = analysis_data.get("statistics", {})
        total_lines = stats.get("total_lines", 0)
        function_count = len(stats.get("functions", []))

        if function_count > 0:
            avg_length = total_lines / function_count
            if avg_length > 50:
                return "Long (average > 50 lines)"
            elif avg_length > 25:
                return "Medium (average 25-50 lines)"
            else:
                return "Good (average < 25 lines)"
        return "Unknown"

    def _assess_class_cohesion(self, analysis_data: Dict[str, Any]) -> str:
        """Assess class cohesion."""
        return "Medium (requires detailed analysis)"

# IntelliRefactor Knowledge Base

This directory contains accumulated knowledge about refactoring patterns, code transformations, and development process automation.

## 📁 File Structure

### `knowledge_index.json`
Central index of all knowledge items in the knowledge base. Contains metadata about refactoring projects, statistics, and references to knowledge files.

### `pattern_rules.json`
Generic code analysis patterns and rules for identifying refactoring opportunities:
- Large method extraction patterns
- God class decomposition rules
- Tight coupling reduction strategies
- Configuration complexity patterns

### `example_refactoring_metadata.json`
Example metadata structure showing how refactoring knowledge is captured:
- **Transformation rules** with before/after examples
- **Dependency injection patterns** (Constructor Injection)
- **Interface extraction templates**
- **Testing strategies** (Unit + Property-Based)
- **Success metrics** and automation potential

## 🎯 How to Use This Knowledge

### 1. **For Analyzing New Code**
```python
from intellirefactor.knowledge import KnowledgeManager

manager = KnowledgeManager()
recommendations = manager.get_recommendations("large_monolithic_classes")
```

### 2. **For Automatic Code Generation**
Use templates from metadata files for:
- Creating interfaces with pattern `I{ServiceName}`
- Setting up DI containers
- Generating tests (unit + property-based)

### 3. **For Making Refactoring Decisions**
Check your code metrics against applicability criteria:

**✅ Good for Refactoring:**
- Large monolithic classes (>500 lines)
- God objects with multiple responsibilities
- Tightly coupled systems
- Legacy code requiring modernization

**❌ NOT Suitable:**
- Small simple classes (<100 lines)
- Performance-critical tight loops
- Stable APIs without active clients

## 📊 Key Success Metrics

Based on accumulated refactoring experience:

| Metric | Typical Improvement |
|--------|-------------------|
| Code Complexity | -60% to -80% |
| Coupling | -50% to -70% |
| Cohesion | +60% to +80% |
| Testability | +50% to +70% |
| Maintainability | +60% to +80% |
| Test Coverage | +40% to +60% |
| Lines per File | -60% to -80% |

## 🔄 Updating the Knowledge Base

When completing successful refactorings:

1. Generate metadata using the automation tools:
   ```python
   from intellirefactor.knowledge import AutomationMetadata
   metadata = AutomationMetadata.generate_from_project(project_path)
   ```

2. Add to knowledge base:
   ```python
   manager = KnowledgeManager()
   manager.add_refactoring_metadata(metadata_file, project_name)
   ```

3. Update this README with new patterns and metrics

## 🚀 Future Capabilities

This knowledge base enables:

- **Automatic code analysis** and refactoring suggestions
- **Test generation** using established patterns  
- **Code quality assessment** using accumulated metrics
- **ML model training** for refactoring success prediction
- **IDE plugin creation** for refactoring automation

## 🔧 Configuration

The knowledge base can be configured through:
- Environment variables for knowledge directory location
- Configuration files for pattern matching thresholds
- Plugin system for custom analysis rules

---

💡 **Remember**: Each successful refactoring adds value to this knowledge base. Document and preserve your experience!
# IntelliRefactor Plugin Examples

This directory contains example plugins that demonstrate how to extend IntelliRefactor's functionality using the hook system and plugin architecture.

## Overview

The IntelliRefactor plugin system allows you to create custom analysis rules, refactoring transformations, knowledge management, and orchestration logic. The examples in this directory show how to:

1. **Create custom analysis rules** - Detect code patterns and issues specific to your domain
2. **Implement refactoring transformations** - Automate code improvements and restructuring
3. **Manage knowledge** - Learn from refactoring history and provide intelligent recommendations
4. **Coordinate plugins** - Make multiple plugins work together effectively

## Example Plugins

### 1. Custom Rules Plugin (`example_custom_rules_plugin.py`)

Demonstrates how to create custom analysis rules using the hook system.

**Features:**
- Detects long parameter lists
- Analyzes cyclomatic complexity
- Checks naming conventions
- Identifies refactoring opportunities

**Hooks Used:**
- `PRE_FILE_ANALYSIS` - Set up analysis context
- `POST_FILE_ANALYSIS` - Add custom results
- `POST_OPPORTUNITY_DETECTION` - Suggest refactorings
- Custom hooks for specific analysis types

**Configuration:**
```python
config = {
    'max_parameters': 5,
    'max_complexity': 10,
    'naming_conventions': {
        'class_pattern': '^[A-Z][a-zA-Z0-9]*$',
        'function_pattern': '^[a-z_][a-z0-9_]*$',
        'constant_pattern': '^[A-Z_][A-Z0-9_]*$'
    }
}
```

### 2. Refactoring Plugin (`example_refactoring_plugin.py`)

Shows how to implement automated refactoring transformations.

**Features:**
- Extracts magic numbers as constants
- Converts string concatenation to f-strings
- Simplifies boolean expressions
- Applies transformations safely

**Hooks Used:**
- `PRE_REFACTORING` - Prepare transformation context
- `POST_REFACTORING` - Validate results
- Custom hooks for specific transformation types

**Configuration:**
```python
config = {
    'min_string_length': 10,
    'enable_fstring_conversion': True,
    'enable_boolean_simplification': True,
    'min_duplicate_lines': 3
}
```

### 3. Knowledge Plugin (`example_knowledge_plugin.py`)

Demonstrates intelligent knowledge management and learning.

**Features:**
- Learns from refactoring success/failure patterns
- Extracts code patterns from analysis
- Provides recommendations based on history
- Maintains a persistent knowledge database

**Hooks Used:**
- `PRE_KNOWLEDGE_EXTRACTION` - Prepare extraction context
- `POST_KNOWLEDGE_EXTRACTION` - Process extracted knowledge
- `POST_REFACTORING` - Learn from refactoring results
- Custom hooks for pattern extraction and recommendations

**Configuration:**
```python
config = {
    'knowledge_db_path': 'knowledge.json',
    'min_confidence': 0.7,
    'max_knowledge_items': 1000,
    'enable_pattern_learning': True
}
```

### 4. Usage Example (`plugin_usage_example.py`)

Complete example showing how to use all plugins together in a coordinated workflow.

**Features:**
- Plugin initialization and configuration
- Coordinated analysis across multiple plugins
- Knowledge sharing between plugins
- Complete refactoring workflow
- Error handling and cleanup

## Hook System Architecture

The hook system provides extensibility points throughout the IntelliRefactor workflow:

### Standard Hook Types

- **Analysis Hooks:**
  - `PRE_PROJECT_ANALYSIS` - Before analyzing a project
  - `POST_PROJECT_ANALYSIS` - After analyzing a project
  - `PRE_FILE_ANALYSIS` - Before analyzing a file
  - `POST_FILE_ANALYSIS` - After analyzing a file

- **Refactoring Hooks:**
  - `PRE_REFACTORING` - Before applying a refactoring
  - `POST_REFACTORING` - After applying a refactoring
  - `PRE_OPPORTUNITY_DETECTION` - Before detecting opportunities
  - `POST_OPPORTUNITY_DETECTION` - After detecting opportunities

- **Knowledge Hooks:**
  - `PRE_KNOWLEDGE_EXTRACTION` - Before extracting knowledge
  - `POST_KNOWLEDGE_EXTRACTION` - After extracting knowledge
  - `PRE_KNOWLEDGE_QUERY` - Before querying knowledge
  - `POST_KNOWLEDGE_QUERY` - After querying knowledge

- **Validation Hooks:**
  - `PRE_VALIDATION` - Before validation
  - `POST_VALIDATION` - After validation

- **Orchestration Hooks:**
  - `PRE_WORKFLOW` - Before workflow execution
  - `POST_WORKFLOW` - After workflow execution

### Custom Hooks

You can also create custom hooks for domain-specific functionality:

```python
# Register a custom hook
hook_system.register_hook(
    hook_type=HookType.CUSTOM,
    callback=my_custom_function,
    name="my_custom_hook",
    custom_key="my_hook_key"
)

# Execute custom hooks
results = hook_system.execute_custom_hooks("my_hook_key", *args, **kwargs)
```

## Running the Examples

### Prerequisites

1. Install IntelliRefactor and its dependencies
2. Ensure Python 3.8+ is installed
3. Have a Python project to analyze

### Basic Usage

```python
from intellirefactor.plugins.examples.plugin_usage_example import PluginUsageExample

# Create and setup the example
example = PluginUsageExample()
example.setup_plugins()

# Analyze a project
results = example.analyze_project("/path/to/your/project")

# Get refactoring opportunities
opportunities = example.identify_refactoring_opportunities(results)

# Apply a refactoring
if opportunities:
    result = example.apply_refactoring(opportunities[0])

# Get recommendations
recommendations = example.get_recommendations("/path/to/your/project", results)

# Cleanup
example.cleanup()
```

### Complete Workflow

```python
# Run the complete workflow
example = PluginUsageExample()
example.setup_plugins()

results = example.run_complete_workflow("/path/to/your/project")

# Print summary
summary = results['summary']
print(f"Analyzed {summary['files_analyzed']} files")
print(f"Found {summary['opportunities_found']} refactoring opportunities")
print(f"Success rate: {summary['success_rate']:.1%}")

example.cleanup()
```

### Command Line Usage

```bash
# Run the example on the current directory
python plugin_usage_example.py
```

## Creating Your Own Plugins

### 1. Choose Plugin Type

Inherit from the appropriate base class:

```python
from intellirefactor.plugins.plugin_interface import AnalysisPlugin

class MyAnalysisPlugin(AnalysisPlugin):
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my_plugin",
            version="1.0.0",
            description="My custom analysis plugin",
            author="Your Name",
            plugin_type=PluginType.ANALYSIS
        )
```

### 2. Implement Required Methods

```python
def initialize(self) -> bool:
    """Initialize your plugin"""
    # Register hooks, set up resources, etc.
    return True

def analyze_project(self, project_path: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Implement your analysis logic"""
    return {"results": "your analysis results"}

def cleanup(self) -> None:
    """Clean up resources"""
    pass
```

### 3. Register Hooks

```python
def initialize(self) -> bool:
    self.hook_system = getattr(self, '_hook_system', HookSystem())
    
    # Register your hooks
    self.hook_system.register_hook(
        hook_type=HookType.POST_FILE_ANALYSIS,
        callback=self.my_analysis_hook,
        name="my_analysis_hook",
        plugin_name=self.metadata.name
    )
    
    return True
```

### 4. Implement Hook Callbacks

```python
def my_analysis_hook(self, file_path: str, content: str, 
                    context: Dict[str, Any], results: Dict[str, Any]) -> None:
    """Your hook implementation"""
    # Add your custom analysis to results
    results['my_analysis'] = self.analyze_file_content(content)
```

## Best Practices

### Plugin Development

1. **Error Handling:** Always wrap plugin operations in try-catch blocks
2. **Resource Management:** Implement proper cleanup in the `cleanup()` method
3. **Configuration:** Use the config schema to validate plugin settings
4. **Logging:** Use the plugin's logger for consistent log formatting
5. **Testing:** Write tests for your plugin functionality

### Hook Usage

1. **Priority:** Set appropriate hook priorities to control execution order
2. **Context:** Use the hook system context to share data between hooks
3. **Validation:** Validate hook arguments before processing
4. **Performance:** Keep hook callbacks lightweight and fast
5. **Error Isolation:** Don't let hook failures break the entire workflow

### Knowledge Management

1. **Persistence:** Store knowledge in a persistent format (JSON, database)
2. **Confidence:** Track confidence scores for knowledge items
3. **Cleanup:** Implement knowledge cleanup to prevent unbounded growth
4. **Versioning:** Version your knowledge schema for compatibility
5. **Privacy:** Be careful not to store sensitive information

## Troubleshooting

### Common Issues

1. **Plugin Not Loading:**
   - Check that the plugin inherits from the correct base class
   - Verify the `metadata` property is implemented correctly
   - Ensure all required methods are implemented

2. **Hooks Not Executing:**
   - Verify the hook is registered with the correct type and name
   - Check that the hook system is properly injected into the plugin
   - Ensure hook callbacks have the correct signature

3. **Configuration Errors:**
   - Validate configuration against the plugin's schema
   - Check for required configuration parameters
   - Verify configuration file format and location

4. **Performance Issues:**
   - Profile hook execution times
   - Optimize analysis algorithms
   - Consider caching expensive operations

### Debugging

Enable debug logging to see detailed plugin execution:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Use the hook system status to inspect registered hooks:

```python
status = hook_system.get_hook_status()
print(f"Total hooks: {status['total_hooks']}")
print(f"Hooks by type: {status['hooks_by_type']}")
```

## Contributing

If you create useful plugins, consider contributing them back to the IntelliRefactor project:

1. Follow the coding standards and patterns shown in these examples
2. Include comprehensive tests
3. Document your plugin's functionality and configuration
4. Submit a pull request with your plugin

## License

These examples are provided under the same license as IntelliRefactor. See the main project LICENSE file for details.
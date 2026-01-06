# IntelliRefactor Configuration Documentation

This document describes all configuration options available in IntelliRefactor and provides guidance on choosing the right configuration for your project.

## Configuration File Formats

IntelliRefactor supports both JSON and YAML configuration formats:
- JSON: `intellirefactor.json`, `.intellirefactor.json`
- YAML: `intellirefactor.yaml`, `intellirefactor.yml`, `.intellirefactor.yaml`, `.intellirefactor.yml`

## Configuration Loading Priority

Configuration is loaded in the following order (later sources override earlier ones):
1. Default values
2. Configuration file (searched in current directory, then home directory)
3. Environment variables (if enabled)

## Configuration Sections

### Analysis Settings

Controls how IntelliRefactor analyzes your project.

#### `max_file_size` (integer, default: 1048576)
Maximum file size in bytes to analyze. Files larger than this are skipped.
- **Environment variable**: `INTELLIREFACTOR_MAX_FILE_SIZE`
- **Example**: `2097152` (2MB)

#### `excluded_patterns` (list of strings)
File patterns to exclude from analysis.
- **Environment variable**: `INTELLIREFACTOR_EXCLUDED_PATTERNS` (comma-separated)
- **Default**: `["*.pyc", "__pycache__", ".git", ".venv", "venv", "node_modules"]`

#### `metrics_thresholds` (object)
Thresholds for code quality metrics:
- `cyclomatic_complexity` (float, default: 10.0): Maximum acceptable cyclomatic complexity
- `maintainability_index` (float, default: 20.0): Minimum maintainability index
- `lines_of_code` (integer, default: 500): Maximum lines of code per file

#### `analysis_depth` (integer, default: 10)
Maximum depth for recursive analysis.
- **Environment variable**: `INTELLIREFACTOR_ANALYSIS_DEPTH`

#### `include_patterns` (list of strings, default: ["**/*.py"])
File patterns to include in analysis.

#### `exclude_patterns` (list of strings)
Additional patterns to exclude from analysis.

#### `large_file_threshold` (integer, default: 500)
Threshold in lines of code to consider a file "large".

#### `complexity_threshold` (float, default: 15.0)
Cyclomatic complexity threshold for refactoring candidates.

#### `responsibilities_threshold` (integer, default: 5)
Maximum number of responsibilities before suggesting refactoring.

#### `god_object_threshold` (integer, default: 15)
Maximum number of methods in a class before flagging as "god object".

#### `min_candidate_size` (integer, default: 100)
Minimum lines of code for a refactoring candidate.

#### `max_candidates` (integer, default: 10)
Maximum number of refactoring candidates to return.

### Refactoring Settings

Controls refactoring behavior and safety.

#### `safety_level` (string, default: "moderate")
Safety level for refactoring operations:
- `"conservative"`: Maximum safety, minimal changes
- `"moderate"`: Balanced approach
- `"aggressive"`: More extensive refactoring
- **Environment variable**: `INTELLIREFACTOR_SAFETY_LEVEL`

#### `auto_apply` (boolean, default: false)
Whether to automatically apply refactoring suggestions.
- **Environment variable**: `INTELLIREFACTOR_AUTO_APPLY`

#### `backup_enabled` (boolean, default: true)
Whether to create backups before refactoring.
- **Environment variable**: `INTELLIREFACTOR_BACKUP_ENABLED`

#### `validation_required` (boolean, default: true)
Whether to validate refactoring results.
- **Environment variable**: `INTELLIREFACTOR_VALIDATION_REQUIRED`

#### `max_operations_per_session` (integer, default: 50)
Maximum number of refactoring operations per session.

#### `stop_on_failure` (boolean, default: true)
Whether to stop refactoring on first failure.

### Knowledge Settings

Controls the knowledge base and learning system.

#### `knowledge_base_path` (string, default: "knowledge")
Path to the knowledge base directory.
- **Environment variable**: `INTELLIREFACTOR_KNOWLEDGE_PATH`

#### `auto_learn` (boolean, default: true)
Whether to automatically learn from refactoring results.
- **Environment variable**: `INTELLIREFACTOR_AUTO_LEARN`

#### `confidence_threshold` (float, default: 0.7)
Minimum confidence threshold for applying learned knowledge.
- **Environment variable**: `INTELLIREFACTOR_CONFIDENCE_THRESHOLD`
- **Range**: 0.0 to 1.0

#### `max_knowledge_items` (integer, default: 10000)
Maximum number of knowledge items to store.

### Plugin Settings

Controls plugin loading and management.

#### `plugin_directories` (list of strings, default: ["plugins"])
Directories to search for plugins.
- **Environment variable**: `INTELLIREFACTOR_PLUGIN_DIRECTORIES` (comma-separated)

#### `auto_discover` (boolean, default: true)
Whether to automatically discover plugins.
- **Environment variable**: `INTELLIREFACTOR_AUTO_DISCOVER`

#### `enabled_plugins` (list of strings, default: [])
List of explicitly enabled plugins.
- **Environment variable**: `INTELLIREFACTOR_ENABLED_PLUGINS` (comma-separated)

## Project Type Templates

IntelliRefactor provides pre-configured templates for different project types:

### `config_template.json/yaml`
Default configuration suitable for most Python projects.

### `web_application.json/yaml`
Optimized for web applications (Django, Flask, FastAPI):
- Lower complexity thresholds
- Excludes static files, migrations, media
- Conservative safety level
- Web-specific plugins enabled

### `data_science.json/yaml`
Optimized for data science projects:
- Higher file size limits
- Excludes data files, models, notebooks checkpoints
- Includes Jupyter notebook analysis
- Data science specific plugins

### `library_package.json/yaml`
Optimized for Python libraries and packages:
- Strict quality thresholds
- Excludes documentation builds
- Conservative refactoring approach
- Package-specific analysis

### `enterprise_application.json/yaml`
Optimized for large enterprise applications:
- Very strict quality thresholds
- Conservative safety settings
- Extensive exclusion patterns
- Enterprise-specific plugins

### `microservices.json/yaml`
Optimized for microservices architecture:
- Small file thresholds
- Excludes container/deployment files
- Service-oriented analysis
- Microservices-specific plugins

## Usage Examples

### Basic Configuration File

```json
{
  "analysis": {
    "max_file_size": 1048576,
    "complexity_threshold": 12.0
  },
  "refactoring": {
    "safety_level": "conservative",
    "auto_apply": false
  }
}
```

### Environment Variables

```bash
export INTELLIREFACTOR_SAFETY_LEVEL=conservative
export INTELLIREFACTOR_AUTO_APPLY=false
export INTELLIREFACTOR_MAX_FILE_SIZE=2097152
```

### Loading Configuration in Code

```python
from intellirefactor.config import IntelliRefactorConfig

# Load with default search
config = IntelliRefactorConfig.load()

# Load specific file
config = IntelliRefactorConfig.load("my_config.yaml")

# Load without environment variables
config = IntelliRefactorConfig.load(use_env=False)

# Create from template
config = IntelliRefactorConfig.load("templates/web_application.json")
```

## Best Practices

1. **Start with a template**: Choose the template closest to your project type
2. **Adjust thresholds gradually**: Start conservative and adjust based on results
3. **Use environment variables for CI/CD**: Override settings for different environments
4. **Enable validation**: Always keep `validation_required: true` in production
5. **Regular backups**: Keep `backup_enabled: true` for safety
6. **Monitor knowledge base**: Regularly review learned patterns for accuracy

## Troubleshooting

### Configuration Not Found
- Check file names and locations
- Verify file format (valid JSON/YAML)
- Check file permissions

### Invalid Configuration Values
- Verify numeric ranges (e.g., confidence_threshold: 0.0-1.0)
- Check enum values (e.g., safety_level: conservative/moderate/aggressive)
- Validate file paths exist

### Environment Variables Not Working
- Ensure correct variable names (prefix: `INTELLIREFACTOR_`)
- Check boolean values are "true"/"false"
- Verify numeric values are valid

For more help, see the main documentation or create an issue on GitHub.
"""
Configuration system for IntelliRefactor

Provides configuration management with support for files and environment variables.
Includes validation, default value handling, and configuration merging.
"""

import os
import json
import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety levels for refactoring operations."""

    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""

    pass


class ConfigurationManager:
    """Manages configuration loading, validation, and merging."""

    DEFAULT_CONFIG_PATHS = [
        "intellirefactor.json",
        "intellirefactor.yaml",
        "intellirefactor.yml",
        ".intellirefactor.json",
        ".intellirefactor.yaml",
        ".intellirefactor.yml",
        os.path.expanduser("~/.intellirefactor.json"),
        os.path.expanduser("~/.intellirefactor.yaml"),
    ]

    @staticmethod
    def find_config_file(search_paths: Optional[List[str]] = None) -> Optional[str]:
        """Find the first existing configuration file."""
        paths = search_paths or ConfigurationManager.DEFAULT_CONFIG_PATHS

        for path in paths:
            if os.path.exists(path):
                return path
        return None

    @staticmethod
    def load_config_file(config_path: str) -> Dict[str, Any]:
        """Load configuration from file (JSON or YAML)."""
        if not os.path.exists(config_path):
            raise ConfigurationError(f"Configuration file not found: {config_path}")

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.endswith((".yaml", ".yml")):
                    return yaml.safe_load(f) or {}
                else:
                    return json.load(f)
        except (json.JSONDecodeError, yaml.YAMLError) as e:
            raise ConfigurationError(f"Invalid configuration file format: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error reading configuration file: {e}")

    @staticmethod
    def load_env_config() -> Dict[str, Any]:
        """Load configuration from environment variables."""
        config = {}

        # Analysis settings
        analysis = {}
        if os.getenv("INTELLIREFACTOR_MAX_FILE_SIZE"):
            try:
                analysis["max_file_size"] = int(os.getenv("INTELLIREFACTOR_MAX_FILE_SIZE"))
            except ValueError:
                logger.warning("Invalid INTELLIREFACTOR_MAX_FILE_SIZE value, using default")

        if os.getenv("INTELLIREFACTOR_ANALYSIS_DEPTH"):
            try:
                analysis["analysis_depth"] = int(os.getenv("INTELLIREFACTOR_ANALYSIS_DEPTH"))
            except ValueError:
                logger.warning("Invalid INTELLIREFACTOR_ANALYSIS_DEPTH value, using default")

        if os.getenv("INTELLIREFACTOR_EXCLUDED_PATTERNS"):
            analysis["excluded_patterns"] = os.getenv("INTELLIREFACTOR_EXCLUDED_PATTERNS").split(
                ","
            )

        if analysis:
            config["analysis"] = analysis

        # Refactoring settings
        refactoring = {}
        if os.getenv("INTELLIREFACTOR_SAFETY_LEVEL"):
            safety_level = os.getenv("INTELLIREFACTOR_SAFETY_LEVEL").lower()
            if safety_level in ["conservative", "moderate", "aggressive"]:
                refactoring["safety_level"] = safety_level
            else:
                logger.warning("Invalid INTELLIREFACTOR_SAFETY_LEVEL value, using default")

        if os.getenv("INTELLIREFACTOR_AUTO_APPLY"):
            refactoring["auto_apply"] = os.getenv("INTELLIREFACTOR_AUTO_APPLY").lower() == "true"

        if os.getenv("INTELLIREFACTOR_BACKUP_ENABLED"):
            refactoring["backup_enabled"] = (
                os.getenv("INTELLIREFACTOR_BACKUP_ENABLED").lower() == "true"
            )

        if os.getenv("INTELLIREFACTOR_VALIDATION_REQUIRED"):
            refactoring["validation_required"] = (
                os.getenv("INTELLIREFACTOR_VALIDATION_REQUIRED").lower() == "true"
            )

        if refactoring:
            config["refactoring"] = refactoring

        # Knowledge settings
        knowledge = {}
        if os.getenv("INTELLIREFACTOR_KNOWLEDGE_PATH"):
            knowledge["knowledge_base_path"] = os.getenv("INTELLIREFACTOR_KNOWLEDGE_PATH")

        if os.getenv("INTELLIREFACTOR_AUTO_LEARN"):
            knowledge["auto_learn"] = os.getenv("INTELLIREFACTOR_AUTO_LEARN").lower() == "true"

        if os.getenv("INTELLIREFACTOR_CONFIDENCE_THRESHOLD"):
            try:
                knowledge["confidence_threshold"] = float(
                    os.getenv("INTELLIREFACTOR_CONFIDENCE_THRESHOLD")
                )
            except ValueError:
                logger.warning("Invalid INTELLIREFACTOR_CONFIDENCE_THRESHOLD value, using default")

        if knowledge:
            config["knowledge"] = knowledge

        # Plugin settings
        plugins = {}
        if os.getenv("INTELLIREFACTOR_PLUGIN_DIRECTORIES"):
            plugins["plugin_directories"] = os.getenv("INTELLIREFACTOR_PLUGIN_DIRECTORIES").split(
                ","
            )

        if os.getenv("INTELLIREFACTOR_AUTO_DISCOVER"):
            plugins["auto_discover"] = os.getenv("INTELLIREFACTOR_AUTO_DISCOVER").lower() == "true"

        if os.getenv("INTELLIREFACTOR_ENABLED_PLUGINS"):
            plugins["enabled_plugins"] = os.getenv("INTELLIREFACTOR_ENABLED_PLUGINS").split(",")

        if plugins:
            config["plugins"] = plugins

        return config

    @staticmethod
    def merge_configs(*configs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge multiple configuration dictionaries, with later ones taking precedence."""
        result = {}

        for config in configs:
            if not config:
                continue

            for key, value in config.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = ConfigurationManager.merge_configs(result[key], value)
                else:
                    result[key] = value

        return result

    @staticmethod
    def validate_config(config_data: Dict[str, Any]) -> None:
        """Validate configuration data."""
        # Validate analysis settings
        if "analysis" in config_data:
            analysis = config_data["analysis"]

            if "max_file_size" in analysis and analysis["max_file_size"] <= 0:
                raise ConfigurationError("max_file_size must be positive")

            if "analysis_depth" in analysis and analysis["analysis_depth"] <= 0:
                raise ConfigurationError("analysis_depth must be positive")

            if "metrics_thresholds" in analysis:
                thresholds = analysis["metrics_thresholds"]
                for key, value in thresholds.items():
                    if not isinstance(value, (int, float)) or value < 0:
                        raise ConfigurationError(
                            f"metrics_thresholds.{key} must be a non-negative number"
                        )

        # Validate refactoring settings
        if "refactoring" in config_data:
            refactoring = config_data["refactoring"]

            if "safety_level" in refactoring:
                valid_levels = ["conservative", "moderate", "aggressive"]
                if refactoring["safety_level"] not in valid_levels:
                    raise ConfigurationError(f"safety_level must be one of: {valid_levels}")

            if (
                "max_operations_per_session" in refactoring
                and refactoring["max_operations_per_session"] <= 0
            ):
                raise ConfigurationError("max_operations_per_session must be positive")

        # Validate knowledge settings
        if "knowledge" in config_data:
            knowledge = config_data["knowledge"]

            if "confidence_threshold" in knowledge:
                threshold = knowledge["confidence_threshold"]
                if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
                    raise ConfigurationError("confidence_threshold must be between 0 and 1")

            if "max_knowledge_items" in knowledge and knowledge["max_knowledge_items"] <= 0:
                raise ConfigurationError("max_knowledge_items must be positive")


@dataclass
class AnalysisConfig:
    """Configuration for analysis operations."""

    max_file_size: int = 1024 * 1024  # 1MB
    excluded_patterns: List[str] = field(
        default_factory=lambda: [
            "*.pyc",
            "__pycache__",
            ".git",
            ".venv",
            "venv",
            "node_modules",
        ]
    )
    metrics_thresholds: Dict[str, float] = field(
        default_factory=lambda: {
            "cyclomatic_complexity": 10.0,
            "maintainability_index": 20.0,
            "lines_of_code": 500,
        }
    )
    analysis_depth: int = 10

    # Project analyzer specific settings
    include_patterns: List[str] = field(default_factory=lambda: ["**/*.py"])
    exclude_patterns: List[str] = field(
        default_factory=lambda: [
            "**/test_*.py",
            "**/*_test.py",
            "**/__pycache__/**",
            "**/.*",
            "**/build/**",
            "**/dist/**",
        ]
    )
    large_file_threshold: int = 500  # Lines of code
    complexity_threshold: float = 15.0  # Cyclomatic complexity
    responsibilities_threshold: int = 5  # Number of responsibilities
    god_object_threshold: int = 15  # Number of methods in a class
    min_candidate_size: int = 100  # Minimum lines for refactoring candidate
    max_candidates: int = 10  # Maximum number of candidates to return


@dataclass
class RefactoringConfig:
    """Configuration for refactoring operations."""

    safety_level: SafetyLevel = SafetyLevel.MODERATE
    auto_apply: bool = False
    backup_enabled: bool = True
    validation_required: bool = True
    max_operations_per_session: int = 50
    stop_on_failure: bool = True


@dataclass
class KnowledgeConfig:
    """Configuration for knowledge management."""

    knowledge_base_path: str = "knowledge"
    auto_learn: bool = True
    confidence_threshold: float = 0.7
    max_knowledge_items: int = 10000


@dataclass
class PluginConfig:
    """Configuration for plugin system."""

    plugin_directories: List[str] = field(default_factory=lambda: ["plugins"])
    auto_discover: bool = True
    enabled_plugins: List[str] = field(default_factory=list)


@dataclass
class IntelliRefactorConfig:
    """Main configuration class for IntelliRefactor."""

    analysis_settings: AnalysisConfig = field(default_factory=AnalysisConfig)
    refactoring_settings: RefactoringConfig = field(default_factory=RefactoringConfig)
    knowledge_settings: KnowledgeConfig = field(default_factory=KnowledgeConfig)
    plugin_settings: PluginConfig = field(default_factory=PluginConfig)

    @classmethod
    def default(cls) -> "IntelliRefactorConfig":
        """Create default configuration."""
        return cls()

    @classmethod
    def load(
        cls,
        config_path: Optional[str] = None,
        use_env: bool = True,
        validate: bool = True,
    ) -> "IntelliRefactorConfig":
        """
        Load configuration from multiple sources with precedence:
        1. Default values
        2. Configuration file
        3. Environment variables (if use_env=True)

        Args:
            config_path: Path to configuration file. If None, searches for default files.
            use_env: Whether to load environment variables
            validate: Whether to validate the configuration
        """
        # Start with empty config for merging
        configs_to_merge = []

        # Load from file
        file_config = {}
        if config_path:
            if not os.path.exists(config_path):
                raise ConfigurationError(f"Configuration file not found: {config_path}")
            file_config = ConfigurationManager.load_config_file(config_path)
        else:
            # Search for default config files
            found_config = ConfigurationManager.find_config_file()
            if found_config:
                file_config = ConfigurationManager.load_config_file(found_config)
                logger.info(f"Loaded configuration from: {found_config}")

        configs_to_merge.append(file_config)

        # Load from environment variables
        if use_env:
            env_config = ConfigurationManager.load_env_config()
            configs_to_merge.append(env_config)

        # Merge all configurations
        merged_config = ConfigurationManager.merge_configs(*configs_to_merge)

        # Validate merged configuration
        if validate:
            ConfigurationManager.validate_config(merged_config)

        # Create configuration objects
        analysis_config = AnalysisConfig()
        if "analysis" in merged_config:
            analysis_data = merged_config["analysis"]
            for key, value in analysis_data.items():
                if hasattr(analysis_config, key):
                    setattr(analysis_config, key, value)

        refactoring_config = RefactoringConfig()
        if "refactoring" in merged_config:
            refactoring_data = merged_config["refactoring"]
            for key, value in refactoring_data.items():
                if hasattr(refactoring_config, key):
                    if key == "safety_level" and isinstance(value, str):
                        setattr(refactoring_config, key, SafetyLevel(value))
                    else:
                        setattr(refactoring_config, key, value)

        knowledge_config = KnowledgeConfig()
        if "knowledge" in merged_config:
            knowledge_data = merged_config["knowledge"]
            for key, value in knowledge_data.items():
                if hasattr(knowledge_config, key):
                    setattr(knowledge_config, key, value)

        plugin_config = PluginConfig()
        if "plugins" in merged_config:
            plugin_data = merged_config["plugins"]
            for key, value in plugin_data.items():
                if hasattr(plugin_config, key):
                    setattr(plugin_config, key, value)

        return cls(
            analysis_settings=analysis_config,
            refactoring_settings=refactoring_config,
            knowledge_settings=knowledge_config,
            plugin_settings=plugin_config,
        )

    @classmethod
    def from_file(cls, config_path: str) -> "IntelliRefactorConfig":
        """Load configuration from JSON file."""
        return cls.load(config_path=config_path, use_env=False)

    @classmethod
    def from_env(cls) -> "IntelliRefactorConfig":
        """Load configuration from environment variables."""
        return cls.load(config_path=None, use_env=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "analysis": asdict(self.analysis_settings),
            "refactoring": {
                **asdict(self.refactoring_settings),
                "safety_level": self.refactoring_settings.safety_level.value,
            },
            "knowledge": asdict(self.knowledge_settings),
            "plugins": asdict(self.plugin_settings),
        }

    def to_file(self, config_path: str, format: str = "json") -> None:
        """
        Save configuration to file.

        Args:
            config_path: Path to save configuration
            format: File format ('json' or 'yaml')
        """
        config_data = self.to_dict()

        try:
            with open(config_path, "w", encoding="utf-8") as f:
                if format.lower() == "yaml":
                    yaml.dump(config_data, f, default_flow_style=False, indent=2)
                else:
                    json.dump(config_data, f, indent=2)
        except Exception as e:
            raise ConfigurationError(f"Error saving configuration file: {e}")

    def validate(self) -> None:
        """Validate the current configuration."""
        ConfigurationManager.validate_config(self.to_dict())

    def get_config_summary(self) -> str:
        """Get a human-readable summary of the configuration."""
        return f"""IntelliRefactor Configuration Summary:
Analysis:
  - Max file size: {self.analysis_settings.max_file_size} bytes
  - Analysis depth: {self.analysis_settings.analysis_depth}
  - Excluded patterns: {len(self.analysis_settings.excluded_patterns)} patterns
  - Complexity threshold: {self.analysis_settings.complexity_threshold}

Refactoring:
  - Safety level: {self.refactoring_settings.safety_level.value}
  - Auto apply: {self.refactoring_settings.auto_apply}
  - Backup enabled: {self.refactoring_settings.backup_enabled}
  - Validation required: {self.refactoring_settings.validation_required}

Knowledge:
  - Knowledge base path: {self.knowledge_settings.knowledge_base_path}
  - Auto learn: {self.knowledge_settings.auto_learn}
  - Confidence threshold: {self.knowledge_settings.confidence_threshold}

Plugins:
  - Plugin directories: {self.plugin_settings.plugin_directories}
  - Auto discover: {self.plugin_settings.auto_discover}
  - Enabled plugins: {len(self.plugin_settings.enabled_plugins)} plugins
"""


# Convenience function for backward compatibility
def load_config(config_path: Optional[str] = None, use_env: bool = True) -> IntelliRefactorConfig:
    """
    Load configuration from file and/or environment variables.

    Args:
        config_path: Path to configuration file
        use_env: Whether to load environment variables

    Returns:
        IntelliRefactorConfig: Loaded configuration
    """
    return IntelliRefactorConfig.load(config_path=config_path, use_env=use_env)

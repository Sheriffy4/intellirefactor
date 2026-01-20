"""
Configuration management for AutoRefactor.

This module provides the RefactorConfig dataclass that centralizes all configuration
settings for the AutoRefactor system. It handles:

- Configuration parsing from dictionaries or objects
- Validation of configuration values
- Python version requirement checking
- Codebase analysis for inferring conventions
- Default value management

The RefactorConfig class replaces scattered configuration logic that was previously
embedded throughout the AutoRefactor class, improving maintainability and testability.

Example:
    >>> config = RefactorConfig.from_dict({
    ...     'god_class_threshold': 15,
    ...     'auto_apply': False,
    ...     'backup_enabled': True
    ... })
    >>> print(config.god_class_threshold)
    15

Classes:
    RefactorConfig: Main configuration dataclass with validation and analysis
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Set

logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from ..knowledge.import_fixing_patterns import CodebaseAnalysisPatterns

    IMPORT_FIXING_AVAILABLE = True
except ImportError:
    IMPORT_FIXING_AVAILABLE = False
    logger.debug("Import fixing patterns not available")


# Default constants
DEFAULT_COHESION_STOP_FEATURES: FrozenSet[str] = frozenset(
    {
        "config",
        "logger",
        "settings",
        "options",
        "args",
        "kwargs",
        "env",
        "context",
        "state",
        "data",
    }
)

DEFAULT_RESPONSIBILITY_KEYWORDS: Dict[str, List[str]] = {
    "console": [
        "print",
        "log",
        "display",
        "show",
        "console",
        "output",
        "render",
        "table",
        "progress",
    ],
    "validation": ["validate", "verify", "check", "ensure", "is_valid", "normalize"],
    "analysis": ["analyze", "parse", "examine", "inspect", "scan", "stats", "metric"],
    "export": ["export", "dump", "serialize", "write", "save", "json", "csv"],
    "storage": ["load", "read", "file", "path", "store", "cache", "persist", "db"],
    "network": [
        "request",
        "connect",
        "http",
        "api",
        "fetch",
        "download",
        "send",
        "receive",
    ],
    "config": ["config", "setting", "option", "setup", "init", "env"],
    "utility": ["util", "helper", "format", "convert", "transform", "build"],
}


@dataclass
class RefactorConfig:
    """
    Configuration dataclass for AutoRefactor operations.

    This class centralizes all configuration settings for the refactoring system,
    providing validation, defaults, and codebase analysis capabilities.

    Attributes:
        safety_level: Safety level for operations ('strict', 'moderate', 'permissive')
        auto_apply: Whether to automatically apply refactorings without confirmation
        backup_enabled: Whether to create backups before modifications
        validation_required: Whether to validate generated code
        require_python39: Whether to enforce Python 3.9+ requirement

        output_directory: Directory name for generated components
        component_template: Template name for component classes (e.g., 'Service')
        interface_prefix: Prefix for interface class names (e.g., 'I')
        preserve_original: Whether to keep original files
        facade_suffix: Suffix for facade files (e.g., '_refactored')

        reserved_prefix: Prefix for internal attributes (e.g., '__ar_')

        extract_decorated_public_methods: Whether to extract decorated methods
        extract_private_methods: Whether to extract private methods
        keep_private_methods_in_facade: Whether to keep private methods in facade

        skip_methods_with_module_level_deps: Skip methods with module dependencies
        skip_methods_with_bare_self_usage: Skip methods with bare self usage
        skip_methods_with_dangerous_patterns: Skip methods with dangerous patterns

        responsibility_keywords: Keywords for grouping methods by responsibility
        cohesion_cluster_other: Whether to cluster ungrouped methods by cohesion
        cohesion_similarity_threshold: Threshold for cohesion clustering (0.0-1.0)
        cohesion_stop_features: Stop words for cohesion analysis

        god_class_threshold: Minimum methods to consider a class a god object
        min_methods_for_extraction: Minimum methods required for extraction

        effort_per_component: Estimated effort per component (hours)
        base_effort: Base effort for refactoring (hours)

        disable_contextual_analysis: Whether to disable contextual analysis
        analysis_results_dir: Directory for analysis results

        codebase_analysis: Results of codebase convention analysis

    Example:
        >>> config = RefactorConfig(
        ...     god_class_threshold=15,
        ...     auto_apply=False,
        ...     backup_enabled=True
        ... )
        >>> config.container_attr
        '__ar_container'
    """

    # Safety settings
    safety_level: str = "moderate"
    auto_apply: bool = False
    backup_enabled: bool = True
    validation_required: bool = True
    require_python39: bool = True

    # Output settings
    output_directory: str = "components"
    component_template: str = "Service"
    interface_prefix: str = "I"
    preserve_original: bool = True
    facade_suffix: str = "_refactored"

    # Internal naming
    reserved_prefix: str = "__ar_"

    # Extraction policy
    extract_decorated_public_methods: bool = False
    extract_private_methods: bool = True
    keep_private_methods_in_facade: bool = True

    # Skip patterns
    skip_methods_with_module_level_deps: bool = True
    skip_methods_with_bare_self_usage: bool = True
    skip_methods_with_dangerous_patterns: bool = True

    # Grouping settings
    responsibility_keywords: Dict[str, List[str]] = field(
        default_factory=lambda: DEFAULT_RESPONSIBILITY_KEYWORDS.copy()
    )
    cohesion_cluster_other: bool = True
    cohesion_similarity_threshold: float = 0.30
    cohesion_stop_features: FrozenSet[str] = field(
        default_factory=lambda: DEFAULT_COHESION_STOP_FEATURES
    )

    # Thresholds
    god_class_threshold: int = 10
    min_methods_for_extraction: int = 1

    # Effort estimation
    effort_per_component: float = 2.5
    base_effort: float = 4.0

    # Optional integrations
    disable_contextual_analysis: bool = False
    analysis_results_dir: Optional[Path] = None

    # Codebase analysis
    codebase_analysis: Optional[Dict[str, Any]] = None

    @property
    def container_attr(self) -> str:
        """Get the container attribute name."""
        return f"{self.reserved_prefix}container"

    @property
    def components_attr(self) -> str:
        """Get the components attribute name."""
        return f"{self.reserved_prefix}components"

    @classmethod
    def from_dict(cls, config_dict: Optional[Any] = None) -> RefactorConfig:
        """
        Create RefactorConfig from a dictionary or config object.

        Args:
            config_dict: Dictionary or object with configuration attributes

        Returns:
            RefactorConfig instance

        Raises:
            RuntimeError: If Python version requirement not met
        """
        if config_dict is None:
            config_dict = {}

        # Handle object with __dict__ attribute
        if hasattr(config_dict, "__dict__") and not isinstance(config_dict, dict):
            config_dict = {
                "safety_level": getattr(config_dict, "safety_level", "moderate"),
                "auto_apply": getattr(config_dict, "auto_apply", False),
                "backup_enabled": getattr(config_dict, "backup_enabled", True),
                "validation_required": getattr(
                    config_dict, "validation_required", True
                ),
            }
        elif not isinstance(config_dict, dict):
            config_dict = {}

        # Validate Python version
        require_39 = bool(config_dict.get("require_python39", True))
        if require_39 and sys.version_info < (3, 9):
            raise RuntimeError("AutoRefactor requires Python 3.9+ (uses ast.unparse).")

        # Extract and validate configuration values
        config = cls(
            safety_level=config_dict.get("safety_level", "moderate"),
            auto_apply=bool(config_dict.get("auto_apply", False)),
            backup_enabled=bool(config_dict.get("backup_enabled", True)),
            validation_required=bool(config_dict.get("validation_required", True)),
            require_python39=require_39,
            output_directory=config_dict.get("output_directory", "components"),
            component_template=config_dict.get("component_template", "Service"),
            interface_prefix=config_dict.get("interface_prefix", "I"),
            preserve_original=bool(config_dict.get("preserve_original", True)),
            facade_suffix=config_dict.get("facade_suffix", "_refactored"),
            reserved_prefix=config_dict.get("reserved_prefix", "__ar_"),
            extract_decorated_public_methods=bool(
                config_dict.get("extract_decorated_public_methods", False)
            ),
            extract_private_methods=bool(
                config_dict.get("extract_private_methods", True)
            ),
            keep_private_methods_in_facade=bool(
                config_dict.get("keep_private_methods_in_facade", True)
            ),
            skip_methods_with_module_level_deps=bool(
                config_dict.get("skip_methods_with_module_level_deps", True)
            ),
            skip_methods_with_bare_self_usage=bool(
                config_dict.get("skip_methods_with_bare_self_usage", True)
            ),
            skip_methods_with_dangerous_patterns=bool(
                config_dict.get("skip_methods_with_dangerous_patterns", True)
            ),
            responsibility_keywords=config_dict.get(
                "responsibility_keywords", DEFAULT_RESPONSIBILITY_KEYWORDS.copy()
            ),
            cohesion_cluster_other=bool(
                config_dict.get("cohesion_cluster_other", True)
            ),
            cohesion_similarity_threshold=float(
                config_dict.get("cohesion_similarity_threshold", 0.30)
            ),
            cohesion_stop_features=frozenset(
                config_dict.get(
                    "cohesion_stop_features", list(DEFAULT_COHESION_STOP_FEATURES)
                )
            ),
            god_class_threshold=cls._validate_positive_int(
                config_dict.get("god_class_threshold", 10),
                "god_class_threshold",
                10,
            ),
            min_methods_for_extraction=cls._validate_positive_int(
                config_dict.get("min_methods_for_extraction", 1),
                "min_methods_for_extraction",
                1,
            ),
            effort_per_component=float(config_dict.get("effort_per_component", 2.5)),
            base_effort=float(config_dict.get("base_effort", 4.0)),
            disable_contextual_analysis=bool(
                config_dict.get("disable_contextual_analysis", False)
            ),
            analysis_results_dir=(
                Path(config_dict["analysis_results_dir"])
                if config_dict.get("analysis_results_dir")
                else None
            ),
        )

        # Perform codebase analysis if available
        if IMPORT_FIXING_AVAILABLE:
            config.codebase_analysis = cls._analyze_codebase_standards()

        return config

    @staticmethod
    def _validate_positive_int(value: Any, name: str, default: int) -> int:
        """
        Validate and convert value to positive integer.

        Args:
            value: Value to validate
            name: Parameter name for logging
            default: Default value if validation fails

        Returns:
            Validated positive integer
        """
        try:
            v = int(value)
            return v if v > 0 else default
        except (TypeError, ValueError):
            logger.debug(
                "Invalid int for %s=%r, using default=%d", name, value, default
            )
            return default

    @staticmethod
    def _analyze_codebase_standards() -> Optional[Dict[str, Any]]:
        """
        Analyze codebase to infer logging/import conventions.

        Returns:
            Dictionary with analysis results or None if analysis fails
        """
        try:
            current_dir = Path.cwd()
            python_files = list(current_dir.rglob("*.py"))[:50]

            if not python_files:
                logger.warning("No Python files found for codebase analysis")
                return None

            file_contents: List[str] = []
            for py_file in python_files:
                try:
                    file_contents.append(py_file.read_text(encoding="utf-8-sig"))
                except Exception as e:
                    logger.debug("Failed to read %s: %s", py_file, e)

            if not file_contents:
                logger.warning("No readable Python files found for analysis")
                return None

            logging_standard = CodebaseAnalysisPatterns.recommend_logging_standard(
                file_contents
            )
            existing_modules = CodebaseAnalysisPatterns.find_existing_modules(
                file_contents
            )

            analysis_result = {
                "logging_standard": logging_standard,
                "existing_modules": set(existing_modules),
                "analyzed_files": len(file_contents),
            }

            logger.info(
                "Analyzed %d files, recommended logging standard: %s",
                len(file_contents),
                logging_standard,
            )

            return analysis_result

        except Exception as e:
            logger.warning("Failed to analyze codebase standards: %s", e)
            return None

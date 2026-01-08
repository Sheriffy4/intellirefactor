"""
Configuration template generator for IntelliRefactor.

Provides utilities to generate and customize configuration templates
for different project types.
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


class TemplateGenerator:
    """Generates configuration templates for different project types."""

    TEMPLATE_DIR = Path(__file__).parent

    AVAILABLE_TEMPLATES = {
        "default": "config_template",
        "web": "web_application",
        "data_science": "data_science",
        "library": "library_package",
        "enterprise": "enterprise_application",
        "microservices": "microservices",
    }

    @classmethod
    def list_templates(cls) -> Dict[str, str]:
        """List all available templates."""
        return cls.AVAILABLE_TEMPLATES.copy()

    @classmethod
    def get_template_path(cls, template_name: str, format: str = "json") -> Path:
        """Get the path to a template file."""
        if template_name not in cls.AVAILABLE_TEMPLATES:
            raise ValueError(
                f"Unknown template: {template_name}. Available: {list(cls.AVAILABLE_TEMPLATES.keys())}"
            )

        template_file = cls.AVAILABLE_TEMPLATES[template_name]
        return cls.TEMPLATE_DIR / f"{template_file}.{format}"

    @classmethod
    def load_template(cls, template_name: str, format: str = "json") -> Dict[str, Any]:
        """Load a template configuration."""
        template_path = cls.get_template_path(template_name, format)

        if not template_path.exists():
            raise FileNotFoundError(f"Template file not found: {template_path}")

        with open(template_path, "r", encoding="utf-8") as f:
            if format.lower() in ["yaml", "yml"]:
                return yaml.safe_load(f)
            else:
                return json.load(f)

    @classmethod
    def generate_config(
        cls,
        template_name: str,
        output_path: str,
        format: str = "json",
        customizations: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Generate a configuration file from a template.

        Args:
            template_name: Name of the template to use
            output_path: Path where to save the configuration
            format: Output format ('json' or 'yaml')
            customizations: Additional customizations to apply
        """
        # Load base template
        template_data = cls.load_template(template_name, format)

        # Apply customizations
        if customizations:
            template_data = cls._merge_configs(template_data, customizations)

        # Remove description field if present
        template_data.pop("_description", None)

        # Save to file
        with open(output_path, "w", encoding="utf-8") as f:
            if format.lower() in ["yaml", "yml"]:
                yaml.dump(template_data, f, default_flow_style=False, indent=2)
            else:
                json.dump(template_data, f, indent=2)

    @classmethod
    def create_custom_template(
        cls,
        base_template: str,
        project_name: str,
        project_patterns: Optional[Dict[str, List[str]]] = None,
        quality_thresholds: Optional[Dict[str, float]] = None,
        safety_level: str = "moderate",
    ) -> Dict[str, Any]:
        """
        Create a custom template based on project characteristics.

        Args:
            base_template: Base template to start from
            project_name: Name of the project (for documentation)
            project_patterns: Custom include/exclude patterns
            quality_thresholds: Custom quality thresholds
            safety_level: Refactoring safety level
        """
        # Load base template
        template_data = cls.load_template(base_template)

        # Add project-specific description
        template_data["_description"] = f"Custom configuration for {project_name} project"

        # Apply custom patterns
        if project_patterns:
            if "include" in project_patterns:
                template_data["analysis"]["include_patterns"] = project_patterns["include"]
            if "exclude" in project_patterns:
                template_data["analysis"]["exclude_patterns"] = project_patterns["exclude"]
            if "excluded" in project_patterns:
                template_data["analysis"]["excluded_patterns"].extend(project_patterns["excluded"])

        # Apply custom quality thresholds
        if quality_thresholds:
            template_data["analysis"]["metrics_thresholds"].update(quality_thresholds)

            # Update related thresholds
            if "cyclomatic_complexity" in quality_thresholds:
                template_data["analysis"]["complexity_threshold"] = (
                    quality_thresholds["cyclomatic_complexity"] * 1.5
                )

        # Set safety level
        template_data["refactoring"]["safety_level"] = safety_level

        return template_data

    @classmethod
    def get_template_description(cls, template_name: str) -> str:
        """Get the description of a template."""
        try:
            template_data = cls.load_template(template_name)
            return template_data.get("_description", f"Configuration template: {template_name}")
        except (FileNotFoundError, KeyError):
            return f"Configuration template: {template_name}"

    @classmethod
    def validate_template(cls, template_data: Dict[str, Any]) -> List[str]:
        """
        Validate a template configuration.

        Returns:
            List of validation errors (empty if valid)
        """
        errors = []

        # Check required sections
        required_sections = ["analysis", "refactoring", "knowledge", "plugins"]
        for section in required_sections:
            if section not in template_data:
                errors.append(f"Missing required section: {section}")

        # Validate analysis section
        if "analysis" in template_data:
            analysis = template_data["analysis"]

            if "max_file_size" in analysis and analysis["max_file_size"] <= 0:
                errors.append("analysis.max_file_size must be positive")

            if "analysis_depth" in analysis and analysis["analysis_depth"] <= 0:
                errors.append("analysis.analysis_depth must be positive")

        # Validate refactoring section
        if "refactoring" in template_data:
            refactoring = template_data["refactoring"]

            if "safety_level" in refactoring:
                valid_levels = ["conservative", "moderate", "aggressive"]
                if refactoring["safety_level"] not in valid_levels:
                    errors.append(f"refactoring.safety_level must be one of: {valid_levels}")

        return errors

    @staticmethod
    def _merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configuration dictionaries."""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = TemplateGenerator._merge_configs(result[key], value)
            else:
                result[key] = value

        return result


def main():
    """CLI interface for template generation."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate IntelliRefactor configuration templates")
    parser.add_argument(
        "template",
        choices=list(TemplateGenerator.AVAILABLE_TEMPLATES.keys()),
        help="Template type to generate",
    )
    parser.add_argument("output", help="Output file path")
    parser.add_argument("--format", choices=["json", "yaml"], default="json", help="Output format")
    parser.add_argument("--list-templates", action="store_true", help="List available templates")

    args = parser.parse_args()

    if args.list_templates:
        print("Available templates:")
        for name, description in TemplateGenerator.AVAILABLE_TEMPLATES.items():
            desc = TemplateGenerator.get_template_description(name)
            print(f"  {name}: {desc}")
        return

    try:
        TemplateGenerator.generate_config(args.template, args.output, args.format)
        print(f"Generated {args.template} template at {args.output}")
    except Exception as e:
        print(f"Error generating template: {e}")


if __name__ == "__main__":
    main()

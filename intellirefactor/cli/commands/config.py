"""
Configuration and template management commands for IntelliRefactor CLI.

This module contains command handlers for:
- Configuration management (show, init, validate)
- Template management (list, generate, show)
- System status reporting
"""

import json
import sys
from typing import Any

from intellirefactor.api import IntelliRefactor
from intellirefactor.config import IntelliRefactorConfig
from intellirefactor.templates import TemplateGenerator


def cmd_status(args, intellirefactor: IntelliRefactor) -> None:
    """Handle status command."""
    from intellirefactor.cli_entry import _is_machine_readable, _print_json_to_stdout, _maybe_print_output
    
    status = intellirefactor.get_system_status()

    if args.format == "json":
        # Normal JSON mode: use regular stdout (so piping/capture works)
        _maybe_print_output(args, json.dumps(status, indent=2, default=str))
        return

    if _is_machine_readable(args):
        _print_json_to_stdout(args, status)
        return
    else:
        print("IntelliRefactor System Status:")
        print(f"Initialized: {status['initialized']}")
        print(f"Safety Level: {status['configuration']['safety_level']}")
        print(f"Auto Apply: {status['configuration']['auto_apply']}")
        print(f"Knowledge Path: {status['configuration']['knowledge_path']}")

        print("\nComponent Status:")
        for name, component_status in status["components"].items():
            print(f"  {name}: {component_status}")


def cmd_config(args) -> None:
    """Handle config command."""
    if args.config_action == "show":
        config = IntelliRefactorConfig.from_env()
        print("Current IntelliRefactor Configuration:")
        print(config.get_config_summary())

    elif args.config_action == "init":
        if args.template:
            # Use template
            try:
                TemplateGenerator.generate_config(args.template, args.path, args.format)
                print(f"Configuration file created from template '{args.template}' at {args.path}")
            except Exception as e:
                print(
                    f"Error: Failed to create configuration from template: {e}",
                    file=sys.stderr,
                )
                sys.exit(1)
        else:
            # Use default configuration
            config = IntelliRefactorConfig.default()
            config.to_file(args.path, args.format)
            print(f"Default configuration file created at {args.path}")

        print("Edit the file to customize your IntelliRefactor settings.")

    elif args.config_action == "validate":
        try:
            config = IntelliRefactorConfig.load(args.config_file, validate=True)
            print(f"Configuration file {args.config_file} is valid")
        except Exception as e:
            print(f"Error: Configuration file is invalid: {e}", file=sys.stderr)
            sys.exit(1)


def cmd_template(args) -> None:
    """Handle template command."""
    if args.template_action == "list":
        templates = TemplateGenerator.list_templates()
        print("Available configuration templates:")
        print()
        for name, template_file in templates.items():
            description = TemplateGenerator.get_template_description(name)
            print(f"  {name}:")
            print(f"    {description}")
            print()

    elif args.template_action == "generate":
        try:
            customizations = {}
            if args.project_name:
                customizations["_description"] = f"Configuration for {args.project_name} project"
            if args.safety_level != "moderate":
                customizations["refactoring"] = {"safety_level": args.safety_level}

            TemplateGenerator.generate_config(
                args.template_name,
                args.output_path,
                args.format,
                customizations if customizations else None,
            )
            print(f"Generated {args.template_name} template at {args.output_path}")

            if args.project_name:
                print(f"Customized for project: {args.project_name}")
            if args.safety_level != "moderate":
                print(f"Safety level set to: {args.safety_level}")

        except Exception as e:
            print(f"Error: Failed to generate template: {e}", file=sys.stderr)
            sys.exit(1)

    elif args.template_action == "show":
        try:
            template_data = TemplateGenerator.load_template(args.template_name, args.format)

            if args.format.lower() in ["yaml", "yml"]:
                import yaml
                print(yaml.dump(template_data, default_flow_style=False, indent=2))
            else:
                print(json.dumps(template_data, indent=2))

        except Exception as e:
            print(f"Error: Failed to show template: {e}", file=sys.stderr)
            sys.exit(1)

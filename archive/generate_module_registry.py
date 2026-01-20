#!/usr/bin/env python3
"""
Generate MODULE_REGISTRY.md documentation for the project.

This script scans all Python modules in the project, categorizes them by functionality,
and generates comprehensive documentation in MODULE_REGISTRY.md.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.refactoring.module_registry_builder import ModuleRegistryBuilder


def setup_logging():
    """Set up logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler("module_registry_generation.log")],
    )


def main():
    """Main function to generate module registry."""
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("Starting module registry generation...")

    try:
        # Initialize builder
        builder = ModuleRegistryBuilder(project_root)

        # Build the registry
        builder.build_registry()

        logger.info("Module registry generation completed successfully!")
        logger.info("Generated files:")
        logger.info("- MODULE_REGISTRY.md: Complete module documentation")
        logger.info("- module_registry_generation.log: Generation log")

        print("✅ MODULE_REGISTRY.md generated successfully!")
        print("📁 Check the project root for the generated documentation.")

    except Exception as e:
        logger.error(f"Failed to generate module registry: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

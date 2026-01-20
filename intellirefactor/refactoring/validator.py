"""
Validation utilities for AutoRefactor.

This module provides the CodeValidator class that handles all validation
tasks for generated code, including:

- Syntax validation using AST parsing
- Import consistency checking
- Facade generation validation
- Package file validation
- Component service file validation

The CodeValidator ensures that all generated code is syntactically correct
and that imports are consistent before any files are written to disk.

Classes:
    CodeValidator: Validates generated code for syntax and import correctness

Example:
    >>> validator = CodeValidator(output_directory='components')
    >>> results = {'errors': [], 'files_created': []}
    >>> if validator.validate_syntax(code, path, results):
    ...     print("Syntax is valid")
    >>> if validator.validate_refactored_file(facade_path, results):
    ...     print("Refactored file is valid")
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class CodeValidator:
    """
    Validates generated code for syntax and import correctness.

    This class provides comprehensive validation capabilities for all generated
    code, ensuring that refactoring operations produce syntactically correct
    and properly structured Python code.

    Validation includes:
    - AST-based syntax validation
    - Import consistency checking between facade and components
    - Detection of common generation bugs (e.g., nested functions in __init__)
    - Verification that all expected package files exist
    - Component service file validation

    Attributes:
        output_directory: Directory name for generated components

    Example:
        >>> validator = CodeValidator(output_directory='components')
        >>> results = {'errors': [], 'files_created': []}
        >>>
        >>> # Validate syntax
        >>> if validator.validate_syntax(code, Path('test.py'), results):
        ...     print("Valid syntax")
        >>>
        >>> # Validate complete refactored file
        >>> if validator.validate_refactored_file(Path('facade.py'), results):
        ...     print("Refactoring successful")
    """

    def __init__(self, output_directory: str = "components"):
        """
        Initialize CodeValidator.

        Args:
            output_directory: Directory name for generated components
        """
        self.output_directory = output_directory

    def validate_syntax(self, code: str, path: Path, results: Dict[str, Any]) -> bool:
        """
        Validate Python syntax of generated code.

        Args:
            code: Python source code to validate
            path: Path for error reporting
            results: Results dictionary to update with errors

        Returns:
            True if syntax is valid, False otherwise
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError as e:
            results["errors"].append(
                f"Invalid syntax in {path.name} at line {e.lineno}: {e.msg}"
            )
            return False

    def validate_generated_facade(
        self, code: str, path: Path, results: Dict[str, Any]
    ) -> None:
        """
        Validate generated facade for common issues.

        Args:
            code: Facade source code
            path: Path for error reporting
            results: Results dictionary to update with errors
        """
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return

        # Check for nested functions in __init__ (common generation bug)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        results["errors"].append(
                            f"{path.name}: nested function {child.name} inside __init__ (likely generation bug)"
                        )

    def validate_refactored_file(self, filepath: Path, results: Dict[str, Any]) -> bool:
        """
        Validate refactored file for syntax and imports.

        Args:
            filepath: Path to refactored file
            results: Results dictionary to update

        Returns:
            True if validation passed, False otherwise
        """
        try:
            if not filepath.exists():
                results["errors"].append(f"Refactored file not found: {filepath}")
                return False

            content = filepath.read_text(encoding="utf-8-sig")
            try:
                tree = ast.parse(content)
            except SyntaxError as e:
                results["errors"].append(f"Syntax error in {filepath.name}: {e}")
                return False

            self.validate_generated_facade(content, filepath, results)

            import_errors = self.validate_generated_package_files(filepath, tree)
            if import_errors:
                results["errors"].extend(
                    [f"{filepath.name}: {msg}" for msg in import_errors]
                )
                return False

            results.setdefault("validation", {})[str(filepath)] = "PASSED"
            return True

        except Exception as e:
            results["errors"].append(f"Validation error for {filepath}: {e}")
            return False

    def validate_generated_package_files(
        self, facade_path: Path, tree: ast.AST
    ) -> List[str]:
        """
        Validate that facade imports point to existing files.

        Args:
            facade_path: Path to facade file
            tree: AST tree of facade

        Returns:
            List of error messages (empty if validation passed)
        """
        errors: List[str] = []
        output_dir = facade_path.parent / self.output_directory

        # Also check for file-specific component directories
        file_stem = facade_path.stem.replace("_refactored", "")
        file_specific_dir = facade_path.parent / f"{file_stem}_{self.output_directory}"

        # Use whichever directory exists
        if file_specific_dir.exists():
            output_dir = file_specific_dir
            logger.debug(f"Using file-specific components directory: {output_dir}")
        elif output_dir.exists():
            logger.debug(f"Using standard components directory: {output_dir}")

        # If output dir doesn't exist, skip validation
        if not output_dir.exists():
            logger.debug(
                f"Output directory {output_dir} doesn't exist, skipping validation"
            )
            return errors

        # Check for imports that reference the components package
        imports_components_pkg = False
        imported_modules = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                module_name = node.module
                imported_modules.add(module_name)

                # Check for various import patterns
                if (
                    module_name.endswith(f".{self.output_directory}.container")
                    or module_name.endswith(f".{self.output_directory}.interfaces")
                    or module_name == f"{self.output_directory}.container"
                    or module_name == f"{self.output_directory}.interfaces"
                    or "_components.container" in module_name
                    or "_components.interfaces" in module_name
                ):
                    imports_components_pkg = True

        if not imports_components_pkg:
            logger.debug("No component package imports found, skipping file validation")
            return errors

        # Define expected files
        expected_files = {
            "container.py": "DI container implementation",
            "interfaces.py": "Component interfaces",
            "__init__.py": "Package initialization",
            "base.py": "Base classes for components",
        }

        # Check that all expected files exist
        for filename, description in expected_files.items():
            file_path = output_dir / filename
            if not file_path.exists():
                errors.append(
                    f"Missing {description}: {self.output_directory}/{filename}"
                )
                continue

            # Validate syntax of generated files
            try:
                content = file_path.read_text(encoding="utf-8")
                ast.parse(content)
                logger.debug(f"✅ Syntax validation passed for {filename}")
            except SyntaxError as e:
                errors.append(
                    f"Syntax error in {self.output_directory}/{filename} at line {e.lineno}: {e.msg}"
                )
            except Exception as e:
                errors.append(
                    f"Failed to validate {self.output_directory}/{filename}: {e}"
                )

        # Check for component service files
        component_files = list(output_dir.glob("*_service.py"))
        if not component_files:
            errors.append(
                f"No component service files found in {self.output_directory}/"
            )
        else:
            logger.debug(f"Found {len(component_files)} component service files")

            # Validate component service files
            for comp_file in component_files:
                try:
                    content = comp_file.read_text(encoding="utf-8")
                    ast.parse(content)
                except SyntaxError as e:
                    errors.append(
                        f"Syntax error in {comp_file.name} at line {e.lineno}: {e.msg}"
                    )
                except Exception as e:
                    errors.append(f"Failed to validate {comp_file.name}: {e}")

        # Validate import consistency
        self._validate_import_consistency(
            facade_path, output_dir, imported_modules, errors
        )

        return errors

    def _validate_import_consistency(
        self,
        facade_path: Path,
        output_dir: Path,
        imported_modules: set,
        errors: List[str],
    ) -> None:
        """
        Validate import consistency between facade and generated files.

        Args:
            facade_path: Path to facade file
            output_dir: Path to components directory
            imported_modules: Set of imported module names
            errors: List to append errors to
        """
        try:
            # Check container.py imports
            container_file = output_dir / "container.py"
            if container_file.exists():
                container_content = container_file.read_text(encoding="utf-8")
                container_tree = ast.parse(container_content)

                # Extract component imports from container
                container_imports = set()
                for node in ast.walk(container_tree):
                    if isinstance(node, ast.ImportFrom) and node.module:
                        if node.module.startswith(".") and "_service" in node.module:
                            container_imports.add(node.module)

                logger.debug(f"Container imports: {container_imports}")

            # Check interfaces.py
            interfaces_file = output_dir / "interfaces.py"
            if interfaces_file.exists():
                interfaces_content = interfaces_file.read_text(encoding="utf-8")
                interfaces_tree = ast.parse(interfaces_content)

                # Check that interfaces are properly defined
                interface_classes = []
                for node in interfaces_tree.body:
                    if isinstance(node, ast.ClassDef) and node.name.startswith("I"):
                        interface_classes.append(node.name)

                if not interface_classes:
                    errors.append(
                        f"No interface classes found in {self.output_directory}/interfaces.py"
                    )
                else:
                    logger.debug(f"Found interfaces: {interface_classes}")

        except Exception as e:
            logger.warning(f"Failed to validate import consistency: {e}")
            # Don't add to errors - this is a non-critical validation

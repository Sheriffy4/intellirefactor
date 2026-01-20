"""
Execution utilities for AutoRefactor.

This module provides the RefactoringExecutor class that handles all execution
and file I/O operations for refactoring, including:

- Atomic write sessions with rollback capability
- Backup creation and restoration
- Validated file writing with syntax checking
- Dry run execution for validation-only mode
- State management for created files and backups

The RefactoringExecutor implements the safety-first philosophy of IntelliRefactor
by providing multiple layers of protection against code corruption.

Classes:
    RefactoringExecutor: Executes refactoring operations with safety mechanisms

Example:
    >>> executor = RefactoringExecutor(
    ...     backup_enabled=True,
    ...     preserve_original=True
    ... )
    >>>
    >>> # Atomic write session with automatic rollback on error
    >>> with executor.atomic_write_session():
    ...     executor.write_validated(path, content, results, validator)
    ...     executor.write_validated(path2, content2, results, validator)
    >>>
    >>> # Create backup before modifications
    >>> backup_path = executor.create_backup(filepath, results)
"""

from __future__ import annotations

import ast
import logging
import shutil
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class RefactoringExecutor:
    """
    Executes refactoring operations with comprehensive safety mechanisms.

    This class handles all file I/O operations for refactoring with multiple
    layers of safety protection:

    - Atomic write sessions with automatic rollback on failure
    - Backup creation before any modifications
    - Syntax validation before writing files
    - State tracking for all created files
    - Restoration from backup on errors

    The executor implements the safety-first philosophy by ensuring that
    any failure during refactoring can be completely rolled back, leaving
    the codebase in its original state.

    Attributes:
        backup_enabled: Whether to create backups before modifications
        preserve_original: Whether to preserve original files
        facade_suffix: Suffix for facade files

    State Attributes:
        _created_files: List of files created during current session
        _original_filepath: Path to original file being refactored
        _backup_path: Path to backup file if created

    Example:
        >>> executor = RefactoringExecutor(backup_enabled=True)
        >>> results = {'errors': [], 'files_created': []}
        >>>
        >>> # Safe atomic write session
        >>> with executor.atomic_write_session():
        ...     executor.create_backup(original_file, results)
        ...     executor.write_validated(new_file, content, results, validator)
        ...     # Automatic rollback if any exception occurs
        >>>
        >>> # Dry run for validation only
        >>> results = executor.execute_dry_run(
        ...     filepath, plan, output_dir, results,
        ...     code_generator, validator, ...
        ... )
    """

    def __init__(
        self,
        backup_enabled: bool = True,
        preserve_original: bool = True,
        facade_suffix: str = "_refactored",
    ):
        """
        Initialize RefactoringExecutor.

        Args:
            backup_enabled: Whether to create backups
            preserve_original: Whether to preserve original files
            facade_suffix: Suffix for facade files
        """
        self.backup_enabled = backup_enabled
        self.preserve_original = preserve_original
        self.facade_suffix = facade_suffix

        self._created_files: List[Path] = []
        self._original_filepath: Optional[Path] = None
        self._backup_path: Optional[Path] = None

    @contextmanager
    def atomic_write_session(self):
        """
        Context manager for atomic write operations with rollback.

        Yields:
            None

        Raises:
            Exception: Re-raises any exception after rollback
        """
        self._created_files = []
        try:
            yield
        except Exception:
            self.rollback()
            raise

    def rollback(self) -> None:
        """Rollback all changes made during the session."""
        logger.warning("Rolling back changes...")
        for p in reversed(self._created_files):
            try:
                if p.exists() and p != self._original_filepath:
                    p.unlink()
            except Exception as e:
                logger.error("Failed to remove %s: %s", p, e)

        if self._backup_path and self._backup_path.exists() and self._original_filepath:
            try:
                shutil.copy2(self._backup_path, self._original_filepath)
                logger.info("Restored from backup: %s", self._original_filepath)
            except Exception as e:
                logger.error("Failed to restore backup: %s", e)

        self._created_files = []

    def write_validated(
        self, path: Path, content: str, results: Dict[str, Any], validator
    ) -> None:
        """
        Write file after syntax validation.

        Args:
            path: Path to write to
            content: Content to write
            results: Results dictionary to update
            validator: CodeValidator instance

        Raises:
            ValueError: If syntax validation fails
        """
        if not validator.validate_syntax(content, path, results):
            msg = f"Cannot write {path.name}: syntax validation failed"
            results["errors"].append(msg)
            raise ValueError(msg)

        path.write_text(content, encoding="utf-8")
        self._created_files.append(path)
        results["files_created"].append(str(path))

    def write_file_direct(
        self, path: Path, content: str, results: Dict[str, Any], apply_fixes_func=None
    ) -> None:
        """
        Write file directly with optional automatic fixes.

        Args:
            path: Path to write to
            content: Content to write
            results: Results dictionary to update
            apply_fixes_func: Optional function to apply automatic fixes
        """
        if apply_fixes_func:
            content = apply_fixes_func(content, path)

        ast.parse(content)  # Validate syntax
        path.write_text(content, encoding="utf-8")

    def create_backup(self, filepath: Path, results: Dict[str, Any]) -> Optional[Path]:
        """
        Create backup of original file.

        Args:
            filepath: Path to file to backup
            results: Results dictionary to update

        Returns:
            Path to backup file or None if backup disabled
        """
        if not self.backup_enabled:
            return None

        backup_path = (
            filepath.parent
            / f"{filepath.stem}.backup_{int(time.time())}{filepath.suffix}"
        )
        shutil.copy2(filepath, backup_path)
        self._backup_path = backup_path
        results["backup_created"] = str(backup_path)
        logger.info("Backup created: %s", backup_path)
        return backup_path

    def execute_dry_run(
        self,
        filepath: Path,
        plan,
        output_dir: Path,
        results: Dict[str, Any],
        code_generator,
        validator,
        component_class_name_func,
        get_group_name_func,
        is_public_extractable_func,
        min_methods_for_extraction: int,
        extract_private_methods: bool,
    ) -> Dict[str, Any]:
        """
        Execute dry run (validation only, no file writes).

        Args:
            filepath: Path to file being refactored
            plan: RefactoringPlan instance
            output_dir: Output directory for components
            results: Results dictionary
            code_generator: CodeGenerator instance
            validator: CodeValidator instance
            component_class_name_func: Function to generate component names
            get_group_name_func: Function to get group names
            is_public_extractable_func: Function to check if method is extractable
            min_methods_for_extraction: Minimum methods for extraction
            extract_private_methods: Whether to extract private methods

        Returns:
            Updated results dictionary
        """
        if not plan.extracted_components:
            results["success"] = True
            return results

        interface_classes: List[str] = []
        method_map: Dict[str, tuple] = {}

        # Validate base code
        base_code = code_generator.generate_owner_proxy_base()
        if validator.validate_syntax(base_code, output_dir / "base.py", results):
            results["files_created"].append(str(output_dir / "base.py"))

        # Validate component implementations
        for group_name, methods in plan._method_groups.items():
            extractable_public = [m for m in methods if is_public_extractable_func(m)]
            if len(extractable_public) < min_methods_for_extraction:
                continue

            comp_name = component_class_name_func(group_name)
            private_methods = (
                plan._private_methods_by_group.get(group_name, [])
                if extract_private_methods
                else []
            )

            for m in extractable_public:
                method_map[m.name] = (comp_name, m.is_async, m)

            # Validate interface
            iface = code_generator.generate_interface(comp_name, extractable_public)
            if iface:
                interface_classes.append(iface)

            # Validate implementation
            impl = code_generator.generate_component_implementation(
                comp_name, group_name, extractable_public, private_methods, plan
            )
            impl_file = output_dir / f"{group_name}_service.py"
            if validator.validate_syntax(impl, impl_file, results):
                results["files_created"].append(str(impl_file))

        # Validate interfaces file
        iface_content = code_generator.generate_interfaces_file(interface_classes, plan)
        if validator.validate_syntax(
            iface_content, output_dir / "interfaces.py", results
        ):
            results["files_created"].append(str(output_dir / "interfaces.py"))

        # Validate container
        container = code_generator.generate_di_container(
            plan.extracted_components, get_group_name_func
        )
        if validator.validate_syntax(container, output_dir / "container.py", results):
            results["files_created"].append(str(output_dir / "container.py"))

        # Validate package init
        init_code = code_generator.generate_package_init(
            plan.extracted_components, True, get_group_name_func
        )
        if validator.validate_syntax(init_code, output_dir / "__init__.py", results):
            results["files_created"].append(str(output_dir / "__init__.py"))

        # Validate facade
        facade_code = self._create_facade_code(
            plan.target_class_name,
            plan.extracted_components,
            method_map,
            plan,
            code_generator,
        )
        facade_file = filepath.with_name(
            f"{filepath.stem}{self.facade_suffix}{filepath.suffix}"
        )
        if validator.validate_syntax(facade_code, facade_file, results):
            results["files_created"].append(str(facade_file))
            validator.validate_generated_facade(facade_code, facade_file, results)

        results["success"] = len(results["errors"]) == 0
        return results

    def _create_facade_code(
        self,
        original_class_name: str,
        components: List[str],
        method_map: Dict[str, tuple],
        plan,
        code_generator,
    ) -> str:
        """
        Create facade code (helper for dry run).

        Args:
            original_class_name: Name of original class
            components: List of component names
            method_map: Mapping of methods to components
            plan: RefactoringPlan instance
            code_generator: CodeGenerator instance

        Returns:
            Facade source code
        """
        # This is a simplified version - actual implementation would use
        # the full _create_facade method from auto_refactor
        return f"# Facade for {original_class_name}\n# Components: {components}\n"

"""
Refactoring Results Validator for IntelliRefactor

Validates refactoring results and project documentation.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import RefactoringConfig


@dataclass
class ValidationResult:
    """Result of a validation check."""

    check_name: str
    success: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class RefactoringValidator:
    """Validator for refactoring results and project documentation."""

    def __init__(self, project_root: Optional[Path] = None, config: Optional[RefactoringConfig] = None):
        """
        Initialize the validator.

        Args:
            project_root: Root directory of the project.
            config: Configuration for validation operations.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.config = config or RefactoringConfig()

        self.expected_docs: List[str] = [
            "PROJECT_STRUCTURE.md",
            "MODULE_REGISTRY.md",
            "LLM_CONTEXT.md",
        ]

    def set_expected_documents(self, documents: List[str]) -> None:
        """Set the list of expected documents to validate."""
        self.expected_docs = list(documents)

    def validate_all(self) -> List[ValidationResult]:
        """
        Execute all validation checks.

        Returns:
            List of validation results.
        """
        results: List[ValidationResult] = []
        results.append(self._check_documents_exist())
        results.append(self._check_document_links())

        if "PROJECT_STRUCTURE.md" in self.expected_docs:
            results.append(self._check_project_structure_content())
        if "MODULE_REGISTRY.md" in self.expected_docs:
            results.append(self._check_module_registry_content())
        if "LLM_CONTEXT.md" in self.expected_docs:
            results.append(self._check_llm_context_content())

        results.append(self._check_cleanup_results())
        return results

    def validate_refactoring(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate a refactoring result for correctness and safety.

        Args:
            result: Refactoring result to validate.

        Returns:
            Validation report dict.
        """
        validation_results: List[ValidationResult] = []

        if not isinstance(result, dict):
            validation_results.append(
                ValidationResult(
                    check_name="Input Type",
                    success=False,
                    message="Result must be a dict",
                    details={"type": str(type(result))},
                )
            )
        else:
            validation_results.append(self._validate_result_structure(result))

            if "changes" in result:
                changes = result.get("changes")
                if isinstance(changes, list):
                    validation_results.append(self._validate_file_changes(changes))
                else:
                    validation_results.append(
                        ValidationResult(
                            check_name="File Changes",
                            success=False,
                            message="Field 'changes' must be a list when present",
                            details={"type": str(type(changes))},
                        )
                    )

            if "metrics_before" in result and "metrics_after" in result:
                if isinstance(result["metrics_before"], dict) and isinstance(result["metrics_after"], dict):
                    validation_results.append(
                        self._validate_metrics_improvement(
                            result["metrics_before"], result["metrics_after"]
                        )
                    )
                else:
                    validation_results.append(
                        ValidationResult(
                            check_name="Metrics Improvement",
                            success=False,
                            message="'metrics_before' and 'metrics_after' must be dicts",
                        )
                    )

        overall_success = all(vr.success for vr in validation_results)

        return {
            "status": "passed" if overall_success else "failed",
            "overall_success": overall_success,
            "validation_results": [
                {
                    "check": vr.check_name,
                    "success": vr.success,
                    "message": vr.message,
                    "details": vr.details or {},
                }
                for vr in validation_results
            ],
            "summary": {
                "total_checks": len(validation_results),
                "passed_checks": sum(1 for vr in validation_results if vr.success),
                "failed_checks": sum(1 for vr in validation_results if not vr.success),
            },
        }

    def _validate_result_structure(self, result: Dict[str, Any]) -> ValidationResult:
        """Validate the structure of a refactoring result."""
        required_fields = ["operation_id", "status"]
        missing_fields = [field for field in required_fields if field not in result]

        if missing_fields:
            return ValidationResult(
                check_name="Result Structure",
                success=False,
                message=f"Missing required fields: {', '.join(missing_fields)}",
                details={"missing_fields": missing_fields},
            )

        return ValidationResult(
            check_name="Result Structure",
            success=True,
            message="Refactoring result has valid structure",
            details={"fields_present": list(result.keys())},
        )

    def _validate_file_changes(self, changes: List[Dict[str, Any]]) -> ValidationResult:
        """Validate file changes in a refactoring result."""
        if not changes:
            return ValidationResult(
                check_name="File Changes",
                success=True,
                message="No file changes to validate",
                details={"changes_count": 0},
            )

        invalid_changes: List[str] = []
        for i, change in enumerate(changes):
            if not isinstance(change, dict):
                invalid_changes.append(f"Change {i}: change must be a dict")
                continue
            if "file_path" not in change or "change_type" not in change:
                invalid_changes.append(f"Change {i}: missing file_path or change_type")

        if invalid_changes:
            return ValidationResult(
                check_name="File Changes",
                success=False,
                message=f"Invalid changes found: {len(invalid_changes)}",
                details={"invalid_changes": invalid_changes},
            )

        return ValidationResult(
            check_name="File Changes",
            success=True,
            message=f"All {len(changes)} file changes are valid",
            details={"changes_count": len(changes)},
        )

    def _validate_metrics_improvement(
        self, metrics_before: Dict[str, Any], metrics_after: Dict[str, Any]
    ) -> ValidationResult:
        """Validate that refactoring improved code metrics."""
        improvements: List[str] = []
        regressions: List[str] = []

        improvement_metrics = [
            "cyclomatic_complexity",
            "maintainability_index",
            "code_duplication",
        ]

        for metric in improvement_metrics:
            if metric not in metrics_before or metric not in metrics_after:
                continue

            before = metrics_before[metric]
            after = metrics_after[metric]

            if metric == "maintainability_index":
                if after > before:
                    improvements.append(f"{metric}: {before} -> {after}")
                elif after < before:
                    regressions.append(f"{metric}: {before} -> {after}")
            else:
                if after < before:
                    improvements.append(f"{metric}: {before} -> {after}")
                elif after > before:
                    regressions.append(f"{metric}: {before} -> {after}")

        if regressions and not improvements:
            return ValidationResult(
                check_name="Metrics Improvement",
                success=False,
                message=f"Metrics regressed: {', '.join(regressions)}",
                details={"regressions": regressions, "improvements": improvements},
            )

        return ValidationResult(
            check_name="Metrics Improvement",
            success=True,
            message=(
                "Metrics improved or maintained: "
                f"{len(improvements)} improvements, {len(regressions)} regressions"
            ),
            details={"improvements": improvements, "regressions": regressions},
        )

    def _check_documents_exist(self) -> ValidationResult:
        """Check existence of all expected documents."""
        missing_docs: List[str] = []
        existing_docs: List[str] = []

        for doc_name in self.expected_docs:
            doc_path = self.project_root / doc_name
            if doc_path.exists():
                existing_docs.append(doc_name)
            else:
                missing_docs.append(doc_name)

        if missing_docs:
            return ValidationResult(
                check_name="Document Existence",
                success=False,
                message=f"Missing documents: {', '.join(missing_docs)}",
                details={"missing": missing_docs, "existing": existing_docs},
            )

        return ValidationResult(
            check_name="Document Existence",
            success=True,
            message=f"All {len(self.expected_docs)} documents found",
            details={"existing": existing_docs},
        )

    def _check_document_links(self) -> ValidationResult:
        """Check links between documents."""
        broken_links: List[str] = []
        working_links: List[str] = []

        llm_context_path = self.project_root / "LLM_CONTEXT.md"
        if llm_context_path.exists():
            try:
                content = llm_context_path.read_text(encoding="utf-8")
            except OSError as exc:
                return ValidationResult(
                    check_name="Document Links",
                    success=False,
                    message=f"Failed to read LLM_CONTEXT.md: {exc}",
                )

            for doc_name in ("PROJECT_STRUCTURE.md", "MODULE_REGISTRY.md"):
                if doc_name in content:
                    target_path = self.project_root / doc_name
                    if target_path.exists():
                        working_links.append(f"LLM_CONTEXT.md -> {doc_name}")
                    else:
                        broken_links.append(f"LLM_CONTEXT.md -> {doc_name}")

        if broken_links:
            return ValidationResult(
                check_name="Document Links",
                success=False,
                message=f"Broken links found: {', '.join(broken_links)}",
                details={"broken": broken_links, "working": working_links},
            )

        return ValidationResult(
            check_name="Document Links",
            success=True,
            message=f"All links working ({len(working_links)} checked)",
            details={"working": working_links},
        )

    def _check_project_structure_content(self) -> ValidationResult:
        """Check content of PROJECT_STRUCTURE.md."""
        doc_path = self.project_root / "PROJECT_STRUCTURE.md"
        if not doc_path.exists():
            return ValidationResult(
                check_name="PROJECT_STRUCTURE.md Content",
                success=False,
                message="PROJECT_STRUCTURE.md file not found",
            )

        try:
            content = doc_path.read_text(encoding="utf-8")
        except OSError as exc:
            return ValidationResult(
                check_name="PROJECT_STRUCTURE.md Content",
                success=False,
                message=f"Failed to read PROJECT_STRUCTURE.md: {exc}",
            )

        required_patterns = [
            (r"(#.*structur.*project|project.*structure)", "Project Structure header"),
            (r"(entry points?|entry.*points?)", "Entry Points section"),
            (r"(configuration files?|config.*files?)", "Configuration Files section"),
        ]

        missing_sections: List[str] = []
        for pattern, description in required_patterns:
            if not re.search(pattern, content, re.IGNORECASE):
                missing_sections.append(description)

        if missing_sections:
            return ValidationResult(
                check_name="PROJECT_STRUCTURE.md Content",
                success=False,
                message=f"Missing sections: {', '.join(missing_sections)}",
                details={"missing_sections": missing_sections},
            )

        return ValidationResult(
            check_name="PROJECT_STRUCTURE.md Content",
            success=True,
            message="All required sections present",
            details={"content_length": len(content)},
        )

    def _check_module_registry_content(self) -> ValidationResult:
        """Check content of MODULE_REGISTRY.md."""
        doc_path = self.project_root / "MODULE_REGISTRY.md"
        if not doc_path.exists():
            return ValidationResult(
                check_name="MODULE_REGISTRY.md Content",
                success=False,
                message="MODULE_REGISTRY.md file not found",
            )

        try:
            content = doc_path.read_text(encoding="utf-8")
        except OSError as exc:
            return ValidationResult(
                check_name="MODULE_REGISTRY.md Content",
                success=False,
                message=f"Failed to read MODULE_REGISTRY.md: {exc}",
            )

        required_patterns = [
            (r"(#.*module.*registry|module registry)", "Module Registry header"),
            (r"(categories)", "Categories section"),
            (r"(modules)", "Modules section"),
        ]

        missing_sections: List[str] = []
        for pattern, description in required_patterns:
            if not re.search(pattern, content, re.IGNORECASE):
                missing_sections.append(description)

        module_patterns = [
            r"^##\s+.*\.py",
            r"^\*\*.*\.py\*\*",
            r"###\s+.*\.py",
        ]
        module_count = sum(len(re.findall(p, content, re.MULTILINE)) for p in module_patterns)

        if missing_sections:
            return ValidationResult(
                check_name="MODULE_REGISTRY.md Content",
                success=False,
                message=f"Missing sections: {', '.join(missing_sections)}",
                details={"missing_sections": missing_sections, "module_count": module_count},
            )

        return ValidationResult(
            check_name="MODULE_REGISTRY.md Content",
            success=True,
            message=f"All sections present, found {module_count} modules",
            details={"module_count": module_count, "content_length": len(content)},
        )

    def _check_llm_context_content(self) -> ValidationResult:
        """Check content of LLM_CONTEXT.md."""
        doc_path = self.project_root / "LLM_CONTEXT.md"
        if not doc_path.exists():
            return ValidationResult(
                check_name="LLM_CONTEXT.md Content",
                success=False,
                message="LLM_CONTEXT.md file not found",
            )

        try:
            content = doc_path.read_text(encoding="utf-8")
        except OSError as exc:
            return ValidationResult(
                check_name="LLM_CONTEXT.md Content",
                success=False,
                message=f"Failed to read LLM_CONTEXT.md: {exc}",
            )

        required_patterns = [
            (r"MODULE_REGISTRY\.md", "MODULE_REGISTRY.md reference"),
            (r"PROJECT_STRUCTURE\.md", "PROJECT_STRUCTURE.md reference"),
            (r"(before creating.*functionality|check.*before.*creating)", "check before creating functionality rule"),
            (r"(where to place.*code|code.*placement)", "where to place code rule"),
        ]

        missing_rules: List[str] = []
        for pattern, description in required_patterns:
            if not re.search(pattern, content, re.IGNORECASE):
                missing_rules.append(description)

        if missing_rules:
            return ValidationResult(
                check_name="LLM_CONTEXT.md Content",
                success=False,
                message=f"Missing rules: {', '.join(missing_rules)}",
                details={"missing_rules": missing_rules},
            )

        return ValidationResult(
            check_name="LLM_CONTEXT.md Content",
            success=True,
            message="All required rules present",
            details={"content_length": len(content)},
        )

    def _check_cleanup_results(self) -> ValidationResult:
        """Check cleanup results."""
        to_delete_path = self.project_root / "_to_delete"
        if not to_delete_path.exists():
            return ValidationResult(
                check_name="Cleanup Results",
                success=True,
                message="No _to_delete folder found - cleanup may not have been performed",
            )

        moved_files = [p for p in to_delete_path.rglob("*") if p.is_file()]
        return ValidationResult(
            check_name="Cleanup Results",
            success=True,
            message=f"_to_delete folder contains {len(moved_files)} moved files",
            details={"moved_files_count": len(moved_files)},
        )
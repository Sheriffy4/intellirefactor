"""
Refactoring Final Reporter for IntelliRefactor

Creates comprehensive reports on refactoring activities.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..config import RefactoringConfig


@dataclass
class RefactoringStats:
    """Statistics for refactoring operations."""

    garbage_files_found: int = 0
    garbage_files_moved: int = 0
    size_freed_bytes: int = 0
    directories_analyzed: int = 0
    entry_points_found: int = 0
    config_files_found: int = 0
    modules_analyzed: int = 0
    categories_found: int = 0
    documents_created: int = 0
    refactoring_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0


class RefactoringReporter:
    """Generator for comprehensive refactoring reports."""

    def __init__(self, project_root: Optional[Path] = None, config: Optional[RefactoringConfig] = None):
        """
        Initialize the reporter.

        Args:
            project_root: Root directory of the project.
            config: Configuration for reporting operations.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.config = config or RefactoringConfig()
        self.stats = RefactoringStats()

    def generate_report(
        self,
        results: Optional[List[Dict[str, Any]]] = None,
        output_format: str = "dict",
    ) -> Union[Dict[str, Any], str]:
        """
        Generate a comprehensive report on refactoring activities.

        Args:
            results: List of refactoring results (optional).
            output_format: Output format ('dict', 'json', 'text', 'html').

        Returns:
            Report as dict (for 'dict') or formatted string (json/text/html).
        """
        report: Dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "stats": {},
            "documents": {},
            "cleanup": {},
            "refactoring_results": {},
            "validation": {},
            "summary": {},
        }

        if results:
            self._analyze_refactoring_results(report, results)

        self._analyze_cleanup_results(report)
        self._analyze_created_documents(report)
        self._analyze_project_structure(report)
        self._analyze_module_registry(report)
        self._create_summary(report)

        fmt = (output_format or "dict").lower().strip()
        if fmt == "dict":
            return report
        if fmt == "json":
            return json.dumps(report, indent=2, default=str, ensure_ascii=False)
        if fmt == "text":
            return self._format_text_report(report)
        if fmt == "html":
            return self._format_html_report(report)

        # Backward compatibility: unknown formats -> JSON string
        return json.dumps(report, indent=2, default=str, ensure_ascii=False)

    def _analyze_refactoring_results(self, report: Dict[str, Any], results: List[Dict[str, Any]]) -> None:
        """Analyze refactoring operation results."""
        refactoring_data: Dict[str, Any] = {
            "total_operations": len(results),
            "successful_operations": 0,
            "failed_operations": 0,
            "operations_by_type": {},
            "metrics_improvements": {},
            "file_changes": 0,
        }

        for result in results:
            ok = result.get("status") == "completed" or bool(result.get("success", False))
            if ok:
                refactoring_data["successful_operations"] += 1
                self.stats.successful_operations += 1
            else:
                refactoring_data["failed_operations"] += 1
                self.stats.failed_operations += 1

            operation_type = result.get("operation_type", "unknown")
            ops_by_type = refactoring_data["operations_by_type"]
            ops_by_type[operation_type] = ops_by_type.get(operation_type, 0) + 1

            changes = result.get("changes", [])
            if isinstance(changes, list):
                refactoring_data["file_changes"] += len(changes)

            if "metrics_before" in result and "metrics_after" in result:
                self._analyze_metrics_improvement(
                    result["metrics_before"],
                    result["metrics_after"],
                    refactoring_data["metrics_improvements"],
                )

        self.stats.refactoring_operations = len(results)
        report["refactoring_results"] = refactoring_data

    def _analyze_metrics_improvement(
        self,
        before: Dict[str, Any],
        after: Dict[str, Any],
        improvements: Dict[str, Any],
    ) -> None:
        """Analyze metrics improvements from refactoring."""
        metrics_to_check = [
            "cyclomatic_complexity",
            "maintainability_index",
            "code_duplication",
            "lines_of_code",
        ]

        for metric in metrics_to_check:
            if metric not in before or metric not in after:
                continue

            before_val = before[metric]
            after_val = after[metric]

            bucket = improvements.setdefault(
                metric, {"improved": 0, "degraded": 0, "unchanged": 0}
            )

            # maintainability_index: higher is better; others: lower is better
            if metric == "maintainability_index":
                if after_val > before_val:
                    bucket["improved"] += 1
                elif after_val < before_val:
                    bucket["degraded"] += 1
                else:
                    bucket["unchanged"] += 1
            else:
                if after_val < before_val:
                    bucket["improved"] += 1
                elif after_val > before_val:
                    bucket["degraded"] += 1
                else:
                    bucket["unchanged"] += 1

    def _analyze_cleanup_results(self, report: Dict[str, Any]) -> None:
        """Analyze cleanup results."""
        to_delete_path = self.project_root / "_to_delete"

        cleanup_data: Dict[str, Any] = {
            "to_delete_exists": to_delete_path.exists(),
            "moved_files": 0,
            "categories": {},
            "total_size": 0,
        }

        if to_delete_path.exists():
            moved_files = [p for p in to_delete_path.rglob("*") if p.is_file()]
            cleanup_data["moved_files"] = len(moved_files)

            total_size = 0
            for p in moved_files:
                try:
                    total_size += p.stat().st_size
                except OSError:
                    continue

            cleanup_data["total_size"] = total_size
            self.stats.size_freed_bytes = total_size

            categories: Dict[str, int] = {}
            for file_path in moved_files:
                parent = file_path.parent.name
                if parent != "_to_delete":
                    categories[parent] = categories.get(parent, 0) + 1

            cleanup_data["categories"] = categories
            self.stats.garbage_files_moved = len(moved_files)

        report["cleanup"] = cleanup_data

    def _analyze_created_documents(self, report: Dict[str, Any]) -> None:
        """Analyze created documents."""
        expected_docs = ["PROJECT_STRUCTURE.md", "MODULE_REGISTRY.md", "LLM_CONTEXT.md"]
        documents_data: Dict[str, Any] = {"expected": len(expected_docs), "created": 0, "details": {}}

        for doc_name in expected_docs:
            doc_path = self.project_root / doc_name
            doc_info: Dict[str, Any] = {
                "exists": doc_path.exists(),
                "size": 0,
                "lines": 0,
                "created_time": None,
            }

            if doc_path.exists():
                documents_data["created"] += 1
                try:
                    st = doc_path.stat()
                    doc_info["size"] = st.st_size
                    doc_info["created_time"] = datetime.fromtimestamp(st.st_mtime).isoformat()
                except OSError:
                    pass

                try:
                    content = doc_path.read_text(encoding="utf-8")
                    doc_info["lines"] = len(content.splitlines())
                except OSError:
                    doc_info["lines"] = 0

            documents_data["details"][doc_name] = doc_info

        self.stats.documents_created = documents_data["created"]
        report["documents"] = documents_data

    def _analyze_project_structure(self, report: Dict[str, Any]) -> None:
        """Analyze PROJECT_STRUCTURE.md."""
        doc_path = self.project_root / "PROJECT_STRUCTURE.md"
        structure_data: Dict[str, Any] = {
            "exists": doc_path.exists(),
            "directories_documented": 0,
            "entry_points_found": 0,
            "config_files_found": 0,
        }

        if doc_path.exists():
            try:
                content = doc_path.read_text(encoding="utf-8")

                structure_data["directories_documented"] = len(
                    re.findall(r"^###\s+", content, re.MULTILINE)
                )
                self.stats.directories_analyzed = structure_data["directories_documented"]

                structure_data["entry_points_found"] = len(
                    re.findall(r"entry.?point", content, re.IGNORECASE)
                )
                self.stats.entry_points_found = structure_data["entry_points_found"]

                structure_data["config_files_found"] = len(
                    re.findall(r"\.(json|yaml|yml|ini|conf|toml)\b", content)
                )
                self.stats.config_files_found = structure_data["config_files_found"]

            except Exception as exc:
                structure_data["error"] = str(exc)

        report["structure"] = structure_data

    def _analyze_module_registry(self, report: Dict[str, Any]) -> None:
        """Analyze MODULE_REGISTRY.md."""
        doc_path = self.project_root / "MODULE_REGISTRY.md"
        registry_data: Dict[str, Any] = {
            "exists": doc_path.exists(),
            "modules_documented": 0,
            "categories_found": 0,
            "categories": {},
        }

        if doc_path.exists():
            try:
                content = doc_path.read_text(encoding="utf-8")

                module_patterns = [
                    r"^##\s+.*\.py",
                    r"^\*\*.*\.py\*\*",
                    r"###\s+.*\.py",
                ]

                modules = sum(len(re.findall(p, content, re.MULTILINE)) for p in module_patterns)
                registry_data["modules_documented"] = modules
                self.stats.modules_analyzed = modules

                categories = re.findall(r"^#\s+(.+)", content, re.MULTILINE)
                categories = [
                    cat for cat in categories
                    if not any(
                        word in cat.lower()
                        for word in ("module registry", "overview", "modules", "categories")
                    )
                ]

                registry_data["categories_found"] = len(categories)
                self.stats.categories_found = len(categories)

                for category in categories:
                    category_pattern = rf"^#\s+{re.escape(category)}.*?(?=^#|\Z)"
                    category_section = re.search(
                        category_pattern, content, re.MULTILINE | re.DOTALL
                    )
                    if not category_section:
                        continue

                    category_content = category_section.group(0)
                    module_count = sum(
                        len(re.findall(p, category_content, re.MULTILINE)) for p in module_patterns
                    )
                    registry_data["categories"][category] = module_count

            except Exception as exc:
                registry_data["error"] = str(exc)

        report["registry"] = registry_data

    def _create_summary(self, report: Dict[str, Any]) -> None:
        """Create final summary."""
        cleanup = report.get("cleanup", {})
        documents = report.get("documents", {})
        structure = report.get("structure", {})
        registry = report.get("registry", {})
        refactoring = report.get("refactoring_results", {}) or {}

        summary: Dict[str, Any] = {
            "overall_success": True,
            "completed_stages": [],
            "issues": [],
            "recommendations": [],
        }

        if cleanup.get("to_delete_exists"):
            summary["completed_stages"].append("Project cleanup")
            if cleanup.get("moved_files", 0) == 0:
                summary["issues"].append("No garbage files found or moved")
        else:
            summary["issues"].append("Cleanup stage not executed")

        docs_created = int(documents.get("created", 0))
        docs_expected = int(documents.get("expected", 0))
        if docs_created == docs_expected and docs_expected > 0:
            summary["completed_stages"].append("Documentation creation")
        else:
            summary["issues"].append(f"Created {docs_created} of {docs_expected} documents")
            if docs_created == 0 and docs_expected > 0:
                summary["overall_success"] = False

        if structure.get("exists"):
            summary["completed_stages"].append("Project structure analysis")
        else:
            summary["issues"].append("PROJECT_STRUCTURE.md not created")

        if registry.get("exists"):
            summary["completed_stages"].append("Module registry creation")
        else:
            summary["issues"].append("MODULE_REGISTRY.md not created")

        if refactoring:
            if refactoring.get("successful_operations", 0) > 0:
                summary["completed_stages"].append("Refactoring operations")
            if refactoring.get("failed_operations", 0) > 0:
                summary["issues"].append(
                    f"{refactoring['failed_operations']} refactoring operations failed"
                )

        if self.stats.modules_analyzed > 100:
            summary["recommendations"].append(
                "Consider splitting large modules for better maintainability"
            )

        if self.stats.categories_found < 5:
            summary["recommendations"].append("Consider reviewing module categorization")

        if cleanup.get("moved_files", 0) > 200:
            summary["recommendations"].append(
                "Consider setting up automatic cleanup processes"
            )

        if refactoring:
            total = max(int(refactoring.get("total_operations", 0)), 1)
            success = int(refactoring.get("successful_operations", 0))
            success_rate = success / total
            if success_rate < 0.8:
                summary["recommendations"].append(
                    "Review failed refactoring operations for patterns"
                )

        report["summary"] = summary
        report["stats"] = asdict(self.stats)

    def save_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """
        Save report to file as JSON.

        Args:
            report: Report data.
            filename: Filename (auto-generated if not provided).

        Returns:
            Path to saved file.
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"refactoring_report_{timestamp}.json"

        report_path = self.project_root / filename
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with report_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

        return report_path

    def _format_size(self, size_bytes: int) -> str:
        """Format size in human-readable form."""
        size = float(size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"

    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """Format report as plain text."""
        lines: List[str] = []
        lines.append("=" * 60)
        lines.append("COMPREHENSIVE REFACTORING REPORT")
        lines.append("=" * 60)

        lines.append(f"Project: {report.get('project_root', '')}")
        lines.append(f"Report generated: {report.get('timestamp', '')}")
        lines.append("")

        stats = report.get("stats", {})
        lines.append("STATISTICS:")
        lines.append(f"  Refactoring operations: {stats.get('refactoring_operations', 0)}")
        lines.append(f"  Successful operations: {stats.get('successful_operations', 0)}")
        lines.append(f"  Failed operations: {stats.get('failed_operations', 0)}")
        lines.append("")

        refactoring = report.get("refactoring_results", {})
        if refactoring.get("total_operations", 0) > 0:
            lines.append("REFACTORING RESULTS:")
            lines.append(f"  Total operations: {refactoring.get('total_operations', 0)}")
            lines.append(f"  Successful: {refactoring.get('successful_operations', 0)}")
            lines.append(f"  Failed: {refactoring.get('failed_operations', 0)}")
            lines.append("")

        summary = report.get("summary", {})
        lines.append("SUMMARY:")
        lines.append(
            f"  Overall result: {'SUCCESS' if summary.get('overall_success', False) else 'ISSUES FOUND'}"
        )

        if summary.get("issues"):
            lines.append(f"  Issues found: {len(summary['issues'])}")
            for issue in summary["issues"]:
                lines.append(f"    - {issue}")

        if summary.get("recommendations"):
            lines.append(f"  Recommendations: {len(summary['recommendations'])}")
            for rec in summary["recommendations"]:
                lines.append(f"    - {rec}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def _format_html_report(self, report: Dict[str, Any]) -> str:
        """Format report as HTML."""
        overall_success = bool(report.get("summary", {}).get("overall_success", False))
        status_class = "success" if overall_success else "warning"
        status_text = "SUCCESS" if overall_success else "ISSUES FOUND"

        ref_ops = report.get("stats", {}).get("refactoring_operations", 0)
        ok_ops = report.get("stats", {}).get("successful_operations", 0)
        bad_ops = report.get("stats", {}).get("failed_operations", 0)

        return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Refactoring Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    .header {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; }}
    .section {{ margin: 20px 0; }}
    .stats {{ background-color: #e8f4f8; padding: 10px; border-radius: 5px; }}
    .success {{ color: green; }}
    .warning {{ color: orange; }}
    .error {{ color: red; }}
  </style>
</head>
<body>
  <div class="header">
    <h1>Comprehensive Refactoring Report</h1>
    <p><strong>Project:</strong> {report.get("project_root", "")}</p>
    <p><strong>Generated:</strong> {report.get("timestamp", "")}</p>
  </div>

  <div class="section stats">
    <h2>Statistics</h2>
    <ul>
      <li>Refactoring operations: {ref_ops}</li>
      <li>Successful operations: {ok_ops}</li>
      <li>Failed operations: {bad_ops}</li>
    </ul>
  </div>

  <div class="section">
    <h2>Summary</h2>
    <p class="{status_class}">Overall result: {status_text}</p>
  </div>
</body>
</html>
"""
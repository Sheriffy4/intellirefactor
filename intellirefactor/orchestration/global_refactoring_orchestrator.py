"""
Global Refactoring Orchestrator for IntelliRefactor

Coordinates complex multi-step refactoring workflows with logging and error handling.

Stages:
1. Project cleanup (log files, temporary files, debug scripts)
2. Project structure analysis (creating PROJECT_STRUCTURE.md)
3. Module registry creation (creating MODULE_REGISTRY.md)
4. Context generation for analysis tools
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import RefactoringConfig


@dataclass
class StageResult:
    """Result of executing a refactoring stage."""

    stage_name: str
    success: bool
    duration: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[BaseException] = None


@dataclass
class RefactoringReport:
    """Final report of refactoring execution."""

    start_time: datetime
    end_time: datetime
    total_duration: float
    stages: List[StageResult]
    overall_success: bool

    @property
    def successful_stages(self) -> int:
        return sum(1 for s in self.stages if s.success)

    @property
    def failed_stages(self) -> int:
        return sum(1 for s in self.stages if not s.success)


class GlobalRefactoringOrchestrator:
    """Orchestrator for executing all stages of global refactoring."""

    def __init__(
        self,
        project_root: Optional[Path] = None,
        dry_run: bool = False,
        config: Optional[RefactoringConfig] = None,
    ) -> None:
        """
        Initialize the orchestrator.

        Args:
            project_root: Root directory of the project.
            dry_run: Dry run mode - show what would be done without executing.
            config: Configuration for refactoring operations.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.dry_run = dry_run
        self.config = config or RefactoringConfig()

        # Components are optional and can be injected by user/config.
        self.components: Dict[str, Any] = {}

        self.logger = self._setup_logging()

        self.logger.info("Initialized refactoring orchestrator for %s", self.project_root)
        if self.dry_run:
            self.logger.info("DRY RUN MODE - changes will not be applied")

    def _setup_logging(self) -> logging.Logger:
        """
        Set up logging configuration.

        Notes:
            - Logger is made unique per orchestrator instance to avoid handler duplication.
            - File logs are stored in: <project_root>/.intellirefactor/logs/
        """
        logger_name = f"{__name__}.{self.__class__.__name__}.{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

        # Avoid duplicates if something re-initializes the same logger.
        if logger.handlers:
            return logger

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File logging is optional; failures should not break orchestration.
        try:
            base_dir = self.project_root
            base_dir.mkdir(parents=True, exist_ok=True)

            logs_dir = base_dir / ".intellirefactor" / "logs"
            logs_dir.mkdir(parents=True, exist_ok=True)

            log_file = logs_dir / f"refactoring_{datetime.now():%Y%m%d_%H%M%S}.log"
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning("Could not setup file logging: %s", exc)

        return logger

    def register_component(self, name: str, component: Any) -> None:
        """
        Register a component for use in orchestration stages.

        Args:
            name: Component name (e.g. "module_scanner").
            component: Object implementing required interface for the stage.
        """
        if not name or not isinstance(name, str):
            raise ValueError("Component name must be a non-empty string.")
        self.components[name] = component
        self.logger.debug("Registered component: %s", name)

    def analyze_and_refactor_file(
        self, file_path: Path, project_root: Optional[Path] = None
    ) -> StageResult:
        """
        Analyze and refactor a single file.

        Args:
            file_path: Path to the file.
            project_root: Project root (optional).

        Returns:
            StageResult indicating success/failure.
        """
        # Lazy import to avoid circular dependencies.
        from ..refactoring.auto_refactor import AutoRefactor

        started = time.time()
        file_path = Path(file_path)

        if project_root is not None:
            self.project_root = Path(project_root)

        self.logger.info("Analyzing and refactoring file: %s", file_path)

        try:
            if not file_path.exists() or not file_path.is_file():
                return StageResult(
                    stage_name="file_refactoring",
                    success=False,
                    duration=0.0,
                    message=f"File not found: {file_path}",
                    details={"file": str(file_path)},
                )

            refactorer = AutoRefactor(self.config)

            plan = refactorer.analyze_god_object(file_path)
            transformations = getattr(plan, "transformations", None)

            if not transformations:
                self.logger.info("No refactoring opportunities found for %s", file_path)
                return StageResult(
                    stage_name="file_refactoring",
                    success=True,
                    duration=time.time() - started,
                    message="No refactoring needed",
                    details={"file": str(file_path)},
                )

            risk_level = getattr(plan, "risk_level", "unknown")
            self.logger.info(
                "Generated plan: %s transformations, Risk: %s",
                len(transformations),
                risk_level,
            )

            if self.dry_run:
                self.logger.info("Dry run - skipping execution")
                return StageResult(
                    stage_name="file_refactoring",
                    success=True,
                    duration=time.time() - started,
                    message="Dry run completed",
                    details={"plan": str(plan), "file": str(file_path)},
                )

            result = refactorer.execute_refactoring(file_path, plan, dry_run=False)
            if not isinstance(result, dict):
                return StageResult(
                    stage_name="file_refactoring",
                    success=False,
                    duration=time.time() - started,
                    message="Refactoring returned unexpected result type",
                    details={"type": str(type(result))},
                )

            success = bool(result.get("success", False))
            message = "Refactoring successful" if success else "Refactoring failed"
            errors = result.get("errors")
            if errors:
                if isinstance(errors, list):
                    message += f": {', '.join(map(str, errors))}"
                else:
                    message += f": {errors}"

            return StageResult(
                stage_name="file_refactoring",
                success=success,
                duration=time.time() - started,
                message=message,
                details=result,
            )

        except Exception as exc:
            self.logger.exception("Error refactoring file %s", file_path)
            return StageResult(
                stage_name="file_refactoring",
                success=False,
                duration=time.time() - started,
                message=f"Exception: {exc}",
                error=exc,
                details={"file": str(file_path)},
            )

    def run_all_stages(
        self, stage_configs: Optional[List[Dict[str, Any]]] = None
    ) -> RefactoringReport:
        """
        Execute all refactoring stages.

        Args:
            stage_configs: Optional list of stage configurations.

        Returns:
            RefactoringReport with results of all stages.
        """
        start_time = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("STARTING GLOBAL REFACTORING")
        self.logger.info("=" * 60)

        stages: List[StageResult] = []

        if stage_configs is None:
            stage_configs = [
                {"name": "cleanup", "description": "Project cleanup"},
                {"name": "structure", "description": "Project structure analysis"},
                {"name": "modules", "description": "Module registry creation"},
                {"name": "context", "description": "Context generation"},
                {"name": "visualization", "description": "Visualization generation"},
            ]

        for i, stage_config in enumerate(stage_configs, 1):
            stage_result = self._run_stage(i, stage_config)
            stages.append(stage_result)

            if not stage_result.success and getattr(self.config, "stop_on_failure", False):
                self.logger.error(
                    "Stopping execution due to failure in %s", stage_result.stage_name
                )
                break

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()

        report = RefactoringReport(
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            stages=stages,
            overall_success=all(stage.success for stage in stages),
        )

        self._print_final_report(report)
        return report

    def run_single_stage(
        self, stage_number: int, stage_config: Optional[Dict[str, Any]] = None
    ) -> StageResult:
        """
        Execute a single refactoring stage.

        Args:
            stage_number: Stage number.
            stage_config: Configuration for the stage.

        Returns:
            StageResult with execution result.
        """
        self.logger.info("Running stage %s", stage_number)

        if stage_config is None:
            default_configs = {
                1: {"name": "cleanup", "description": "Project cleanup"},
                2: {"name": "structure", "description": "Project structure analysis"},
                3: {"name": "modules", "description": "Module registry creation"},
                4: {"name": "context", "description": "Context generation"},
                5: {"name": "visualization", "description": "Visualization generation"},
            }
            stage_config = default_configs.get(
                stage_number, {"name": "unknown", "description": "Unknown stage"}
            )

        return self._run_stage(stage_number, stage_config)

    def _run_stage(self, stage_number: int, stage_config: Dict[str, Any]) -> StageResult:
        """Execute a single stage with error handling."""
        stage_title = f"Stage {stage_number}: {stage_config.get('description', 'Unknown')}"
        started = time.time()

        try:
            self.logger.info("Running %s", stage_title)

            handler_name = stage_config.get("name", "unknown")
            handler = getattr(self, f"_handle_{handler_name}_stage", None)

            if handler is None:
                return StageResult(
                    stage_name=stage_title,
                    success=False,
                    duration=time.time() - started,
                    message=f"No handler found for stage: {handler_name}",
                    error=ValueError(f"Unknown stage handler: {handler_name}"),
                )

            result = handler(stage_config)
            if not isinstance(result, StageResult):
                return StageResult(
                    stage_name=stage_title,
                    success=False,
                    duration=time.time() - started,
                    message="Stage handler returned unexpected result type",
                    details={"type": str(type(result)), "handler": str(handler_name)},
                )

            result.duration = time.time() - started
            result.stage_name = stage_title
            return result

        except Exception as exc:
            self.logger.exception("Error in %s", stage_title)
            return StageResult(
                stage_name=stage_title,
                success=False,
                duration=time.time() - started,
                message=f"Error: {exc}",
                error=exc,
            )

    def _handle_cleanup_stage(self, config: Dict[str, Any]) -> StageResult:
        """Handle project cleanup stage."""
        self.logger.info("Executing cleanup stage...")

        if self.dry_run:
            return StageResult(
                stage_name="cleanup",
                success=True,
                duration=0.0,
                message="[DRY RUN] Would clean up project files",
                details={"dry_run": True},
            )

        cleanup_component = self.components.get("file_scanner")
        if cleanup_component:
            try:
                result = cleanup_component.scan_and_clean(self.project_root)
                files_cleaned = int(result.get("files_cleaned", 0)) if isinstance(result, dict) else 0
                return StageResult(
                    stage_name="cleanup",
                    success=True,
                    duration=0.0,
                    message=f"Cleaned up {files_cleaned} files",
                    details=result if isinstance(result, dict) else {"result": result},
                )
            except Exception as exc:
                return StageResult(
                    stage_name="cleanup",
                    success=False,
                    duration=0.0,
                    message=f"Cleanup failed: {exc}",
                    error=exc,
                )

        return StageResult(
            stage_name="cleanup",
            success=True,
            duration=0.0,
            message="Cleanup stage completed (no cleanup component registered)",
            details={"files_cleaned": 0},
        )

    def _handle_structure_stage(self, config: Dict[str, Any]) -> StageResult:
        """Handle project structure analysis stage."""
        self.logger.info("Executing structure analysis stage...")

        if self.dry_run:
            return StageResult(
                stage_name="structure",
                success=True,
                duration=0.0,
                message="[DRY RUN] Would analyze project structure",
                details={"dry_run": True},
            )

        analyzer = self.components.get("structure_analyzer")
        if analyzer:
            try:
                result = analyzer.analyze_structure(self.project_root)
                directories_count = (
                    int(result.get("directories_count", 0)) if isinstance(result, dict) else 0
                )
                return StageResult(
                    stage_name="structure",
                    success=True,
                    duration=0.0,
                    message=f"Analyzed {directories_count} directories",
                    details=result if isinstance(result, dict) else {"result": result},
                )
            except Exception as exc:
                return StageResult(
                    stage_name="structure",
                    success=False,
                    duration=0.0,
                    message=f"Structure analysis failed: {exc}",
                    error=exc,
                )

        return StageResult(
            stage_name="structure",
            success=True,
            duration=0.0,
            message="Structure analysis completed (no analyzer component registered)",
            details={"directories_count": 0},
        )

    def _handle_modules_stage(self, config: Dict[str, Any]) -> StageResult:
        """Handle module registry creation stage."""
        self.logger.info("Executing module registry stage...")

        if self.dry_run:
            return StageResult(
                stage_name="modules",
                success=True,
                duration=0.0,
                message="[DRY RUN] Would create module registry",
                details={"dry_run": True},
            )

        scanner = self.components.get("module_scanner")
        if scanner:
            try:
                result = scanner.scan_and_register(self.project_root)
                modules_count = int(result.get("modules_count", 0)) if isinstance(result, dict) else 0
                return StageResult(
                    stage_name="modules",
                    success=True,
                    duration=0.0,
                    message=f"Registered {modules_count} modules",
                    details=result if isinstance(result, dict) else {"result": result},
                )
            except Exception as exc:
                return StageResult(
                    stage_name="modules",
                    success=False,
                    duration=0.0,
                    message=f"Module scan failed: {exc}",
                    error=exc,
                )

        return StageResult(
            stage_name="modules",
            success=True,
            duration=0.0,
            message="Module registry completed (no scanner component registered)",
            details={"modules_count": 0},
        )

    def _handle_context_stage(self, config: Dict[str, Any]) -> StageResult:
        """Handle context generation stage."""
        self.logger.info("Executing context generation stage...")

        if self.dry_run:
            return StageResult(
                stage_name="context",
                success=True,
                duration=0.0,
                message="[DRY RUN] Would generate context",
                details={"dry_run": True},
            )

        generator = self.components.get("context_generator")
        if generator:
            try:
                result = generator.generate_context(self.project_root)
                return StageResult(
                    stage_name="context",
                    success=True,
                    duration=0.0,
                    message="Generated project context",
                    details=result if isinstance(result, dict) else {"result": result},
                )
            except Exception as exc:
                return StageResult(
                    stage_name="context",
                    success=False,
                    duration=0.0,
                    message=f"Context generation failed: {exc}",
                    error=exc,
                )

        return StageResult(
            stage_name="context",
            success=True,
            duration=0.0,
            message="Context generation completed (no generator component registered)",
            details={"context_generated": True},
        )

    def _build_index_if_needed(self, db_path: Path) -> Dict[str, Any]:
        """
        Ensure index exists for visualization.

        Returns:
            dict with keys: built(bool), existed(bool), details(dict)
        """
        existed = db_path.exists()
        if existed:
            return {"built": False, "existed": True, "details": {}}

        # Lazy import to avoid heavy deps and circular imports.
        from ..analysis.index_builder import IndexBuilder

        db_path.parent.mkdir(parents=True, exist_ok=True)

        builder = IndexBuilder(db_path)
        try:
            build_result = builder.build_index(self.project_root, incremental=False)

            # Normalize possible return types.
            if isinstance(build_result, dict):
                success = bool(build_result.get("success", True))
                details = build_result
            else:
                success = bool(getattr(build_result, "success", True))
                details = {"result": str(build_result)}

            return {"built": success, "existed": False, "details": details}
        finally:
            close = getattr(builder, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    # Don't fail the stage because of close issues.
                    pass

    def _generate_visualizations(self, output_dir: Path, entry_point: Optional[str]) -> Dict[str, Any]:
        """
        Generate and save visualization diagrams.

        Raises:
            Any exception from underlying visualization components.
        """
        from ..analysis.index_store import IndexStore
        from ..visualization.diagram_generator import FlowchartGenerator

        output_dir.mkdir(parents=True, exist_ok=True)

        db_path = self.project_root / ".intellirefactor" / "index.db"
        index_info = self._build_index_if_needed(db_path)

        if not db_path.exists():
            raise FileNotFoundError(f"Index DB not found at {db_path}")

        store = IndexStore(db_path)
        visualizer = FlowchartGenerator(store)

        files_generated: List[str] = []

        structure_diagram = visualizer.generate_project_diagram(self.project_root.name)
        structure_file = output_dir / "ARCHITECTURE_DIAGRAM.md"
        visualizer.save_diagram(structure_diagram, structure_file)
        files_generated.append(structure_file.name)

        if entry_point:
            # Pretty relative entry point if possible
            entry_str = entry_point
            try:
                entry_str = str(Path(entry_point).relative_to(self.project_root))
            except Exception:
                pass

            flow_diagram = visualizer.generate_execution_flow(entry_str)
            flow_file = output_dir / "EXECUTION_FLOW.md"
            visualizer.save_diagram(flow_diagram, flow_file)
            files_generated.append(flow_file.name)

        return {
            "output_dir": str(output_dir),
            "files_generated": files_generated,
            "index": index_info,
        }

    def run_visualization_stage(self, output_dir: Path, entry_point: Optional[str] = None) -> StageResult:
        """
        Runs the visualization generation stage.

        Args:
            output_dir: Directory where diagrams will be saved.
            entry_point: Optional entry point for execution flow diagram.

        Returns:
            StageResult.
        """
        started = time.time()
        self.logger.info("Executing visualization generation stage...")

        if self.dry_run:
            return StageResult(
                stage_name="visualization",
                success=True,
                duration=0.0,
                message="[DRY RUN] Would generate visualizations",
                details={"dry_run": True, "output_dir": str(output_dir)},
            )

        try:
            details = self._generate_visualizations(Path(output_dir), entry_point)
            return StageResult(
                stage_name="visualization",
                success=True,
                duration=time.time() - started,
                message=f"Generated visualizations in {output_dir}",
                details=details,
            )
        except Exception as exc:
            self.logger.exception("Error in visualization stage")
            return StageResult(
                stage_name="visualization",
                success=False,
                duration=time.time() - started,
                message=f"Error generating visualizations: {exc}",
                error=exc,
                details={"output_dir": str(output_dir)},
            )

    def _handle_visualization_stage(self, config: Dict[str, Any]) -> StageResult:
        """Handle visualization generation stage."""
        self.logger.info("Executing visualization generation stage...")

        if self.dry_run:
            return StageResult(
                stage_name="visualization",
                success=True,
                duration=0.0,
                message="[DRY RUN] Would generate visualizations",
                details={"dry_run": True},
            )

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = self.project_root.name
        output_dir = self.project_root / "reports" / f"{project_name}_{timestamp}"

        entry_point = config.get("entry_point")

        # If a custom generator is registered, use it.
        generator = self.components.get("visualization_generator")
        if generator:
            try:
                result = generator.generate_visualization(self.project_root, output_dir)
                return StageResult(
                    stage_name="visualization",
                    success=True,
                    duration=0.0,
                    message=f"Generated visualizations in {output_dir}",
                    details=result if isinstance(result, dict) else {"result": result},
                )
            except Exception as exc:
                return StageResult(
                    stage_name="visualization",
                    success=False,
                    duration=0.0,
                    message=f"Custom visualization generator failed: {exc}",
                    error=exc,
                )

        # Default visualization
        return self.run_visualization_stage(output_dir=output_dir, entry_point=entry_point)

    def _print_final_report(self, report: RefactoringReport) -> None:
        """Print final refactoring report."""
        self.logger.info("=" * 60)
        self.logger.info("REFACTORING FINAL REPORT")
        self.logger.info("=" * 60)

        self.logger.info("Start time: %s", report.start_time.strftime("%Y-%m-%d %H:%M:%S"))
        self.logger.info("End time: %s", report.end_time.strftime("%Y-%m-%d %H:%M:%S"))
        self.logger.info("Total time: %s", self._format_duration(report.total_duration))
        self.logger.info("Overall result: %s", "SUCCESS" if report.overall_success else "ERROR")
        self.logger.info("Successful stages: %s/%s", report.successful_stages, len(report.stages))

        self.logger.info("Stage details:")
        for i, stage in enumerate(report.stages, 1):
            status = "OK" if stage.success else "FAIL"
            self.logger.info("%s. [%s] %s", i, status, stage.stage_name)
            self.logger.info("   Time: %s", self._format_duration(stage.duration))
            self.logger.info("   Result: %s", stage.message)

            for key, value in (stage.details or {}).items():
                if key != "dry_run":
                    self.logger.info("   %s: %s", key, value)

            if stage.error:
                self.logger.error("   Error: %s", stage.error)

        if report.overall_success:
            self.logger.info("Global refactoring completed successfully!")
        else:
            self.logger.error("Refactoring completed with errors. Check logs for details.")

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        if seconds < 3600:
            return f"{(seconds / 60):.1f}m"
        return f"{(seconds / 3600):.1f}h"

    def orchestrate_refactoring(self, refactoring_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate a complex multi-step refactoring workflow.

        Args:
            refactoring_plan: Plan describing the refactoring workflow.

        Returns:
            Orchestration result.
        """
        self.logger.info("Starting refactoring orchestration")

        try:
            stages = refactoring_plan.get("stages", [])
            if stages is None:
                stages = []

            report = self.run_all_stages(stages)

            return {
                "status": "completed" if report.overall_success else "failed",
                "report": {
                    "start_time": report.start_time.isoformat(),
                    "end_time": report.end_time.isoformat(),
                    "total_duration": report.total_duration,
                    "successful_stages": report.successful_stages,
                    "failed_stages": report.failed_stages,
                    "overall_success": report.overall_success,
                },
                "stages": [
                    {
                        "name": stage.stage_name,
                        "success": stage.success,
                        "duration": stage.duration,
                        "message": stage.message,
                        "details": stage.details or {},
                    }
                    for stage in report.stages
                ],
            }

        except Exception as exc:
            self.logger.exception("Orchestration failed")
            return {
                "status": "error",
                "error": str(exc),
                "message": "Refactoring orchestration failed",
            }
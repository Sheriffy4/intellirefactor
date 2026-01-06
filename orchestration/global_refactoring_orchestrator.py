"""
Global Refactoring Orchestrator for IntelliRefactor

Coordinates complex multi-step refactoring workflows with logging and error handling.
Extracted and adapted from the recon project to work with any Python project.

Stages:
1. Project cleanup (log files, temporary files, debug scripts)
2. Project structure analysis (creating PROJECT_STRUCTURE.md)
3. Module registry creation (creating MODULE_REGISTRY.md)
4. Context generation for analysis tools
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

from ..config import RefactoringConfig


@dataclass
class StageResult:
    """Result of executing a refactoring stage."""
    stage_name: str
    success: bool
    duration: float
    message: str
    details: Dict[str, Any] = None
    error: Optional[Exception] = None


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
        return len([s for s in self.stages if s.success])
    
    @property
    def failed_stages(self) -> int:
        return len([s for s in self.stages if not s.success])


class GlobalRefactoringOrchestrator:
    """Orchestrator for executing all stages of global refactoring."""
    
    def __init__(self, project_root: Path = None, dry_run: bool = False, config: Optional[RefactoringConfig] = None):
        """
        Initialize the orchestrator.
        
        Args:
            project_root: Root directory of the project
            dry_run: Dry run mode - show what would be done without executing
            config: Configuration for refactoring operations
        """
        self.project_root = project_root or Path.cwd()
        self.dry_run = dry_run
        self.config = config or RefactoringConfig()
        self.logger = self._setup_logging()
        
        # Initialize components - these will be injected or created based on config
        self.components = {}
        
        self.logger.info(f"Initialized refactoring orchestrator for {self.project_root}")
        if self.dry_run:
            self.logger.info("DRY RUN MODE - changes will not be applied")
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger('refactoring_orchestrator')
        logger.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        # Ensure project root exists before creating log file
        if not self.project_root.exists():
            try:
                self.project_root.mkdir(parents=True, exist_ok=True)
            except Exception:
                # Fallback to current directory if project root creation fails
                pass
                
        try:
            log_file = self.project_root / f"refactoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}")
        
        return logger
    
    def register_component(self, name: str, component: Any) -> None:
        """Register a component for use in orchestration stages."""
        self.components[name] = component
        self.logger.debug(f"Registered component: {name}")
    
    def analyze_and_refactor_file(self, file_path: Path, project_root: Path = None) -> StageResult:
        """
        Analyze and refactor a single file.
        
        Args:
            file_path: Path to the file
            project_root: Project root (optional)
            
        Returns:
            StageResult indicating success/failure
        """
        # Lazy import to avoid circular dependencies
        from ..refactoring.auto_refactor import AutoRefactor
        
        start_time = time.time()
        file_path = Path(file_path)
        if project_root:
            self.project_root = Path(project_root)
            
        self.logger.info(f"Analyzing and refactoring file: {file_path}")
        
        try:
            if not file_path.exists():
                return StageResult(
                    stage_name="file_refactoring",
                    success=False,
                    duration=0,
                    message=f"File not found: {file_path}"
                )

            # Initialize AutoRefactor
            refactorer = AutoRefactor(self.config)
            
            # Analyze
            plan = refactorer.analyze_god_object(file_path)
            
            if not plan.transformations:
                self.logger.info(f"No refactoring opportunities found for {file_path}")
                return StageResult(
                    stage_name="file_refactoring",
                    success=True,
                    duration=time.time() - start_time,
                    message="No refactoring needed",
                    details={'file': str(file_path)}
                )
            
            self.logger.info(f"Generated plan: {len(plan.transformations)} transformations, Risk: {plan.risk_level}")
            
            # Execute
            if not self.dry_run:
                result = refactorer.execute_refactoring(file_path, plan, dry_run=False)
                success = result.get('success', False)
                message = "Refactoring successful" if success else "Refactoring failed"
                if result.get('errors'):
                    message += f": {', '.join(result['errors'])}"
                
                return StageResult(
                    stage_name="file_refactoring",
                    success=success,
                    duration=time.time() - start_time,
                    message=message,
                    details=result
                )
            else:
                self.logger.info("Dry run - skipping execution")
                return StageResult(
                    stage_name="file_refactoring",
                    success=True,
                    duration=time.time() - start_time,
                    message="Dry run completed",
                    details={'plan': str(plan)}
                )
                
        except Exception as e:
            self.logger.error(f"Error refactoring file {file_path}: {e}")
            return StageResult(
                stage_name="file_refactoring",
                success=False,
                duration=time.time() - start_time,
                message=f"Exception: {str(e)}",
                error=e
            )

    def run_all_stages(self, stage_configs: List[Dict[str, Any]] = None) -> RefactoringReport:
        """
        Execute all refactoring stages.
        
        Args:
            stage_configs: Optional list of stage configurations
            
        Returns:
            RefactoringReport with results of all stages
        """
        start_time = datetime.now()
        self.logger.info("="*60)
        self.logger.info("STARTING GLOBAL REFACTORING")
        self.logger.info("="*60)
        
        stages = []
        
        # Default stages if none provided
        if stage_configs is None:
            stage_configs = [
                {"name": "cleanup", "description": "Project cleanup"},
                {"name": "structure", "description": "Project structure analysis"},
                {"name": "modules", "description": "Module registry creation"},
                {"name": "context", "description": "Context generation"},
                {"name": "visualization", "description": "Visualization generation"}
            ]
        
        # Execute each stage
        for i, stage_config in enumerate(stage_configs, 1):
            stage_result = self._run_stage(i, stage_config)
            stages.append(stage_result)
            
            # Stop on failure if configured to do so
            if not stage_result.success and self.config.stop_on_failure:
                self.logger.error(f"Stopping execution due to failure in {stage_result.stage_name}")
                break
        
        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        overall_success = all(stage.success for stage in stages)
        
        report = RefactoringReport(
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            stages=stages,
            overall_success=overall_success
        )
        
        self._print_final_report(report)
        return report
    
    def run_visualization_stage(self, output_dir: Path, entry_point: Optional[str] = None):
        """Runs the visualization generation stage."""
        # Импорты лучше делать внутри метода, чтобы избежать циклических зависимостей
        from ..visualization.diagram_generator import FlowchartGenerator
        from ..analysis.index_store import IndexStore
        from ..analysis.index_builder import IndexBuilder
        
        self.logger.info("Executing visualization generation stage...")
        
        # Гарантируем создание директории для БД
        db_dir = self.project_root / '.intellirefactor'
        db_dir.mkdir(parents=True, exist_ok=True)
        
        db_path = db_dir / 'index.db'
        
        # Если индекса нет, строим его
        if not db_path.exists():
            self.logger.info("No index.db found, building index for visualization...")
            try:
                # Инициализируем билдер
                builder = IndexBuilder(db_path)
                # Запускаем построение
                builder.build_index(self.project_root, incremental=False)
                # Важно: закрываем соединение билдера, чтобы IndexStore мог открыть своё
                builder.close()
            except Exception as e:
                self.logger.error(f"Failed to build index: {e}")
                return

        try:
            store = IndexStore(db_path)
            visualizer = FlowchartGenerator(store)
            
            # 1. General Project Structure
            structure_diagram = visualizer.generate_project_diagram(self.project_root.name)
            visualizer.save_diagram(structure_diagram, output_dir / "ARCHITECTURE_DIAGRAM.md")
            
            # 2. Execution Flow (if entry point provided)
            if entry_point:
                # Убираем абсолютный путь, если он передан, оставляем относительный для красоты
                try:
                    rel_entry = Path(entry_point).relative_to(self.project_root)
                    entry_str = str(rel_entry)
                except ValueError:
                    entry_str = entry_point

                flow_diagram = visualizer.generate_execution_flow(entry_str)
                visualizer.save_diagram(flow_diagram, output_dir / "EXECUTION_FLOW.md")
                
            self.logger.info(f"Diagrams saved to {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error in visualization stage: {e}")
    
    def run_single_stage(self, stage_number: int, stage_config: Dict[str, Any] = None) -> StageResult:
        """
        Execute a single refactoring stage.
        
        Args:
            stage_number: Stage number
            stage_config: Configuration for the stage
            
        Returns:
            StageResult with execution result
        """
        self.logger.info(f"Running stage {stage_number}")
        
        if stage_config is None:
            # Default stage configurations
            default_configs = {
                1: {"name": "cleanup", "description": "Project cleanup"},
                2: {"name": "structure", "description": "Project structure analysis"},
                3: {"name": "modules", "description": "Module registry creation"},
                4: {"name": "context", "description": "Context generation"}
            }
            stage_config = default_configs.get(stage_number, {"name": "unknown", "description": "Unknown stage"})
        
        return self._run_stage(stage_number, stage_config)
    
    def _run_stage(self, stage_number: int, stage_config: Dict[str, Any]) -> StageResult:
        """Execute a single stage with error handling."""
        stage_name = f"Stage {stage_number}: {stage_config.get('description', 'Unknown')}"
        start_time = time.time()
        
        try:
            self.logger.info(f"🔄 {stage_name}")
            
            # Get the appropriate handler for this stage
            handler_name = stage_config.get('name', 'unknown')
            handler = getattr(self, f'_handle_{handler_name}_stage', None)
            
            if handler is None:
                return StageResult(
                    stage_name=stage_name,
                    success=False,
                    duration=time.time() - start_time,
                    message=f"No handler found for stage: {handler_name}",
                    error=ValueError(f"Unknown stage handler: {handler_name}")
                )
            
            # Execute the stage handler
            result = handler(stage_config)
            
            # Update timing
            result.duration = time.time() - start_time
            result.stage_name = stage_name
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in {stage_name}: {e}")
            return StageResult(
                stage_name=stage_name,
                success=False,
                duration=time.time() - start_time,
                message=f"Error: {str(e)}",
                error=e
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
                details={'dry_run': True}
            )
        
        # Use registered cleanup component if available
        cleanup_component = self.components.get('file_scanner')
        if cleanup_component:
            # Execute cleanup using the component
            result = cleanup_component.scan_and_clean(self.project_root)
            return StageResult(
                stage_name="cleanup",
                success=True,
                duration=0.0,
                message=f"Cleaned up {result.get('files_cleaned', 0)} files",
                details=result
            )
        
        # Default cleanup behavior
        return StageResult(
            stage_name="cleanup",
            success=True,
            duration=0.0,
            message="Cleanup stage completed (no cleanup component registered)",
            details={'files_cleaned': 0}
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
                details={'dry_run': True}
            )
        
        # Use registered structure analyzer if available
        analyzer = self.components.get('structure_analyzer')
        if analyzer:
            result = analyzer.analyze_structure(self.project_root)
            return StageResult(
                stage_name="structure",
                success=True,
                duration=0.0,
                message=f"Analyzed {result.get('directories_count', 0)} directories",
                details=result
            )
        
        # Default structure analysis
        return StageResult(
            stage_name="structure",
            success=True,
            duration=0.0,
            message="Structure analysis completed (no analyzer component registered)",
            details={'directories_count': 0}
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
                details={'dry_run': True}
            )
        
        # Use registered module scanner if available
        scanner = self.components.get('module_scanner')
        if scanner:
            result = scanner.scan_and_register(self.project_root)
            return StageResult(
                stage_name="modules",
                success=True,
                duration=0.0,
                message=f"Registered {result.get('modules_count', 0)} modules",
                details=result
            )
        
        # Default module registry
        return StageResult(
            stage_name="modules",
            success=True,
            duration=0.0,
            message="Module registry completed (no scanner component registered)",
            details={'modules_count': 0}
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
                details={'dry_run': True}
            )
        
        # Use registered context generator if available
        generator = self.components.get('context_generator')
        if generator:
            result = generator.generate_context(self.project_root)
            return StageResult(
                stage_name="context",
                success=True,
                duration=0.0,
                message="Generated project context",
                details=result
            )
        
        # Default context generation
        return StageResult(
            stage_name="context",
            success=True,
            duration=0.0,
            message="Context generation completed (no generator component registered)",
            details={'context_generated': True}
        )
    
    def _handle_visualization_stage(self, config: Dict[str, Any]) -> StageResult:
        """Handle visualization generation stage."""
        from datetime import datetime
        from pathlib import Path
        import os
        
        self.logger.info("Executing visualization generation stage...")
        
        if self.dry_run:
            return StageResult(
                stage_name="visualization",
                success=True,
                duration=0.0,
                message="[DRY RUN] Would generate visualizations",
                details={'dry_run': True}
            )
        
        try:
            # Create output directory for analysis results
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            project_name = self.project_root.name
            output_dir = self.project_root / "reports" / f"{project_name}_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Use registered visualization generator if available
            generator = self.components.get('visualization_generator')
            if generator:
                result = generator.generate_visualization(self.project_root, output_dir)
                return StageResult(
                    stage_name="visualization",
                    success=True,
                    duration=0.0,
                    message=f"Generated visualizations in {output_dir}",
                    details=result
                )
            
            # Import visualization module and generate diagrams
            from ..visualization.diagram_generator import FlowchartGenerator
            from ..analysis.index_store import IndexStore
            
            # Try to create visualization using index store
            db_path = self.project_root / '.intellirefactor' / 'index.db'
            if db_path.exists():
                store = IndexStore(db_path)
                visualizer = FlowchartGenerator(store)
                
                # 1. General Project Structure
                structure_diagram = visualizer.generate_project_diagram(project_name)
                structure_file = output_dir / "ARCHITECTURE_DIAGRAM.md"
                visualizer.save_diagram(structure_diagram, structure_file)
                
                # 2. Execution Flow (if entry point provided in config)
                entry_point = config.get('entry_point')
                if entry_point:
                    flow_diagram = visualizer.generate_execution_flow(entry_point)
                    flow_file = output_dir / "EXECUTION_FLOW.md"
                    visualizer.save_diagram(flow_diagram, flow_file)
                
                return StageResult(
                    stage_name="visualization",
                    success=True,
                    duration=0.0,
                    message=f"Generated visualizations in {output_dir}",
                    details={
                        'output_dir': str(output_dir),
                        'files_generated': ['ARCHITECTURE_DIAGRAM.md', 'EXECUTION_FLOW.md' if entry_point else '']
                    }
                )
            else:
                # If no index exists, try to build it first
                try:
                    from ..analysis.index_builder import IndexBuilder
                    
                    self.logger.info(f"No index.db found, building index for visualization...")
                    builder = IndexBuilder(self.project_root)
                    build_result = builder.build_index(self.project_root)
                    
                    if build_result.success:
                        # Now try to create visualizations
                        store = IndexStore(db_path)
                        visualizer = FlowchartGenerator(store)
                        
                        # 1. General Project Structure
                        structure_diagram = visualizer.generate_project_diagram(project_name)
                        structure_file = output_dir / "ARCHITECTURE_DIAGRAM.md"
                        visualizer.save_diagram(structure_diagram, structure_file)
                        
                        # 2. Execution Flow (if entry point provided in config)
                        entry_point = config.get('entry_point')
                        if entry_point:
                            flow_diagram = visualizer.generate_execution_flow(entry_point)
                            flow_file = output_dir / "EXECUTION_FLOW.md"
                            visualizer.save_diagram(flow_diagram, flow_file)
                        
                        return StageResult(
                            stage_name="visualization",
                            success=True,
                            duration=0.0,
                            message=f"Generated visualizations in {output_dir}",
                            details={
                                'output_dir': str(output_dir),
                                'files_generated': ['ARCHITECTURE_DIAGRAM.md', 'EXECUTION_FLOW.md' if entry_point else '']
                            }
                        )
                    else:
                        return StageResult(
                            stage_name="visualization",
                            success=False,
                            duration=0.0,
                            message="Failed to build index for visualizations",
                            details={'output_dir': str(output_dir), 'index_exists': False, 'build_result': build_result}
                        )
                except ImportError:
                    # If index builder is not available, return with warning
                    return StageResult(
                        stage_name="visualization",
                        success=False,
                        duration=0.0,
                        message="No index.db found and index builder unavailable, cannot generate visualizations",
                        details={'output_dir': str(output_dir), 'index_exists': False, 'index_builder_available': False}
                    )
        
        except Exception as e:
            self.logger.error(f"Error in visualization stage: {e}")
            return StageResult(
                stage_name="visualization",
                success=False,
                duration=0.0,
                message=f"Error generating visualizations: {str(e)}",
                error=e
            )
    
    def _print_final_report(self, report: RefactoringReport) -> None:
        """Print final refactoring report."""
        self.logger.info("="*60)
        self.logger.info("REFACTORING FINAL REPORT")
        self.logger.info("="*60)
        
        self.logger.info(f"Start time: {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"End time: {report.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Total time: {self._format_duration(report.total_duration)}")
        self.logger.info(f"Overall result: {'✅ SUCCESS' if report.overall_success else '❌ ERROR'}")
        self.logger.info(f"Successful stages: {report.successful_stages}/{len(report.stages)}")
        
        self.logger.info("\nStage details:")
        for i, stage in enumerate(report.stages, 1):
            status = "✅" if stage.success else "❌"
            self.logger.info(f"{i}. {status} {stage.stage_name}")
            self.logger.info(f"   Time: {self._format_duration(stage.duration)}")
            self.logger.info(f"   Result: {stage.message}")
            
            if stage.details:
                for key, value in stage.details.items():
                    if key != 'dry_run':
                        self.logger.info(f"   {key}: {value}")
            
            if stage.error:
                self.logger.error(f"   Error: {stage.error}")
        
        if report.overall_success:
            self.logger.info("\n🎉 Global refactoring completed successfully!")
        else:
            self.logger.error("\n❌ Refactoring completed with errors")
            self.logger.error("Check logs for detailed information")
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable form."""
        if seconds < 60:
            return f"{seconds:.1f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"
    
    def orchestrate_refactoring(self, refactoring_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Orchestrate a complex multi-step refactoring workflow.
        
        Args:
            refactoring_plan: Plan describing the refactoring workflow
            
        Returns:
            Orchestration result
        """
        self.logger.info("Starting refactoring orchestration")
        
        try:
            # Extract stages from the plan
            stages = refactoring_plan.get('stages', [])
            
            # Execute the refactoring workflow
            report = self.run_all_stages(stages)
            
            return {
                "status": "completed" if report.overall_success else "failed",
                "report": {
                    "start_time": report.start_time.isoformat(),
                    "end_time": report.end_time.isoformat(),
                    "total_duration": report.total_duration,
                    "successful_stages": report.successful_stages,
                    "failed_stages": report.failed_stages,
                    "overall_success": report.overall_success
                },
                "stages": [
                    {
                        "name": stage.stage_name,
                        "success": stage.success,
                        "duration": stage.duration,
                        "message": stage.message,
                        "details": stage.details or {}
                    }
                    for stage in report.stages
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Orchestration failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "message": "Refactoring orchestration failed"
            }
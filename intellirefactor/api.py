"""
Main API interface for IntelliRefactor

Provides a unified facade for all IntelliRefactor functionality.
This module implements the facade pattern to hide internal complexity
and provide a clean, simple interface for all IntelliRefactor operations.
"""

import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

from .analysis import ProjectAnalyzer, FileAnalyzer
from .analysis.error_handler import AnalysisErrorHandler, ErrorReporter
from .refactoring import IntelligentRefactoringSystem, AutoRefactor
from .knowledge import KnowledgeManager
from .orchestration import (
    GlobalRefactoringOrchestrator,
    RefactoringValidator,
    RefactoringReporter,
)
from .safety import SafetyManager
from .config import IntelliRefactorConfig

# Configure logging
logger = logging.getLogger(__name__)


@dataclass
class AnalysisResult:
    """Standardized analysis result structure."""

    success: bool
    data: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


@dataclass
class RefactoringResult:
    """Standardized refactoring result structure."""

    success: bool
    operations_applied: int
    changes_made: List[Dict[str, Any]]
    validation_results: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    metadata: Dict[str, Any]


class IntelliRefactor:
    """
    Main API class for IntelliRefactor system.

    This class implements the facade pattern to provide a unified, simplified
    interface for all IntelliRefactor functionality.
    """

    def __init__(self, config: Optional[IntelliRefactorConfig] = None):
        """
        Initialize IntelliRefactor with optional configuration.

        Args:
            config: Optional configuration object. If None, uses default configuration.
        """
        self.config = config or IntelliRefactorConfig.default()
        self._initialized = False
        self._components = {}

        # Initialize error handling system
        self.error_handler = AnalysisErrorHandler()
        self.error_reporter = ErrorReporter()

        # Lazy initialization of components
        self._initialize_components()

        logger.info("IntelliRefactor initialized with configuration")

    def _initialize_components(self) -> None:
        """Initialize all core components with error handling."""
        try:
            self._components["project_analyzer"] = ProjectAnalyzer(self.config.analysis_settings)
            self._components["file_analyzer"] = FileAnalyzer(self.config.analysis_settings)
            self._components["refactoring_system"] = IntelligentRefactoringSystem(
                self.config.refactoring_settings
            )
            self._components["auto_refactor"] = AutoRefactor(self.config.refactoring_settings)
            self._components["knowledge_manager"] = KnowledgeManager(self.config.knowledge_settings)
            self._components["validator"] = RefactoringValidator(
                project_root=None, config=self.config.refactoring_settings
            )
            self._components["reporter"] = RefactoringReporter(
                project_root=None, config=self.config.refactoring_settings
            )
            self._components["safety_manager"] = SafetyManager(self.config.refactoring_settings)
            self._components["orchestrator"] = GlobalRefactoringOrchestrator(
                project_root=None,  # Will use current working directory
                dry_run=False,
                config=self.config.refactoring_settings,
            )
            self._initialized = True
            logger.debug("All components initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise RuntimeError(f"IntelliRefactor initialization failed: {e}")

    @property
    def is_initialized(self) -> bool:
        """Check if the system is properly initialized."""
        return self._initialized

    def initialize_project(self, project_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Initialize IntelliRefactor for a specific project.
        Sets up safety directories, knowledge base context, etc.

        Args:
            project_path: Path to the project directory

        Returns:
            Dictionary containing initialization status
        """
        if not self._initialized:
            raise RuntimeError("IntelliRefactor system not initialized")

        project_path = str(project_path)
        results = {
            "project_path": project_path,
            "timestamp": datetime.now().isoformat(),
            "components": {},
        }

        try:
            # Initialize Safety Manager (creates .intellirefactor/safety)
            if "safety_manager" in self._components:
                results["components"]["safety"] = self._components[
                    "safety_manager"
                ].initialize_project(project_path)

            # Initialize Knowledge Manager
            if "knowledge_manager" in self._components:
                km = self._components["knowledge_manager"]
                # Check if method exists (it might be added in extensions)
                if hasattr(km, "initialize_project_knowledge"):
                    results["components"]["knowledge"] = km.initialize_project_knowledge(
                        project_path
                    )
                else:
                    # Default behavior if specific init not needed
                    results["components"]["knowledge"] = {
                        "status": "ready",
                        "path": km.knowledge_dir,
                    }

            logger.info(f"Project initialized at {project_path}")
            return results
        except Exception as e:
            logger.error(f"Project initialization failed: {e}")
            raise

    def initialize(self, project_path: Union[str, Path]) -> Dict[str, Any]:
        """Alias for initialize_project."""
        return self.initialize_project(project_path)

    # Analysis Methods

    def analyze_project(
        self,
        project_path: Union[str, Path],
        include_metrics: bool = True,
        include_opportunities: bool = True,
    ) -> AnalysisResult:
        """
        Analyze a project and return comprehensive analysis results.
        """
        if not self._initialized:
            error = self.error_handler.handle_analysis_error(
                RuntimeError("System not initialized"),
                "project_analysis_initialization",
            )
            return AnalysisResult(
                False, {}, [error.message], [], {"error_details": error.to_dict()}
            )

        try:
            project_path = Path(project_path)
            if not project_path.exists():
                error = self.error_handler.handle_analysis_error(
                    FileNotFoundError(f"Project path does not exist: {project_path}"),
                    "project_analysis_path_validation",
                    str(project_path),
                )
                return AnalysisResult(
                    False, {}, [error.message], [], {"error_details": error.to_dict()}
                )

            logger.info(f"Starting project analysis for: {project_path}")

            # Perform basic analysis
            project_analysis_result = self._components["project_analyzer"].analyze_project(
                str(project_path)
            )

            # Extract data from GenericAnalysisResult
            if project_analysis_result.success:
                result_data = project_analysis_result.data

                analysis_data = {
                    "total_files": result_data.get("total_files", 0),
                    "total_lines": result_data.get("total_lines", 0),
                    "large_files": result_data.get("large_files", []),
                    "complex_files": result_data.get("complex_files", []),
                    "god_objects": result_data.get("god_objects", []),
                    "refactoring_candidates": result_data.get("refactoring_candidates", []),
                    "overall_recommendations": result_data.get("overall_recommendations", []),
                    "automation_potential": result_data.get("automation_potential", 0.0),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                analysis_data = {
                    "total_files": 0,
                    "total_lines": 0,
                    "large_files": [],
                    "complex_files": [],
                    "god_objects": [],
                    "refactoring_candidates": [],
                    "overall_recommendations": [],
                    "automation_potential": 0.0,
                    "timestamp": datetime.now().isoformat(),
                }

            # Add opportunities if requested
            if include_opportunities:
                try:
                    raw_opportunities = self._components[
                        "refactoring_system"
                    ].identify_opportunities(analysis_data)
                    # Convert dataclasses to dicts
                    analysis_data["refactoring_opportunities"] = [
                        asdict(opp) for opp in raw_opportunities
                    ]
                except Exception as e:
                    error = self.error_handler.handle_analysis_error(
                        e, "opportunity_identification", str(project_path)
                    )
                    logger.warning(f"Failed to identify opportunities: {error.message}")
                    analysis_data["refactoring_opportunities"] = []

            metadata = {
                "analysis_timestamp": analysis_data["timestamp"],
                "project_path": str(project_path),
                "include_metrics": include_metrics,
                "include_opportunities": include_opportunities,
            }

            return AnalysisResult(True, analysis_data, [], [], metadata)

        except Exception as e:
            error = self.error_handler.handle_analysis_error(
                e, "project_analysis", str(project_path)
            )
            logger.error(f"Project analysis failed: {error.message}")
            return AnalysisResult(
                False, {}, [error.message], [], {"error_details": error.to_dict()}
            )

    def analyze_file(
        self, file_path: Union[str, Path], project_root: Optional[Path] = None
    ) -> AnalysisResult:
        """
        Analyze a single file and return analysis results.
        """
        if not self._initialized:
            error = self.error_handler.handle_analysis_error(
                RuntimeError("System not initialized"), "file_analysis_initialization"
            )
            return AnalysisResult(
                False, {}, [error.message], [], {"error_details": error.to_dict()}
            )

        try:
            file_path = Path(file_path)
            if not file_path.exists():
                error = self.error_handler.handle_analysis_error(
                    FileNotFoundError(f"File does not exist: {file_path}"),
                    "file_analysis_path_validation",
                    str(file_path),
                )
                return AnalysisResult(
                    False, {}, [error.message], [], {"error_details": error.to_dict()}
                )

            logger.info(f"Starting file analysis for: {file_path}")

            # Auto-determine project root if not provided
            if not project_root:
                project_root = self._find_project_root(file_path)

            # Try to load external index if project root is available
            external_index = None
            if project_root:
                index_path = project_root / ".intellirefactor" / "index.db"
                if index_path.exists():
                    try:
                        from .analysis.index_store import IndexStore

                        external_index = IndexStore(index_path)
                    except Exception as e:
                        error = self.error_handler.handle_analysis_error(
                            e, "external_index_loading", str(index_path)
                        )
                        logger.warning(f"Failed to load external index: {error.message}")

            # Pass external index to file analyzer
            file_analysis_result = self._components["file_analyzer"].analyze_file(
                str(file_path), external_context=external_index
            )

            # Extract data from GenericAnalysisResult
            if file_analysis_result.success:
                result_data = file_analysis_result.data

                analysis_data = {
                    "filepath": result_data.get("filepath", str(file_path)),
                    "lines_count": result_data.get("lines_count", 0),
                    "classes": result_data.get("classes", []),
                    "imports_count": result_data.get("imports_count", 0),
                    "complexity_score": file_analysis_result.metrics.get("complexity_score", 0.0),
                    "issues": file_analysis_result.issues,
                    "recommendations": file_analysis_result.recommendations,
                    "refactoring_priority": result_data.get("refactoring_priority", 0),
                    "automation_potential": result_data.get("automation_potential", 0.0),
                    "timestamp": datetime.now().isoformat(),
                }
            else:
                analysis_data = {
                    "filepath": str(file_path),
                    "lines_count": 0,
                    "classes": [],
                    "imports_count": 0,
                    "complexity_score": 0.0,
                    "issues": file_analysis_result.issues,
                    "recommendations": file_analysis_result.recommendations,
                    "refactoring_priority": 0,
                    "automation_potential": 0.0,
                    "timestamp": datetime.now().isoformat(),
                }

            metadata = {
                "analysis_timestamp": analysis_data["timestamp"],
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size if file_path.exists() else 0,
                "project_root": str(project_root) if project_root else None,
                "external_index_used": external_index is not None,
            }

            return AnalysisResult(True, analysis_data, [], [], metadata)

        except Exception as e:
            error = self.error_handler.handle_analysis_error(e, "file_analysis", str(file_path))
            logger.error(f"File analysis failed: {error.message}")
            return AnalysisResult(
                False, {}, [error.message], [], {"error_details": error.to_dict()}
            )

    # Refactoring Methods

    def identify_opportunities(
        self, project_path: Union[str, Path], max_opportunities: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Identify refactoring opportunities in a project."""
        if not self._initialized:
            logger.error("System not initialized")
            return []

        try:
            analysis_result = self.analyze_project(project_path, include_opportunities=True)
            if not analysis_result.success:
                logger.error(f"Failed to analyze project: {analysis_result.errors}")
                return []

            opportunities = analysis_result.data.get("refactoring_opportunities", [])

            if max_opportunities:
                opportunities = opportunities[:max_opportunities]

            logger.info(f"Identified {len(opportunities)} refactoring opportunities")
            return opportunities

        except Exception as e:
            logger.error(f"Failed to identify opportunities: {e}")
            return []

    def apply_refactoring(
        self,
        opportunity: Dict[str, Any],
        dry_run: bool = False,
        validate_results: bool = True,
    ) -> RefactoringResult:
        """Apply a specific refactoring opportunity."""
        if not self._initialized:
            return RefactoringResult(False, 0, [], {}, ["System not initialized"], [], {})

        try:
            logger.info(f"Applying refactoring: {opportunity.get('description', 'Unknown')}")

            # Apply the refactoring
            result = self._components["auto_refactor"].apply_opportunity(
                opportunity, dry_run=dry_run
            )

            # Validate results if requested
            validation_results = {}
            if validate_results and not dry_run:
                validation_results = self._components["validator"].validate_refactoring(result)

            # Learn from the result
            if not dry_run and self.config.knowledge_settings.auto_learn:
                self._components["knowledge_manager"].learn_from_result(result)

            metadata = {
                "opportunity_id": opportunity.get("id"),
                "dry_run": dry_run,
                "validation_enabled": validate_results,
                "timestamp": result.get("timestamp"),
            }

            return RefactoringResult(
                success=result.get("success", False),
                operations_applied=result.get("operations_applied", 0),
                changes_made=result.get("changes", []),
                validation_results=validation_results,
                errors=result.get("errors", []),
                warnings=result.get("warnings", []),
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Refactoring application failed: {e}")
            return RefactoringResult(False, 0, [], {}, [str(e)], [], {})

    def auto_refactor_project(
        self,
        project_path: Union[str, Path],
        strategy: Optional[str] = None,
        max_operations: Optional[int] = None,
        dry_run: bool = False,
    ) -> RefactoringResult:
        """Automatically refactor a project using the specified strategy."""
        if not self._initialized:
            return RefactoringResult(False, 0, [], {}, ["System not initialized"], [], {})

        try:
            logger.info(f"Starting auto-refactoring for project: {project_path}")

            # Set operation limit
            if max_operations:
                original_limit = self.config.refactoring_settings.max_operations_per_session
                self.config.refactoring_settings.max_operations_per_session = max_operations

            result = self._components["auto_refactor"].refactor_project(
                str(project_path), strategy, dry_run=dry_run
            )

            # Restore original limit
            if max_operations:
                self.config.refactoring_settings.max_operations_per_session = original_limit

            metadata = {
                "project_path": str(project_path),
                "strategy": strategy,
                "max_operations": max_operations,
                "dry_run": dry_run,
                "timestamp": result.get("timestamp"),
            }

            return RefactoringResult(
                success=result.get("success", False),
                operations_applied=result.get("operations_applied", 0),
                changes_made=result.get("changes", []),
                validation_results=result.get("validation_results", {}),
                errors=result.get("errors", []),
                warnings=result.get("warnings", []),
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Auto-refactoring failed: {e}")
            return RefactoringResult(False, 0, [], {}, [str(e)], [], {})

    def orchestrate_refactoring(self, refactoring_plan: Dict[str, Any]) -> RefactoringResult:
        """Orchestrate a complex multi-step refactoring workflow."""
        if not self._initialized:
            return RefactoringResult(False, 0, [], {}, ["System not initialized"], [], {})

        try:
            logger.info("Starting refactoring orchestration")

            result = self._components["orchestrator"].orchestrate_refactoring(refactoring_plan)

            metadata = {
                "plan_id": refactoring_plan.get("id"),
                "plan_steps": len(refactoring_plan.get("steps", [])),
                "timestamp": result.get("timestamp"),
            }

            return RefactoringResult(
                success=result.get("success", False),
                operations_applied=result.get("operations_applied", 0),
                changes_made=result.get("changes", []),
                validation_results=result.get("validation_results", {}),
                errors=result.get("errors", []),
                warnings=result.get("warnings", []),
                metadata=metadata,
            )

        except Exception as e:
            logger.error(f"Refactoring orchestration failed: {e}")
            return RefactoringResult(False, 0, [], {}, [str(e)], [], {})

    # Knowledge Management Methods

    def query_knowledge(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Query the knowledge base."""
        if not self._initialized:
            logger.error("System not initialized")
            return []

        try:
            logger.info(f"Querying knowledge base: {query}")
            results = self._components["knowledge_manager"].query_knowledge(query)
            return results[:limit] if results else []

        except Exception as e:
            logger.error(f"Knowledge query failed: {e}")
            return []

    def add_knowledge(self, knowledge_item: Dict[str, Any]) -> bool:
        """Add a new knowledge item to the knowledge base."""
        if not self._initialized:
            logger.error("System not initialized")
            return False

        try:
            return self._components["knowledge_manager"].add_knowledge(knowledge_item)
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}")
            return False

    def learn_from_refactoring(self, refactoring_result: RefactoringResult) -> None:
        """Learn from a refactoring result."""
        if not self._initialized:
            logger.error("System not initialized")
            return

        try:
            # Convert RefactoringResult to dict for knowledge manager
            result_dict = {
                "success": refactoring_result.success,
                "operations_applied": refactoring_result.operations_applied,
                "changes_made": refactoring_result.changes_made,
                "validation_results": refactoring_result.validation_results,
                "errors": refactoring_result.errors,
                "warnings": refactoring_result.warnings,
                "metadata": refactoring_result.metadata,
            }

            self._components["knowledge_manager"].learn_from_result(result_dict)
            logger.info("Learning from refactoring result completed")

        except Exception as e:
            logger.error(f"Failed to learn from refactoring: {e}")

    # Utility Methods

    def generate_report(self, results: List[RefactoringResult], output_format: str = "text") -> str:
        """Generate a comprehensive report from refactoring results."""
        if not self._initialized:
            return "Error: System not initialized"

        try:
            # Convert RefactoringResult objects to dicts
            result_dicts = []
            for result in results:
                result_dict = {
                    "success": result.success,
                    "operations_applied": result.operations_applied,
                    "changes_made": result.changes_made,
                    "validation_results": result.validation_results,
                    "errors": result.errors,
                    "warnings": result.warnings,
                    "metadata": result.metadata,
                }
                result_dicts.append(result_dict)

            return self._components["reporter"].generate_report(result_dicts, output_format)

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return f"Error generating report: {e}"

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and health information."""
        status = {
            "initialized": self._initialized,
            "components": {},
            "configuration": {
                "safety_level": self.config.refactoring_settings.safety_level.value,
                "auto_apply": self.config.refactoring_settings.auto_apply,
                "knowledge_path": self.config.knowledge_settings.knowledge_base_path,
            },
            "error_handling": {
                "handler_available": self.error_handler is not None,
                "reporter_available": self.error_reporter is not None,
            },
        }

        if self._initialized:
            for name, component in self._components.items():
                try:
                    # Try to get component status if available
                    if hasattr(component, "get_status"):
                        status["components"][name] = component.get_status()
                    else:
                        status["components"][name] = "active"
                except Exception as e:
                    status["components"][name] = f"error: {e}"

        return status

    def format_error(self, error_details: Dict[str, Any], format_type: str = "text") -> str:
        """
        Format error details for display.

        Args:
            error_details: Error details dictionary from AnalysisResult
            format_type: Output format ('text', 'json', 'markdown')

        Returns:
            Formatted error string
        """
        if not self.error_reporter:
            return f"Error formatting not available: {error_details}"

        try:
            # Convert dict back to AnalysisError if needed
            from .analysis.error_handler import (
                AnalysisError,
                ErrorCategory,
                ErrorSeverity,
            )

            if isinstance(error_details, dict):
                # Reconstruct AnalysisError from dict
                error = AnalysisError(
                    category=ErrorCategory[error_details.get("category", "UNKNOWN")],
                    severity=ErrorSeverity(error_details.get("severity", "medium")),
                    message=error_details.get("message", "Unknown error"),
                    context=error_details.get("context", "Unknown context"),
                    file_path=error_details.get("file_path"),
                    line_number=error_details.get("line_number"),
                    column_number=error_details.get("column_number"),
                    stack_trace=error_details.get("stack_trace"),
                    suggested_fixes=error_details.get("suggested_fixes", []),
                    diagnostic_info=error_details.get("diagnostic_info", {}),
                    timestamp=error_details.get("timestamp"),
                )
                return self.error_reporter.format_error(error, format_type)
            else:
                return str(error_details)

        except Exception as e:
            return f"Error formatting failed: {e}. Original error: {error_details}"

    def get_error_summary(self, analysis_results: List[AnalysisResult]) -> Dict[str, Any]:
        """
        Generate a summary of errors from multiple analysis results.

        Args:
            analysis_results: List of AnalysisResult objects

        Returns:
            Error summary dictionary
        """
        if not self.error_reporter:
            return {"error": "Error reporting not available"}

        try:
            from .analysis.error_handler import (
                AnalysisError,
                ErrorCategory,
                ErrorSeverity,
            )

            errors = []
            for result in analysis_results:
                if not result.success and "error_details" in result.metadata:
                    error_details = result.metadata["error_details"]

                    # Reconstruct AnalysisError from dict
                    error = AnalysisError(
                        category=ErrorCategory[error_details.get("category", "UNKNOWN")],
                        severity=ErrorSeverity(error_details.get("severity", "medium")),
                        message=error_details.get("message", "Unknown error"),
                        context=error_details.get("context", "Unknown context"),
                        file_path=error_details.get("file_path"),
                        line_number=error_details.get("line_number"),
                        column_number=error_details.get("column_number"),
                        stack_trace=error_details.get("stack_trace"),
                        suggested_fixes=error_details.get("suggested_fixes", []),
                        diagnostic_info=error_details.get("diagnostic_info", {}),
                        timestamp=error_details.get("timestamp"),
                    )
                    errors.append(error)

            return self.error_reporter.generate_error_summary(errors)

        except Exception as e:
            return {"error": f"Error summary generation failed: {e}"}

    def _find_project_root(self, file_path: Path) -> Optional[Path]:
        """Auto-determine project root by looking for common project markers."""
        current_dir = file_path.parent

        # Look for common project markers up to 5 levels up
        for _ in range(5):
            # Check for common project files/directories
            project_markers = [
                "setup.py",
                "pyproject.toml",
                "requirements.txt",
                "__init__.py",
                ".git",
                "README.md",
                "Pipfile",
                "poetry.lock",
                "tox.ini",
                "pytest.ini",
                "MANIFEST.in",
                "setup.cfg",
            ]

            for marker in project_markers:
                if (current_dir / marker).exists():
                    return current_dir

            # Move up one directory
            parent_dir = current_dir.parent
            if parent_dir == current_dir:  # Reached root
                break
            current_dir = parent_dir

        # If no markers found, use current directory
        return file_path.parent

    def get_available_plugins(self) -> List[Dict[str, Any]]:
        """Get list of available plugins."""
        if not self._initialized:
            return []

        try:
            # Check if plugin manager exists
            if hasattr(self._components.get("knowledge_manager"), "get_available_plugins"):
                return self._components["knowledge_manager"].get_available_plugins()
            else:
                # Return empty list if no plugin system available
                return []
        except Exception as e:
            logger.error(f"Failed to get available plugins: {e}")
            return []

    def update_configuration(self, new_config: IntelliRefactorConfig) -> bool:
        """Update the system configuration."""
        try:
            self.config = new_config
            # Reinitialize components with new configuration
            self._initialize_components()
            logger.info("Configuration updated successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to update configuration: {e}")
            return False

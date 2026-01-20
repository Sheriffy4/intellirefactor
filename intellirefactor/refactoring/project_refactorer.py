"""Project-level refactoring orchestration.

This module provides the ProjectRefactorer class that orchestrates refactoring
operations across multiple files in a project, managing analysis, planning,
and execution phases.

The ProjectRefactorer handles:
- Discovery of Python files in projects
- Batch analysis of multiple files
- Coordinated execution of refactoring plans
- Result aggregation and reporting
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .auto_refactor import AutoRefactor, RefactoringPlan

logger = logging.getLogger(__name__)


class ProjectRefactorer:
    """Orchestrates refactoring across multiple files in a project.
    
    The ProjectRefactorer provides high-level coordination for refactoring
    operations that span multiple files or entire projects. It handles:
    - File discovery and filtering
    - Parallel analysis planning
    - Sequential execution with error handling
    - Result aggregation and reporting
    
    Attributes:
        analyzer: AutoRefactor instance for analysis and execution
    
    Example:
        >>> from intellirefactor.refactoring import AutoRefactor
        >>> ar = AutoRefactor()
        >>> refactorer = ProjectRefactorer(ar)
        >>> results = refactorer.refactor_project('/path/to/project', dry_run=True)
        >>> print(f"Found {len(results['changes'])} refactoring opportunities")
    """
    
    def __init__(self, analyzer: "AutoRefactor"):
        """Initialize ProjectRefactorer.
        
        Args:
            analyzer: AutoRefactor instance to use for analysis and execution
        """
        self._analyzer = analyzer
    
    def refactor_project(
        self,
        project_path: Union[str, Path],
        strategy: Optional[str] = None,
        dry_run: bool = True,
    ) -> Dict[str, Any]:
        """Refactor all files in a project or directory.
        
        This method orchestrates the complete refactoring workflow:
        1. Discover Python files in the project
        2. Analyze each file for refactoring opportunities
        3. Plan refactoring operations
        4. Execute refactorings (if not dry_run)
        5. Aggregate and return results
        
        Args:
            project_path: Path to project directory or single Python file
            strategy: Reserved for future refactoring strategy selection
            dry_run: If True, only analyze without making changes
            
        Returns:
            Dictionary with refactoring results:
                - success: Overall success status
                - operations_applied: Number of refactorings executed
                - changes: List of planned/executed changes
                - errors: List of error messages
                - warnings: List of warning messages
                - timestamp: ISO format timestamp
                - planned_operations: (dry_run only) Number of planned operations
                
        Example:
            >>> refactorer = ProjectRefactorer(analyzer)
            >>> # Dry run to see what would be done
            >>> results = refactorer.refactor_project('/project', dry_run=True)
            >>> print(f"Would refactor {results['planned_operations']} files")
            >>> 
            >>> # Execute refactoring
            >>> results = refactorer.refactor_project('/project', dry_run=False)
            >>> print(f"Refactored {results['operations_applied']} files")
        """
        _ = strategy  # reserved for future extensions

        project_path = Path(project_path)
        results: Dict[str, Any] = {
            "success": True,
            "operations_applied": 0,
            "changes": [],
            "errors": [],
            "warnings": [],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        # Validate project path
        if not project_path.exists():
            results["success"] = False
            results["errors"].append(f"Project path does not exist: {project_path}")
            return results

        try:
            # Discover Python files
            files_to_analyze = self._collect_python_files(project_path, results)
            if not files_to_analyze:
                return results

            # Analyze files and plan refactorings
            planned_changes = self._analyze_files(files_to_analyze, results)

            if dry_run:
                # In dry-run mode, only return analysis results
                results["operations_applied"] = 0
                results["planned_operations"] = len(planned_changes)
                return results

            # Execute refactorings
            actual_operations = self._execute_planned_changes(planned_changes, results)
            results["operations_applied"] = actual_operations
            
            return results

        except Exception as e:
            results["success"] = False
            results["errors"].append(str(e))
            logger.exception("Project refactoring failed")
            return results
    
    def _collect_python_files(
        self,
        project_path: Path,
        results: Dict[str, Any]
    ) -> List[Path]:
        """Discover Python files to analyze.
        
        Args:
            project_path: Path to project or file
            results: Results dictionary to update with errors
            
        Returns:
            List of Python file paths to analyze
        """
        try:
            if project_path.is_file() and project_path.suffix == ".py":
                # Single Python file
                return [project_path]
            elif project_path.is_dir():
                # Directory - find all Python files recursively
                return list(project_path.rglob("*.py"))
            else:
                results["success"] = False
                results["errors"].append(
                    f"Path must be a Python file or directory: {project_path}"
                )
                return []
        except Exception as e:
            results["errors"].append(f"Failed to collect Python files: {e}")
            logger.error("File collection failed: %s", e)
            return []
    
    def _analyze_files(
        self,
        files: List[Path],
        results: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Analyze files for refactoring opportunities.
        
        Args:
            files: List of Python files to analyze
            results: Results dictionary to update
            
        Returns:
            List of planned changes with refactoring plans
        """
        planned_changes = []
        
        for file_path in files:
            try:
                plan = self._analyzer.analyze_god_object(file_path)
                
                if plan.transformations:
                    change_info = {
                        "file": str(file_path),
                        "target_class": plan.target_class_name,
                        "transformations": plan.transformations,
                        "new_files": plan.new_files,
                        "estimated_effort": plan.estimated_effort,
                        "risk_level": plan.risk_level,
                        "plan": plan,  # Keep the plan for execution
                    }
                    planned_changes.append(change_info)
                    
                    # Add to results (without the plan object for serialization)
                    results["changes"].append({
                        "file": str(file_path),
                        "target_class": plan.target_class_name,
                        "transformations": plan.transformations,
                        "new_files": plan.new_files,
                        "estimated_effort": plan.estimated_effort,
                        "risk_level": plan.risk_level,
                    })
                    
            except Exception as e:
                results["warnings"].append(f"Failed to analyze {file_path}: {e}")
                logger.warning("Analysis failed for %s: %s", file_path, e)
        
        return planned_changes
    
    def _execute_planned_changes(
        self,
        planned_changes: List[Dict[str, Any]],
        results: Dict[str, Any]
    ) -> int:
        """Execute planned refactoring changes.
        
        Args:
            planned_changes: List of planned changes with plans
            results: Results dictionary to update
            
        Returns:
            Number of successfully executed refactorings
        """
        actual_operations = 0
        
        for change_info in planned_changes:
            try:
                file_path = Path(change_info["file"])
                plan = change_info["plan"]

                # Execute refactoring for this file
                execution_result = self._analyzer.execute_refactoring(
                    file_path, plan, dry_run=False
                )

                if execution_result.get("success", False):
                    actual_operations += 1
                    # Update change info with execution results
                    change_info.update({
                        "files_created": execution_result.get("files_created", []),
                        "files_modified": execution_result.get("files_modified", []),
                        "backup_created": execution_result.get("backup_created"),
                    })
                else:
                    results["errors"].extend(execution_result.get("errors", []))
                    results["warnings"].extend(execution_result.get("warnings", []))

            except Exception as e:
                results["errors"].append(
                    f"Failed to execute refactoring for {change_info['file']}: {e}"
                )
                logger.error(
                    "Execution failed for %s: %s",
                    change_info['file'],
                    e
                )
        
        return actual_operations
    
    def apply_opportunity(
        self,
        opportunity: Dict[str, Any],
        dry_run: bool = False
    ) -> Dict[str, Any]:
        """Apply a single refactoring opportunity.
        
        This method provides a convenient interface for applying individual
        refactoring opportunities identified by analysis tools.
        
        Args:
            opportunity: Dictionary describing the opportunity:
                - filepath: Path to file to refactor
                - description: Human-readable description
                - priority: Priority level (optional)
            dry_run: If True, validate without making changes
            
        Returns:
            Dictionary with application results:
                - success: Whether application succeeded
                - operations_applied: Number of operations (0 or 1)
                - changes_made: List of files created/modified
                - errors: List of error messages
                - warnings: List of warning messages
                
        Example:
            >>> opportunity = {
            ...     'filepath': '/path/to/file.py',
            ...     'description': 'Extract god class',
            ...     'priority': 1
            ... }
            >>> result = refactorer.apply_opportunity(opportunity, dry_run=True)
            >>> if result['success']:
            ...     print("Opportunity is valid")
        """
        try:
            description = opportunity.get("description", "Unknown opportunity")
            priority = opportunity.get("priority", 0)
            filepath = opportunity.get("filepath")

            logger.info(
                "Applying opportunity: %s (Priority: %s)", description, priority
            )

            # Validate filepath
            if not filepath:
                return {
                    "success": False,
                    "error": "No filepath specified",
                    "operations_applied": 0,
                    "changes_made": [],
                }

            file_path = Path(filepath) if isinstance(filepath, str) else filepath
            if not file_path.exists():
                return {
                    "success": False,
                    "error": f"File not found: {file_path}",
                    "operations_applied": 0,
                    "changes_made": [],
                }

            # Analyze file
            plan = self._analyzer.analyze_god_object(file_path)
            if not plan.transformations:
                return {
                    "success": True,
                    "message": "No refactoring needed",
                    "operations_applied": 0,
                    "changes_made": [],
                }

            # Execute refactoring
            execution_results = self._analyzer.execute_refactoring(
                file_path, plan, dry_run=dry_run
            )
            
            return {
                "success": execution_results.get("success", False),
                "message": execution_results.get("message", ""),
                "operations_applied": len(plan.transformations),
                "changes_made": execution_results.get("files_created", [])
                + execution_results.get("files_modified", []),
                "validation_results": execution_results.get("validation", {}),
                "errors": execution_results.get("errors", []),
                "warnings": execution_results.get("warnings", []),
            }

        except Exception as e:
            logger.error("Failed to apply opportunity: %s", e)
            return {
                "success": False,
                "error": str(e),
                "operations_applied": 0,
                "changes_made": [],
            }

"""
Incremental Debugging Workflow for IntelliRefactor.

This module provides checkpoint and resume functionality, progress tracking,
and validation for incremental debugging operations.
"""

import logging
import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto

from intellirefactor.analysis.foundation.debug.debug_cycle_manager import (
    DebugCycleManager,
    Issue,
    IssueType,
)
from intellirefactor.analysis.foundation.models import Severity

logger = logging.getLogger(__name__)

_DEFAULT_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".intellirefactor",
    "intellirefactor_out",
}


class WorkflowState(Enum):
    """States of the incremental debugging workflow."""

    INITIALIZED = auto()
    ANALYZING = auto()
    DEBUGGING = auto()
    FIXING = auto()
    VALIDATING = auto()
    CHECKPOINTING = auto()
    COMPLETED = auto()
    FAILED = auto()
    SUSPENDED = auto()


@dataclass
class WorkflowCheckpoint:
    """Represents a checkpoint in the debugging workflow."""

    checkpoint_id: str
    timestamp: str
    workflow_state: WorkflowState
    progress_data: Dict[str, Any]
    debug_manager_state: Dict[str, Any]
    refactoring_state: Dict[str, Any]
    validation_state: Dict[str, Any]
    file_states: Dict[str, str] = field(default_factory=dict)  # file_path -> hash
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary format."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "timestamp": self.timestamp,
            "workflow_state": self.workflow_state.name,
            "progress_data": self.progress_data,
            "debug_manager_state": self.debug_manager_state,
            "refactoring_state": self.refactoring_state,
            "validation_state": self.validation_state,
            "file_states": self.file_states,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowCheckpoint":
        """Create checkpoint from dictionary."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            timestamp=data["timestamp"],
            workflow_state=WorkflowState[data["workflow_state"]],
            progress_data=data["progress_data"],
            debug_manager_state=data["debug_manager_state"],
            refactoring_state=data["refactoring_state"],
            validation_state=data["validation_state"],
            file_states=data.get("file_states", {}),
            metadata=data.get("metadata", {}),
        )


@dataclass
class WorkflowProgress:
    """Tracks progress of the incremental debugging workflow."""

    workflow_id: str
    current_state: WorkflowState
    start_time: str
    last_update_time: str
    total_steps: int = 0
    completed_steps: int = 0
    failed_steps: int = 0
    checkpoints_created: int = 0
    issues_detected: int = 0
    issues_resolved: int = 0
    files_processed: List[str] = field(default_factory=list)
    operations_completed: List[str] = field(default_factory=list)
    validation_results: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert progress to dictionary format."""
        d = asdict(self)
        # Enum -> str for JSON
        d["current_state"] = self.current_state.name
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WorkflowProgress":
        """Create progress from dictionary."""
        d = dict(data or {})
        cs = d.get("current_state")
        if isinstance(cs, str):
            d["current_state"] = WorkflowState[cs]
        return cls(**d)


class IncrementalDebuggingWorkflow:
    """
    Manages incremental debugging workflow with checkpoint and resume functionality.

    Provides progress tracking, validation, and state management for debugging operations.
    """

    def __init__(
        self,
        workspace_path: Union[str, Path],
        checkpoint_interval: int = 5,
        auto_checkpoint: bool = True,
    ):
        """
        Initialize the incremental debugging workflow.

        Args:
            workspace_path: Path to the workspace directory
            checkpoint_interval: Number of operations between automatic checkpoints
            auto_checkpoint: Whether to create checkpoints automatically
        """
        self.workspace_path = Path(workspace_path)
        self.checkpoint_interval = checkpoint_interval
        self.auto_checkpoint = auto_checkpoint

        # Create workflow directories
        self.workflow_dir = self.workspace_path / ".intellirefactor" / "workflow"
        self.checkpoints_dir = self.workflow_dir / "checkpoints"
        self.progress_dir = self.workflow_dir / "progress"

        self._ensure_directories()

        # Initialize components
        self.debug_manager = DebugCycleManager()

        # Workflow state
        self.current_progress: Optional[WorkflowProgress] = None
        self.current_checkpoint: Optional[WorkflowCheckpoint] = None
        self.operation_count = 0

        # State tracking
        self.refactoring_state: Dict[str, Any] = {}
        self.validation_state: Dict[str, Any] = {}
        self.file_states: Dict[str, str] = {}

        logger.info(f"IncrementalDebuggingWorkflow initialized at {workspace_path}")

    def start_workflow(
        self,
        workflow_id: Optional[str] = None,
        initial_state: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Start a new incremental debugging workflow.

        Args:
            workflow_id: Optional workflow ID, generated if not provided
            initial_state: Optional initial state data

        Returns:
            Workflow ID
        """
        if not workflow_id:
            workflow_id = self._generate_workflow_id()

        # Initialize progress tracking
        self.current_progress = WorkflowProgress(
            workflow_id=workflow_id,
            current_state=WorkflowState.INITIALIZED,
            start_time=self._get_timestamp(),
            last_update_time=self._get_timestamp(),
        )

        # Initialize debug manager session
        debug_session_id = self.debug_manager.start_debug_session(f"workflow_{workflow_id}")

        # Initialize states
        self.refactoring_state = initial_state.copy() if initial_state else {}
        self.validation_state = {"validation_enabled": True, "strict_mode": False}
        self.validation_state["debug_session_id"] = debug_session_id
        self.file_states = {}

        # Create initial checkpoint
        if self.auto_checkpoint:
            self.create_checkpoint("Initial workflow state")

        # Save progress
        self._save_progress()

        logger.info(f"Started incremental debugging workflow {workflow_id}")
        return workflow_id

    def resume_workflow(self, workflow_id: str, checkpoint_id: Optional[str] = None) -> bool:
        """
        Resume a workflow from a checkpoint.

        Args:
            workflow_id: ID of the workflow to resume
            checkpoint_id: Optional specific checkpoint ID, uses latest if not provided

        Returns:
            True if resume was successful, False otherwise
        """
        try:
            # Load progress
            progress_file = self.progress_dir / f"{workflow_id}.json"
            if not progress_file.exists():
                logger.error(f"No progress file found for workflow {workflow_id}")
                return False

            with open(progress_file, "r", encoding="utf-8") as f:
                progress_data = json.load(f)

            self.current_progress = WorkflowProgress.from_dict(progress_data)

            # Load checkpoint
            if checkpoint_id:
                checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"
            else:
                # Find latest checkpoint for this workflow
                checkpoint_files = list(self.checkpoints_dir.glob(f"{workflow_id}_*.json"))
                if not checkpoint_files:
                    logger.error(f"No checkpoints found for workflow {workflow_id}")
                    return False

                # Sort by timestamp and get latest
                checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                checkpoint_file = checkpoint_files[0]
                checkpoint_id = checkpoint_file.stem

            if not checkpoint_file.exists():
                logger.error(f"Checkpoint file not found: {checkpoint_file}")
                return False

            with open(checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint_data = json.load(f)

            self.current_checkpoint = WorkflowCheckpoint.from_dict(checkpoint_data)

            # Restore states
            self.refactoring_state = self.current_checkpoint.refactoring_state.copy()
            self.validation_state = self.current_checkpoint.validation_state.copy()
            self.file_states = self.current_checkpoint.file_states.copy()

            # Restore debug manager state
            debug_manager_state = self.current_checkpoint.debug_manager_state
            if "debug_progress" in debug_manager_state:
                debug_progress_data = debug_manager_state["debug_progress"]
                if debug_progress_data:
                    # Resume debug manager from checkpoint
                    checkpoint_for_debug = {
                        "checkpoint_id": checkpoint_id,
                        "debug_progress": debug_progress_data,
                        "debug_history_count": debug_manager_state.get("debug_history_count", 0),
                    }
                    self.debug_manager.resume_from_checkpoint(checkpoint_for_debug)

            # Update workflow state
            self.current_progress.current_state = self.current_checkpoint.workflow_state
            self.current_progress.last_update_time = self._get_timestamp()

            logger.info(f"Resumed workflow {workflow_id} from checkpoint {checkpoint_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to resume workflow {workflow_id}: {e}")
            return False

    def create_checkpoint(self, description: str = "") -> str:
        """
        Create a checkpoint of the current workflow state.

        Args:
            description: Optional description of the checkpoint

        Returns:
            Checkpoint ID
        """
        try:
            if not self.current_progress:
                raise ValueError("No active workflow to checkpoint")

            checkpoint_id = (
                f"{self.current_progress.workflow_id}_{int(datetime.now().timestamp() * 1000)}"
            )

            # Capture current file states
            self._update_file_states()

            # Capture debug manager state
            debug_progress = self.debug_manager.get_debug_progress()
            debug_history = self.debug_manager.get_debug_history()

            debug_manager_state = {
                "debug_progress": debug_progress.to_dict() if debug_progress else None,
                "debug_history_count": len(debug_history),
                "session_active": debug_progress is not None,
            }

            # Create checkpoint
            checkpoint = WorkflowCheckpoint(
                checkpoint_id=checkpoint_id,
                timestamp=self._get_timestamp(),
                workflow_state=self.current_progress.current_state,
                progress_data=self.current_progress.to_dict(),
                debug_manager_state=debug_manager_state,
                refactoring_state=self.refactoring_state.copy(),
                validation_state=self.validation_state.copy(),
                file_states=self.file_states.copy(),
                metadata={
                    "description": description,
                    "operation_count": self.operation_count,
                },
            )

            # Save checkpoint
            checkpoint_file = self.checkpoints_dir / f"{checkpoint_id}.json"
            with open(checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint.to_dict(), f, indent=2, ensure_ascii=False)

            self.current_checkpoint = checkpoint
            self.current_progress.checkpoints_created += 1
            self.current_progress.last_update_time = self._get_timestamp()

            # Save updated progress
            self._save_progress()

            logger.info(f"Created checkpoint {checkpoint_id}: {description}")
            return checkpoint_id

        except Exception as e:
            logger.error(f"Failed to create checkpoint: {e}")
            return ""

    def execute_debug_operation(
        self,
        operation_type: str,
        operation_data: Dict[str, Any],
        validate_after: bool = True,
    ) -> Dict[str, Any]:
        """
        Execute a debugging operation with progress tracking.

        Args:
            operation_type: Type of operation to execute
            operation_data: Data for the operation
            validate_after: Whether to validate after the operation

        Returns:
            Operation result
        """
        if not self.current_progress:
            raise ValueError("No active workflow")

        try:
            logger.info(f"Executing debug operation: {operation_type}")

            # Update workflow state
            self.current_progress.current_state = WorkflowState.DEBUGGING
            self.current_progress.total_steps += 1
            self.operation_count += 1

            # Execute operation based on type
            result = {}

            if operation_type == "detect_issues":
                result = self._execute_issue_detection(operation_data)
            elif operation_type == "apply_fix":
                result = self._execute_fix_application(operation_data)
            elif operation_type == "debug_cycle":
                result = self._execute_debug_cycle(operation_data)
            elif operation_type == "validate_state":
                result = self._execute_state_validation(operation_data)
            else:
                result = {
                    "success": False,
                    "error": f"Unknown operation type: {operation_type}",
                }

            # Update progress based on result
            if result.get("success", False):
                self.current_progress.completed_steps += 1
                self.current_progress.operations_completed.append(
                    f"{operation_type}:{self.operation_count}"
                )
            else:
                self.current_progress.failed_steps += 1

            # Validate after operation if requested
            if validate_after and result.get("success", False):
                validation_result = self.validate_workflow_state()
                result["validation"] = validation_result

            # Auto-checkpoint if configured
            if self.auto_checkpoint and self.operation_count % self.checkpoint_interval == 0:
                checkpoint_id = self.create_checkpoint(f"Auto checkpoint after {operation_type}")
                result["checkpoint_created"] = checkpoint_id

            # Update progress
            self.current_progress.last_update_time = self._get_timestamp()
            self._save_progress()

            logger.info(
                f"Debug operation {operation_type} completed: {result.get('success', False)}"
            )
            return result

        except Exception as e:
            logger.error(f"Debug operation {operation_type} failed: {e}")
            self.current_progress.failed_steps += 1
            self.current_progress.last_update_time = self._get_timestamp()
            self._save_progress()

            return {
                "success": False,
                "error": str(e),
                "operation_type": operation_type,
                "operation_count": self.operation_count,
            }

    def validate_workflow_state(self) -> Dict[str, Any]:
        """
        Validate the current workflow state.

        Returns:
            Validation result
        """
        try:
            validation_result = {
                "valid": True,
                "issues": [],
                "warnings": [],
                "timestamp": self._get_timestamp(),
            }

            # Validate progress consistency
            if self.current_progress:
                if (
                    self.current_progress.completed_steps + self.current_progress.failed_steps
                    > self.current_progress.total_steps
                ):
                    validation_result["issues"].append("Step counts are inconsistent")
                    validation_result["valid"] = False

                if self.current_progress.issues_resolved > self.current_progress.issues_detected:
                    validation_result["warnings"].append("More issues resolved than detected")

            # Validate file states
            file_validation = self._validate_file_states()
            if not file_validation["valid"]:
                validation_result["issues"].extend(file_validation["issues"])
                validation_result["valid"] = False

            # Validate debug manager state
            debug_progress = self.debug_manager.get_debug_progress()
            if debug_progress:
                if (
                    debug_progress.successful_cycles + debug_progress.failed_cycles
                    > debug_progress.total_cycles
                ):
                    validation_result["issues"].append("Debug cycle counts are inconsistent")
                    validation_result["valid"] = False

            # Update validation state
            self.validation_state["last_validation"] = validation_result
            self.validation_state["validation_count"] = (
                self.validation_state.get("validation_count", 0) + 1
            )

            return validation_result

        except Exception as e:
            logger.error(f"Workflow state validation failed: {e}")
            return {
                "valid": False,
                "issues": [f"Validation error: {str(e)}"],
                "warnings": [],
                "timestamp": self._get_timestamp(),
            }

    def get_workflow_progress(self) -> Optional[WorkflowProgress]:
        """Get current workflow progress."""
        return self.current_progress

    def get_available_checkpoints(self, workflow_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of available checkpoints.

        Args:
            workflow_id: Optional workflow ID to filter by

        Returns:
            List of checkpoint information
        """
        try:
            checkpoints = []
            pattern = f"{workflow_id}_*.json" if workflow_id else "*.json"

            for checkpoint_file in self.checkpoints_dir.glob(pattern):
                try:
                    with open(checkpoint_file, "r", encoding="utf-8") as f:
                        checkpoint_data = json.load(f)

                    checkpoint_info = {
                        "checkpoint_id": checkpoint_data["checkpoint_id"],
                        "timestamp": checkpoint_data["timestamp"],
                        "workflow_state": checkpoint_data["workflow_state"],
                        "description": checkpoint_data.get("metadata", {}).get("description", ""),
                        "operation_count": checkpoint_data.get("metadata", {}).get(
                            "operation_count", 0
                        ),
                        "file_path": str(checkpoint_file),
                    }
                    checkpoints.append(checkpoint_info)

                except Exception as e:
                    logger.warning(f"Failed to read checkpoint {checkpoint_file}: {e}")

            # Sort by timestamp (newest first)
            checkpoints.sort(key=lambda x: x["timestamp"], reverse=True)
            return checkpoints

        except Exception as e:
            logger.error(f"Failed to get available checkpoints: {e}")
            return []

    def cleanup_old_checkpoints(self, keep_count: int = 10) -> int:
        """
        Clean up old checkpoints, keeping only the most recent ones.

        Args:
            keep_count: Number of checkpoints to keep

        Returns:
            Number of checkpoints removed
        """
        try:
            checkpoints = self.get_available_checkpoints()

            if len(checkpoints) <= keep_count:
                return 0

            # Remove oldest checkpoints
            checkpoints_to_remove = checkpoints[keep_count:]
            removed_count = 0

            for checkpoint in checkpoints_to_remove:
                try:
                    checkpoint_file = Path(checkpoint["file_path"])
                    if checkpoint_file.exists():
                        checkpoint_file.unlink()
                        removed_count += 1
                        logger.debug(f"Removed old checkpoint: {checkpoint['checkpoint_id']}")
                except Exception as e:
                    logger.warning(
                        f"Failed to remove checkpoint {checkpoint['checkpoint_id']}: {e}"
                    )

            logger.info(f"Cleaned up {removed_count} old checkpoints")
            return removed_count

        except Exception as e:
            logger.error(f"Failed to cleanup old checkpoints: {e}")
            return 0

    # Private helper methods

    def _ensure_directories(self):
        """Ensure workflow directories exist."""
        self.workflow_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.progress_dir.mkdir(parents=True, exist_ok=True)

    def _generate_workflow_id(self) -> str:
        """Generate unique workflow ID."""
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"workflow_{timestamp}_{id(self) % 10000}"

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now().isoformat()

    def _save_progress(self):
        """Save current progress to file."""
        if not self.current_progress:
            return

        try:
            progress_file = self.progress_dir / f"{self.current_progress.workflow_id}.json"
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(self.current_progress.to_dict(), f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save progress: {e}")

    def _hash_bytes(self, data: bytes) -> str:
        """Stable, non-cryptographic digest for change detection (Bandit-safe)."""
        return hashlib.blake2b(data, digest_size=16).hexdigest()

    def _update_file_states(self):
        """Update file state hashes."""
        try:
            # Track Python files in the workspace
            for py_file in self.workspace_path.rglob("*.py"):
                if py_file.is_file():
                    try:
                        rel_parts = set(py_file.relative_to(self.workspace_path).parts)
                    except Exception:
                        rel_parts = set(py_file.parts)
                    if rel_parts.intersection(_DEFAULT_SKIP_DIRS):
                        continue
                    try:
                        with open(py_file, "rb") as f:
                            file_hash = self._hash_bytes(f.read())
                        self.file_states[str(py_file.relative_to(self.workspace_path))] = file_hash
                    except Exception as e:
                        logger.warning(f"Failed to hash file {py_file}: {e}")
        except Exception as e:
            logger.error(f"Failed to update file states: {e}")

    def _validate_file_states(self) -> Dict[str, Any]:
        """Validate current file states against stored hashes."""
        validation_result = {"valid": True, "issues": [], "changed_files": []}

        try:
            current_states = {}

            # Get current file hashes
            for py_file in self.workspace_path.rglob("*.py"):
                if py_file.is_file():
                    try:
                        rel_parts = set(py_file.relative_to(self.workspace_path).parts)
                    except Exception:
                        rel_parts = set(py_file.parts)
                    if rel_parts.intersection(_DEFAULT_SKIP_DIRS):
                        continue
                    try:
                        with open(py_file, "rb") as f:
                            file_hash = self._hash_bytes(f.read())
                        relative_path = str(py_file.relative_to(self.workspace_path))
                        current_states[relative_path] = file_hash
                    except Exception as e:
                        logger.warning(f"Failed to hash file {py_file}: {e}")

            # Compare with stored states
            for file_path, stored_hash in self.file_states.items():
                current_hash = current_states.get(file_path)
                if current_hash != stored_hash:
                    validation_result["changed_files"].append(file_path)
                    if current_hash is None:
                        validation_result["issues"].append(f"File deleted: {file_path}")
                    else:
                        validation_result["issues"].append(f"File modified: {file_path}")

            # Check for new files
            for file_path in current_states:
                if file_path not in self.file_states:
                    validation_result["changed_files"].append(file_path)
                    validation_result["issues"].append(f"New file: {file_path}")

            if validation_result["issues"]:
                validation_result["valid"] = False

        except Exception as e:
            validation_result["valid"] = False
            validation_result["issues"].append(f"File validation error: {str(e)}")

        return validation_result

    def _execute_issue_detection(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute issue detection operation."""
        try:
            error = operation_data.get("error")
            context = operation_data.get("context", "issue_detection")
            additional_info = operation_data.get("additional_info", {})

            if not error:
                return {
                    "success": False,
                    "error": "No error provided for issue detection",
                }

            # Use debug manager to detect issues
            issues = self.debug_manager.detect_intellirefactor_issues(
                error, context, additional_info
            )

            # Update progress
            self.current_progress.issues_detected += len(issues)

            return {
                "success": True,
                "issues_detected": len(issues),
                "issues": [issue.to_dict() for issue in issues],
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_fix_application(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fix application operation."""
        try:
            issue_data = operation_data.get("issue")
            if not issue_data:
                return {
                    "success": False,
                    "error": "No issue provided for fix application",
                }

            # Convert dict to Issue object if needed
            if isinstance(issue_data, dict):
                issue = Issue(
                    issue_id=issue_data["issue_id"],
                    issue_type=IssueType[issue_data["issue_type"]],
                    severity=Severity(issue_data["severity"]),
                    description=issue_data["description"],
                    context=issue_data["context"],
                    file_path=issue_data.get("file_path"),
                    line_number=issue_data.get("line_number"),
                    suggested_fixes=issue_data.get("suggested_fixes", []),
                    diagnostic_info=issue_data.get("diagnostic_info", {}),
                    timestamp=issue_data.get("timestamp"),
                )
            else:
                issue = issue_data

            # Apply fix using debug manager
            fix = self.debug_manager.apply_fix(issue)
            validation_passed = self.debug_manager.validate_fix(fix)

            # Update progress
            if validation_passed:
                self.current_progress.issues_resolved += 1

            return {
                "success": validation_passed,
                "fix_applied": fix.to_dict(),
                "validation_passed": validation_passed,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_debug_cycle(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute debug cycle operation."""
        try:
            error = operation_data.get("error")
            context = operation_data.get("context", "debug_cycle")
            additional_info = operation_data.get("additional_info", {})

            if not error:
                return {"success": False, "error": "No error provided for debug cycle"}

            # Execute debug cycle using debug manager
            cycle = self.debug_manager.execute_debug_cycle(error, context, additional_info)

            # Update progress
            self.current_progress.issues_detected += len(cycle.issues_detected)
            successful_fixes = sum(
                1 for result in cycle.validation_results if result.get("validation_passed", False)
            )
            self.current_progress.issues_resolved += successful_fixes

            return {
                "success": cycle.success,
                "cycle": cycle.to_dict(),
                "issues_detected": len(cycle.issues_detected),
                "fixes_applied": len(cycle.fixes_applied),
                "successful_fixes": successful_fixes,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

    def _execute_state_validation(self, operation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute state validation operation."""
        try:
            validation_result = self.validate_workflow_state()

            return {
                "success": validation_result["valid"],
                "validation_result": validation_result,
            }

        except Exception as e:
            return {"success": False, "error": str(e)}

"""
Safety Manager for IntelliRefactor

Provides comprehensive safety checks, backup mechanisms, and rollback capabilities
to prevent destructive operations and ensure safe refactoring operations.
"""

import os
import shutil
import tempfile
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum
import logging

from .backup_manager import BackupManager, BackupResult
from .rollback_manager import RollbackManager, RollbackResult
from .destructive_operation_detector import DestructiveOperationDetector, OperationRisk
from .validation_tools import RefactoringValidator, ValidationLevel

logger = logging.getLogger(__name__)


class SafetyLevel(Enum):
    """Safety levels for refactoring operations"""
    MINIMAL = "minimal"      # Basic checks only
    STANDARD = "standard"    # Standard safety checks
    PARANOID = "paranoid"    # Maximum safety checks


@dataclass
class SafetyCheck:
    """Represents a safety check result"""
    name: str
    passed: bool
    message: str
    risk_level: OperationRisk
    details: Dict[str, Any] = None


@dataclass
class SafetyResult:
    """Result of a comprehensive safety analysis"""
    safe_to_proceed: bool
    checks: List[SafetyCheck]
    backup_created: bool
    backup_path: Optional[str] = None
    warnings: List[str] = None
    errors: List[str] = None


class SafetyManager:
    """
    Comprehensive safety manager for refactoring operations.
    
    Provides backup creation, destructive operation detection, validation,
    and rollback capabilities to ensure safe refactoring operations.
    """
    
    def __init__(self, config: Optional[Any] = None):
        """Initialize the safety manager."""
        self.config = config
        self.safety_level = getattr(config, 'safety_level', SafetyLevel.STANDARD)
        
        # Initialize component managers
        self.backup_manager = BackupManager()
        self.rollback_manager = RollbackManager()
        self.operation_detector = DestructiveOperationDetector()
        self.validator = RefactoringValidator()
        
        self.logger = logging.getLogger(__name__)
    
    def initialize_project(self, project_path: str) -> Dict[str, Any]:
        """Initialize safety systems for a project."""
        self.logger.info(f"Initializing safety systems for project: {project_path}")
        
        project_path = Path(project_path)
        if not project_path.exists():
            raise ValueError(f"Project path does not exist: {project_path}")
        
        # Create safety directory
        safety_dir = project_path / '.intellirefactor' / 'safety'
        safety_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize backup system
        backup_config = {
            'backup_dir': str(safety_dir / 'backups'),
            'max_backups': 10,
            'compress_backups': True
        }
        
        return {
            'safety_dir': str(safety_dir),
            'backup_config': backup_config,
            'safety_level': self.safety_level.value,
            'initialized': True
        }
    
    def pre_refactoring_check(self, project_path: str, refactoring_plan: Dict[str, Any]) -> SafetyResult:
        """Perform comprehensive safety checks before refactoring."""
        self.logger.info(f"Performing pre-refactoring safety check for: {project_path}")
        
        checks = []
        warnings = []
        errors = []
        
        try:
            # Check 1: Project structure validation
            structure_check = self._check_project_structure(project_path)
            checks.append(structure_check)
            
            # Check 2: Destructive operation detection
            destructive_check = self._check_destructive_operations(refactoring_plan)
            checks.append(destructive_check)
            
            # Check 3: Backup feasibility
            backup_check = self._check_backup_feasibility(project_path)
            checks.append(backup_check)
            
            # Check 4: Version control status
            vcs_check = self._check_version_control_status(project_path)
            checks.append(vcs_check)
            
            # Determine overall safety
            failed_checks = [c for c in checks if not c.passed]
            critical_failures = [c for c in failed_checks if c.risk_level == OperationRisk.CRITICAL]
            
            safe_to_proceed = len(critical_failures) == 0
            
            if failed_checks:
                for check in failed_checks:
                    if check.risk_level == OperationRisk.CRITICAL:
                        errors.append(f"Critical: {check.message}")
                    else:
                        warnings.append(f"Warning: {check.message}")
            
            return SafetyResult(
                safe_to_proceed=safe_to_proceed,
                checks=checks,
                backup_created=False,
                warnings=warnings,
                errors=errors
            )
            
        except Exception as e:
            self.logger.error(f"Safety check failed: {e}")
            return SafetyResult(
                safe_to_proceed=False,
                checks=checks,
                backup_created=False,
                errors=[f"Safety check failed: {str(e)}"]
            )
    
    def create_backup(self, project_path: str) -> Dict[str, Any]:
        """Create a backup of the project."""
        self.logger.info(f"Creating backup for project: {project_path}")
        
        try:
            backup_result = self.backup_manager.create_backup(project_path)
            
            return {
                'success': backup_result.success,
                'backup_path': backup_result.backup_path,
                'backup_size': backup_result.backup_size,
                'files_backed_up': backup_result.files_backed_up,
                'timestamp': backup_result.timestamp
            }
            
        except Exception as e:
            self.logger.error(f"Backup creation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def analyze_project_safety(self, project_path: str) -> Dict[str, Any]:
        """Analyze overall project safety for refactoring."""
        self.logger.info(f"Analyzing project safety: {project_path}")
        
        try:
            project_path = Path(project_path)
            
            # Basic project analysis
            analysis = {
                'project_path': str(project_path),
                'timestamp': datetime.now().isoformat(),
                'safety_level': self.safety_level.value,
                'safe_to_refactor': True,
                'concerns': [],
                'recommendations': []
            }
            
            # Check project structure
            if not project_path.exists():
                analysis['safe_to_refactor'] = False
                analysis['concerns'].append("Project path does not exist")
                return analysis
            
            # Check for critical files
            critical_files = ['.git', 'setup.py', 'pyproject.toml', 'requirements.txt']
            missing_critical = []
            
            for file in critical_files:
                if not (project_path / file).exists():
                    missing_critical.append(file)
            
            if missing_critical:
                analysis['concerns'].append(f"Missing critical files: {', '.join(missing_critical)}")
            
            # Check project size
            try:
                total_files = sum(1 for _ in project_path.rglob('*.py'))
                if total_files > 1000:
                    analysis['concerns'].append(f"Large project ({total_files} Python files) - consider incremental refactoring")
                    analysis['recommendations'].append("Use incremental refactoring approach")
            except Exception:
                pass
            
            # Check for version control
            if not (project_path / '.git').exists():
                analysis['concerns'].append("No version control detected - backup is critical")
                analysis['recommendations'].append("Initialize git repository before refactoring")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Safety analysis failed: {e}")
            return {
                'project_path': str(project_path),
                'safe_to_refactor': False,
                'error': str(e)
            }
    
    def _check_project_structure(self, project_path: str) -> SafetyCheck:
        """Check project structure validity."""
        try:
            project_path = Path(project_path)
            
            if not project_path.exists():
                return SafetyCheck(
                    name="project_structure",
                    passed=False,
                    message="Project path does not exist",
                    risk_level=OperationRisk.CRITICAL
                )
            
            if not project_path.is_dir():
                return SafetyCheck(
                    name="project_structure",
                    passed=False,
                    message="Project path is not a directory",
                    risk_level=OperationRisk.CRITICAL
                )
            
            # Check for Python files
            python_files = list(project_path.rglob('*.py'))
            if not python_files:
                return SafetyCheck(
                    name="project_structure",
                    passed=False,
                    message="No Python files found in project",
                    risk_level=OperationRisk.HIGH
                )
            
            return SafetyCheck(
                name="project_structure",
                passed=True,
                message=f"Project structure valid ({len(python_files)} Python files)",
                risk_level=OperationRisk.LOW
            )
            
        except Exception as e:
            return SafetyCheck(
                name="project_structure",
                passed=False,
                message=f"Structure check failed: {str(e)}",
                risk_level=OperationRisk.HIGH
            )
    
    def _check_destructive_operations(self, refactoring_plan: Dict[str, Any]) -> SafetyCheck:
        """Check for potentially destructive operations in the plan."""
        try:
            # Analyze refactoring plan for destructive operations
            destructive_ops = []
            
            steps = refactoring_plan.get('steps', [])
            for step in steps:
                operation_type = step.get('type', '')
                
                # Check for high-risk operations
                if operation_type in ['delete_file', 'remove_class', 'remove_function']:
                    destructive_ops.append(f"{operation_type}: {step.get('target', 'unknown')}")
                elif operation_type in ['rename_file', 'move_file']:
                    # Medium risk operations
                    pass
            
            if destructive_ops:
                return SafetyCheck(
                    name="destructive_operations",
                    passed=False,
                    message=f"Destructive operations detected: {', '.join(destructive_ops)}",
                    risk_level=OperationRisk.HIGH,
                    details={'operations': destructive_ops}
                )
            
            return SafetyCheck(
                name="destructive_operations",
                passed=True,
                message="No destructive operations detected",
                risk_level=OperationRisk.LOW
            )
            
        except Exception as e:
            return SafetyCheck(
                name="destructive_operations",
                passed=False,
                message=f"Destructive operation check failed: {str(e)}",
                risk_level=OperationRisk.MEDIUM
            )
    
    def _check_backup_feasibility(self, project_path: str) -> SafetyCheck:
        """Check if backup creation is feasible."""
        try:
            project_path = Path(project_path)
            
            # Check available disk space
            total_size = sum(f.stat().st_size for f in project_path.rglob('*') if f.is_file())
            
            # Get available space (simplified check)
            available_space = shutil.disk_usage(project_path).free
            
            if available_space < total_size * 2:  # Need at least 2x project size
                return SafetyCheck(
                    name="backup_feasibility",
                    passed=False,
                    message=f"Insufficient disk space for backup (need {total_size * 2}, have {available_space})",
                    risk_level=OperationRisk.HIGH
                )
            
            return SafetyCheck(
                name="backup_feasibility",
                passed=True,
                message="Sufficient space available for backup",
                risk_level=OperationRisk.LOW
            )
            
        except Exception as e:
            return SafetyCheck(
                name="backup_feasibility",
                passed=False,
                message=f"Backup feasibility check failed: {str(e)}",
                risk_level=OperationRisk.MEDIUM
            )
    
    def _check_version_control_status(self, project_path: str) -> SafetyCheck:
        """Check version control status."""
        try:
            project_path = Path(project_path)
            git_dir = project_path / '.git'
            
            if not git_dir.exists():
                return SafetyCheck(
                    name="version_control",
                    passed=False,
                    message="No version control system detected",
                    risk_level=OperationRisk.MEDIUM
                )
            
            # Check for uncommitted changes (simplified)
            # In a real implementation, you'd use GitPython or similar
            return SafetyCheck(
                name="version_control",
                passed=True,
                message="Version control system detected",
                risk_level=OperationRisk.LOW
            )
            
        except Exception as e:
            return SafetyCheck(
                name="version_control",
                passed=False,
                message=f"Version control check failed: {str(e)}",
                risk_level=OperationRisk.MEDIUM
            )
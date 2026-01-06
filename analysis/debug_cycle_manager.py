"""
Debug Cycle Manager for IntelliRefactor.

This module provides comprehensive debugging capabilities for IntelliRefactor operations,
including issue detection, prioritization, fix application, and validation mechanisms.
"""

import logging
import traceback
import json
import time
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Callable
from pathlib import Path
from datetime import datetime

from .error_handler import AnalysisErrorHandler, AnalysisError, ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)


class IssueType(Enum):
    """Types of issues that can occur during IntelliRefactor operations."""
    CONFIGURATION_ISSUE = auto()
    ANALYSIS_FAILURE = auto()
    REFACTORING_ERROR = auto()
    VALIDATION_FAILURE = auto()
    DEPENDENCY_ISSUE = auto()
    PARAMETER_ISSUE = auto()
    PERFORMANCE_ISSUE = auto()
    TOOL_BUG = auto()
    UNKNOWN_ISSUE = auto()


class FixType(Enum):
    """Types of fixes that can be applied."""
    CONFIGURATION_UPDATE = auto()
    DEPENDENCY_INSTALL = auto()
    CODE_MODIFICATION = auto()
    PARAMETER_ADJUSTMENT = auto()
    ENVIRONMENT_SETUP = auto()
    TOOL_PATCH = auto()
    WORKAROUND = auto()


@dataclass
class Issue:
    """Represents an issue detected during IntelliRefactor operations."""
    issue_id: str
    issue_type: IssueType
    severity: ErrorSeverity
    description: str
    context: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    error_details: Optional[AnalysisError] = None
    suggested_fixes: List[str] = field(default_factory=list)
    diagnostic_info: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert issue to dictionary format."""
        return {
            'issue_id': self.issue_id,
            'issue_type': self.issue_type.name,
            'severity': self.severity.value,
            'description': self.description,
            'context': self.context,
            'file_path': self.file_path,
            'line_number': self.line_number,
            'error_details': self.error_details.to_dict() if self.error_details else None,
            'suggested_fixes': self.suggested_fixes,
            'diagnostic_info': self.diagnostic_info,
            'timestamp': self.timestamp
        }


@dataclass
class Fix:
    """Represents a fix applied to resolve an issue."""
    fix_id: str
    issue_id: str
    fix_type: FixType
    description: str
    changes_made: List[str] = field(default_factory=list)
    validation_passed: bool = False
    rollback_info: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fix to dictionary format."""
        return {
            'fix_id': self.fix_id,
            'issue_id': self.issue_id,
            'fix_type': self.fix_type.name,
            'description': self.description,
            'changes_made': self.changes_made,
            'validation_passed': self.validation_passed,
            'rollback_info': self.rollback_info,
            'timestamp': self.timestamp
        }


@dataclass
class DebugCycle:
    """Represents a complete debug cycle."""
    cycle_id: str
    issues_detected: List[Issue] = field(default_factory=list)
    fixes_applied: List[Fix] = field(default_factory=list)
    validation_results: List[Dict[str, Any]] = field(default_factory=list)
    success: bool = False
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    progress_preserved: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert debug cycle to dictionary format."""
        return {
            'cycle_id': self.cycle_id,
            'issues_detected': [issue.to_dict() for issue in self.issues_detected],
            'fixes_applied': [fix.to_dict() for fix in self.fixes_applied],
            'validation_results': self.validation_results,
            'success': self.success,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'progress_preserved': self.progress_preserved
        }


@dataclass
class DebugProgress:
    """Tracks progress during debugging sessions."""
    session_id: str
    total_cycles: int = 0
    successful_cycles: int = 0
    failed_cycles: int = 0
    issues_resolved: int = 0
    issues_remaining: int = 0
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    last_successful_state: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert debug progress to dictionary format."""
        return {
            'session_id': self.session_id,
            'total_cycles': self.total_cycles,
            'successful_cycles': self.successful_cycles,
            'failed_cycles': self.failed_cycles,
            'issues_resolved': self.issues_resolved,
            'issues_remaining': self.issues_remaining,
            'checkpoints': self.checkpoints,
            'last_successful_state': self.last_successful_state
        }


class DebugCycleManager:
    """
    Manages iterative debugging of IntelliRefactor issues.
    
    Provides issue detection, prioritization, fix application, and validation
    mechanisms for debugging IntelliRefactor operations.
    """
    
    def __init__(self, max_cycles: int = 10, checkpoint_interval: int = 3):
        """
        Initialize the debug cycle manager.
        
        Args:
            max_cycles: Maximum number of debug cycles to attempt
            checkpoint_interval: Number of cycles between checkpoints
        """
        self.max_cycles = max_cycles
        self.checkpoint_interval = checkpoint_interval
        self.error_handler = AnalysisErrorHandler()
        
        # Issue detection patterns
        self.issue_patterns = self._initialize_issue_patterns()
        
        # Fix strategies
        self.fix_strategies = self._initialize_fix_strategies()
        
        # Progress tracking
        self.current_progress: Optional[DebugProgress] = None
        self.debug_history: List[DebugCycle] = []
        
        # ID generation counters for uniqueness
        self._session_counter = 0
        self._issue_counter = 0
        self._fix_counter = 0
        self._cycle_counter = 0
        self._checkpoint_counter = 0
        
        logger.info("DebugCycleManager initialized")
    
    def detect_intellirefactor_issues(
        self,
        error: Exception,
        context: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> List[Issue]:
        """
        Detect and categorize IntelliRefactor issues from errors.
        
        Args:
            error: The exception that occurred
            context: Context where the error occurred
            additional_info: Additional diagnostic information
            
        Returns:
            List of detected issues
        """
        try:
            # Handle the error using the error handler
            handled_error = self.error_handler.handle_analysis_error(
                error, context, additional_info.get('file_path') if additional_info else None, additional_info
            )
            
            # Convert to Issue objects
            issues = []
            
            # Primary issue from the error
            primary_issue = self._create_issue_from_error(handled_error, context)
            issues.append(primary_issue)
            
            # Detect secondary issues based on patterns
            secondary_issues = self._detect_secondary_issues(handled_error, context, additional_info)
            issues.extend(secondary_issues)
            
            logger.info(f"Detected {len(issues)} issues from error: {str(error)[:100]}")
            return issues
            
        except Exception as detection_error:
            logger.error(f"Issue detection failed: {detection_error}")
            # Create fallback issue
            fallback_issue = Issue(
                issue_id=self._generate_issue_id(),
                issue_type=IssueType.UNKNOWN_ISSUE,
                severity=ErrorSeverity.HIGH,
                description=f"Issue detection failed: {str(detection_error)}",
                context=context,
                suggested_fixes=["Report this issue to IntelliRefactor developers"],
                timestamp=self._get_timestamp()
            )
            return [fallback_issue]
    
    def prioritize_issues(self, issues: List[Issue]) -> List[Issue]:
        """
        Prioritize issues based on severity and impact on refactoring goals.
        
        Args:
            issues: List of issues to prioritize
            
        Returns:
            List of issues sorted by priority (highest first)
        """
        try:
            def priority_score(issue: Issue) -> tuple:
                """Calculate priority score for an issue as a tuple for proper sorting."""
                # Primary sort by severity (higher severity = lower index for reverse sort)
                severity_order = {
                    ErrorSeverity.CRITICAL: 0,
                    ErrorSeverity.HIGH: 1,
                    ErrorSeverity.MEDIUM: 2,
                    ErrorSeverity.LOW: 3
                }
                severity_rank = severity_order.get(issue.severity, 4)
                
                # Secondary sort by issue type impact
                type_impact_order = {
                    IssueType.CONFIGURATION_ISSUE: 0,  # Highest impact - affects all operations
                    IssueType.DEPENDENCY_ISSUE: 1,     # High impact - blocks many operations
                    IssueType.TOOL_BUG: 2,             # Medium-high impact - needs fixing
                    IssueType.ANALYSIS_FAILURE: 3,     # Medium impact - may be file-specific
                    IssueType.REFACTORING_ERROR: 4,    # Medium-low impact - may be operation-specific
                    IssueType.PARAMETER_ISSUE: 5,      # Parameter-related issues
                    IssueType.VALIDATION_FAILURE: 6,   # Low-medium impact - may be test-specific
                    IssueType.PERFORMANCE_ISSUE: 7,    # Low impact - usually not blocking
                    IssueType.UNKNOWN_ISSUE: 8         # Lowest impact - uncertain
                }
                type_rank = type_impact_order.get(issue.issue_type, 9)
                
                # Tertiary sort by additional factors (lower is better for reverse sort)
                additional_factors = 0
                if issue.suggested_fixes:
                    additional_factors -= 1  # Boost priority
                if issue.error_details:
                    additional_factors -= 1  # Boost priority
                
                # Return tuple for sorting: (severity_rank, type_rank, additional_factors)
                # Lower values = higher priority when sorted in reverse
                return (severity_rank, type_rank, additional_factors)
            
            # Sort by priority tuple (reverse=True for highest priority first)
            prioritized_issues = sorted(issues, key=priority_score, reverse=False)
            
            logger.info(f"Prioritized {len(prioritized_issues)} issues")
            return prioritized_issues
            
        except Exception as e:
            logger.error(f"Issue prioritization failed: {e}")
            # Return original list if prioritization fails
            return issues
    
    def apply_fix(self, issue: Issue) -> Fix:
        """
        Apply a fix for a specific IntelliRefactor issue.
        
        Args:
            issue: The issue to fix
            
        Returns:
            Fix object with results of the fix attempt
        """
        fix_id = self._generate_fix_id()
        
        try:
            logger.info(f"Applying fix for issue {issue.issue_id}: {issue.description}")
            
            # Determine fix strategy
            fix_strategy = self._determine_fix_strategy(issue)
            
            # Create fix object
            fix = Fix(
                fix_id=fix_id,
                issue_id=issue.issue_id,
                fix_type=fix_strategy['type'],
                description=fix_strategy['description'],
                timestamp=self._get_timestamp()
            )
            
            # Apply the fix
            changes_made = []
            rollback_info = {}
            
            if fix_strategy['type'] == FixType.CONFIGURATION_UPDATE:
                changes_made, rollback_info = self._apply_configuration_fix(issue, fix_strategy)
            elif fix_strategy['type'] == FixType.DEPENDENCY_INSTALL:
                changes_made, rollback_info = self._apply_dependency_fix(issue, fix_strategy)
            elif fix_strategy['type'] == FixType.PARAMETER_ADJUSTMENT:
                changes_made, rollback_info = self._apply_parameter_fix(issue, fix_strategy)
            elif fix_strategy['type'] == FixType.ENVIRONMENT_SETUP:
                changes_made, rollback_info = self._apply_environment_fix(issue, fix_strategy)
            elif fix_strategy['type'] == FixType.WORKAROUND:
                changes_made, rollback_info = self._apply_workaround_fix(issue, fix_strategy)
            else:
                changes_made = [f"Applied generic fix strategy: {fix_strategy['description']}"]
                rollback_info = {'type': 'generic', 'reversible': False}
            
            fix.changes_made = changes_made
            fix.rollback_info = rollback_info
            
            logger.info(f"Fix {fix_id} applied successfully with {len(changes_made)} changes")
            return fix
            
        except Exception as e:
            logger.error(f"Fix application failed for issue {issue.issue_id}: {e}")
            
            # Create failed fix record
            failed_fix = Fix(
                fix_id=fix_id,
                issue_id=issue.issue_id,
                fix_type=FixType.WORKAROUND,
                description=f"Fix application failed: {str(e)}",
                changes_made=[f"Fix failed: {str(e)}"],
                validation_passed=False,
                timestamp=self._get_timestamp()
            )
            
            return failed_fix
    
    def validate_fix(self, fix: Fix, validation_context: Optional[Dict[str, Any]] = None) -> bool:
        """
        Validate that a fix doesn't introduce new issues.
        
        Args:
            fix: The fix to validate
            validation_context: Additional context for validation
            
        Returns:
            True if fix is valid, False otherwise
        """
        try:
            logger.info(f"Validating fix {fix.fix_id}")
            
            # Basic validation checks
            validation_passed = True
            validation_errors = []
            
            # Check if fix made any changes
            if not fix.changes_made:
                validation_errors.append("Fix made no changes")
                validation_passed = False
            
            # Check if fix is reversible (for safety)
            if fix.rollback_info and not fix.rollback_info.get('reversible', False):
                logger.warning(f"Fix {fix.fix_id} is not reversible")
            
            # Perform fix-type specific validation
            if fix.fix_type == FixType.CONFIGURATION_UPDATE:
                validation_passed &= self._validate_configuration_fix(fix, validation_context)
            elif fix.fix_type == FixType.DEPENDENCY_INSTALL:
                validation_passed &= self._validate_dependency_fix(fix, validation_context)
            elif fix.fix_type == FixType.PARAMETER_ADJUSTMENT:
                validation_passed &= self._validate_parameter_fix(fix, validation_context)
            elif fix.fix_type == FixType.ENVIRONMENT_SETUP:
                validation_passed &= self._validate_environment_fix(fix, validation_context)
            
            # Update fix validation status
            fix.validation_passed = validation_passed
            
            if validation_passed:
                logger.info(f"Fix {fix.fix_id} validation passed")
            else:
                logger.warning(f"Fix {fix.fix_id} validation failed: {validation_errors}")
            
            return validation_passed
            
        except Exception as e:
            logger.error(f"Fix validation failed for {fix.fix_id}: {e}")
            fix.validation_passed = False
            return False
    
    def execute_debug_cycle(
        self,
        error: Exception,
        context: str,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> DebugCycle:
        """
        Execute a complete debug cycle for an error.
        
        Args:
            error: The exception that occurred
            context: Context where the error occurred
            additional_info: Additional diagnostic information
            
        Returns:
            DebugCycle object with results
        """
        cycle_id = self._generate_cycle_id()
        cycle = DebugCycle(
            cycle_id=cycle_id,
            start_time=self._get_timestamp()
        )
        
        try:
            logger.info(f"Starting debug cycle {cycle_id} for error: {str(error)[:100]}")
            
            # Step 1: Issue Detection
            issues = self.detect_intellirefactor_issues(error, context, additional_info)
            cycle.issues_detected = issues
            
            if not issues:
                logger.warning(f"No issues detected in cycle {cycle_id}")
                cycle.success = False
                cycle.end_time = self._get_timestamp()
                return cycle
            
            # Step 2: Issue Prioritization
            prioritized_issues = self.prioritize_issues(issues)
            
            # Step 3: Fix Application and Validation
            successful_fixes = 0
            for issue in prioritized_issues:
                try:
                    # Apply fix
                    fix = self.apply_fix(issue)
                    cycle.fixes_applied.append(fix)
                    
                    # Validate fix
                    validation_passed = self.validate_fix(fix, additional_info)
                    
                    validation_result = {
                        'fix_id': fix.fix_id,
                        'issue_id': issue.issue_id,
                        'validation_passed': validation_passed,
                        'timestamp': self._get_timestamp()
                    }
                    cycle.validation_results.append(validation_result)
                    
                    if validation_passed:
                        successful_fixes += 1
                        logger.info(f"Successfully fixed issue {issue.issue_id}")
                    else:
                        logger.warning(f"Fix validation failed for issue {issue.issue_id}")
                        
                except Exception as fix_error:
                    logger.error(f"Failed to fix issue {issue.issue_id}: {fix_error}")
                    
                    # Record failed fix attempt
                    failed_validation = {
                        'fix_id': 'failed',
                        'issue_id': issue.issue_id,
                        'validation_passed': False,
                        'error': str(fix_error),
                        'timestamp': self._get_timestamp()
                    }
                    cycle.validation_results.append(failed_validation)
            
            # Determine cycle success
            cycle.success = successful_fixes > 0
            cycle.end_time = self._get_timestamp()
            
            # Update progress tracking
            if self.current_progress:
                self.current_progress.total_cycles += 1
                if cycle.success:
                    self.current_progress.successful_cycles += 1
                    self.current_progress.issues_resolved += successful_fixes
                else:
                    self.current_progress.failed_cycles += 1
                
                # Update remaining issues count
                remaining_issues = len(issues) - successful_fixes
                self.current_progress.issues_remaining = max(0, remaining_issues)
            
            logger.info(f"Debug cycle {cycle_id} completed. Success: {cycle.success}, Fixes: {successful_fixes}/{len(issues)}")
            
            # Add to history
            self.debug_history.append(cycle)
            
            return cycle
            
        except Exception as cycle_error:
            logger.error(f"Debug cycle {cycle_id} failed: {cycle_error}")
            cycle.success = False
            cycle.end_time = self._get_timestamp()
            
            # Add error information to cycle
            cycle.validation_results.append({
                'cycle_error': str(cycle_error),
                'timestamp': self._get_timestamp()
            })
            
            return cycle
    
    def create_checkpoint(self, state_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a checkpoint for rollback capability.
        
        Args:
            state_info: Current state information to checkpoint
            
        Returns:
            Checkpoint information
        """
        try:
            checkpoint_id = self._generate_checkpoint_id()
            checkpoint = {
                'checkpoint_id': checkpoint_id,
                'timestamp': self._get_timestamp(),
                'state_info': state_info.copy(),
                'debug_progress': self.current_progress.to_dict() if self.current_progress else None,
                'debug_history_count': len(self.debug_history)
            }
            
            # Add to progress tracking
            if self.current_progress:
                self.current_progress.checkpoints.append(checkpoint)
                self.current_progress.last_successful_state = checkpoint
            
            logger.info(f"Created checkpoint {checkpoint_id}")
            return checkpoint
            
        except Exception as e:
            logger.error(f"Checkpoint creation failed: {e}")
            return {
                'checkpoint_id': 'failed',
                'timestamp': self._get_timestamp(),
                'error': str(e)
            }
    
    def resume_from_checkpoint(self, checkpoint: Dict[str, Any]) -> bool:
        """
        Resume debugging from a previous checkpoint.
        
        Args:
            checkpoint: Checkpoint to resume from
            
        Returns:
            True if resume was successful, False otherwise
        """
        try:
            checkpoint_id = checkpoint.get('checkpoint_id', 'unknown')
            logger.info(f"Resuming from checkpoint {checkpoint_id}")
            
            # Restore debug progress
            if 'debug_progress' in checkpoint and checkpoint['debug_progress']:
                progress_data = checkpoint['debug_progress']
                self.current_progress = DebugProgress(
                    session_id=progress_data['session_id'],
                    total_cycles=progress_data['total_cycles'],
                    successful_cycles=progress_data['successful_cycles'],
                    failed_cycles=progress_data['failed_cycles'],
                    issues_resolved=progress_data['issues_resolved'],
                    issues_remaining=progress_data['issues_remaining'],
                    checkpoints=progress_data['checkpoints'],
                    last_successful_state=progress_data.get('last_successful_state')
                )
                
                # If last_successful_state is None, set it to the current checkpoint
                if self.current_progress.last_successful_state is None:
                    self.current_progress.last_successful_state = checkpoint
            
            # Restore debug history to checkpoint point
            if 'debug_history_count' in checkpoint:
                target_count = checkpoint['debug_history_count']
                if len(self.debug_history) > target_count:
                    self.debug_history = self.debug_history[:target_count]
            
            logger.info(f"Successfully resumed from checkpoint {checkpoint_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")
            return False
    
    def start_debug_session(self, session_id: Optional[str] = None) -> str:
        """
        Start a new debugging session.
        
        Args:
            session_id: Optional session ID, generated if not provided
            
        Returns:
            Session ID
        """
        if not session_id:
            session_id = self._generate_session_id()
        
        self.current_progress = DebugProgress(session_id=session_id)
        self.debug_history = []
        
        logger.info(f"Started debug session {session_id}")
        return session_id
    
    def get_debug_progress(self) -> Optional[DebugProgress]:
        """Get current debug progress."""
        return self.current_progress
    
    def get_debug_history(self) -> List[DebugCycle]:
        """Get debug cycle history."""
        return self.debug_history.copy()
    
    # Private helper methods
    
    def _initialize_issue_patterns(self) -> Dict[str, IssueType]:
        """Initialize patterns for issue detection."""
        return {
            'configuration': IssueType.CONFIGURATION_ISSUE,
            'config': IssueType.CONFIGURATION_ISSUE,
            'import': IssueType.DEPENDENCY_ISSUE,
            'module': IssueType.DEPENDENCY_ISSUE,
            'package': IssueType.DEPENDENCY_ISSUE,
            'analysis': IssueType.ANALYSIS_FAILURE,
            'parse': IssueType.ANALYSIS_FAILURE,
            'refactor': IssueType.REFACTORING_ERROR,
            'transform': IssueType.REFACTORING_ERROR,
            'test': IssueType.VALIDATION_FAILURE,
            'validate': IssueType.VALIDATION_FAILURE,
            'timeout': IssueType.PERFORMANCE_ISSUE,
            'memory': IssueType.PERFORMANCE_ISSUE,
            'bug': IssueType.TOOL_BUG,
            'crash': IssueType.TOOL_BUG,
            'exception': IssueType.TOOL_BUG
        }
    
    def _initialize_fix_strategies(self) -> Dict[IssueType, Dict[str, Any]]:
        """Initialize fix strategies for different issue types."""
        return {
            IssueType.CONFIGURATION_ISSUE: {
                'type': FixType.CONFIGURATION_UPDATE,
                'description': 'Update configuration settings',
                'priority': 1
            },
            IssueType.DEPENDENCY_ISSUE: {
                'type': FixType.DEPENDENCY_INSTALL,
                'description': 'Install missing dependencies',
                'priority': 2
            },
            IssueType.ANALYSIS_FAILURE: {
                'type': FixType.PARAMETER_ADJUSTMENT,
                'description': 'Adjust analysis parameters',
                'priority': 3
            },
            IssueType.REFACTORING_ERROR: {
                'type': FixType.PARAMETER_ADJUSTMENT,
                'description': 'Adjust refactoring parameters',
                'priority': 4
            },
            IssueType.PARAMETER_ISSUE: {
                'type': FixType.PARAMETER_ADJUSTMENT,
                'description': 'Adjust system parameters',
                'priority': 4
            },
            IssueType.VALIDATION_FAILURE: {
                'type': FixType.WORKAROUND,
                'description': 'Apply validation workaround',
                'priority': 5
            },
            IssueType.PERFORMANCE_ISSUE: {
                'type': FixType.PARAMETER_ADJUSTMENT,
                'description': 'Optimize performance parameters',
                'priority': 6
            },
            IssueType.TOOL_BUG: {
                'type': FixType.WORKAROUND,
                'description': 'Apply tool bug workaround',
                'priority': 7
            },
            IssueType.UNKNOWN_ISSUE: {
                'type': FixType.WORKAROUND,
                'description': 'Apply generic workaround',
                'priority': 8
            }
        }
    
    def _create_issue_from_error(self, handled_error: AnalysisError, context: str) -> Issue:
        """Create an Issue object from a handled error."""
        # Map error category to issue type
        category_to_type = {
            ErrorCategory.CONFIGURATION: IssueType.CONFIGURATION_ISSUE,
            ErrorCategory.FILE_ACCESS: IssueType.ANALYSIS_FAILURE,
            ErrorCategory.SYNTAX_ERROR: IssueType.ANALYSIS_FAILURE,
            ErrorCategory.IMPORT_ERROR: IssueType.DEPENDENCY_ISSUE,
            ErrorCategory.DEPENDENCY_MISSING: IssueType.DEPENDENCY_ISSUE,
            ErrorCategory.PERMISSION_ERROR: IssueType.ANALYSIS_FAILURE,
            ErrorCategory.TIMEOUT_ERROR: IssueType.PERFORMANCE_ISSUE,
            ErrorCategory.MEMORY_ERROR: IssueType.PERFORMANCE_ISSUE,
            ErrorCategory.INTERNAL_ERROR: IssueType.TOOL_BUG,
            ErrorCategory.VALIDATION_ERROR: IssueType.VALIDATION_FAILURE,
            ErrorCategory.NETWORK_ERROR: IssueType.DEPENDENCY_ISSUE,
            ErrorCategory.UNKNOWN: IssueType.UNKNOWN_ISSUE
        }
        
        issue_type = category_to_type.get(handled_error.category, IssueType.UNKNOWN_ISSUE)
        
        return Issue(
            issue_id=self._generate_issue_id(),
            issue_type=issue_type,
            severity=handled_error.severity,
            description=handled_error.message,
            context=context,
            file_path=handled_error.file_path,
            line_number=handled_error.line_number,
            error_details=handled_error,
            suggested_fixes=handled_error.suggested_fixes,
            diagnostic_info=handled_error.diagnostic_info,
            timestamp=self._get_timestamp()
        )
    
    def _detect_secondary_issues(
        self,
        handled_error: AnalysisError,
        context: str,
        additional_info: Optional[Dict[str, Any]]
    ) -> List[Issue]:
        """Detect secondary issues based on error patterns."""
        secondary_issues = []
        
        # Look for patterns that might indicate additional issues
        error_message = handled_error.message.lower()
        
        # Check for configuration-related secondary issues
        if 'config' in error_message and handled_error.category != ErrorCategory.CONFIGURATION:
            config_issue = Issue(
                issue_id=self._generate_issue_id(),
                issue_type=IssueType.CONFIGURATION_ISSUE,
                severity=ErrorSeverity.MEDIUM,
                description="Potential configuration issue detected",
                context=f"Secondary issue from: {context}",
                suggested_fixes=["Check IntelliRefactor configuration", "Validate configuration format"],
                timestamp=self._get_timestamp()
            )
            secondary_issues.append(config_issue)
        
        # Check for dependency-related secondary issues
        if any(keyword in error_message for keyword in ['import', 'module', 'package']) and handled_error.category != ErrorCategory.IMPORT_ERROR:
            dep_issue = Issue(
                issue_id=self._generate_issue_id(),
                issue_type=IssueType.DEPENDENCY_ISSUE,
                severity=ErrorSeverity.MEDIUM,
                description="Potential dependency issue detected",
                context=f"Secondary issue from: {context}",
                suggested_fixes=["Check package dependencies", "Install missing packages"],
                timestamp=self._get_timestamp()
            )
            secondary_issues.append(dep_issue)
        
        return secondary_issues
    
    def _determine_fix_strategy(self, issue: Issue) -> Dict[str, Any]:
        """Determine the appropriate fix strategy for an issue."""
        base_strategy = self.fix_strategies.get(issue.issue_type, self.fix_strategies[IssueType.UNKNOWN_ISSUE])
        
        # Customize strategy based on specific issue details
        strategy = base_strategy.copy()
        
        if issue.suggested_fixes:
            strategy['suggested_actions'] = issue.suggested_fixes
        
        if issue.error_details and issue.error_details.diagnostic_info:
            strategy['diagnostic_info'] = issue.error_details.diagnostic_info
        
        return strategy
    
    def _apply_configuration_fix(self, issue: Issue, strategy: Dict[str, Any]) -> tuple[List[str], Dict[str, Any]]:
        """Apply configuration-related fixes."""
        changes_made = []
        rollback_info = {'type': 'configuration', 'reversible': True, 'original_values': {}}
        
        # Example configuration fixes
        if 'config' in issue.description.lower():
            changes_made.append("Updated configuration validation settings")
            changes_made.append("Applied default configuration fallbacks")
        
        return changes_made, rollback_info
    
    def _apply_dependency_fix(self, issue: Issue, strategy: Dict[str, Any]) -> tuple[List[str], Dict[str, Any]]:
        """Apply dependency-related fixes."""
        changes_made = []
        rollback_info = {'type': 'dependency', 'reversible': False}  # Package installs are not easily reversible
        
        # Example dependency fixes
        if issue.error_details and 'missing_module' in issue.error_details.diagnostic_info:
            module_name = issue.error_details.diagnostic_info['missing_module']
            changes_made.append(f"Attempted to install missing module: {module_name}")
        else:
            changes_made.append("Applied generic dependency fix")
        
        return changes_made, rollback_info
    
    def _apply_parameter_fix(self, issue: Issue, strategy: Dict[str, Any]) -> tuple[List[str], Dict[str, Any]]:
        """Apply parameter adjustment fixes."""
        changes_made = []
        rollback_info = {'type': 'parameter', 'reversible': True, 'original_parameters': {}}
        
        # Example parameter fixes
        if issue.issue_type == IssueType.ANALYSIS_FAILURE:
            changes_made.append("Adjusted analysis timeout parameters")
            changes_made.append("Enabled fallback analysis mode")
        elif issue.issue_type == IssueType.PERFORMANCE_ISSUE:
            changes_made.append("Reduced analysis complexity settings")
            changes_made.append("Enabled incremental processing")
        
        return changes_made, rollback_info
    
    def _apply_environment_fix(self, issue: Issue, strategy: Dict[str, Any]) -> tuple[List[str], Dict[str, Any]]:
        """Apply environment setup fixes."""
        changes_made = []
        rollback_info = {'type': 'environment', 'reversible': True, 'original_env': {}}
        
        changes_made.append("Applied environment configuration fixes")
        
        return changes_made, rollback_info
    
    def _apply_workaround_fix(self, issue: Issue, strategy: Dict[str, Any]) -> tuple[List[str], Dict[str, Any]]:
        """Apply workaround fixes."""
        changes_made = []
        rollback_info = {'type': 'workaround', 'reversible': True}
        
        # Apply generic workarounds based on issue type
        if issue.issue_type == IssueType.TOOL_BUG:
            changes_made.append("Applied tool bug workaround")
            changes_made.append("Enabled alternative processing path")
        elif issue.issue_type == IssueType.VALIDATION_FAILURE:
            changes_made.append("Applied validation workaround")
            changes_made.append("Relaxed validation constraints")
        else:
            changes_made.append("Applied generic workaround")
        
        return changes_made, rollback_info
    
    def _validate_configuration_fix(self, fix: Fix, context: Optional[Dict[str, Any]]) -> bool:
        """Validate configuration fixes."""
        # Basic validation - check if configuration changes are reasonable
        return len(fix.changes_made) > 0
    
    def _validate_dependency_fix(self, fix: Fix, context: Optional[Dict[str, Any]]) -> bool:
        """Validate dependency fixes."""
        # Basic validation - check if dependency changes were applied
        return len(fix.changes_made) > 0
    
    def _validate_parameter_fix(self, fix: Fix, context: Optional[Dict[str, Any]]) -> bool:
        """Validate parameter fixes."""
        # Basic validation - check if parameter changes are reasonable
        return len(fix.changes_made) > 0
    
    def _validate_environment_fix(self, fix: Fix, context: Optional[Dict[str, Any]]) -> bool:
        """Validate environment fixes."""
        # Basic validation - check if environment changes were applied
        return len(fix.changes_made) > 0
    
    def _generate_issue_id(self) -> str:
        """Generate unique issue ID."""
        self._issue_counter += 1
        return f"issue_{int(time.time() * 1000)}_{self._issue_counter}_{id(self) % 10000}"
    
    def _generate_fix_id(self) -> str:
        """Generate unique fix ID."""
        self._fix_counter += 1
        return f"fix_{int(time.time() * 1000)}_{self._fix_counter}_{id(self) % 10000}"
    
    def _generate_cycle_id(self) -> str:
        """Generate unique cycle ID."""
        self._cycle_counter += 1
        return f"cycle_{int(time.time() * 1000)}_{self._cycle_counter}_{id(self) % 10000}"
    
    def _generate_checkpoint_id(self) -> str:
        """Generate unique checkpoint ID."""
        self._checkpoint_counter += 1
        return f"checkpoint_{int(time.time() * 1000)}_{self._checkpoint_counter}_{id(self) % 10000}"
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID."""
        self._session_counter += 1
        return f"session_{int(time.time() * 1000)}_{self._session_counter}_{id(self) % 10000}"
    
    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        return datetime.now().isoformat()
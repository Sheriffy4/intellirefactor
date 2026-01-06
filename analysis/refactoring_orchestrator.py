"""
Refactoring Orchestrator - Manages incremental refactoring with validation and rollback.
"""

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import ast
import time

logger = logging.getLogger(__name__)


@dataclass
class RefactoringStep:
    """Represents a single refactoring step."""
    step_id: str
    description: str
    target_file: Path
    operation_type: str  # 'extract_method', 'create_component', 'update_imports', etc.
    parameters: Dict[str, Any]
    backup_path: Optional[Path] = None
    completed: bool = False
    validation_passed: bool = False
    error_message: Optional[str] = None


@dataclass
class RefactoringPlan:
    """Complete refactoring plan with steps and validation."""
    plan_id: str
    target_file: Path
    steps: List[RefactoringStep]
    backup_directory: Path
    created_files: List[Path]
    modified_files: List[Path]
    rollback_available: bool = True


class RefactoringOrchestrator:
    """
    Orchestrates incremental refactoring with comprehensive validation and rollback.
    
    This class implements the orchestration system from the design document,
    providing safe, incremental refactoring with automatic validation and rollback.
    """
    
    def __init__(self, backup_enabled: bool = True, validation_enabled: bool = True):
        """
        Initialize the refactoring orchestrator.
        
        Args:
            backup_enabled: Whether to create backups before refactoring
            validation_enabled: Whether to validate each step
        """
        self.backup_enabled = backup_enabled
        self.validation_enabled = validation_enabled
        self.active_plans: Dict[str, RefactoringPlan] = {}
        self.temp_dir = Path(tempfile.mkdtemp(prefix="intellirefactor_"))
        
        logger.info(f"RefactoringOrchestrator initialized with temp dir: {self.temp_dir}")
    
    def create_refactoring_plan(
        self, 
        target_file: Path, 
        refactoring_opportunities: List[Dict[str, Any]]
    ) -> RefactoringPlan:
        """
        Create a comprehensive refactoring plan from opportunities.
        
        Args:
            target_file: File to refactor
            refactoring_opportunities: List of identified opportunities
            
        Returns:
            RefactoringPlan: Complete plan with validation steps
        """
        plan_id = f"refactor_{target_file.stem}_{int(time.time())}"
        backup_dir = self.temp_dir / plan_id
        backup_dir.mkdir(exist_ok=True)
        
        steps = []
        
        # Step 1: Create backup
        if self.backup_enabled:
            steps.append(RefactoringStep(
                step_id=f"{plan_id}_backup",
                description=f"Create backup of {target_file.name}",
                target_file=target_file,
                operation_type="create_backup",
                parameters={"backup_dir": str(backup_dir)}
            ))
        
        # Step 2: Validate original file
        steps.append(RefactoringStep(
            step_id=f"{plan_id}_validate_original",
            description=f"Validate original file syntax",
            target_file=target_file,
            operation_type="validate_syntax",
            parameters={}
        ))
        
        # Step 3-N: Apply refactoring opportunities
        for i, opportunity in enumerate(refactoring_opportunities):
            steps.append(RefactoringStep(
                step_id=f"{plan_id}_apply_{i}",
                description=f"Apply: {opportunity.get('description', 'Unknown')}",
                target_file=target_file,
                operation_type="apply_opportunity",
                parameters={"opportunity": opportunity}
            ))
        
        # Final step: Validate result
        steps.append(RefactoringStep(
            step_id=f"{plan_id}_validate_final",
            description="Validate final refactored code",
            target_file=target_file,
            operation_type="validate_final",
            parameters={}
        ))
        
        plan = RefactoringPlan(
            plan_id=plan_id,
            target_file=target_file,
            steps=steps,
            backup_directory=backup_dir,
            created_files=[],
            modified_files=[]
        )
        
        self.active_plans[plan_id] = plan
        logger.info(f"Created refactoring plan {plan_id} with {len(steps)} steps")
        
        return plan
    
    def execute_plan(self, plan: RefactoringPlan) -> Dict[str, Any]:
        """
        Execute a refactoring plan with incremental validation.
        
        Args:
            plan: RefactoringPlan to execute
            
        Returns:
            Dict with execution results
        """
        results = {
            'success': False,
            'completed_steps': 0,
            'total_steps': len(plan.steps),
            'errors': [],
            'warnings': [],
            'created_files': [],
            'modified_files': []
        }
        
        try:
            for i, step in enumerate(plan.steps):
                logger.info(f"Executing step {i+1}/{len(plan.steps)}: {step.description}")
                
                # Execute the step
                step_result = self._execute_step(step, plan)
                
                if not step_result['success']:
                    results['errors'].append(f"Step {i+1} failed: {step_result.get('error', 'Unknown error')}")
                    
                    # Attempt rollback
                    if plan.rollback_available:
                        logger.warning(f"Step {i+1} failed, attempting rollback")
                        rollback_result = self.rollback_plan(plan)
                        if rollback_result['success']:
                            results['warnings'].append("Rollback completed successfully")
                        else:
                            results['errors'].append(f"Rollback failed: {rollback_result.get('error', 'Unknown error')}")
                    
                    return results
                
                step.completed = True
                results['completed_steps'] += 1
                
                # Validate step if enabled
                if self.validation_enabled and step.operation_type != 'validate_syntax':
                    validation_result = self._validate_step(step, plan)
                    step.validation_passed = validation_result['success']
                    
                    if not validation_result['success']:
                        results['errors'].append(f"Step {i+1} validation failed: {validation_result.get('error', 'Unknown error')}")
                        
                        # Attempt rollback
                        if plan.rollback_available:
                            logger.warning(f"Step {i+1} validation failed, attempting rollback")
                            rollback_result = self.rollback_plan(plan)
                            if rollback_result['success']:
                                results['warnings'].append("Rollback completed successfully")
                            else:
                                results['errors'].append(f"Rollback failed: {rollback_result.get('error', 'Unknown error')}")
                        
                        return results
            
            # All steps completed successfully
            results['success'] = True
            results['created_files'] = [str(f) for f in plan.created_files]
            results['modified_files'] = [str(f) for f in plan.modified_files]
            
            logger.info(f"Refactoring plan {plan.plan_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Unexpected error during plan execution: {e}")
            results['errors'].append(f"Unexpected error: {str(e)}")
            
            # Attempt emergency rollback
            if plan.rollback_available:
                try:
                    rollback_result = self.rollback_plan(plan)
                    if rollback_result['success']:
                        results['warnings'].append("Emergency rollback completed")
                    else:
                        results['errors'].append(f"Emergency rollback failed: {rollback_result.get('error', 'Unknown error')}")
                except Exception as rollback_error:
                    results['errors'].append(f"Emergency rollback exception: {str(rollback_error)}")
        
        return results
    
    def _execute_step(self, step: RefactoringStep, plan: RefactoringPlan) -> Dict[str, Any]:
        """Execute a single refactoring step."""
        try:
            if step.operation_type == "create_backup":
                return self._create_backup(step, plan)
            elif step.operation_type == "validate_syntax":
                return self._validate_syntax(step)
            elif step.operation_type == "apply_opportunity":
                return self._apply_opportunity(step, plan)
            elif step.operation_type == "validate_final":
                return self._validate_final(step, plan)
            else:
                return {
                    'success': False,
                    'error': f"Unknown operation type: {step.operation_type}"
                }
        except Exception as e:
            logger.error(f"Error executing step {step.step_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_backup(self, step: RefactoringStep, plan: RefactoringPlan) -> Dict[str, Any]:
        """Create backup of the target file."""
        try:
            backup_dir = Path(step.parameters["backup_dir"])
            backup_file = backup_dir / f"{step.target_file.name}.backup"
            
            shutil.copy2(step.target_file, backup_file)
            step.backup_path = backup_file
            
            logger.info(f"Backup created: {backup_file}")
            return {'success': True, 'backup_path': str(backup_file)}
            
        except Exception as e:
            return {'success': False, 'error': f"Backup creation failed: {str(e)}"}
    
    def _validate_syntax(self, step: RefactoringStep) -> Dict[str, Any]:
        """Validate Python syntax of the target file."""
        try:
            with open(step.target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Try to parse the file
            ast.parse(content)
            
            logger.debug(f"Syntax validation passed for {step.target_file}")
            return {'success': True}
            
        except SyntaxError as e:
            error_msg = f"Syntax error in {step.target_file}: {e}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
        except Exception as e:
            error_msg = f"Validation error for {step.target_file}: {str(e)}"
            logger.error(error_msg)
            return {'success': False, 'error': error_msg}
    
    def _apply_opportunity(self, step: RefactoringStep, plan: RefactoringPlan) -> Dict[str, Any]:
        """Apply a refactoring opportunity."""
        try:
            # This is a placeholder - in real implementation, this would
            # call the appropriate refactoring method based on opportunity type
            opportunity = step.parameters["opportunity"]
            
            logger.info(f"Applying opportunity: {opportunity.get('description', 'Unknown')}")
            
            # For now, just mark as successful
            # In real implementation, this would call AutoRefactor methods
            return {'success': True, 'message': 'Opportunity applied (placeholder)'}
            
        except Exception as e:
            return {'success': False, 'error': f"Failed to apply opportunity: {str(e)}"}
    
    def _validate_final(self, step: RefactoringStep, plan: RefactoringPlan) -> Dict[str, Any]:
        """Validate the final refactored code."""
        try:
            # Validate syntax
            syntax_result = self._validate_syntax(step)
            if not syntax_result['success']:
                return syntax_result
            
            # Additional validations could go here:
            # - Import validation
            # - Class structure validation
            # - Method signature validation
            
            logger.info(f"Final validation passed for {step.target_file}")
            return {'success': True}
            
        except Exception as e:
            return {'success': False, 'error': f"Final validation failed: {str(e)}"}
    
    def _validate_step(self, step: RefactoringStep, plan: RefactoringPlan) -> Dict[str, Any]:
        """Validate a completed step."""
        # For most steps, syntax validation is sufficient
        return self._validate_syntax(step)
    
    def rollback_plan(self, plan: RefactoringPlan) -> Dict[str, Any]:
        """
        Rollback a refactoring plan to its original state.
        
        Args:
            plan: RefactoringPlan to rollback
            
        Returns:
            Dict with rollback results
        """
        try:
            # Find the backup step
            backup_step = None
            for step in plan.steps:
                if step.operation_type == "create_backup" and step.backup_path:
                    backup_step = step
                    break
            
            if not backup_step:
                return {'success': False, 'error': 'No backup available for rollback'}
            
            # Restore from backup
            shutil.copy2(backup_step.backup_path, plan.target_file)
            
            # Remove created files
            for created_file in plan.created_files:
                if created_file.exists():
                    created_file.unlink()
            
            # Reset step states
            for step in plan.steps:
                step.completed = False
                step.validation_passed = False
            
            logger.info(f"Rollback completed for plan {plan.plan_id}")
            return {'success': True, 'message': 'Rollback completed successfully'}
            
        except Exception as e:
            logger.error(f"Rollback failed for plan {plan.plan_id}: {e}")
            return {'success': False, 'error': f"Rollback failed: {str(e)}"}
    
    def cleanup_plan(self, plan: RefactoringPlan) -> None:
        """Clean up temporary files for a completed plan."""
        try:
            if plan.backup_directory.exists():
                shutil.rmtree(plan.backup_directory)
            
            if plan.plan_id in self.active_plans:
                del self.active_plans[plan.plan_id]
            
            logger.info(f"Cleaned up plan {plan.plan_id}")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup plan {plan.plan_id}: {e}")
    
    def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a refactoring plan."""
        if plan_id not in self.active_plans:
            return None
        
        plan = self.active_plans[plan_id]
        completed_steps = sum(1 for step in plan.steps if step.completed)
        validated_steps = sum(1 for step in plan.steps if step.validation_passed)
        
        return {
            'plan_id': plan_id,
            'target_file': str(plan.target_file),
            'total_steps': len(plan.steps),
            'completed_steps': completed_steps,
            'validated_steps': validated_steps,
            'rollback_available': plan.rollback_available,
            'created_files': [str(f) for f in plan.created_files],
            'modified_files': [str(f) for f in plan.modified_files]
        }
    
    def __del__(self):
        """Cleanup temporary directory on destruction."""
        try:
            if hasattr(self, 'temp_dir') and self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception:
            pass  # Ignore cleanup errors during destruction
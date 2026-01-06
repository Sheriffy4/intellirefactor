"""
Rollback Manager for IntelliRefactor

Provides comprehensive rollback capabilities for safe refactoring operations.
"""

import os
import shutil
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class RollbackResult:
    """Result of rollback operation"""
    success: bool
    restored_files: Optional[List[str]] = None
    failed_files: Optional[List[str]] = None
    rollback_info: Optional[Dict[str, any]] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None


@dataclass
class RollbackPoint:
    """Represents a rollback point in the system"""
    id: str
    operation_id: str
    description: str
    backup_path: str
    target_files: List[str]
    timestamp: datetime
    status: str  # 'active', 'completed', 'rolled_back'


class RollbackManager:
    """
    Manages rollback operations and rollback points for safe refactoring
    """

    def __init__(self, rollback_history_file: Optional[str] = None):
        """
        Initialize rollback manager
        
        Args:
            rollback_history_file: File to store rollback history (optional)
        """
        self.rollback_history_file = rollback_history_file or os.path.join(
            os.path.expanduser('~'), '.intellirefactor', 'rollback_history.json'
        )
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.rollback_history_file), exist_ok=True)
        
        # Load rollback history
        self.rollback_history = self._load_rollback_history()
        
        # Track active rollback points
        self.active_rollback_points: Dict[str, RollbackPoint] = {}

    def create_rollback_point(self, operation_id: str, backup_path: str, 
                            target_files: List[str], description: str = "") -> str:
        """
        Create a rollback point for an operation
        
        Args:
            operation_id: Unique identifier for the operation
            backup_path: Path to the backup for this operation
            target_files: Files that will be modified
            description: Description of the operation
            
        Returns:
            Rollback point ID
        """
        rollback_id = f"rollback_{operation_id}_{int(datetime.now().timestamp())}"
        
        rollback_point = RollbackPoint(
            id=rollback_id,
            operation_id=operation_id,
            description=description,
            backup_path=backup_path,
            target_files=target_files.copy(),
            timestamp=datetime.now(),
            status='active'
        )
        
        self.active_rollback_points[rollback_id] = rollback_point
        
        # Add to history
        self.rollback_history[rollback_id] = {
            'operation_id': operation_id,
            'description': description,
            'backup_path': backup_path,
            'target_files': target_files,
            'timestamp': rollback_point.timestamp.isoformat(),
            'status': 'active'
        }
        
        self._save_rollback_history()
        logger.info(f"Created rollback point {rollback_id} for operation {operation_id}")
        
        return rollback_id

    def rollback_to_point(self, rollback_id: str, 
                         selective_files: Optional[List[str]] = None) -> RollbackResult:
        """
        Rollback to a specific rollback point
        
        Args:
            rollback_id: ID of the rollback point
            selective_files: Specific files to rollback (None = all files)
            
        Returns:
            RollbackResult with rollback details
        """
        try:
            # Get rollback point info
            if rollback_id in self.active_rollback_points:
                rollback_point = self.active_rollback_points[rollback_id]
            elif rollback_id in self.rollback_history:
                # Reconstruct from history
                history_entry = self.rollback_history[rollback_id]
                rollback_point = RollbackPoint(
                    id=rollback_id,
                    operation_id=history_entry['operation_id'],
                    description=history_entry['description'],
                    backup_path=history_entry['backup_path'],
                    target_files=history_entry['target_files'],
                    timestamp=datetime.fromisoformat(history_entry['timestamp']),
                    status=history_entry['status']
                )
            else:
                return RollbackResult(
                    success=False,
                    error=f"Rollback point {rollback_id} not found"
                )

            # Check if backup exists
            if not os.path.exists(rollback_point.backup_path):
                return RollbackResult(
                    success=False,
                    error=f"Backup not found at {rollback_point.backup_path}"
                )

            # Perform rollback
            result = self.rollback_from_backup(
                rollback_point.backup_path,
                selective_files or rollback_point.target_files
            )

            if result.success:
                # Update rollback point status
                rollback_point.status = 'rolled_back'
                if rollback_id in self.active_rollback_points:
                    self.active_rollback_points[rollback_id] = rollback_point
                
                self.rollback_history[rollback_id]['status'] = 'rolled_back'
                self.rollback_history[rollback_id]['rollback_timestamp'] = datetime.now().isoformat()
                self._save_rollback_history()
                
                logger.info(f"Successfully rolled back to point {rollback_id}")

            return result

        except Exception as e:
            logger.error(f"Failed to rollback to point {rollback_id}: {e}")
            return RollbackResult(
                success=False,
                error=str(e)
            )

    def rollback_from_backup(self, backup_path: str, target_files: List[str]) -> RollbackResult:
        """
        Rollback files from a backup directory
        
        Args:
            backup_path: Path to backup directory
            target_files: Files to restore
            
        Returns:
            RollbackResult with restoration details
        """
        try:
            # Load backup manifest
            manifest_path = os.path.join(backup_path, 'manifest.json')
            if not os.path.exists(manifest_path):
                return RollbackResult(
                    success=False,
                    error="Backup manifest not found"
                )

            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            restored_files = []
            failed_files = []
            
            # Create pre-rollback backup of current state
            pre_rollback_backup = self._create_pre_rollback_backup(target_files)

            for file_path in target_files:
                try:
                    rel_path = os.path.relpath(file_path)
                    backup_file_path = os.path.join(backup_path, rel_path)
                    
                    if not os.path.exists(backup_file_path):
                        # File might have been deleted in the operation
                        if os.path.exists(file_path):
                            # Current file exists but wasn't in backup - remove it
                            os.remove(file_path)
                            restored_files.append(f"{file_path} (removed)")
                        continue

                    # Verify backup file integrity if possible
                    if self._verify_backup_file_integrity(backup_file_path, manifest, file_path):
                        # Create target directory if needed
                        os.makedirs(os.path.dirname(file_path), exist_ok=True)
                        
                        # Restore file
                        shutil.copy2(backup_file_path, file_path)
                        restored_files.append(file_path)
                    else:
                        failed_files.append(f"{file_path} (integrity check failed)")
                        
                except Exception as e:
                    failed_files.append(f"{file_path} ({str(e)})")

            success = len(failed_files) == 0
            
            rollback_info = {
                'backup_path': backup_path,
                'manifest': manifest,
                'pre_rollback_backup': pre_rollback_backup,
                'restored_count': len(restored_files),
                'failed_count': len(failed_files)
            }

            return RollbackResult(
                success=success,
                restored_files=restored_files,
                failed_files=failed_files,
                rollback_info=rollback_info,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Failed to rollback from backup {backup_path}: {e}")
            return RollbackResult(
                success=False,
                error=str(e)
            )

    def complete_rollback_point(self, rollback_id: str) -> bool:
        """
        Mark a rollback point as completed (operation finished successfully)
        
        Args:
            rollback_id: ID of the rollback point to complete
            
        Returns:
            True if completed successfully
        """
        try:
            if rollback_id in self.active_rollback_points:
                self.active_rollback_points[rollback_id].status = 'completed'
                
            if rollback_id in self.rollback_history:
                self.rollback_history[rollback_id]['status'] = 'completed'
                self.rollback_history[rollback_id]['completion_timestamp'] = datetime.now().isoformat()
                
            self._save_rollback_history()
            logger.info(f"Completed rollback point {rollback_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to complete rollback point {rollback_id}: {e}")
            return False

    def list_rollback_points(self, include_completed: bool = True) -> List[Dict[str, any]]:
        """
        List available rollback points
        
        Args:
            include_completed: Whether to include completed rollback points
            
        Returns:
            List of rollback point information
        """
        rollback_points = []
        
        for rollback_id, info in self.rollback_history.items():
            if not include_completed and info['status'] == 'completed':
                continue
                
            point_info = info.copy()
            point_info['rollback_id'] = rollback_id
            point_info['backup_exists'] = os.path.exists(info['backup_path'])
            rollback_points.append(point_info)
        
        # Sort by timestamp (newest first)
        rollback_points.sort(key=lambda x: x['timestamp'], reverse=True)
        return rollback_points

    def get_rollback_point_info(self, rollback_id: str) -> Optional[Dict[str, any]]:
        """Get detailed information about a rollback point"""
        if rollback_id not in self.rollback_history:
            return None

        info = self.rollback_history[rollback_id].copy()
        info['rollback_id'] = rollback_id
        info['backup_exists'] = os.path.exists(info['backup_path'])
        
        # Add active rollback point info if available
        if rollback_id in self.active_rollback_points:
            active_point = self.active_rollback_points[rollback_id]
            info['active_point'] = {
                'status': active_point.status,
                'target_files_count': len(active_point.target_files)
            }

        return info

    def cleanup_rollback_point(self, rollback_id: str, remove_backup: bool = False) -> bool:
        """
        Clean up a rollback point
        
        Args:
            rollback_id: ID of rollback point to clean up
            remove_backup: Whether to also remove the backup
            
        Returns:
            True if cleanup successful
        """
        try:
            # Remove from active rollback points
            if rollback_id in self.active_rollback_points:
                del self.active_rollback_points[rollback_id]

            # Optionally remove backup
            if remove_backup and rollback_id in self.rollback_history:
                backup_path = self.rollback_history[rollback_id]['backup_path']
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path)

            # Remove from history
            if rollback_id in self.rollback_history:
                del self.rollback_history[rollback_id]
                self._save_rollback_history()

            logger.info(f"Cleaned up rollback point {rollback_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup rollback point {rollback_id}: {e}")
            return False

    def cleanup_old_rollback_points(self, max_age_days: int = 30, max_count: int = 100) -> int:
        """
        Clean up old rollback points
        
        Args:
            max_age_days: Maximum age in days for rollback points
            max_count: Maximum number of rollback points to keep
            
        Returns:
            Number of rollback points cleaned up
        """
        cleaned_count = 0
        current_time = datetime.now()
        
        # Sort rollback points by timestamp (oldest first)
        sorted_points = sorted(
            self.rollback_history.items(),
            key=lambda x: x[1]['timestamp']
        )
        
        # Clean up by age
        for rollback_id, info in sorted_points:
            try:
                point_time = datetime.fromisoformat(info['timestamp'])
                age_days = (current_time - point_time).days
                
                if age_days > max_age_days:
                    if self.cleanup_rollback_point(rollback_id, remove_backup=True):
                        cleaned_count += 1
            except Exception as e:
                logger.error(f"Error cleaning up rollback point {rollback_id}: {e}")
        
        # Clean up by count (keep only the newest max_count points)
        remaining_points = list(self.rollback_history.items())
        if len(remaining_points) > max_count:
            # Sort again after age cleanup
            remaining_points.sort(key=lambda x: x[1]['timestamp'])
            points_to_remove = len(remaining_points) - max_count
            
            for i in range(points_to_remove):
                rollback_id = remaining_points[i][0]
                if self.cleanup_rollback_point(rollback_id, remove_backup=True):
                    cleaned_count += 1

        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old rollback points")
            
        return cleaned_count

    def _create_pre_rollback_backup(self, target_files: List[str]) -> Optional[str]:
        """Create a backup of current state before rollback"""
        try:
            import tempfile
            pre_rollback_dir = tempfile.mkdtemp(prefix='pre_rollback_')
            
            for file_path in target_files:
                if os.path.exists(file_path):
                    rel_path = os.path.relpath(file_path)
                    backup_file_path = os.path.join(pre_rollback_dir, rel_path)
                    os.makedirs(os.path.dirname(backup_file_path), exist_ok=True)
                    shutil.copy2(file_path, backup_file_path)
            
            return pre_rollback_dir
            
        except Exception as e:
            logger.warning(f"Failed to create pre-rollback backup: {e}")
            return None

    def _verify_backup_file_integrity(self, backup_file_path: str, 
                                    manifest: Dict[str, any], 
                                    original_file_path: str) -> bool:
        """Verify backup file integrity using manifest hashes"""
        try:
            file_hashes = manifest.get('file_hashes', {})
            if original_file_path not in file_hashes:
                # No hash available, assume valid
                return True
            
            expected_hash = file_hashes[original_file_path]
            
            # Calculate actual hash
            import hashlib
            hash_sha256 = hashlib.sha256()
            with open(backup_file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            actual_hash = hash_sha256.hexdigest()
            
            return expected_hash == actual_hash
            
        except Exception as e:
            logger.warning(f"Failed to verify backup file integrity for {backup_file_path}: {e}")
            return True  # Assume valid if verification fails

    def _load_rollback_history(self) -> Dict[str, Dict[str, any]]:
        """Load rollback history from disk"""
        if not os.path.exists(self.rollback_history_file):
            return {}

        try:
            with open(self.rollback_history_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load rollback history: {e}")
            return {}

    def _save_rollback_history(self):
        """Save rollback history to disk"""
        try:
            with open(self.rollback_history_file, 'w', encoding='utf-8') as f:
                json.dump(self.rollback_history, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save rollback history: {e}")
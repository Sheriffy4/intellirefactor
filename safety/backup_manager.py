"""
Backup Manager for IntelliRefactor

Provides comprehensive backup and restoration capabilities for safe refactoring operations.
"""

import os
import shutil
import tempfile
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class BackupResult:
    """Result of backup operation"""
    success: bool
    backup_path: Optional[str] = None
    backed_up_files: Optional[List[str]] = None
    backup_size: Optional[int] = None
    error: Optional[str] = None
    timestamp: Optional[datetime] = None


class BackupManager:
    """
    Manages backup creation, restoration, and cleanup for safe refactoring operations
    """

    def __init__(self, backup_dir: Optional[str] = None, max_backups: int = 50):
        """
        Initialize backup manager
        
        Args:
            backup_dir: Directory to store backups (defaults to temp directory)
            max_backups: Maximum number of backups to keep (oldest are cleaned up)
        """
        self.backup_dir = backup_dir or os.path.join(tempfile.gettempdir(), 'intellirefactor_backups')
        self.max_backups = max_backups
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Track backup metadata
        self.metadata_file = os.path.join(self.backup_dir, 'backup_metadata.json')
        self.metadata = self._load_metadata()

    def create_backup(self, target_files: List[str], operation_id: str, 
                     description: str = "") -> BackupResult:
        """
        Create backup of target files
        
        Args:
            target_files: List of files to backup
            operation_id: Unique identifier for the operation
            description: Optional description of the operation
            
        Returns:
            BackupResult with backup details
        """
        try:
            timestamp = datetime.now()
            backup_id = f"backup_{operation_id}_{int(timestamp.timestamp())}"
            backup_path = os.path.join(self.backup_dir, backup_id)
            os.makedirs(backup_path, exist_ok=True)

            backed_up_files = []
            total_size = 0
            file_hashes = {}

            for file_path in target_files:
                if not os.path.exists(file_path):
                    logger.warning(f"File does not exist, skipping: {file_path}")
                    continue

                # Calculate file hash for integrity checking
                file_hash = self._calculate_file_hash(file_path)
                file_hashes[file_path] = file_hash

                # Create directory structure in backup
                rel_path = os.path.relpath(file_path)
                backup_file_path = os.path.join(backup_path, rel_path)
                os.makedirs(os.path.dirname(backup_file_path), exist_ok=True)
                
                # Copy file with metadata preservation
                shutil.copy2(file_path, backup_file_path)
                backed_up_files.append(file_path)
                total_size += os.path.getsize(file_path)

            # Create backup manifest
            manifest = {
                'backup_id': backup_id,
                'operation_id': operation_id,
                'description': description,
                'timestamp': timestamp.isoformat(),
                'files': backed_up_files,
                'file_hashes': file_hashes,
                'backup_path': backup_path,
                'total_size': total_size,
                'version': '1.0'
            }
            
            manifest_path = os.path.join(backup_path, 'manifest.json')
            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)

            # Update metadata
            self.metadata[backup_id] = {
                'operation_id': operation_id,
                'description': description,
                'timestamp': timestamp.isoformat(),
                'file_count': len(backed_up_files),
                'total_size': total_size,
                'backup_path': backup_path
            }
            self._save_metadata()

            # Cleanup old backups if needed
            self._cleanup_old_backups()

            logger.info(f"Created backup {backup_id} with {len(backed_up_files)} files ({total_size} bytes)")

            return BackupResult(
                success=True,
                backup_path=backup_path,
                backed_up_files=backed_up_files,
                backup_size=total_size,
                timestamp=timestamp
            )

        except Exception as e:
            logger.error(f"Failed to create backup for operation {operation_id}: {e}")
            return BackupResult(
                success=False,
                error=str(e)
            )

    def restore_backup(self, backup_path: str, target_files: Optional[List[str]] = None) -> 'RestoreResult':
        """
        Restore files from backup
        
        Args:
            backup_path: Path to backup directory
            target_files: Specific files to restore (None = restore all)
            
        Returns:
            RestoreResult with restoration details
        """
        try:
            # Load backup manifest
            manifest_path = os.path.join(backup_path, 'manifest.json')
            if not os.path.exists(manifest_path):
                return RestoreResult(
                    success=False,
                    error="Backup manifest not found"
                )

            with open(manifest_path, 'r', encoding='utf-8') as f:
                manifest = json.load(f)

            # Determine files to restore
            files_to_restore = target_files or manifest['files']
            restored_files = []
            failed_files = []

            for file_path in files_to_restore:
                try:
                    rel_path = os.path.relpath(file_path)
                    backup_file_path = os.path.join(backup_path, rel_path)
                    
                    if not os.path.exists(backup_file_path):
                        failed_files.append(f"{file_path} (backup file not found)")
                        continue

                    # Verify backup file integrity if hash available
                    if file_path in manifest.get('file_hashes', {}):
                        expected_hash = manifest['file_hashes'][file_path]
                        actual_hash = self._calculate_file_hash(backup_file_path)
                        if expected_hash != actual_hash:
                            failed_files.append(f"{file_path} (backup file corrupted)")
                            continue

                    # Create target directory if needed
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    
                    # Restore file
                    shutil.copy2(backup_file_path, file_path)
                    restored_files.append(file_path)
                    
                except Exception as e:
                    failed_files.append(f"{file_path} ({str(e)})")

            success = len(failed_files) == 0
            if not success:
                logger.warning(f"Restoration completed with {len(failed_files)} failures")

            return RestoreResult(
                success=success,
                restored_files=restored_files,
                failed_files=failed_files,
                backup_info=manifest
            )

        except Exception as e:
            logger.error(f"Failed to restore backup from {backup_path}: {e}")
            return RestoreResult(
                success=False,
                error=str(e)
            )

    def list_backups(self) -> List[Dict[str, any]]:
        """List all available backups"""
        backups = []
        for backup_id, info in self.metadata.items():
            backup_info = info.copy()
            backup_info['backup_id'] = backup_id
            backup_info['exists'] = os.path.exists(info['backup_path'])
            backups.append(backup_info)
        
        # Sort by timestamp (newest first)
        backups.sort(key=lambda x: x['timestamp'], reverse=True)
        return backups

    def get_backup_info(self, backup_id: str) -> Optional[Dict[str, any]]:
        """Get detailed information about a specific backup"""
        if backup_id not in self.metadata:
            return None

        info = self.metadata[backup_id].copy()
        info['backup_id'] = backup_id
        info['exists'] = os.path.exists(info['backup_path'])

        # Load manifest if backup exists
        if info['exists']:
            try:
                manifest_path = os.path.join(info['backup_path'], 'manifest.json')
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                info['manifest'] = manifest
            except Exception as e:
                info['manifest_error'] = str(e)

        return info

    def delete_backup(self, backup_id: str) -> bool:
        """Delete a specific backup"""
        if backup_id not in self.metadata:
            logger.warning(f"Backup {backup_id} not found in metadata")
            return False

        backup_path = self.metadata[backup_id]['backup_path']
        
        try:
            if os.path.exists(backup_path):
                shutil.rmtree(backup_path)
            
            del self.metadata[backup_id]
            self._save_metadata()
            
            logger.info(f"Deleted backup {backup_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False

    def can_create_backup(self, target_files: List[str]) -> bool:
        """
        Check if backup can be created for target files
        
        Args:
            target_files: Files to check for backup capability
            
        Returns:
            True if backup can be created
        """
        try:
            # Check if all files exist and are readable
            for file_path in target_files:
                if not os.path.exists(file_path):
                    logger.debug(f"File does not exist: {file_path}")
                    continue
                if not os.access(file_path, os.R_OK):
                    logger.debug(f"File not readable: {file_path}")
                    return False

            # Check disk space
            total_size = sum(
                os.path.getsize(f) for f in target_files 
                if os.path.exists(f)
            )
            
            free_space = shutil.disk_usage(self.backup_dir).free
            
            # Require at least 2x the file size in free space (safety margin)
            if free_space < (total_size * 2):
                logger.debug(f"Insufficient disk space: need {total_size * 2}, have {free_space}")
                return False

            # Check write permissions on backup directory
            if not os.access(self.backup_dir, os.W_OK):
                logger.debug(f"Backup directory not writable: {self.backup_dir}")
                return False

            return True

        except Exception as e:
            logger.debug(f"Error checking backup capability: {e}")
            return False

    def cleanup_backup(self, backup_path: str) -> bool:
        """Clean up a specific backup directory"""
        try:
            if not os.path.exists(backup_path):
                return True

            # Find backup ID from path
            backup_id = os.path.basename(backup_path)
            
            # Remove from metadata if present
            if backup_id in self.metadata:
                del self.metadata[backup_id]
                self._save_metadata()

            # Remove directory
            shutil.rmtree(backup_path)
            logger.info(f"Cleaned up backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup backup {backup_path}: {e}")
            return False

    def _cleanup_old_backups(self):
        """Remove old backups if we exceed the maximum count"""
        if len(self.metadata) <= self.max_backups:
            return

        # Sort backups by timestamp (oldest first)
        sorted_backups = sorted(
            self.metadata.items(),
            key=lambda x: x[1]['timestamp']
        )

        # Remove oldest backups
        backups_to_remove = len(self.metadata) - self.max_backups
        for i in range(backups_to_remove):
            backup_id, backup_info = sorted_backups[i]
            try:
                backup_path = backup_info['backup_path']
                if os.path.exists(backup_path):
                    shutil.rmtree(backup_path)
                del self.metadata[backup_id]
                logger.info(f"Cleaned up old backup: {backup_id}")
            except Exception as e:
                logger.error(f"Failed to cleanup old backup {backup_id}: {e}")

        self._save_metadata()

    def _load_metadata(self) -> Dict[str, Dict[str, any]]:
        """Load backup metadata from disk"""
        if not os.path.exists(self.metadata_file):
            return {}

        try:
            with open(self.metadata_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load backup metadata: {e}")
            return {}

    def _save_metadata(self):
        """Save backup metadata to disk"""
        try:
            with open(self.metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to save backup metadata: {e}")

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of a file for integrity checking"""
        hash_sha256 = hashlib.sha256()
        try:
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_sha256.update(chunk)
            return hash_sha256.hexdigest()
        except Exception as e:
            logger.warning(f"Failed to calculate hash for {file_path}: {e}")
            return ""


@dataclass
class RestoreResult:
    """Result of restore operation"""
    success: bool
    restored_files: Optional[List[str]] = None
    failed_files: Optional[List[str]] = None
    backup_info: Optional[Dict[str, any]] = None
    error: Optional[str] = None
"""
Git History Analyzer for expert refactoring analysis.

Analyzes Git history to find co-change patterns and evolution trends.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..models import GitChangePattern

logger = logging.getLogger(__name__)


class GitHistoryAnalyzer:
    """Analyzes Git history for change patterns."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)

    def analyze_change_patterns(self) -> Optional[GitChangePattern]:
        """
        Analyze Git history for change patterns.
        
        Returns:
            GitChangePattern with co-change analysis or None if Git not available
        """
        logger.info("Analyzing Git change patterns...")
        
        if not self._is_git_repository():
            logger.warning("Not a Git repository - skipping Git analysis")
            return None
        
        try:
            # Get commit history for the target file
            commits = self._get_file_commits()
            
            # Analyze co-changes
            co_changes = self._analyze_co_changes(commits)
            
            # Find change frequency
            change_frequency = self._calculate_change_frequency(commits)
            
            # Identify hotspots
            hotspots = self._identify_hotspots(change_frequency)
            
            # Find hidden dependencies
            hidden_deps = self._find_hidden_dependencies(co_changes)
            
            pattern = GitChangePattern(
                files_changed_together=co_changes,
                change_frequency=change_frequency,
                hotspots=hotspots,
                hidden_dependencies=hidden_deps
            )
            
            logger.info(f"Git analysis: {len(co_changes)} co-changes, {len(hotspots)} hotspots")
            return pattern
            
        except Exception as e:
            logger.warning(f"Git analysis failed: {e}")
            return None

    def _is_git_repository(self) -> bool:
        """Check if the project is a Git repository."""
        return (self.project_root / '.git').exists()

    def _get_file_commits(self) -> List[Dict[str, Any]]:
        """Get commit history for the target file."""
        try:
            # Get commits that modified the target file
            cmd = [
                'git', 'log', '--oneline', '--follow',
                str(self.target_module.relative_to(self.project_root))
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                logger.warning(f"Git log failed: {result.stderr}")
                return []
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) >= 2:
                        commits.append({
                            'hash': parts[0],
                            'message': parts[1]
                        })
            
            return commits
            
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"Error getting Git commits: {e}")
            return []

    def _analyze_co_changes(self, commits: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
        """Analyze which files changed together with the target file."""
        co_changes = []
        
        for commit in commits[:20]:  # Analyze last 20 commits
            try:
                # Get files changed in this commit
                cmd = ['git', 'show', '--name-only', '--format=', commit['hash']]
                
                result = subprocess.run(
                    cmd,
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if result.returncode == 0:
                    changed_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
                    target_rel = str(self.target_module.relative_to(self.project_root))
                    
                    # Find files that changed together with our target
                    for file_path in changed_files:
                        if file_path != target_rel and file_path.endswith('.py'):
                            co_changes.append((target_rel, file_path))
                            
            except (subprocess.TimeoutExpired, subprocess.SubprocessError):
                continue
        
        # Count co-change frequency and return most frequent pairs
        co_change_counts = {}
        for pair in co_changes:
            co_change_counts[pair] = co_change_counts.get(pair, 0) + 1
        
        # Return pairs that changed together more than once
        frequent_co_changes = [
            pair for pair, count in co_change_counts.items() 
            if count > 1
        ]
        
        return frequent_co_changes

    def _calculate_change_frequency(self, commits: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate how frequently different parts of the codebase change."""
        frequency = {}
        
        # For now, just count commits to the target file
        target_rel = str(self.target_module.relative_to(self.project_root))
        frequency[target_rel] = len(commits)
        
        # Could be extended to analyze line-level changes
        return frequency

    def _identify_hotspots(self, change_frequency: Dict[str, int]) -> List[str]:
        """Identify files that change frequently (hotspots)."""
        if not change_frequency:
            return []
        
        # Files with above-average change frequency are hotspots
        avg_frequency = sum(change_frequency.values()) / len(change_frequency)
        
        hotspots = [
            file_path for file_path, freq in change_frequency.items()
            if freq > avg_frequency and freq > 5  # At least 5 changes
        ]
        
        return hotspots

    def _find_hidden_dependencies(self, co_changes: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Find hidden dependencies based on co-change patterns."""
        # Files that frequently change together might have hidden dependencies
        co_change_counts = {}
        for pair in co_changes:
            co_change_counts[pair] = co_change_counts.get(pair, 0) + 1
        
        # Return pairs with high co-change frequency
        hidden_deps = [
            pair for pair, count in co_change_counts.items()
            if count >= 3  # Changed together 3+ times
        ]
        
        return hidden_deps

    def get_recent_changes(self, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get recent changes to the target file.
        
        Args:
            days: Number of days to look back
            
        Returns:
            List of recent commits
        """
        try:
            cmd = [
                'git', 'log', '--oneline', '--since', f'{days} days ago',
                str(self.target_module.relative_to(self.project_root))
            ]
            
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode != 0:
                return []
            
            commits = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(' ', 1)
                    if len(parts) >= 2:
                        commits.append({
                            'hash': parts[0],
                            'message': parts[1]
                        })
            
            return commits
            
        except Exception as e:
            logger.warning(f"Error getting recent changes: {e}")
            return []

    def analyze_change_velocity(self) -> Dict[str, Any]:
        """
        Analyze the velocity of changes to the target file.
        
        Returns:
            Dictionary with change velocity metrics
        """
        commits = self._get_file_commits()
        
        if not commits:
            return {
                'total_commits': 0,
                'recent_activity': 'none',
                'change_trend': 'stable'
            }
        
        recent_commits = self.get_recent_changes(30)
        very_recent_commits = self.get_recent_changes(7)
        
        # Determine activity level
        if len(very_recent_commits) > 3:
            activity = 'very_high'
        elif len(recent_commits) > 5:
            activity = 'high'
        elif len(recent_commits) > 2:
            activity = 'moderate'
        elif len(recent_commits) > 0:
            activity = 'low'
        else:
            activity = 'none'
        
        # Determine trend (simplified)
        if len(very_recent_commits) > len(recent_commits) / 4:
            trend = 'increasing'
        elif len(recent_commits) == 0:
            trend = 'stable'
        else:
            trend = 'decreasing'
        
        return {
            'total_commits': len(commits),
            'recent_commits_30d': len(recent_commits),
            'recent_commits_7d': len(very_recent_commits),
            'recent_activity': activity,
            'change_trend': trend
        }
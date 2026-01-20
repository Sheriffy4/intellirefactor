"""
Compatibility Analyzer for expert refactoring analysis.

Analyzes compatibility constraints and assesses breaking change impact.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import Dict, List

from ..models import ImpactAssessment, RiskLevel

logger = logging.getLogger(__name__)


class CompatibilityAnalyzer:
    """Analyzes compatibility constraints for safe refactoring."""

    def __init__(self, project_root: str, target_module: str):
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)

    def assess_breaking_change_impact(self, proposed_changes: List[str]) -> ImpactAssessment:
        """
        Assess the impact of proposed breaking changes.
        
        Args:
            proposed_changes: List of proposed changes
            
        Returns:
            ImpactAssessment with risk analysis
        """
        logger.info("Assessing breaking change impact...")
        
        # Find all files that might be affected
        affected_files = self._find_potentially_affected_files()
        
        # Analyze the severity of proposed changes
        risk_level = self._assess_change_risk(proposed_changes, affected_files)
        
        # Identify specific breaking changes
        breaking_changes = self._identify_breaking_changes(proposed_changes)
        
        # Estimate migration effort
        migration_effort = self._estimate_migration_effort(affected_files, breaking_changes)
        
        # Generate recommendations
        recommendations = self._generate_compatibility_recommendations(
            risk_level, affected_files, breaking_changes
        )
        
        assessment = ImpactAssessment(
            affected_files=affected_files,
            risk_level=risk_level,
            breaking_changes=breaking_changes,
            migration_effort=migration_effort,
            recommendations=recommendations
        )
        
        logger.info(f"Impact assessment: {len(affected_files)} files affected, risk {risk_level.value}")
        return assessment

    def _find_potentially_affected_files(self) -> List[str]:
        """Find files that might be affected by changes to the target module."""
        affected_files = []
        
        # Get module name for import detection
        module_name = self.target_module.stem
        
        # Search for files that import or reference the target module
        for py_file in self.project_root.rglob("*.py"):
            if py_file == self.target_module:
                continue
            
            try:
                content = py_file.read_text(encoding='utf-8')
                if module_name in content:
                    # More precise check - look for actual imports
                    if self._file_imports_module(py_file, module_name):
                        rel_path = str(py_file.relative_to(self.project_root))
                        affected_files.append(rel_path)
            except (OSError, UnicodeDecodeError):
                continue
        
        return affected_files

    def _file_imports_module(self, py_file: Path, module_name: str) -> bool:
        """Check if a file actually imports the target module."""
        try:
            content = py_file.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if module_name in alias.name:
                            return True
                elif isinstance(node, ast.ImportFrom):
                    if node.module and module_name in node.module:
                        return True
            
        except (OSError, SyntaxError):
            pass
        
        return False

    def _assess_change_risk(self, proposed_changes: List[str], affected_files: List[str]) -> RiskLevel:
        """Assess the risk level of proposed changes."""
        risk_score = 0
        
        # Risk factors
        if len(affected_files) > 10:
            risk_score += 3
        elif len(affected_files) > 5:
            risk_score += 2
        elif len(affected_files) > 2:
            risk_score += 1
        
        # Analyze change types
        for change in proposed_changes:
            change_lower = change.lower()
            if any(keyword in change_lower for keyword in ['remove', 'delete', 'drop']):
                risk_score += 2
            elif any(keyword in change_lower for keyword in ['rename', 'move', 'change signature']):
                risk_score += 1
            elif any(keyword in change_lower for keyword in ['add', 'extend', 'enhance']):
                risk_score += 0  # Additive changes are safer
        
        # Convert score to risk level
        if risk_score >= 6:
            return RiskLevel.CRITICAL
        elif risk_score >= 4:
            return RiskLevel.HIGH
        elif risk_score >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _identify_breaking_changes(self, proposed_changes: List[str]) -> List[str]:
        """Identify which proposed changes are breaking changes."""
        breaking_changes = []
        
        for change in proposed_changes:
            change_lower = change.lower()
            
            # Definitely breaking changes
            if any(keyword in change_lower for keyword in [
                'remove method', 'delete function', 'remove class',
                'change signature', 'rename method', 'rename class',
                'remove parameter', 'change parameter type'
            ]):
                breaking_changes.append(change)
            
            # Potentially breaking changes
            elif any(keyword in change_lower for keyword in [
                'move method', 'extract class', 'split module',
                'change return type', 'modify behavior'
            ]):
                breaking_changes.append(f"Potentially breaking: {change}")
        
        return breaking_changes

    def _estimate_migration_effort(self, affected_files: List[str], breaking_changes: List[str]) -> str:
        """Estimate the effort required for migration."""
        if not affected_files and not breaking_changes:
            return "minimal"
        
        effort_score = 0
        
        # File count impact
        effort_score += len(affected_files)
        
        # Breaking change impact
        effort_score += len(breaking_changes) * 2
        
        # Convert to effort level
        if effort_score >= 20:
            return "very_high"
        elif effort_score >= 10:
            return "high"
        elif effort_score >= 5:
            return "medium"
        else:
            return "low"

    def _generate_compatibility_recommendations(
        self, 
        risk_level: RiskLevel, 
        affected_files: List[str], 
        breaking_changes: List[str]
    ) -> List[str]:
        """Generate recommendations for maintaining compatibility."""
        recommendations = []
        
        # Risk-based recommendations
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.append("Create comprehensive test suite before making changes")
            recommendations.append("Consider phased rollout with feature flags")
            recommendations.append("Implement deprecation warnings before removing functionality")
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.append("Consider creating a new module instead of modifying existing one")
            recommendations.append("Coordinate with all affected teams before proceeding")
        
        # File-based recommendations
        if len(affected_files) > 5:
            recommendations.append("Update all affected files in a single atomic commit")
            recommendations.append("Create migration scripts for automated updates")
        
        # Breaking change recommendations
        if breaking_changes:
            recommendations.append("Document all breaking changes in CHANGELOG")
            recommendations.append("Provide migration guide with examples")
            recommendations.append("Consider backward compatibility shims")
        
        # General recommendations
        recommendations.append("Run full test suite after changes")
        recommendations.append("Monitor for runtime errors after deployment")
        
        return recommendations

    def determine_compatibility_constraints(self) -> List[str]:
        """
        Determine compatibility constraints for the target module.
        
        Returns:
            List of compatibility constraints
        """
        logger.info("Determining compatibility constraints...")
        
        constraints = []
        
        # Analyze the module to determine constraints
        try:
            content = self.target_module.read_text(encoding='utf-8')
            tree = ast.parse(content)
            
            # Check for public API elements
            public_classes = []
            public_functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef) and not node.name.startswith('_'):
                    public_classes.append(node.name)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and not node.name.startswith('_'):
                    public_functions.append(node.name)
            
            # Generate constraints based on public API
            if public_classes:
                constraints.append(f"Must maintain public classes: {', '.join(public_classes)}")
            
            if public_functions:
                constraints.append(f"Must maintain public functions: {', '.join(public_functions)}")
            
            # Check for __all__ definition
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == '__all__':
                            constraints.append("Must maintain __all__ exported symbols")
                            break
            
            # Check for version constraints
            if self._has_version_info(tree):
                constraints.append("Must maintain version compatibility")
            
            # Check for external dependencies
            external_deps = self._find_external_dependencies(tree)
            if external_deps:
                constraints.append(f"Must maintain compatibility with: {', '.join(external_deps)}")
            
        except (OSError, SyntaxError) as e:
            logger.warning(f"Error analyzing compatibility constraints: {e}")
            constraints.append("Unable to analyze - manual review required")
        
        logger.info(f"Found {len(constraints)} compatibility constraints")
        return constraints

    def _has_version_info(self, tree: ast.Module) -> bool:
        """Check if the module has version information."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in ['__version__', 'VERSION']:
                        return True
        return False

    def _find_external_dependencies(self, tree: ast.Module) -> List[str]:
        """Find external dependencies that might impose constraints."""
        dependencies = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # Skip standard library modules
                    if not self._is_stdlib_module(alias.name):
                        dependencies.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and not self._is_stdlib_module(node.module):
                    dependencies.add(node.module.split('.')[0])
        
        return list(dependencies)

    def _is_stdlib_module(self, module_name: str) -> bool:
        """Check if a module is part of the standard library."""
        stdlib_modules = {
            'os', 'sys', 'json', 'logging', 'datetime', 'time', 'random',
            'collections', 'itertools', 'functools', 'operator', 'typing',
            'pathlib', 'shutil', 'subprocess', 'threading', 'multiprocessing',
            'asyncio', 'concurrent', 'queue', 'socket', 'urllib', 'http',
            'email', 'html', 'xml', 'csv', 'sqlite3', 'pickle', 'base64',
            'hashlib', 'hmac', 'secrets', 'uuid', 'enum', 'dataclasses',
            'contextlib', 'warnings', 'traceback', 'inspect', 'ast', 're'
        }
        
        base_module = module_name.split('.')[0]
        return base_module in stdlib_modules

    def check_backward_compatibility(self, old_api: Dict[str, Any], new_api: Dict[str, Any]) -> List[str]:
        """
        Check backward compatibility between old and new API.
        
        Args:
            old_api: Dictionary describing the old API
            new_api: Dictionary describing the new API
            
        Returns:
            List of compatibility issues
        """
        issues = []
        
        # Check for removed functions/classes
        old_symbols = set(old_api.get('symbols', []))
        new_symbols = set(new_api.get('symbols', []))
        
        removed_symbols = old_symbols - new_symbols
        if removed_symbols:
            issues.extend([f"Removed symbol: {symbol}" for symbol in removed_symbols])
        
        # Check for signature changes
        old_signatures = old_api.get('signatures', {})
        new_signatures = new_api.get('signatures', {})
        
        for symbol in old_signatures:
            if symbol in new_signatures:
                if old_signatures[symbol] != new_signatures[symbol]:
                    issues.append(f"Changed signature: {symbol}")
        
        return issues
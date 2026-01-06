"""
Destructive Operation Detector for IntelliRefactor

Analyzes refactoring operations to detect potentially destructive changes
and assess risk levels for safe refactoring.
"""

import os
import ast
import re
import logging
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class OperationRisk(Enum):
    """Risk levels for refactoring operations"""
    LOW = "low"              # Safe operations like formatting
    MEDIUM = "medium"        # Structural changes
    HIGH = "high"            # Potentially destructive operations
    CRITICAL = "critical"    # Operations that could break the project


@dataclass
class RiskFactor:
    """Represents a risk factor in an operation"""
    name: str
    description: str
    risk_level: OperationRisk
    weight: float  # 0.0 to 1.0, higher means more risky
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class OperationAnalysis:
    """Analysis result for a refactoring operation"""
    operation_type: str
    target_files: List[str]
    risk_level: OperationRisk
    risk_factors: List[RiskFactor]
    is_safe: bool
    recommendations: List[str]


class DestructiveOperationDetector:
    """
    Detects and analyzes potentially destructive refactoring operations.
    
    Provides risk assessment and safety recommendations for various
    types of refactoring operations.
    """
    
    def __init__(self):
        """Initialize the detector."""
        self.logger = logging.getLogger(__name__)
        
        # Define high-risk operation patterns
        self.high_risk_operations = {
            'delete_file', 'remove_class', 'remove_function', 'remove_method',
            'delete_directory', 'remove_import', 'remove_dependency'
        }
        
        # Define medium-risk operation patterns
        self.medium_risk_operations = {
            'rename_file', 'move_file', 'rename_class', 'rename_function',
            'change_signature', 'modify_inheritance', 'restructure_module'
        }
        
        # Define low-risk operation patterns
        self.low_risk_operations = {
            'format_code', 'add_comments', 'add_docstrings', 'fix_imports',
            'add_type_hints', 'extract_variable', 'inline_variable'
        }
    
    def analyze_operation(self, operation: Dict[str, Any]) -> OperationAnalysis:
        """Analyze a single refactoring operation for destructive potential."""
        operation_type = operation.get('type', 'unknown')
        target_files = operation.get('target_files', [])
        
        self.logger.debug(f"Analyzing operation: {operation_type}")
        
        # Determine base risk level
        if operation_type in self.high_risk_operations:
            base_risk = OperationRisk.HIGH
        elif operation_type in self.medium_risk_operations:
            base_risk = OperationRisk.MEDIUM
        elif operation_type in self.low_risk_operations:
            base_risk = OperationRisk.LOW
        else:
            base_risk = OperationRisk.MEDIUM  # Unknown operations are medium risk
        
        # Analyze risk factors
        risk_factors = self._analyze_risk_factors(operation, base_risk)
        
        # Determine final risk level based on factors
        final_risk = self._calculate_final_risk(base_risk, risk_factors)
        
        # Determine if operation is safe
        is_safe = final_risk in [OperationRisk.LOW, OperationRisk.MEDIUM]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(operation, final_risk, risk_factors)
        
        return OperationAnalysis(
            operation_type=operation_type,
            target_files=target_files,
            risk_level=final_risk,
            risk_factors=risk_factors,
            is_safe=is_safe,
            recommendations=recommendations
        )
    
    def analyze_operation_batch(self, operations: List[Dict[str, Any]]) -> List[OperationAnalysis]:
        """Analyze a batch of operations."""
        return [self.analyze_operation(op) for op in operations]
    
    def _analyze_risk_factors(self, operation: Dict[str, Any], base_risk: OperationRisk) -> List[RiskFactor]:
        """Analyze specific risk factors for an operation."""
        risk_factors = []
        
        # Check target file criticality
        target_files = operation.get('target_files', [])
        for file_path in target_files:
            if self._is_critical_file(file_path):
                risk_factors.append(RiskFactor(
                    name="critical_file_target",
                    description=f"Operation targets critical file: {file_path}",
                    risk_level=OperationRisk.HIGH,
                    weight=0.8
                ))
        
        # Check for bulk operations
        if len(target_files) > 10:
            risk_factors.append(RiskFactor(
                name="bulk_operation",
                description=f"Operation affects many files ({len(target_files)})",
                risk_level=OperationRisk.MEDIUM,
                weight=0.6
            ))
        
        # Check for cross-module dependencies
        if self._has_cross_module_impact(operation):
            risk_factors.append(RiskFactor(
                name="cross_module_impact",
                description="Operation may affect multiple modules",
                risk_level=OperationRisk.MEDIUM,
                weight=0.7
            ))
        
        # Check for public API changes
        if self._affects_public_api(operation):
            risk_factors.append(RiskFactor(
                name="public_api_change",
                description="Operation may change public API",
                risk_level=OperationRisk.HIGH,
                weight=0.9
            ))
        
        return risk_factors
    
    def _calculate_final_risk(self, base_risk: OperationRisk, risk_factors: List[RiskFactor]) -> OperationRisk:
        """Calculate final risk level based on base risk and factors."""
        if not risk_factors:
            return base_risk
        
        # Calculate weighted risk score
        total_weight = sum(factor.weight for factor in risk_factors)
        if total_weight == 0:
            return base_risk
        
        # Count high-risk factors
        high_risk_factors = [f for f in risk_factors if f.risk_level == OperationRisk.HIGH]
        critical_risk_factors = [f for f in risk_factors if f.risk_level == OperationRisk.CRITICAL]
        
        # Escalate risk if critical factors present
        if critical_risk_factors:
            return OperationRisk.CRITICAL
        
        # Escalate risk if multiple high-risk factors
        if len(high_risk_factors) >= 2:
            if base_risk == OperationRisk.HIGH:
                return OperationRisk.CRITICAL
            else:
                return OperationRisk.HIGH
        
        # Escalate risk if single high-risk factor and base risk is medium+
        if high_risk_factors and base_risk in [OperationRisk.MEDIUM, OperationRisk.HIGH]:
            return OperationRisk.HIGH
        
        return base_risk
    
    def _generate_recommendations(self, operation: Dict[str, Any], risk_level: OperationRisk, 
                                risk_factors: List[RiskFactor]) -> List[str]:
        """Generate safety recommendations for the operation."""
        recommendations = []
        
        if risk_level == OperationRisk.CRITICAL:
            recommendations.append("CRITICAL: Consider breaking this operation into smaller, safer steps")
            recommendations.append("Create comprehensive backup before proceeding")
            recommendations.append("Test in isolated environment first")
        
        elif risk_level == OperationRisk.HIGH:
            recommendations.append("Create backup before proceeding")
            recommendations.append("Review changes carefully before applying")
            recommendations.append("Consider running in dry-run mode first")
        
        elif risk_level == OperationRisk.MEDIUM:
            recommendations.append("Review changes before applying")
            recommendations.append("Ensure version control is up to date")
        
        # Specific recommendations based on risk factors
        for factor in risk_factors:
            if factor.name == "critical_file_target":
                recommendations.append("Extra caution required for critical file modifications")
            elif factor.name == "bulk_operation":
                recommendations.append("Consider processing files in smaller batches")
            elif factor.name == "public_api_change":
                recommendations.append("Update documentation and notify API consumers")
        
        return recommendations
    
    def _is_critical_file(self, file_path: str) -> bool:
        """Check if a file is considered critical."""
        critical_patterns = [
            '__init__.py', 'setup.py', 'pyproject.toml', 'requirements.txt',
            'main.py', 'app.py', 'config.py', 'settings.py'
        ]
        
        file_name = Path(file_path).name.lower()
        return any(pattern in file_name for pattern in critical_patterns)
    
    def _has_cross_module_impact(self, operation: Dict[str, Any]) -> bool:
        """Check if operation has cross-module impact."""
        # Simplified check - in reality would analyze imports and dependencies
        operation_type = operation.get('type', '')
        
        cross_module_operations = {
            'rename_class', 'rename_function', 'move_class', 'move_function',
            'change_signature', 'remove_class', 'remove_function'
        }
        
        return operation_type in cross_module_operations
    
    def _affects_public_api(self, operation: Dict[str, Any]) -> bool:
        """Check if operation affects public API."""
        # Simplified check - would need more sophisticated analysis
        operation_type = operation.get('type', '')
        target = operation.get('target', '')
        
        # Operations that typically affect public API
        public_api_operations = {
            'rename_class', 'rename_function', 'change_signature',
            'remove_class', 'remove_function', 'modify_inheritance'
        }
        
        if operation_type in public_api_operations:
            # Check if target appears to be public (not starting with _)
            if target and not target.startswith('_'):
                return True
        
        return False
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class DestructiveOperationDetector:
    """
    Detects potentially destructive refactoring operations and assesses their risk levels
    """

    def __init__(self):
        """Initialize the destructive operation detector"""
        # Define base operation risk levels
        self.operation_risks = {
            # Low risk operations - safe formatting and documentation
            'format_code': OperationRisk.LOW,
            'add_comments': OperationRisk.LOW,
            'fix_imports': OperationRisk.LOW,
            'sort_imports': OperationRisk.LOW,
            'add_docstrings': OperationRisk.LOW,
            
            # Medium risk operations - structural changes that are usually safe
            'extract_method': OperationRisk.MEDIUM,
            'rename_variable': OperationRisk.MEDIUM,
            'inline_method': OperationRisk.MEDIUM,
            'extract_class': OperationRisk.MEDIUM,
            'move_method': OperationRisk.MEDIUM,
            'introduce_parameter': OperationRisk.MEDIUM,
            'remove_parameter': OperationRisk.MEDIUM,
            
            # High risk operations - significant structural changes
            'rename_class': OperationRisk.HIGH,
            'move_class': OperationRisk.HIGH,
            'change_interface': OperationRisk.HIGH,
            'remove_method': OperationRisk.HIGH,
            'merge_classes': OperationRisk.HIGH,
            'split_class': OperationRisk.HIGH,
            'change_method_signature': OperationRisk.HIGH,
            
            # Critical operations - potentially destructive
            'delete_file': OperationRisk.CRITICAL,
            'remove_class': OperationRisk.CRITICAL,
            'change_inheritance': OperationRisk.CRITICAL,
            'remove_module': OperationRisk.CRITICAL,
            'restructure_package': OperationRisk.CRITICAL,
        }

        # Define critical file patterns that increase risk
        self.critical_file_patterns = [
            r'__init__\.py$',
            r'setup\.py$',
            r'main\.py$',
            r'app\.py$',
            r'config\.py$',
            r'settings\.py$',
            r'requirements\.txt$',
            r'pyproject\.toml$',
            r'Dockerfile$',
            r'\.github/.*\.yml$',
            r'\.github/.*\.yaml$',
        ]

        # Define high-impact directories
        self.high_impact_directories = [
            'src',
            'lib',
            'core',
            'api',
            'models',
            'services',
            'utils',
            'common',
        ]

    def analyze_operation(self, operation_type: str, target_files: List[str], 
                         operation_details: Dict[str, Any] = None) -> OperationAnalysis:
        """
        Analyze an operation for destructive potential
        
        Args:
            operation_type: Type of operation being performed
            target_files: Files that will be modified
            operation_details: Additional details about the operation
            
        Returns:
            OperationAnalysis with risk assessment
        """
        if operation_details is None:
            operation_details = {}

        risk_factors = []
        recommendations = []

        # Get base risk level
        base_risk = self.operation_risks.get(operation_type, OperationRisk.MEDIUM)
        current_risk = base_risk

        # Analyze file-based risk factors
        file_risk_factors = self._analyze_file_risks(target_files)
        risk_factors.extend(file_risk_factors)

        # Analyze operation scope
        scope_risk_factors = self._analyze_scope_risks(operation_details, target_files)
        risk_factors.extend(scope_risk_factors)

        # Analyze dependency impact
        dependency_risk_factors = self._analyze_dependency_risks(target_files, operation_details)
        risk_factors.extend(dependency_risk_factors)

        # Analyze code complexity
        complexity_risk_factors = self._analyze_complexity_risks(target_files, operation_type)
        risk_factors.extend(complexity_risk_factors)

        # Calculate overall risk level based on risk factors
        current_risk = self._calculate_overall_risk(base_risk, risk_factors)

        # Generate recommendations
        recommendations = self._generate_recommendations(current_risk, risk_factors, operation_type)

        # Determine if operation is safe
        is_safe = current_risk != OperationRisk.CRITICAL

        return OperationAnalysis(
            operation_type=operation_type,
            target_files=target_files,
            risk_level=current_risk,
            risk_factors=risk_factors,
            is_safe=is_safe,
            recommendations=recommendations,
            details={
                'base_risk': base_risk.value,
                'escalated_risk': current_risk.value,
                'file_count': len(target_files),
                'operation_details': operation_details
            }
        )

    def _analyze_file_risks(self, target_files: List[str]) -> List[RiskFactor]:
        """Analyze risk factors based on target files"""
        risk_factors = []

        # Check for critical files
        critical_files = []
        for file_path in target_files:
            for pattern in self.critical_file_patterns:
                if re.search(pattern, file_path):
                    critical_files.append(file_path)
                    break

        if critical_files:
            risk_factors.append(RiskFactor(
                name="critical_files",
                description=f"Operation affects critical files: {', '.join(critical_files)}",
                risk_level=OperationRisk.HIGH,
                weight=0.8,
                details={'critical_files': critical_files}
            ))

        # Check for high file count
        if len(target_files) > 20:
            risk_factors.append(RiskFactor(
                name="high_file_count",
                description=f"Operation affects {len(target_files)} files",
                risk_level=OperationRisk.MEDIUM,
                weight=0.6,
                details={'file_count': len(target_files)}
            ))

        # Check for high-impact directories
        high_impact_files = []
        for file_path in target_files:
            for directory in self.high_impact_directories:
                if f"/{directory}/" in file_path or file_path.startswith(f"{directory}/"):
                    high_impact_files.append(file_path)
                    break

        if high_impact_files:
            risk_factors.append(RiskFactor(
                name="high_impact_directories",
                description=f"Operation affects files in high-impact directories",
                risk_level=OperationRisk.MEDIUM,
                weight=0.5,
                details={'high_impact_files': high_impact_files}
            ))

        return risk_factors

    def _analyze_scope_risks(self, operation_details: Dict[str, Any], 
                           target_files: List[str]) -> List[RiskFactor]:
        """Analyze risk factors based on operation scope"""
        risk_factors = []

        scope = operation_details.get('scope', 'local')
        if scope == 'global':
            risk_factors.append(RiskFactor(
                name="global_scope",
                description="Operation has global scope affecting multiple components",
                risk_level=OperationRisk.HIGH,
                weight=0.7,
                details={'scope': scope}
            ))

        # Check for cross-module operations
        modules = set()
        for file_path in target_files:
            # Extract module path (directory structure)
            module_path = os.path.dirname(file_path)
            if module_path:
                modules.add(module_path)

        if len(modules) > 5:
            risk_factors.append(RiskFactor(
                name="cross_module_operation",
                description=f"Operation spans {len(modules)} different modules",
                risk_level=OperationRisk.MEDIUM,
                weight=0.6,
                details={'module_count': len(modules), 'modules': list(modules)}
            ))

        return risk_factors

    def _analyze_dependency_risks(self, target_files: List[str], 
                                operation_details: Dict[str, Any]) -> List[RiskFactor]:
        """Analyze risk factors based on dependency impact"""
        risk_factors = []

        # Operations that commonly affect external dependencies
        high_dependency_impact_ops = [
            'rename_class', 'move_class', 'change_interface', 
            'remove_method', 'change_method_signature'
        ]

        operation_type = operation_details.get('operation_type', '')
        if operation_type in high_dependency_impact_ops:
            risk_factors.append(RiskFactor(
                name="dependency_impact",
                description=f"Operation '{operation_type}' may impact external dependencies",
                risk_level=OperationRisk.MEDIUM,
                weight=0.5,
                details={'operation_type': operation_type}
            ))

        # Check for public API files (heuristic based on file names)
        api_files = []
        for file_path in target_files:
            if any(keyword in file_path.lower() for keyword in ['api', 'interface', 'public', 'client']):
                api_files.append(file_path)

        if api_files:
            risk_factors.append(RiskFactor(
                name="api_files",
                description="Operation affects potential API files",
                risk_level=OperationRisk.HIGH,
                weight=0.7,
                details={'api_files': api_files}
            ))

        return risk_factors

    def _analyze_complexity_risks(self, target_files: List[str], 
                                operation_type: str) -> List[RiskFactor]:
        """Analyze risk factors based on code complexity"""
        risk_factors = []

        try:
            total_lines = 0
            complex_files = []

            for file_path in target_files:
                if not file_path.endswith('.py') or not os.path.exists(file_path):
                    continue

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        line_count = len([line for line in lines if line.strip() and not line.strip().startswith('#')])
                        total_lines += line_count

                        # Consider files with >500 lines as complex
                        if line_count > 500:
                            complex_files.append((file_path, line_count))

                except Exception as e:
                    logger.debug(f"Could not analyze complexity for {file_path}: {e}")

            if complex_files:
                risk_factors.append(RiskFactor(
                    name="complex_files",
                    description=f"Operation affects {len(complex_files)} complex files",
                    risk_level=OperationRisk.MEDIUM,
                    weight=0.4,
                    details={'complex_files': complex_files}
                ))

            # Large total codebase impact
            if total_lines > 5000:
                risk_factors.append(RiskFactor(
                    name="large_codebase_impact",
                    description=f"Operation affects {total_lines} lines of code",
                    risk_level=OperationRisk.MEDIUM,
                    weight=0.3,
                    details={'total_lines': total_lines}
                ))

        except Exception as e:
            logger.debug(f"Error analyzing complexity risks: {e}")

        return risk_factors

    def _calculate_overall_risk(self, base_risk: OperationRisk, 
                              risk_factors: List[RiskFactor]) -> OperationRisk:
        """Calculate overall risk level based on base risk and risk factors"""
        if not risk_factors:
            return base_risk

        # Calculate weighted risk score
        total_weight = 0
        weighted_risk_score = 0

        risk_values = {
            OperationRisk.LOW: 1,
            OperationRisk.MEDIUM: 2,
            OperationRisk.HIGH: 3,
            OperationRisk.CRITICAL: 4
        }

        base_score = risk_values[base_risk]

        for factor in risk_factors:
            factor_score = risk_values[factor.risk_level]
            weighted_risk_score += factor_score * factor.weight
            total_weight += factor.weight

        if total_weight > 0:
            average_factor_score = weighted_risk_score / total_weight
            # Combine base score with factor score (weighted average)
            final_score = (base_score + average_factor_score) / 2
        else:
            final_score = base_score

        # Map back to risk level
        if final_score >= 3.5:
            return OperationRisk.CRITICAL
        elif final_score >= 2.5:
            return OperationRisk.HIGH
        elif final_score >= 1.5:
            return OperationRisk.MEDIUM
        else:
            return OperationRisk.LOW

    def _generate_recommendations(self, risk_level: OperationRisk, 
                                risk_factors: List[RiskFactor], 
                                operation_type: str) -> List[str]:
        """Generate recommendations based on risk analysis"""
        recommendations = []

        if risk_level == OperationRisk.CRITICAL:
            recommendations.append("⚠️  CRITICAL: This operation is potentially destructive and should be avoided")
            recommendations.append("Consider breaking down the operation into smaller, safer steps")
            recommendations.append("Ensure comprehensive backups are created before proceeding")

        elif risk_level == OperationRisk.HIGH:
            recommendations.append("⚠️  HIGH RISK: Proceed with extreme caution")
            recommendations.append("Create full project backup before executing")
            recommendations.append("Test the operation on a copy of the project first")
            recommendations.append("Consider performing the operation in smaller increments")

        elif risk_level == OperationRisk.MEDIUM:
            recommendations.append("⚠️  MEDIUM RISK: Standard precautions recommended")
            recommendations.append("Create backup of affected files")
            recommendations.append("Ensure all tests pass before and after the operation")

        else:  # LOW risk
            recommendations.append("✅ LOW RISK: Operation appears safe to execute")
            recommendations.append("Standard backup procedures are sufficient")

        # Add specific recommendations based on risk factors
        for factor in risk_factors:
            if factor.name == "critical_files":
                recommendations.append("Extra caution needed: critical system files are affected")
            elif factor.name == "global_scope":
                recommendations.append("Consider limiting scope to reduce impact")
            elif factor.name == "api_files":
                recommendations.append("Verify API compatibility after changes")
            elif factor.name == "dependency_impact":
                recommendations.append("Check for external dependencies that might be affected")

        return recommendations

    def is_operation_safe(self, operation_type: str, target_files: List[str], 
                         operation_details: Dict[str, Any] = None) -> bool:
        """
        Quick check if an operation is safe to execute
        
        Args:
            operation_type: Type of operation
            target_files: Files to be modified
            operation_details: Additional operation details
            
        Returns:
            True if operation is safe (not critical risk)
        """
        analysis = self.analyze_operation(operation_type, target_files, operation_details)
        return analysis.is_safe

    def get_operation_risk_level(self, operation_type: str) -> OperationRisk:
        """Get the base risk level for an operation type"""
        return self.operation_risks.get(operation_type, OperationRisk.MEDIUM)
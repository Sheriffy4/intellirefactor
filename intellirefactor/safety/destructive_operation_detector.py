"""
Destructive Operation Detector for IntelliRefactor

Analyzes refactoring operations to detect potentially destructive changes
and assess risk levels for safe refactoring.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

logger = logging.getLogger(__name__)


class OperationRisk(Enum):
    """Risk levels for refactoring operations"""

    LOW = "low"  # Safe operations like formatting
    MEDIUM = "medium"  # Structural changes
    HIGH = "high"  # Potentially destructive operations
    CRITICAL = "critical"  # Operations that could break the project


@dataclass
class RiskFactor:
    """Represents a risk factor in an operation"""

    name: str
    description: str
    risk_level: OperationRisk
    weight: float  # 0.0 to 1.0, higher means more risky
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationAnalysis:
    """Analysis result for a refactoring operation"""

    operation_type: str
    target_files: List[str]
    risk_level: OperationRisk
    risk_factors: List[RiskFactor]
    is_safe: bool
    recommendations: List[str]
    # optional debugging / explainability payload
    details: Dict[str, Any] = field(default_factory=dict)


class DestructiveOperationDetector:
    """
    Detects and analyzes potentially destructive refactoring operations.

    Provides risk assessment and safety recommendations for various
    types of refactoring operations.
    """

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)

        # Base operation risk levels (single source of truth).
        self.operation_risks: Dict[str, OperationRisk] = {
            # Low risk
            "format_code": OperationRisk.LOW,
            "add_comments": OperationRisk.LOW,
            "fix_imports": OperationRisk.LOW,
            "sort_imports": OperationRisk.LOW,
            "add_docstrings": OperationRisk.LOW,
            "add_type_hints": OperationRisk.LOW,
            "extract_variable": OperationRisk.LOW,
            "inline_variable": OperationRisk.LOW,
            # Medium risk
            "extract_method": OperationRisk.MEDIUM,
            "rename_variable": OperationRisk.MEDIUM,
            "inline_method": OperationRisk.MEDIUM,
            "extract_class": OperationRisk.MEDIUM,
            "move_method": OperationRisk.MEDIUM,
            "introduce_parameter": OperationRisk.MEDIUM,
            "remove_parameter": OperationRisk.MEDIUM,
            "rename_file": OperationRisk.MEDIUM,
            "move_file": OperationRisk.MEDIUM,
            "restructure_module": OperationRisk.MEDIUM,
            # High risk
            "rename_class": OperationRisk.HIGH,
            "rename_function": OperationRisk.HIGH,
            "move_class": OperationRisk.HIGH,
            "move_function": OperationRisk.HIGH,
            "change_signature": OperationRisk.HIGH,
            "change_method_signature": OperationRisk.HIGH,
            "modify_inheritance": OperationRisk.HIGH,
            "change_interface": OperationRisk.HIGH,
            "remove_method": OperationRisk.HIGH,
            "merge_classes": OperationRisk.HIGH,
            "split_class": OperationRisk.HIGH,
            "remove_import": OperationRisk.HIGH,
            "remove_dependency": OperationRisk.HIGH,
            # Critical
            "delete_file": OperationRisk.CRITICAL,
            "delete_directory": OperationRisk.CRITICAL,
            "remove_class": OperationRisk.CRITICAL,
            "remove_function": OperationRisk.CRITICAL,
            "remove_module": OperationRisk.CRITICAL,
            "restructure_package": OperationRisk.CRITICAL,
        }

        # Critical file patterns that increase risk (regex, case-insensitive).
        self.critical_file_patterns: List[str] = [
            r"__init__\.py$",
            r"setup\.py$",
            r"main\.py$",
            r"app\.py$",
            r"config\.py$",
            r"settings\.py$",
            r"requirements\.txt$",
            r"pyproject\.toml$",
            r"dockerfile$",
            r"\.github/.*\.(yml|yaml)$",
        ]
        self._critical_file_regexes = [re.compile(p, re.IGNORECASE) for p in self.critical_file_patterns]

        # High-impact directories (heuristics).
        self.high_impact_directories = {
            "src",
            "lib",
            "core",
            "api",
            "models",
            "services",
            "utils",
            "common",
        }

    def analyze_operation(
        self,
        operation: Union[Dict[str, Any], str],
        target_files: Optional[Sequence[str]] = None,
        operation_details: Optional[Dict[str, Any]] = None,
    ) -> OperationAnalysis:
        """
        Analyze a single operation.

        Supports both call styles:
          1) analyze_operation({"type": "...", "target_files": [...], ...})
          2) analyze_operation("rename_class", ["a.py"], {...})
        """
        if isinstance(operation, dict):
            operation_type = str(operation.get("type", "unknown"))
            files = list(operation.get("target_files", []) or [])
            details = dict(operation)
        else:
            operation_type = str(operation)
            files = list(target_files or [])
            details = dict(operation_details or {})

        self.logger.debug("Analyzing operation: %s", operation_type)

        base_risk = self.operation_risks.get(operation_type, OperationRisk.MEDIUM)

        risk_factors: List[RiskFactor] = []
        risk_factors.extend(self._analyze_file_risks(files))
        risk_factors.extend(self._analyze_scope_risks(details, files))
        risk_factors.extend(self._analyze_dependency_risks(operation_type, files, details))
        risk_factors.extend(self._analyze_complexity_risks(files, operation_type))

        final_risk = self._calculate_overall_risk(base_risk, risk_factors)
        recommendations = self._generate_recommendations(final_risk, risk_factors, operation_type)

        return OperationAnalysis(
            operation_type=operation_type,
            target_files=files,
            risk_level=final_risk,
            risk_factors=risk_factors,
            is_safe=(final_risk != OperationRisk.CRITICAL),
            recommendations=recommendations,
            details={
                "base_risk": base_risk.value,
                "final_risk": final_risk.value,
                "file_count": len(files),
            },
        )

    def analyze_operation_batch(self, operations: List[Dict[str, Any]]) -> List[OperationAnalysis]:
        """Analyze a batch of operations."""
        return [self.analyze_operation(op) for op in operations]

    def _normalize_path(self, file_path: str) -> str:
        # make matching stable across platforms (Windows separators etc.)
        return str(file_path).replace("\\", "/").lower()

    def _is_critical_file(self, file_path: str) -> bool:
        p = self._normalize_path(file_path)
        name = Path(file_path).name.lower()
        if name in {"__init__.py", "setup.py", "main.py", "app.py", "config.py", "settings.py"}:
            return True
        for rx in self._critical_file_regexes:
            if rx.search(p):
                return True
        return False

    def _analyze_file_risks(self, target_files: Sequence[str]) -> List[RiskFactor]:
        risk_factors: List[RiskFactor] = []

        critical_files = [fp for fp in target_files if self._is_critical_file(fp)]
        if critical_files:
            risk_factors.append(
                RiskFactor(
                    name="critical_files",
                    description="Operation affects critical files",
                    risk_level=OperationRisk.HIGH,
                    weight=0.8,
                    details={"critical_files": critical_files},
                )
            )

        if len(target_files) > 20:
            risk_factors.append(
                RiskFactor(
                    name="high_file_count",
                    description=f"Operation affects {len(target_files)} files",
                    risk_level=OperationRisk.MEDIUM,
                    weight=0.6,
                    details={"file_count": len(target_files)},
                )
            )

        high_impact_files: List[str] = []
        for fp in target_files:
            p = self._normalize_path(fp)
            if any(p.startswith(f"{d}/") or f"/{d}/" in p for d in self.high_impact_directories):
                high_impact_files.append(fp)
        if high_impact_files:
            risk_factors.append(
                RiskFactor(
                    name="high_impact_directories",
                    description="Operation affects files in high-impact directories",
                    risk_level=OperationRisk.MEDIUM,
                    weight=0.5,
                    details={"high_impact_files": high_impact_files},
                )
            )

        return risk_factors

    def _analyze_scope_risks(
        self, operation_details: Dict[str, Any], target_files: Sequence[str]
    ) -> List[RiskFactor]:
        """Analyze risk factors based on operation scope"""
        risk_factors = []

        scope = operation_details.get("scope", "local")
        if scope == "global":
            risk_factors.append(
                RiskFactor(
                    name="global_scope",
                    description="Operation has global scope affecting multiple components",
                    risk_level=OperationRisk.HIGH,
                    weight=0.7,
                    details={"scope": scope},
                )
            )

        return risk_factors

    def _analyze_dependency_risks(
        self, operation_type: str, target_files: Sequence[str], operation_details: Dict[str, Any]
    ) -> List[RiskFactor]:
        """Analyze risk factors based on dependency impact"""
        risk_factors = []

        high_dependency_impact_ops = {
            "rename_class",
            "rename_function",
            "move_class",
            "move_function",
            "change_interface",
            "remove_method",
            "change_method_signature",
            "change_signature",
        }
        if operation_type in high_dependency_impact_ops:
            risk_factors.append(
                RiskFactor(
                    name="dependency_impact",
                    description=f"Operation '{operation_type}' may impact external dependencies",
                    risk_level=OperationRisk.MEDIUM,
                    weight=0.5,
                    details={"operation_type": operation_type},
                )
            )

        # Check for public API files (heuristic based on file names)
        api_files = []
        for file_path in target_files:
            if any(
                keyword in file_path.lower() for keyword in ["api", "interface", "public", "client"]
            ):
                api_files.append(file_path)

        if api_files:
            risk_factors.append(
                RiskFactor(
                    name="api_files",
                    description="Operation affects potential API files",
                    risk_level=OperationRisk.HIGH,
                    weight=0.7,
                    details={"api_files": api_files},
                )
            )

        return risk_factors

    def _analyze_complexity_risks(
        self, target_files: Sequence[str], operation_type: str
    ) -> List[RiskFactor]:
        """Analyze risk factors based on code complexity"""
        risk_factors = []

        try:
            total_lines = 0
            complex_files = []

            for file_path in target_files:
                if not str(file_path).endswith(".py"):
                    continue
                p = Path(file_path)
                if not p.exists():
                    continue

                try:
                    with p.open("r", encoding="utf-8") as f:
                        lines = f.readlines()
                        line_count = len(
                            [
                                line
                                for line in lines
                                if line.strip() and not line.strip().startswith("#")
                            ]
                        )
                        total_lines += line_count

                        # Consider files with >500 lines as complex
                        if line_count > 500:
                            complex_files.append((file_path, line_count))

                except Exception as e:
                    logger.debug("Could not analyze complexity for %s: %s", file_path, e)

            if complex_files:
                risk_factors.append(
                    RiskFactor(
                        name="complex_files",
                        description=f"Operation affects {len(complex_files)} complex files",
                        risk_level=OperationRisk.MEDIUM,
                        weight=0.4,
                        details={"complex_files": complex_files},
                    )
                )

            # Large total codebase impact
            if total_lines > 5000:
                risk_factors.append(
                    RiskFactor(
                        name="large_codebase_impact",
                        description=f"Operation affects {total_lines} lines of code",
                        risk_level=OperationRisk.MEDIUM,
                        weight=0.3,
                        details={"total_lines": total_lines},
                    )
                )

        except Exception as e:
            logger.debug("Error analyzing complexity risks: %s", e)

        return risk_factors

    def _calculate_overall_risk(
        self, base_risk: OperationRisk, risk_factors: List[RiskFactor]
    ) -> OperationRisk:
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
            OperationRisk.CRITICAL: 4,
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

    def _generate_recommendations(
        self,
        risk_level: OperationRisk,
        risk_factors: List[RiskFactor],
        operation_type: str,
    ) -> List[str]:
        """Generate recommendations based on risk analysis"""
        recommendations = []

        if risk_level == OperationRisk.CRITICAL:
            recommendations.append(
                "⚠️  CRITICAL: This operation is potentially destructive and should be avoided"
            )
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

    def is_operation_safe(
        self,
        operation_type: str,
        target_files: List[str],
        operation_details: Optional[Dict[str, Any]] = None,
    ) -> bool:
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
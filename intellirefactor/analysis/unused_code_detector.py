"""
Unused Code Detection with Evidence-Based Analysis

This module implements three-level unused code detection:
- Level 1: Module-level reachability from entry points
- Level 2: Symbol-level with usage classification
- Level 3: Dynamic usage patterns with confidence scoring

All findings include comprehensive evidence and confidence scores.
"""

import ast
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict

from .models import FileReference, Evidence
from .file_analyzer import FileAnalyzer
from .lazy_loader import LazyProjectContext


class UnusedCodeType(Enum):
    """Types of unused code detected."""

    MODULE_UNREACHABLE = "module_unreachable"
    SYMBOL_UNUSED = "symbol_unused"
    PRIVATE_METHOD_UNUSED = "private_method_unused"
    PUBLIC_EXPORT_UNUSED = "public_export_unused"
    TESTS_ONLY = "tests_only"
    SCRIPTS_ONLY = "scripts_only"
    UNCERTAIN_DYNAMIC = "uncertain_dynamic"


class UsagePattern(Enum):
    """Usage patterns for code analysis."""

    DIRECT_IMPORT = "direct_import"
    FROM_IMPORT = "from_import"
    DYNAMIC_IMPORT = "dynamic_import"
    GETATTR_ACCESS = "getattr_access"
    EVAL_USAGE = "eval_usage"
    STRING_REFERENCE = "string_reference"
    TEST_USAGE = "test_usage"
    SCRIPT_USAGE = "script_usage"


class ConfidenceLevel(Enum):
    """Confidence levels for unused code detection."""

    HIGH = "high"  # 0.8-1.0
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"  # 0.0-0.5


@dataclass
class UsageReference:
    """Reference to where code is used."""

    file_path: str
    line_number: int
    pattern: UsagePattern
    context: str
    confidence: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "pattern": self.pattern.value,
            "context": self.context,
            "confidence": self.confidence,
        }


@dataclass
class UnusedCodeFinding:
    """Represents a finding of unused code with evidence."""

    symbol_name: str
    file_path: str
    line_start: int
    line_end: int
    unused_type: UnusedCodeType
    confidence: float
    evidence: Evidence
    usage_references: List[UsageReference] = field(default_factory=list)
    dynamic_usage_indicators: List[str] = field(default_factory=list)
    severity: str = "medium"

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level based on confidence score."""
        if self.confidence >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.confidence >= 0.5:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "symbol_name": self.symbol_name,
            "file_path": self.file_path,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "unused_type": self.unused_type.value,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "evidence": self.evidence.to_dict(),
            "usage_references": [ref.to_dict() for ref in self.usage_references],
            "dynamic_usage_indicators": self.dynamic_usage_indicators,
            "severity": self.severity,
        }


@dataclass
class UnusedCodeAnalysisResult:
    """Results of unused code analysis."""

    findings: List[UnusedCodeFinding]
    statistics: Dict[str, Any]
    entry_points: List[str]
    analysis_metadata: Dict[str, Any]

    def get_findings_by_type(self, unused_type: UnusedCodeType) -> List[UnusedCodeFinding]:
        """Get findings filtered by type."""
        return [f for f in self.findings if f.unused_type == unused_type]

    def get_findings_by_confidence(self, min_confidence: float) -> List[UnusedCodeFinding]:
        """Get findings filtered by minimum confidence."""
        return [f for f in self.findings if f.confidence >= min_confidence]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "findings": [f.to_dict() for f in self.findings],
            "statistics": self.statistics,
            "entry_points": self.entry_points,
            "analysis_metadata": self.analysis_metadata,
        }


class UnusedCodeDetector:
    """
    Three-level unused code detector with evidence-based analysis.

    Level 1: Module-level reachability from entry points
    Level 2: Symbol-level with usage classification
    Level 3: Dynamic usage patterns with confidence scoring
    """

    def __init__(self, project_path: Union[str, Path], library_mode: bool = False):
        """Initialize the unused code detector."""
        self.project_path = Path(project_path)
        self.library_mode = library_mode
        self.file_analyzer = FileAnalyzer()
        self.logger = logging.getLogger(__name__)

        # Analysis state
        self.module_graph: Dict[str, Set[str]] = defaultdict(set)
        self.symbol_usage: Dict[str, List[UsageReference]] = defaultdict(list)
        self.dynamic_patterns: Dict[str, List[str]] = defaultdict(list)
        self.entry_points: List[str] = []

        # Configuration
        self.test_patterns = [
            r"test_.*\.py$",
            r".*_test\.py$",
            r".*/tests/.*\.py$",
            r".*/test/.*\.py$",
        ]

        self.script_patterns = [
            r".*script.*\.py$",
            r".*main\.py$",
            r".*cli\.py$",
            r".*run.*\.py$",
        ]

        self.dynamic_usage_patterns = [
            r"getattr\s*\(",
            r"hasattr\s*\(",
            r"setattr\s*\(",
            r"importlib\.",
            r"__import__\s*\(",
            r"eval\s*\(",
            r"exec\s*\(",
            r"globals\s*\(\)",
            r"locals\s*\(\)",
        ]

        # Dynamic import patterns for more specific analysis
        self.dynamic_import_patterns = [
            r'importlib\.import_module\s*\(\s*["\']([^"\']*)["\']',
            r'__import__\s*\(\s*["\']([^"\']*)["\']',
            r'importlib\.util\.spec_from_file_location\s*\(\s*["\']([^"\']*)["\']',
            r'importlib\.load\s*\(\s*["\']([^"\']*)["\']',
        ]

    def detect_unused_code(
        self,
        entry_points: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        min_confidence: float = 0.5,
        external_index: Optional["IndexStore"] = None,
    ) -> UnusedCodeAnalysisResult:
        """
        Perform comprehensive unused code detection.

        Args:
            entry_points: List of entry point files/modules
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude
            min_confidence: Minimum confidence threshold
            external_index: Optional external index store for cross-module dependency analysis

        Returns:
            UnusedCodeAnalysisResult with findings and evidence
        """
        # Set default patterns
        if include_patterns is None:
            include_patterns = ["**/*.py"]
        if exclude_patterns is None:
            exclude_patterns = ["**/__pycache__/**", "**/.*"]

        # Discover Python files
        python_files = self._discover_python_files(include_patterns, exclude_patterns)

        # Determine entry points
        if entry_points is None:
            entry_points = self._discover_entry_points(python_files)
        self.entry_points = entry_points

        # Build analysis data
        self._build_module_graph(python_files)
        self._analyze_symbol_usage(python_files)
        self._detect_dynamic_patterns(python_files)

        # Perform three-level analysis
        findings = []

        # Level 1: Module-level reachability
        findings.extend(
            self._detect_unreachable_modules(python_files, entry_points, external_index)
        )

        # Level 2: Symbol-level analysis
        findings.extend(self._detect_unused_symbols(python_files, external_index))

        # Level 3: Dynamic usage analysis
        findings.extend(self._analyze_dynamic_usage(python_files))

        # Filter by confidence
        findings = [f for f in findings if f.confidence >= min_confidence]

        # Generate statistics
        statistics = self._generate_statistics(findings, python_files)

        # Analysis metadata
        metadata = {
            "total_files_analyzed": len(python_files),
            "entry_points_used": len(entry_points),
            "analysis_timestamp": self._get_timestamp(),
            "confidence_threshold": min_confidence,
            "include_patterns": include_patterns,
            "exclude_patterns": exclude_patterns,
            "external_index_used": external_index is not None,
        }

        return UnusedCodeAnalysisResult(
            findings=findings,
            statistics=statistics,
            entry_points=entry_points,
            analysis_metadata=metadata,
        )

    def _discover_python_files(
        self, include_patterns: List[str], exclude_patterns: List[str]
    ) -> List[Path]:
        """Discover Python files matching patterns."""
        import glob

        python_files = []

        # Find files matching include patterns
        for pattern in include_patterns:
            files = glob.glob(str(self.project_path / pattern), recursive=True)
            python_files.extend([Path(f) for f in files if Path(f).is_file()])

        # Filter out excluded patterns
        for exclude_pattern in exclude_patterns:
            excluded_files = set(
                glob.glob(str(self.project_path / exclude_pattern), recursive=True)
            )
            python_files = [f for f in python_files if str(f) not in excluded_files]

        return list(set(python_files))  # Remove duplicates

    def _discover_entry_points(self, python_files: List[Path]) -> List[str]:
        """Discover likely entry points in the project."""
        entry_points = []

        for file_path in python_files:
            # Check for main execution patterns
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Look for if __name__ == "__main__":
                if 'if __name__ == "__main__"' in content:
                    entry_points.append(str(file_path))

                # Look for setup.py, main.py, cli.py, etc.
                filename = file_path.name.lower()
                if filename in ["setup.py", "main.py", "cli.py", "app.py", "run.py"]:
                    entry_points.append(str(file_path))

            except Exception:
                continue

        # If no entry points found, use common patterns
        if not entry_points:
            for file_path in python_files:
                if any(pattern in str(file_path).lower() for pattern in ["main", "cli", "app"]):
                    entry_points.append(str(file_path))
                    break

        return entry_points

    def _build_module_graph(self, python_files: List[Path]) -> None:
        """Build module dependency graph."""
        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content, filename=str(file_path))
                module_name = self._get_module_name(file_path)

                # Find imports
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imported_module = alias.name
                            self.module_graph[module_name].add(imported_module)

                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imported_module = node.module
                            self.module_graph[module_name].add(imported_module)

            except Exception:
                continue

    def _analyze_symbol_usage(self, python_files: List[Path]) -> None:
        """Analyze symbol usage across the project."""
        # First pass: collect all symbols
        all_symbols = {}

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content, filename=str(file_path))

                # Collect function and class definitions
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                        symbol_key = f"{file_path}:{node.name}"
                        all_symbols[symbol_key] = {
                            "name": node.name,
                            "file_path": str(file_path),
                            "line_start": node.lineno,
                            "line_end": getattr(node, "end_lineno", node.lineno),
                            "is_private": node.name.startswith("_"),
                            "node_type": type(node).__name__,
                        }

            except Exception:
                continue

        # Second pass: find usage references
        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    lines = content.splitlines()

                tree = ast.parse(content, filename=str(file_path))

                # Find function/method calls and attribute access
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            symbol_name = node.func.id
                            self._add_usage_reference(
                                symbol_name,
                                str(file_path),
                                node.lineno,
                                UsagePattern.DIRECT_IMPORT,
                                lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                                0.9,
                            )

                    elif isinstance(node, ast.Attribute):
                        symbol_name = node.attr
                        self._add_usage_reference(
                            symbol_name,
                            str(file_path),
                            node.lineno,
                            UsagePattern.GETATTR_ACCESS,
                            lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                            0.7,
                        )

            except Exception:
                continue

    def _detect_dynamic_patterns(self, python_files: List[Path]) -> None:
        """Detect dynamic usage patterns that might indicate hidden usage."""
        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Check for general dynamic usage patterns
                for pattern in self.dynamic_usage_patterns:
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        context = content[max(0, match.start() - 50) : match.end() + 50]
                        self.dynamic_patterns[str(file_path)].append(
                            f"Line {line_num}: {context.strip()}"
                        )

                # Specifically check for dynamic import patterns
                for import_pattern in self.dynamic_import_patterns:
                    matches = re.finditer(import_pattern, content, re.IGNORECASE)
                    for match in matches:
                        line_num = content[: match.start()].count("\n") + 1
                        context = content[max(0, match.start() - 50) : match.end() + 50]
                        # Extract the module name that's being dynamically imported
                        if match.groups():
                            module_name = match.group(1)
                            self.dynamic_patterns[str(file_path)].append(
                                f"Line {line_num}: Dynamic import of module '{module_name}': {context.strip()}"
                            )

            except Exception:
                continue

    def _detect_unreachable_modules(
        self,
        python_files: List[Path],
        entry_points: List[str],
        external_index: Optional["IndexStore"] = None,
    ) -> List[UnusedCodeFinding]:
        """Level 1: Detect modules unreachable from entry points."""
        findings = []

        # Build reachability graph from entry points
        reachable_modules = set()

        def mark_reachable(module_name: str):
            if module_name in reachable_modules:
                return
            reachable_modules.add(module_name)
            for imported in self.module_graph.get(module_name, []):
                mark_reachable(imported)

        # Mark all modules reachable from entry points
        for entry_point in entry_points:
            entry_module = self._get_module_name(Path(entry_point))
            mark_reachable(entry_module)

        # Find unreachable modules
        for file_path in python_files:
            module_name = self._get_module_name(file_path)

            # Check if module is reachable from entry points
            is_reachable_from_entry_points = module_name in reachable_modules

            # If not reachable from entry points, check external index for usage
            is_used_in_external_project = False
            if external_index and not is_reachable_from_entry_points:
                try:
                    # Check if this module is imported by other modules in the external index
                    is_used_in_external_project = self._is_module_used_in_external_index(
                        module_name, external_index
                    )
                except Exception:
                    # If there's an error accessing external index, continue with local analysis
                    pass

            if not is_reachable_from_entry_points and not is_used_in_external_project:
                # Check if it's a test or script file
                is_test = any(re.search(pattern, str(file_path)) for pattern in self.test_patterns)
                is_script = any(
                    re.search(pattern, str(file_path)) for pattern in self.script_patterns
                )

                if is_test:
                    unused_type = UnusedCodeType.TESTS_ONLY
                    confidence = 0.6  # Lower confidence for test files
                elif is_script:
                    unused_type = UnusedCodeType.SCRIPTS_ONLY
                    confidence = 0.7
                else:
                    unused_type = UnusedCodeType.MODULE_UNREACHABLE
                    confidence = 0.9

                # Create evidence
                evidence = Evidence(
                    description=f"Module {module_name} is not reachable from any entry point",
                    confidence=confidence,
                    file_references=[FileReference(str(file_path), 1, -1)],
                    code_snippets=[],
                    metadata={
                        "entry_points_checked": entry_points,
                        "is_test_file": is_test,
                        "is_script_file": is_script,
                        "module_imports": list(self.module_graph.get(module_name, [])),
                        "used_in_external_project": is_used_in_external_project,
                    },
                )

                finding = UnusedCodeFinding(
                    symbol_name=module_name,
                    file_path=str(file_path),
                    line_start=1,
                    line_end=-1,
                    unused_type=unused_type,
                    confidence=confidence,
                    evidence=evidence,
                    severity="high" if confidence > 0.8 else "medium",
                )

                findings.append(finding)
            elif is_used_in_external_project:
                # Module is used in external project, so it's not unused
                # Add debug logging or handle as needed
                pass

        return findings

    def _detect_unused_symbols(
        self, python_files: List[Path], external_index: Optional["IndexStore"] = None
    ) -> List[UnusedCodeFinding]:
        """Level 2: Detect unused symbols with usage classification."""
        findings = []

        # Collect all defined symbols
        defined_symbols = {}

        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                tree = ast.parse(content, filename=str(file_path))

                # Get __all__ definitions if in library mode
                all_exports = []
                if self.library_mode:
                    all_exports = self._get_all_exports(tree)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                        symbol_key = f"{file_path}:{node.name}"
                        defined_symbols[symbol_key] = {
                            "name": node.name,
                            "file_path": str(file_path),
                            "line_start": node.lineno,
                            "line_end": getattr(node, "end_lineno", node.lineno),
                            "is_private": node.name.startswith("_"),
                            "node_type": type(node).__name__,
                            "is_exported": node.name in all_exports,  # For library mode
                        }

            except Exception:
                continue

        # Check usage for each symbol
        for symbol_key, symbol_info in defined_symbols.items():
            symbol_name = symbol_info["name"]
            usage_refs = self.symbol_usage.get(symbol_name, [])

            # Filter out self-references (usage in the same file)
            external_usage = [
                ref for ref in usage_refs if ref.file_path != symbol_info["file_path"]
            ]

            # Check external index for usage if provided
            is_used_in_external_project = False
            if external_index and not external_usage:
                try:
                    is_used_in_external_project = self._is_symbol_used_in_external_index(
                        symbol_name, symbol_info["file_path"], external_index
                    )
                except Exception:
                    # If there's an error accessing external index, continue with local analysis
                    pass

            if not external_usage and not is_used_in_external_project:
                # In library mode, public symbols and exported symbols are considered potentially used
                is_potentially_used_in_library = False
                if self.library_mode:
                    # Public symbols (not starting with _) and symbols in __all__ are potentially used
                    is_public = not symbol_info["is_private"]
                    is_exported = symbol_info.get("is_exported", False)
                    is_potentially_used_in_library = is_public or is_exported

                if is_potentially_used_in_library:
                    # In library mode, don't flag this as unused
                    continue

                # Determine unused type and confidence
                if symbol_info["is_private"]:
                    unused_type = UnusedCodeType.PRIVATE_METHOD_UNUSED
                    confidence = 0.8
                else:
                    unused_type = UnusedCodeType.PUBLIC_EXPORT_UNUSED
                    confidence = 0.7  # Lower confidence for public symbols

                # Check for dynamic usage indicators
                dynamic_indicators = self.dynamic_patterns.get(symbol_info["file_path"], [])

                # Check specifically for dynamic imports
                dynamic_import_indicators = [
                    indicator
                    for indicator in dynamic_indicators
                    if "Dynamic import of module" in indicator
                ]

                if dynamic_indicators:
                    if dynamic_import_indicators:
                        # If there are dynamic imports, reduce confidence more significantly
                        confidence *= 0.4  # More significant reduction for dynamic imports
                    else:
                        confidence *= 0.6  # Reduce confidence if other dynamic patterns present

                # Create evidence
                evidence = Evidence(
                    description=f"Symbol {symbol_name} has no external usage references",
                    confidence=confidence,
                    file_references=[
                        FileReference(
                            symbol_info["file_path"],
                            symbol_info["line_start"],
                            symbol_info["line_end"],
                        )
                    ],
                    code_snippets=[],
                    metadata={
                        "symbol_type": symbol_info["node_type"],
                        "is_private": symbol_info["is_private"],
                        "internal_usage_count": len(usage_refs) - len(external_usage),
                        "dynamic_indicators_count": len(dynamic_indicators),
                        "dynamic_import_indicators_count": len(dynamic_import_indicators),
                        "used_in_external_project": is_used_in_external_project,
                        "library_mode": self.library_mode,
                        "is_potentially_used_in_library": is_potentially_used_in_library,
                        "has_dynamic_imports": len(dynamic_import_indicators) > 0,
                    },
                )

                finding = UnusedCodeFinding(
                    symbol_name=symbol_name,
                    file_path=symbol_info["file_path"],
                    line_start=symbol_info["line_start"],
                    line_end=symbol_info["line_end"],
                    unused_type=unused_type,
                    confidence=confidence,
                    evidence=evidence,
                    usage_references=usage_refs,
                    dynamic_usage_indicators=dynamic_indicators,
                    severity="medium" if symbol_info["is_private"] else "low",
                )

                findings.append(finding)
            elif is_used_in_external_project:
                # Symbol is used in external project, so it's not unused
                # Add debug logging or handle as needed
                pass

        return findings

    def _analyze_dynamic_usage(self, python_files: List[Path]) -> List[UnusedCodeFinding]:
        """Level 3: Analyze dynamic usage patterns."""
        findings = []

        # Identify potential dynamic usage that could reference symbols
        # This is important for library modules where symbols are accessed dynamically

        for file_path in python_files:
            dynamic_indicators = self.dynamic_patterns.get(str(file_path), [])

            if dynamic_indicators:
                # Check if any of the dynamic indicators are dynamic imports
                dynamic_import_indicators = [
                    indicator
                    for indicator in dynamic_indicators
                    if "Dynamic import of module" in indicator
                ]

                if dynamic_import_indicators:
                    # If this file has dynamic imports, reduce confidence that symbols are truly unused
                    # This is particularly important for plugin systems or dynamic loading
                    for indicator in dynamic_import_indicators:
                        self.logger.info(f"Found dynamic import in {file_path}: {indicator}")

                    # Create findings for files with dynamic imports
                    evidence = Evidence(
                        description=f"File contains dynamic import patterns that may reference symbols ({len(dynamic_import_indicators)} found)",
                        confidence=0.4,  # Low-medium confidence due to uncertainty
                        file_references=[FileReference(str(file_path), 1, -1)],
                        code_snippets=[],
                        metadata={
                            "dynamic_patterns": dynamic_indicators,
                            "dynamic_imports": dynamic_import_indicators,
                            "pattern_count": len(dynamic_indicators),
                            "import_count": len(dynamic_import_indicators),
                        },
                    )

                    finding = UnusedCodeFinding(
                        symbol_name=f"dynamic_import_{file_path.name}",
                        file_path=str(file_path),
                        line_start=1,
                        line_end=-1,
                        unused_type=UnusedCodeType.UNCERTAIN_DYNAMIC,
                        confidence=0.4,
                        evidence=evidence,
                        dynamic_usage_indicators=dynamic_import_indicators,
                        severity="low",
                    )

                    findings.append(finding)
                else:
                    # Create findings for files with other dynamic usage patterns
                    evidence = Evidence(
                        description="File contains dynamic usage patterns that may hide code usage",
                        confidence=0.3,  # Low confidence due to uncertainty
                        file_references=[FileReference(str(file_path), 1, -1)],
                        code_snippets=[],
                        metadata={
                            "dynamic_patterns": dynamic_indicators,
                            "pattern_count": len(dynamic_indicators),
                        },
                    )

                    finding = UnusedCodeFinding(
                        symbol_name=f"dynamic_usage_{file_path.name}",
                        file_path=str(file_path),
                        line_start=1,
                        line_end=-1,
                        unused_type=UnusedCodeType.UNCERTAIN_DYNAMIC,
                        confidence=0.3,
                        evidence=evidence,
                        dynamic_usage_indicators=dynamic_indicators,
                        severity="low",
                    )

                    findings.append(finding)

        return findings

    def _add_usage_reference(
        self,
        symbol_name: str,
        file_path: str,
        line_number: int,
        pattern: UsagePattern,
        context: str,
        confidence: float,
    ) -> None:
        """Add a usage reference for a symbol."""
        ref = UsageReference(
            file_path=file_path,
            line_number=line_number,
            pattern=pattern,
            context=context,
            confidence=confidence,
        )
        self.symbol_usage[symbol_name].append(ref)

    def _get_module_name(self, file_path: Path) -> str:
        """Get module name from file path."""
        try:
            relative_path = file_path.relative_to(self.project_path)
            module_parts = list(relative_path.parts[:-1])  # Remove filename
            if relative_path.stem != "__init__":
                module_parts.append(relative_path.stem)
            return ".".join(module_parts) if module_parts else relative_path.stem
        except ValueError:
            return str(file_path)

    def _generate_statistics(
        self, findings: List[UnusedCodeFinding], python_files: List[Path]
    ) -> Dict[str, Any]:
        """Generate analysis statistics."""
        stats = {
            "total_findings": len(findings),
            "total_files_analyzed": len(python_files),
            "findings_by_type": {},
            "findings_by_confidence": {"high": 0, "medium": 0, "low": 0},
            "findings_by_severity": {"high": 0, "medium": 0, "low": 0},
        }

        # Count by type
        for finding in findings:
            unused_type = finding.unused_type.value
            stats["findings_by_type"][unused_type] = (
                stats["findings_by_type"].get(unused_type, 0) + 1
            )

            # Count by confidence level
            stats["findings_by_confidence"][finding.confidence_level.value] += 1

            # Count by severity
            stats["findings_by_severity"][finding.severity] += 1

        return stats

    def _get_all_exports(self, tree: ast.AST) -> List[str]:
        """Get the list of symbols from __all__ definition if present.

        Args:
            tree: The AST tree of the file

        Returns:
            List of symbol names from __all__
        """
        all_exports = []

        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id == "__all__"
                and isinstance(node.value, (ast.List, ast.Tuple))
            ):
                for elt in node.value.elts:
                    if isinstance(elt, ast.Str):  # Python < 3.8
                        all_exports.append(elt.s)
                    elif (
                        hasattr(ast, "Constant")
                        and isinstance(elt, ast.Constant)
                        and isinstance(elt.value, str)
                    ):  # Python 3.8+
                        all_exports.append(elt.value)

        return all_exports

    def _get_timestamp(self) -> str:
        """Get current timestamp."""
        from datetime import datetime

        return datetime.now().isoformat()

    def _is_module_used_in_external_index(
        self, module_name: str, external_index: "IndexStore"
    ) -> bool:
        """Check if a module is imported/used in the external index database using lazy loading.

        Args:
            module_name: The name of the module to check
            external_index: The external index store to query

        Returns:
            True if the module is used in the external project, False otherwise
        """
        try:
            # Create a lazy context for the external project
            lazy_context = LazyProjectContext(Path("."), external_index)

            # Use the lazy loader to find importers of the module
            importers = lazy_context.find_importers_of_module(module_name)

            # If there are importers, the module is used
            if importers:
                return True

            # Also check the dependencies table if it exists using lazy loading
            try:
                # Check if this module name appears as an imported module in other files
                with external_index._get_connection() as conn:
                    cursor = conn.execute(
                        """SELECT COUNT(*) FROM symbols WHERE ast_fingerprint LIKE ? OR token_fingerprint LIKE ?""",
                        (f"%{module_name}%", f"%{module_name}%"),
                    )
                    count = cursor.fetchone()[0]
                    if count > 0:
                        return True
            except Exception:
                pass

            return False
        except Exception:
            # If there's any error accessing the external index, return False
            return False

    def _is_symbol_used_in_external_index(
        self, symbol_name: str, file_path: str, external_index: "IndexStore"
    ) -> bool:
        """Check if a symbol is used in the external index database using lazy loading.

        Args:
            symbol_name: The name of the symbol to check
            file_path: The file path of the symbol
            external_index: The external index store to query

        Returns:
            True if the symbol is used in the external project, False otherwise
        """
        try:
            # Create a lazy context for the external project
            lazy_context = LazyProjectContext(Path("."), external_index)

            # Use the lazy loader to find usage of the symbol
            symbol_usage = lazy_context.find_usage_of_symbol(symbol_name)

            # If there are usage references, the symbol is used
            if symbol_usage:
                return True

            # If not found through lazy loading, try direct database queries
            with external_index._get_connection() as conn:
                # Check if this symbol name appears in usage patterns in other files
                cursor = conn.execute(
                    """SELECT COUNT(*) FROM symbols WHERE name = ? AND file_path != ?""",
                    (symbol_name, file_path),
                )
                count = cursor.fetchone()[0]

                # If we found matches, the symbol might be used
                if count > 0:
                    return True

                # Check in dependencies table
                try:
                    cursor = conn.execute(
                        """SELECT COUNT(*) FROM dependencies WHERE target_external LIKE ?""",
                        (f"%{symbol_name}%",),
                    )
                    count = cursor.fetchone()[0]
                    return count > 0
                except sqlite3.OperationalError:
                    # dependencies table might not exist
                    pass

                # Check attribute access table
                try:
                    cursor = conn.execute(
                        """SELECT COUNT(*) FROM attribute_access WHERE attribute_name = ?""",
                        (symbol_name,),
                    )
                    count = cursor.fetchone()[0]
                    return count > 0
                except sqlite3.OperationalError:
                    # attribute_access table might not exist
                    pass

                return False
        except Exception:
            # If there's any error accessing the external index, return False
            return False

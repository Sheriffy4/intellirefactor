"""
Unused Code Detection with Evidence-Based Analysis

This module implements three-level unused code detection:
- Level 1: Module-level reachability from entry points
- Level 2: Symbol-level with usage classification
- Level 3: Dynamic usage patterns with confidence scoring

All findings include comprehensive evidence and confidence scores.
"""

from __future__ import annotations

import ast
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional, Any, Union, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
import re
from collections import defaultdict
import fnmatch

from intellirefactor.analysis.foundation.models import FileReference, Evidence, Severity

if TYPE_CHECKING:
    from intellirefactor.analysis.index.store import IndexStore


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
    severity: Severity = Severity.MEDIUM

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
            "severity": self.severity.value,
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

        self.logger = logging.getLogger(__name__)

        # Analysis state
        self.module_graph: Dict[str, Set[str]] = defaultdict(set)
        # Memory-safe usage tracking:
        # - count total uses
        # - remember which files referenced symbol (to detect "external usage")
        # - keep a limited number of examples (for report/evidence)
        self.symbol_usage_count: Dict[str, int] = defaultdict(int)
        self.symbol_usage_files: Dict[str, Set[str]] = defaultdict(set)
        self.symbol_usage_examples: Dict[str, List[UsageReference]] = defaultdict(list)
        self.dynamic_patterns: Dict[str, List[str]] = defaultdict(list)
        self.entry_points: List[str] = []

        # Limits to prevent memory blow-ups on large codebases
        self.max_usage_examples_per_symbol: int = 20
        self.max_dynamic_indicators_per_file: int = 50
        self.max_context_chars: int = 220

        # Skip dirs (like index/collect). Used by internal discovery if user patterns are too broad.
        self._default_skip_dirs: Set[str] = {
            ".git",
            ".hg",
            ".svn",
            ".venv",
            "venv",
            "__pycache__",
            ".mypy_cache",
            ".pytest_cache",
            ".intellirefactor",
            "intellirefactor_out",
        }

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

        # Logging noise control: do not spam logs for tooling / caches / our own outputs
        self._dynamic_log_exclude_substrings = [
            "/.git/",
            "/.hg/",
            "/.svn/",
            "/.venv/",
            "/venv/",
            "/__pycache__/",
            "/.mypy_cache/",
            "/.pytest_cache/",
            "/.ruff_cache/",
            "/.tox/",
            "/.nox/",
            "/.intellirefactor/",
            "/intellirefactor_out",
            "/tests/",
            "/test/",
            "/tools/",
        ]

    def _should_log_dynamic(self, file_path: Union[str, Path]) -> bool:
        p = str(file_path).replace("\\", "/")
        return not any(x in p for x in self._dynamic_log_exclude_substrings)

    def _reset_state(self) -> None:
        """Reset internal state for a fresh run (important if detector reused)."""
        self.module_graph = defaultdict(set)
        self.symbol_usage_count = defaultdict(int)
        self.symbol_usage_files = defaultdict(set)
        self.symbol_usage_examples = defaultdict(list)
        self.dynamic_patterns = defaultdict(list)
        self.entry_points = []

    def _relpath_posix(self, p: Union[str, Path]) -> str:
        """Project-relative, stable POSIX path for reporting and DB matching."""
        pp = Path(p)
        if not pp.is_absolute():
            pp = (self.project_path / pp)
        try:
            return pp.resolve().relative_to(self.project_path.resolve()).as_posix()
        except Exception:
            return pp.as_posix()
    
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
        # reset state to avoid accumulation across runs
        self._reset_state()

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
        """
        Discover Python files matching include/exclude glob patterns.

        Important: this implementation avoids `glob.glob()` + huge intermediate sets,
        which can be very memory-expensive on big repos.
        """
        inc = [str(p or "").replace("\\", "/") for p in (include_patterns or ["**/*.py"])]
        exc = [str(p or "").replace("\\", "/") for p in (exclude_patterns or [])]

        files: List[Path] = []
        root = self.project_path.resolve()

        for p in root.rglob("*.py"):
            # skip well-known non-source dirs fast
            try:
                rel_parts = set(p.relative_to(root).parts)
            except Exception:
                rel_parts = set(p.parts)
            if rel_parts.intersection(self._default_skip_dirs):
                continue

            try:
                rel = p.relative_to(root).as_posix()
            except Exception:
                rel = p.as_posix()

            if inc and not any(fnmatch.fnmatch(rel, pat) for pat in inc):
                continue
            if exc and any(fnmatch.fnmatch(rel, pat) for pat in exc):
                continue

            if p.is_file():
                files.append(p)

        # stable-ish uniq
        seen: Set[str] = set()
        out: List[Path] = []
        for f in files:
            k = str(f)
            if k not in seen:
                seen.add(k)
                out.append(f)
        return out

    def _discover_entry_points(self, python_files: List[Path]) -> List[str]:
        """Discover likely entry points in the project."""
        entry_points = []

        for file_path in python_files:
            # Check for main execution patterns
            try:
                with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
                    content = f.read()

                # Look for if __name__ == "__main__":
                if 'if __name__ == "__main__"' in content:
                    entry_points.append(self._relpath_posix(file_path))

                # Look for setup.py, main.py, cli.py, etc.
                filename = file_path.name.lower()
                if filename in ["setup.py", "main.py", "cli.py", "app.py", "run.py"]:
                    entry_points.append(self._relpath_posix(file_path))

            except Exception:
                continue

        # If no entry points found, use common patterns
        if not entry_points:
            for file_path in python_files:
                if any(pattern in str(file_path).lower() for pattern in ["main", "cli", "app"]):
                    entry_points.append(self._relpath_posix(file_path))
                    break

        return entry_points

    def _build_module_graph(self, python_files: List[Path]) -> None:
        """Build module dependency graph."""
        for file_path in python_files:
            try:
                with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
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
        # Single pass: find usage references.
        # (Previously there was a first pass collecting all_symbols, but it was unused and ate memory.)
        for file_path in python_files:
            try:
                rel_file = self._relpath_posix(file_path)
                with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
                    content = f.read()
                    lines = content.splitlines()

                tree = ast.parse(content, filename=str(file_path))

                # Find function/method calls and attribute access
                for node in ast.walk(tree):
                    if isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            symbol_name = node.func.id
                            examples = self.symbol_usage_examples.get(symbol_name)
                            need_example = (len(examples) if examples else 0) < self.max_usage_examples_per_symbol
                            self._add_usage_reference(
                                symbol_name,
                                rel_file,
                                node.lineno,
                                UsagePattern.DIRECT_IMPORT,
                                (lines[node.lineno - 1] if need_example and node.lineno <= len(lines) else ""),
                                0.9,
                            )

                    elif isinstance(node, ast.Attribute):
                        symbol_name = node.attr
                        examples = self.symbol_usage_examples.get(symbol_name)
                        need_example = (len(examples) if examples else 0) < self.max_usage_examples_per_symbol
                        self._add_usage_reference(
                            symbol_name,
                            rel_file,
                            node.lineno,
                            UsagePattern.GETATTR_ACCESS,
                            (lines[node.lineno - 1] if need_example and node.lineno <= len(lines) else ""),
                            0.7,
                        )

            except Exception:
                continue

    def _detect_dynamic_patterns(self, python_files: List[Path]) -> None:
        """Detect dynamic usage patterns that might indicate hidden usage."""
        for file_path in python_files:
            try:
                file_key = self._relpath_posix(file_path)
                with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
                    content = f.read()

                # Check for general dynamic usage patterns
                for pattern in self.dynamic_usage_patterns:
                    if len(self.dynamic_patterns[file_key]) >= self.max_dynamic_indicators_per_file:
                        break
                    matches = re.finditer(pattern, content, re.IGNORECASE)
                    for match in matches:
                        if len(self.dynamic_patterns[file_key]) >= self.max_dynamic_indicators_per_file:
                            break
                        line_num = content.count("\n", 0, match.start()) + 1
                        context = content[max(0, match.start() - 60) : match.end() + 60]
                        context = " ".join(context.strip().split())
                        if len(context) > self.max_context_chars:
                            context = context[: self.max_context_chars] + "..."
                        self.dynamic_patterns[file_key].append(f"Line {line_num}: {context}")

                # Specifically check for dynamic import patterns
                for import_pattern in self.dynamic_import_patterns:
                    if len(self.dynamic_patterns[file_key]) >= self.max_dynamic_indicators_per_file:
                        break
                    matches = re.finditer(import_pattern, content, re.IGNORECASE)
                    for match in matches:
                        if len(self.dynamic_patterns[file_key]) >= self.max_dynamic_indicators_per_file:
                            break
                        line_num = content.count("\n", 0, match.start()) + 1
                        context = content[max(0, match.start() - 60) : match.end() + 60]
                        context = " ".join(context.strip().split())
                        if len(context) > self.max_context_chars:
                            context = context[: self.max_context_chars] + "..."
                        # Extract the module name that's being dynamically imported
                        if match.groups():
                            module_name = match.group(1)
                            self.dynamic_patterns[file_key].append(
                                f"Line {line_num}: Dynamic import of module '{module_name}': {context}"
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
            rel_file = self._relpath_posix(file_path)

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
                    file_references=[FileReference(rel_file, 1, -1)],
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
                    file_path=rel_file,
                    line_start=1,
                    line_end=-1,
                    unused_type=unused_type,
                    confidence=confidence,
                    evidence=evidence,
                    severity=Severity.HIGH if confidence > 0.8 else Severity.MEDIUM,
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
                rel_file = self._relpath_posix(file_path)
                with open(file_path, "r", encoding="utf-8-sig", errors="replace") as f:
                    content = f.read()

                tree = ast.parse(content, filename=str(file_path))

                # Get __all__ definitions if in library mode
                all_exports = []
                if self.library_mode:
                    all_exports = self._get_all_exports(tree)

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                        symbol_key = f"{rel_file}:{node.name}"
                        defined_symbols[symbol_key] = {
                            "name": node.name,
                            "file_path": rel_file,
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
            usage_refs = self.symbol_usage_examples.get(symbol_name, [])
            usage_total = int(self.symbol_usage_count.get(symbol_name, 0) or 0)
            usage_files = self.symbol_usage_files.get(symbol_name, set())

            # Filter out self-references (usage in the same file)
            has_external_usage = any(fp != symbol_info["file_path"] for fp in usage_files)

            # Check external index for usage if provided
            is_used_in_external_project = False
            if external_index and (not has_external_usage):
                try:
                    is_used_in_external_project = self._is_symbol_used_in_external_index(
                        symbol_name, symbol_info["file_path"], external_index
                    )
                except Exception:
                    # If there's an error accessing external index, continue with local analysis
                    pass

            if (not has_external_usage) and (not is_used_in_external_project):
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
                        "usage_total_count": usage_total,
                        "usage_files_count": len(usage_files),
                        "usage_files_sample": sorted(list(usage_files))[:10],
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
                    usage_references=usage_refs,  # limited examples only (memory-safe)
                    dynamic_usage_indicators=dynamic_indicators,
                    severity=Severity.MEDIUM if symbol_info["is_private"] else Severity.LOW,
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
            rel_file = self._relpath_posix(file_path)
            dynamic_indicators = self.dynamic_patterns.get(rel_file, [])

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
                        # Keep stdout clean: this is diagnostic noise for normal runs.
                        # Use DEBUG so users can enable it when needed.
                        if self._should_log_dynamic(file_path):
                            self.logger.debug("Found dynamic import in %s: %s", file_path, indicator)

                    # Create findings for files with dynamic imports
                    evidence = Evidence(
                        description=f"File contains dynamic import patterns that may reference symbols ({len(dynamic_import_indicators)} found)",
                        confidence=0.4,  # Low-medium confidence due to uncertainty
                        file_references=[FileReference(rel_file, 1, -1)],
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
                        file_path=rel_file,
                        line_start=1,
                        line_end=-1,
                        unused_type=UnusedCodeType.UNCERTAIN_DYNAMIC,
                        confidence=0.4,
                        evidence=evidence,
                        dynamic_usage_indicators=dynamic_import_indicators,
                        severity=Severity.LOW,
                    )

                    findings.append(finding)
                else:
                    # Create findings for files with other dynamic usage patterns
                    evidence = Evidence(
                        description="File contains dynamic usage patterns that may hide code usage",
                        confidence=0.3,  # Low confidence due to uncertainty
                        file_references=[FileReference(rel_file, 1, -1)],
                        code_snippets=[],
                        metadata={
                            "dynamic_patterns": dynamic_indicators,
                            "pattern_count": len(dynamic_indicators),
                        },
                    )

                    finding = UnusedCodeFinding(
                        symbol_name=f"dynamic_usage_{file_path.name}",
                        file_path=rel_file,
                        line_start=1,
                        line_end=-1,
                        unused_type=UnusedCodeType.UNCERTAIN_DYNAMIC,
                        confidence=0.3,
                        evidence=evidence,
                        dynamic_usage_indicators=dynamic_indicators,
                        severity=Severity.LOW,
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
        self.symbol_usage_count[symbol_name] += 1
        self.symbol_usage_files[symbol_name].add(file_path)

        # keep only a few examples to avoid memory blow-ups
        examples = self.symbol_usage_examples[symbol_name]
        if len(examples) >= self.max_usage_examples_per_symbol:
            return

        ctx = " ".join((context or "").strip().split())
        if len(ctx) > self.max_context_chars:
            ctx = ctx[: self.max_context_chars] + "..."

        examples.append(
            UsageReference(
                file_path=file_path,
                line_number=line_number,
                pattern=pattern,
                context=ctx,
                confidence=confidence,
            )
        )

    def _get_module_name(self, file_path: Path) -> str:
        """Get module name from file path."""
        try:
            p = file_path
            if not p.is_absolute():
                p = (self.project_path / p).resolve()
            relative_path = p.relative_to(self.project_path)
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
            # keep string keys in output for backward compatibility
            "findings_by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0, "info": 0},
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
            sev_key = finding.severity.value if hasattr(finding.severity, "value") else str(finding.severity)
            stats["findings_by_severity"][sev_key] = stats["findings_by_severity"].get(sev_key, 0) + 1

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
            with external_index._get_connection() as conn:
                cur = conn.execute(
                    """
                    SELECT COUNT(*)
                    FROM dependencies
                    WHERE dependency_kind = 'imports'
                      AND target_external IS NOT NULL
                      AND (target_external = ? OR target_external LIKE ?)
                    """,
                    (module_name, module_name + ".%"),
                )
                return int(cur.fetchone()[0] or 0) > 0
        except Exception:
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
            with external_index._get_connection() as conn:
                # exclude same-file definitions when possible
                exclude_file_id: Optional[int] = None
                try:
                    cur = conn.execute("SELECT file_id FROM files WHERE file_path = ? LIMIT 1", (file_path,))
                    row = cur.fetchone()
                    if row:
                        exclude_file_id = int(row[0])
                except Exception:
                    exclude_file_id = None

                # 1) symbol with same name exists in other files
                try:
                    if exclude_file_id is not None:
                        cur = conn.execute(
                            "SELECT COUNT(*) FROM symbols WHERE name = ? AND file_id != ?",
                            (symbol_name, exclude_file_id),
                        )
                    else:
                        cur = conn.execute("SELECT COUNT(*) FROM symbols WHERE name = ?", (symbol_name,))
                    if int(cur.fetchone()[0] or 0) > 0:
                        return True
                except Exception:
                    pass

                # 2) calls/imports that mention symbol (best-effort)
                try:
                    cur = conn.execute(
                        """
                        SELECT COUNT(*)
                        FROM dependencies
                        WHERE dependency_kind IN ('calls', 'imports')
                          AND target_external IS NOT NULL
                          AND (
                              target_external = ?
                              OR target_external LIKE ?
                          )
                        """,
                        (symbol_name, "%." + symbol_name),
                    )
                    if int(cur.fetchone()[0] or 0) > 0:
                        return True
                except Exception:
                    pass

                # 3) attribute access
                try:
                    cur = conn.execute(
                        "SELECT COUNT(*) FROM attribute_access WHERE attribute_name = ?",
                        (symbol_name,),
                    )
                    if int(cur.fetchone()[0] or 0) > 0:
                        return True
                except Exception:
                    pass

                return False
        except Exception:
            return False

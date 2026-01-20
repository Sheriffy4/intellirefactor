"""
Unified Audit Engine for IntelliRefactor

This module implements the unified audit workflow that combines:
- Index building and management
- Duplicate code detection (blocks and methods)
- Unused code detection (three levels)
- Analysis result aggregation and cross-references
- Comprehensive reporting with evidence and confidence scores
- Basic specification generation
"""

import time
from pathlib import Path
from typing import List, Optional, Any, Union
import inspect

from intellirefactor.analysis.index.builder import IndexBuilder, IndexBuildResult
from intellirefactor.analysis.index.store import IndexStore
from intellirefactor.analysis.index.query import IndexQuery

from intellirefactor.analysis.dedup.block_extractor import BlockExtractor
from intellirefactor.analysis.dedup.block_clone_detector import BlockCloneDetector

from intellirefactor.analysis.utils.file_discovery import discover_python_files

from intellirefactor.analysis.refactor.unused_code_detector import (
    UnusedCodeDetector,
    UnusedCodeAnalysisResult,
)

from intellirefactor.analysis.foundation.models import Evidence
from intellirefactor.analysis.workflows.audit_models import (
    AuditSeverity,
    AuditFindingType,
    AuditFinding,
    AuditStatistics,
    AuditResult,
)

# Decompose (smells) is optional; audit should still work without it.
try:
    from intellirefactor.analysis.decompose.architectural_smell_detector import (
        ArchitecturalSmellDetector,
        SmellSeverity,
    )
except Exception:  # pragma: no cover
    ArchitecturalSmellDetector = None  # type: ignore[assignment]
    SmellSeverity = None  # type: ignore[assignment]


# default excludes for workflow discovery (avoid scanning env/output folders)
_DEFAULT_EXCLUDE = [
    "**/test_*.py",
    "**/*_test.py",
    "**/tests/**",
    "**/.git/**",
    "**/.hg/**",
    "**/.svn/**",
    "**/.venv/**",
    "**/venv/**",
    "**/__pycache__/**",
    "**/.mypy_cache/**",
    "**/.pytest_cache/**",
    "**/.intellirefactor/**",
    "**/intellirefactor_out/**",
]


def _merge_excludes(exclude_patterns: Optional[List[str]]) -> List[str]:
    seen = set()
    out: List[str] = []
    for p in (exclude_patterns or []) + _DEFAULT_EXCLUDE:
        if not p:
            continue
        pp = str(p).replace("\\", "/")
        if pp not in seen:
            seen.add(pp)
            out.append(pp)
    return out


def _exclude_dirs_from_globs(exclude_globs: List[str]) -> List[str]:
    """
    Convert glob excludes like '**/.git/**' -> '.git' for directory exclusion.
    This is best-effort; discovery still applies exclude_globs to files too.
    """
    out: List[str] = []
    for g in exclude_globs or []:
        gg = str(g).replace("\\", "/")
        parts = [p for p in gg.split("/") if p and p not in ("**", "*")]
        if parts:
            # typically '.git' in '**/.git/**'
            out.append(parts[0])
    return out


def _relpath_posix(project_root: Path, p: Union[str, Path]) -> str:
    """Stable project-relative POSIX path for reports/DB matching."""
    pp = Path(p)
    try:
        if not pp.is_absolute():
            pp = (project_root / pp)
        return pp.resolve().relative_to(project_root.resolve()).as_posix()
    except Exception:
        return pp.as_posix()


def _build_evidence(
    *,
    description: str,
    confidence: float = 1.0,
    file_path: Optional[Union[str, Path]] = None,
    line_start: int = 1,
    line_end: int = 1,
    code_snippets: Optional[List[str]] = None,
    metadata: Optional[dict] = None,
) -> Evidence:
    """
    Evidence model has had multiple schemas in this repo.
    Build it defensively by passing only supported kwargs.
    """
    kwargs: dict = {
        "description": description,
        "confidence": confidence,
        "code_snippets": code_snippets or [],
        "metadata": metadata or {},
    }

    # If Evidence supports file_references, provide it.
    if file_path is not None:
        try:
            from intellirefactor.analysis.foundation.models import FileReference  # canonical

            kwargs["file_references"] = [FileReference(str(file_path), line_start, line_end)]
        except Exception:
            # if FileReference doesn't exist in current models, ignore
            pass

    try:
        sig = inspect.signature(Evidence)
        filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
        return Evidence(**filtered)  # type: ignore[arg-type]
    except Exception:
        # Last-resort: minimal ctor
        return Evidence(description=description, confidence=confidence)  # type: ignore[call-arg]


class AuditEngine:
    """
    Unified audit engine that orchestrates all analysis components.

    Combines index building, duplicate detection, unused code analysis,
    and generates comprehensive reports with cross-references.
    """

    def __init__(self, project_path: Union[str, Path]):
        """Initialize the audit engine."""
        try:
            self.project_path = Path(project_path).resolve()
        except Exception:
            self.project_path = Path(project_path)
        self.findings: List[AuditFinding] = []
        self.finding_counter = 0

        # Analysis components
        self.index_builder: Optional[IndexBuilder] = None
        self.index_store: Optional[IndexStore] = None
        self.index_query: Optional[IndexQuery] = None
        self.block_extractor = BlockExtractor()
        self.block_clone_detector = BlockCloneDetector()
        self.unused_code_detector = UnusedCodeDetector(project_path)
        self.smell_detector = ArchitecturalSmellDetector() if ArchitecturalSmellDetector else None

        # Results storage
        self.index_result: Optional[IndexBuildResult] = None
        self.unused_result: Optional[UnusedCodeAnalysisResult] = None
        self.clone_groups: List[Any] = []
        self.index_stats: Optional[dict] = None

    def _get_next_finding_id(self) -> str:
        """Generate next finding ID."""
        self.finding_counter += 1
        return f"AUDIT-{self.finding_counter:04d}"

    def _get_index_db_path(self) -> Path:
        """Get the index database path for the project."""
        return self.project_path / ".intellirefactor" / "index.db"

    def _ensure_index_directory(self) -> None:
        """Ensure the .intellirefactor directory exists."""
        intellirefactor_dir = self.project_path / ".intellirefactor"
        intellirefactor_dir.mkdir(parents=True, exist_ok=True)

    def run_full_audit(
        self,
        include_index: bool = True,
        include_duplicates: bool = True,
        include_unused: bool = True,
        generate_spec: bool = False,
        incremental_index: bool = True,
        min_confidence: float = 0.5,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> AuditResult:
        """
        Run a comprehensive audit of the project.

        Args:
            include_index: Whether to build/update the index
            include_duplicates: Whether to detect duplicate code
            include_unused: Whether to detect unused code
            generate_spec: Whether to generate specification documents
            incremental_index: Whether to use incremental index updates
            min_confidence: Minimum confidence threshold for findings
            include_patterns: File patterns to include
            exclude_patterns: File patterns to exclude

        Returns:
            AuditResult with all findings and analysis results
        """
        start_time = time.time()

        # Reset findings
        self.findings = []
        self.finding_counter = 0
        self.clone_groups = []
        self.unused_result = None
        self.index_result = None
        self.index_stats = None

        # Ensure directory structure
        self._ensure_index_directory()

        # Step 1: Index building and management
        if include_index:
            self._run_index_analysis(incremental_index)
        else:
            # Reuse existing index DB (if present) without rebuilding.
            try:
                db_path = self._get_index_db_path()
                if db_path.exists():
                    from intellirefactor.analysis.index.store import IndexStore
                    from intellirefactor.analysis.index.query import IndexQuery
                    self.index_store = IndexStore(db_path)
                    self.index_query = IndexQuery(self.index_store)
            except Exception:
                # audit should still work without index
                self.index_store = None
                self.index_query = None

        # capture index statistics if available (used by spec/dashboard correlation)
        if self.index_store is not None:
            try:
                self.index_stats = (
                    self.index_store.get_statistics()
                    if hasattr(self.index_store, "get_statistics")
                    else None
                )
            except Exception:
                self.index_stats = None

        # Step 2: Duplicate code detection
        if include_duplicates:
            self._run_duplicate_analysis(include_patterns, exclude_patterns)

        # Step 3: Unused code detection
        if include_unused:
            self._run_unused_code_analysis(min_confidence, include_patterns, exclude_patterns)

        # Step 3.5: Architectural smells (decompose)
        self._run_smell_analysis(min_confidence, include_patterns, exclude_patterns)

        # Step 4: Cross-reference analysis
        self._run_cross_reference_analysis()

        # Step 5: Generate statistics
        analysis_time = time.time() - start_time
        statistics = self._generate_statistics(analysis_time)

        # Step 6: Create audit result
        audit_result = AuditResult(
            project_path=str(self.project_path),
            findings=self.findings,
            statistics=statistics,
            index_result=self.index_result,
            unused_result=self.unused_result,
            clone_groups=self.clone_groups,
            analysis_metadata={
                "analysis_time_seconds": analysis_time,
                "include_index": include_index,
                "include_duplicates": include_duplicates,
                "include_unused": include_unused,
                "min_confidence": min_confidence,
                "incremental_index": incremental_index,
                "index_db_path": str(self._get_index_db_path()),
                "index_stats": self.index_stats,
            },
        )

        # Step 7: Generate specification if requested
        if generate_spec:
            self._generate_specifications(audit_result)

        return audit_result

    def _run_index_analysis(self, incremental: bool) -> None:
        """Run index building and analysis."""
        try:
            db_path = self._get_index_db_path()

            # Build or update index
            with IndexBuilder(db_path) as builder:
                if incremental and db_path.exists():
                    self.index_result = builder.build_index(self.project_path, incremental=True)
                else:
                    self.index_result = builder.rebuild_index(self.project_path)

            # Initialize index components
            self.index_store = IndexStore(db_path)
            self.index_query = IndexQuery(self.index_store)

            # populate stats and fix missing counters in IndexBuildResult (some impls don't fill in incremental mode)
            try:
                self.index_stats = (
                    self.index_store.get_statistics()
                    if hasattr(self.index_store, "get_statistics")
                    else None
                )
                if self.index_stats and self.index_result is not None:
                    deps = getattr(self.index_result, "dependencies_found", None)
                    if deps in (None, 0) and "dependencies_count" in self.index_stats:
                        try:
                            setattr(self.index_result, "dependencies_found", int(self.index_stats["dependencies_count"]))
                        except Exception:
                            pass
            except Exception:
                self.index_stats = None

            # Analyze index results for issues
            if not self.index_result.success:
                for error in self.index_result.errors:
                    finding = AuditFinding(
                        finding_id=self._get_next_finding_id(),
                        finding_type=AuditFindingType.INDEX_ISSUE,
                        severity=AuditSeverity.HIGH,
                        title="Index Build Error",
                        description=f"Failed to build project index: {error}",
                        file_path=str(self.project_path),
                        line_start=1,
                        line_end=1,
                        confidence=1.0,
                        evidence=_build_evidence(
                            description=f"Index build failed with error: {error}",
                            confidence=1.0,
                            file_path=self.project_path,
                            line_start=1,
                            line_end=1,
                            metadata={"error_type": "index_build_failure", "source": "audit.index_builder", "is_internal_error": True},
                        ),
                        recommendations=[
                            "Check file permissions and disk space",
                            "Verify Python syntax in all files",
                            "Review exclude patterns to skip problematic files",
                        ],
                        metadata={"source": "audit.index_builder", "is_internal_error": True},
                    )
                    self.findings.append(finding)

            # Check for performance issues
            if self.index_result.build_time_seconds > 60:  # More than 1 minute
                finding = AuditFinding(
                    finding_id=self._get_next_finding_id(),
                    finding_type=AuditFindingType.PERFORMANCE_ISSUE,
                    severity=AuditSeverity.MEDIUM,
                    title="Slow Index Build",
                    description=f"Index build took {self.index_result.build_time_seconds:.1f} seconds",
                    file_path=str(self.project_path),
                    line_start=1,
                    line_end=1,
                    confidence=1.0,
                    evidence=_build_evidence(
                        description=(
                            f"Index build performance: {self.index_result.build_time_seconds:.1f}s "
                            f"for {self.index_result.files_processed} files"
                        ),
                        confidence=1.0,
                        file_path=self.project_path,
                        line_start=1,
                        line_end=1,
                        metadata={
                            "build_time": self.index_result.build_time_seconds,
                            "files_processed": self.index_result.files_processed,
                            "source": "audit.index_builder",
                        },
                    ),
                    recommendations=[
                        "Consider using more aggressive exclude patterns",
                        "Check for very large files that could be excluded",
                        "Use incremental updates for subsequent analyses",
                    ],
                    metadata={"source": "audit.index_builder"},
                )
                self.findings.append(finding)

        except Exception as e:
            finding = AuditFinding(
                finding_id=self._get_next_finding_id(),
                finding_type=AuditFindingType.INDEX_ISSUE,
                severity=AuditSeverity.CRITICAL,
                title="Index Analysis Failed",
                description=f"Critical error during index analysis: {str(e)}",
                file_path=str(self.project_path),
                line_start=1,
                line_end=1,
                confidence=1.0,
                evidence=_build_evidence(
                    description=f"Index analysis failed with exception: {str(e)}",
                    confidence=1.0,
                    file_path=self.project_path,
                    line_start=1,
                    line_end=1,
                    metadata={"exception_type": type(e).__name__, "source": "audit.index_builder", "is_internal_error": True},
                ),
                recommendations=[
                    "Check project structure and file permissions",
                    "Verify all Python files have valid syntax",
                    "Review system resources and disk space",
                ],
                metadata={"source": "audit.index_builder", "is_internal_error": True},
            )
            self.findings.append(finding)

    def _run_duplicate_analysis(
        self,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]],
    ) -> None:
        """Run duplicate code detection analysis."""
        try:
            patterns = include_patterns or ["**/*.py"]
            exclude_globs = _merge_excludes(exclude_patterns)
            exclude_dirs = set(_exclude_dirs_from_globs(exclude_globs))
            python_files = discover_python_files(
                self.project_path,
                include=patterns,
                exclude_dirs=exclude_dirs,
                exclude_globs=exclude_globs,
            )
            if not python_files:
                return

            # Extract blocks from all files
            all_blocks = []
            for file_path in python_files:
                try:
                    # utf-8-sig handles BOM safely; replace prevents crashes on odd bytes
                    source_code = file_path.read_text(encoding="utf-8-sig", errors="replace")

                    rel_file = _relpath_posix(self.project_path, file_path)
                    blocks = self.block_extractor.extract_blocks(source_code, rel_file)
                    all_blocks.extend(blocks)
                    del source_code

                except Exception as e:
                    # Create finding for file analysis failure
                    finding = AuditFinding(
                        finding_id=self._get_next_finding_id(),
                        finding_type=AuditFindingType.QUALITY_ISSUE,
                        severity=AuditSeverity.LOW,
                        title="File Analysis Failed",
                        description=f"Could not analyze file for duplicates: {str(e)}",
                        file_path=_relpath_posix(self.project_path, file_path),
                        line_start=1,
                        line_end=1,
                        confidence=0.9,
                        evidence=_build_evidence(
                            description=f"File analysis failed: {str(e)}",
                            confidence=0.9,
                            file_path=file_path,
                            line_start=1,
                            line_end=1,
                            metadata={"exception_type": type(e).__name__, "source": "audit.duplicate_detector", "is_internal_error": True},
                        ),
                        recommendations=[
                            "Check file encoding and syntax",
                            "Verify file is valid Python code",
                        ],
                        metadata={"source": "audit.duplicate_detector", "is_internal_error": True},
                    )
                    self.findings.append(finding)
                    continue

            # Detect clones
            self.clone_groups = self.block_clone_detector.detect_clones(all_blocks)

            # Convert clone groups to audit findings
            for group in self.clone_groups:
                if len(group.instances) < 2:
                    continue

                # Determine severity based on clone type and size
                if group.clone_type.value == "exact":
                    severity = AuditSeverity.HIGH
                elif group.clone_type.value == "structural":
                    severity = AuditSeverity.MEDIUM
                else:
                    severity = AuditSeverity.LOW

                # Create finding for the clone group
                instance_descriptions = []
                for instance in group.instances:
                    instance_descriptions.append(
                        f"{instance.file_path}:{instance.line_start}-{instance.line_end}"
                    )

                finding = AuditFinding(
                    finding_id=self._get_next_finding_id(),
                    finding_type=AuditFindingType.DUPLICATE_BLOCK,
                    severity=severity,
                    title=f"{group.clone_type.value.title()} Code Clone",
                    description=f"Found {len(group.instances)} duplicate code blocks",
                    file_path=_relpath_posix(self.project_path, group.instances[0].file_path),
                    line_start=group.instances[0].line_start,
                    line_end=group.instances[0].line_end,
                    confidence=group.similarity_score,
                    evidence=_build_evidence(
                        description=(
                            f"Duplicate {group.clone_type.value} blocks found with "
                            f"{group.similarity_score:.3f} similarity"
                        ),
                        confidence=group.similarity_score,
                        file_path=group.instances[0].file_path,
                        line_start=group.instances[0].line_start,
                        line_end=group.instances[0].line_end,
                        metadata={
                            "clone_type": group.clone_type.value,
                            "instance_count": len(group.instances),
                            "similarity_score": group.similarity_score,
                            "all_instances": instance_descriptions,
                        },
                    ),
                    recommendations=self._get_clone_recommendations(group),
                    metadata={
                        "source": "audit.duplicate_detector",
                        "group_id": group.group_id,
                        "clone_type": group.clone_type.value,
                        "extraction_strategy": (
                            group.extraction_strategy.value if group.extraction_strategy else None
                        ),
                        "extraction_confidence": group.extraction_confidence,
                    },
                )
                self.findings.append(finding)

        except Exception as e:
            finding = AuditFinding(
                finding_id=self._get_next_finding_id(),
                finding_type=AuditFindingType.QUALITY_ISSUE,
                severity=AuditSeverity.MEDIUM,
                title="Duplicate Analysis Failed",
                description=f"Error during duplicate code analysis: {str(e)}",
                file_path=str(self.project_path),
                line_start=1,
                line_end=1,
                confidence=1.0,
                evidence=_build_evidence(
                    description=f"Duplicate analysis failed: {str(e)}",
                    confidence=1.0,
                    file_path=self.project_path,
                    line_start=1,
                    line_end=1,
                    metadata={"exception_type": type(e).__name__, "source": "audit.duplicate_detector", "is_internal_error": True},
                ),
                recommendations=[
                    "Check project structure and file accessibility",
                    "Verify Python syntax in all files",
                ],
                metadata={"source": "audit.duplicate_detector", "is_internal_error": True},
            )
            self.findings.append(finding)

    def _run_unused_code_analysis(
        self,
        min_confidence: float,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]],
    ) -> None:
        """Run unused code detection analysis."""
        try:
            exc = _merge_excludes(exclude_patterns)
            # Run unused code detection
            self.unused_result = self.unused_code_detector.detect_unused_code(
                entry_points=None,  # Auto-detect
                include_patterns=include_patterns,
                exclude_patterns=exc,
                min_confidence=min_confidence,
            )

            # Convert unused code findings to audit findings
            for unused_finding in self.unused_result.findings:
                # Map unused code severity
                if unused_finding.confidence >= 0.8:
                    severity = AuditSeverity.HIGH
                elif unused_finding.confidence >= 0.6:
                    severity = AuditSeverity.MEDIUM
                else:
                    severity = AuditSeverity.LOW

                finding = AuditFinding(
                    finding_id=self._get_next_finding_id(),
                    finding_type=AuditFindingType.UNUSED_CODE,
                    severity=severity,
                    title=f"Unused {unused_finding.unused_type.value.replace('_', ' ').title()}",
                    description=f"Symbol '{unused_finding.symbol_name}' appears to be unused",
                    file_path=_relpath_posix(self.project_path, unused_finding.file_path),
                    line_start=unused_finding.line_start,
                    line_end=unused_finding.line_end,
                    confidence=unused_finding.confidence,
                    evidence=unused_finding.evidence,
                    recommendations=self._get_unused_code_recommendations(unused_finding),
                    metadata={
                        "source": "audit.unused_detector",
                        "unused_type": unused_finding.unused_type.value,
                        "symbol_name": unused_finding.symbol_name,
                        "usage_references": len(unused_finding.usage_references),
                        "dynamic_indicators": len(unused_finding.dynamic_usage_indicators),
                    },
                )
                self.findings.append(finding)

        except Exception as e:
            finding = AuditFinding(
                finding_id=self._get_next_finding_id(),
                finding_type=AuditFindingType.QUALITY_ISSUE,
                severity=AuditSeverity.MEDIUM,
                title="Unused Code Analysis Failed",
                description=f"Error during unused code analysis: {str(e)}",
                file_path=str(self.project_path),
                line_start=1,
                line_end=1,
                confidence=1.0,
                evidence=_build_evidence(
                    description=f"Unused code analysis failed: {str(e)}",
                    confidence=1.0,
                    file_path=self.project_path,
                    line_start=1,
                    line_end=1,
                    metadata={"exception_type": type(e).__name__, "source": "audit.unused_detector", "is_internal_error": True},
                ),
                recommendations=[
                    "Check project structure and entry points",
                    "Verify Python import paths are correct",
                ],
                metadata={"source": "audit.unused_detector", "is_internal_error": True},
            )
            self.findings.append(finding)

    def _run_smell_analysis(
        self,
        min_confidence: float,
        include_patterns: Optional[List[str]],
        exclude_patterns: Optional[List[str]],
    ) -> None:
        """
        Run architectural smells detection (decompose) and convert to audit findings.
        We map smells to AuditFindingType.QUALITY_ISSUE to avoid expanding audit_models enums.
        """
        if not self.smell_detector:
            return

        try:
            patterns = include_patterns or ["**/*.py"]
            exclude_globs = _merge_excludes(exclude_patterns)
            exclude_dirs = set(_exclude_dirs_from_globs(exclude_globs))
            python_files = discover_python_files(
                self.project_path,
                include=patterns,
                exclude_dirs=exclude_dirs,
                exclude_globs=exclude_globs,
            )
            if not python_files:
                return

            # severity mapping
            sev_map = {
                "critical": AuditSeverity.CRITICAL,
                "high": AuditSeverity.HIGH,
                "medium": AuditSeverity.MEDIUM,
                "low": AuditSeverity.LOW,
            }

            for file_path in python_files:
                try:
                    src = file_path.read_text(encoding="utf-8", errors="replace")
                    rel_file = _relpath_posix(self.project_path, file_path)
                    smells = self.smell_detector.detect_smells(src, rel_file)
                except Exception:
                    continue

                for smell in smells or []:
                    try:
                        if float(getattr(smell, "confidence", 0.0) or 0.0) < float(min_confidence):
                            continue
                    except Exception:
                        pass

                    sev_val = getattr(getattr(smell, "severity", None), "value", None) or "low"
                    severity = sev_map.get(str(sev_val), AuditSeverity.LOW)
                    confidence = float(getattr(smell, "confidence", 0.5) or 0.5)

                    evidence = getattr(smell, "evidence", None)
                    if evidence is None:
                        evidence = _build_evidence(
                            description="Architectural smell detected",
                            confidence=confidence,
                            file_path=_relpath_posix(self.project_path, getattr(smell, "file_path", rel_file)),
                            line_start=int(getattr(smell, "line_start", 1) or 1),
                            line_end=int(getattr(smell, "line_end", 1) or 1),
                            metadata={},
                        )

                    finding = AuditFinding(
                        finding_id=self._get_next_finding_id(),
                        finding_type=AuditFindingType.QUALITY_ISSUE,
                        severity=severity,
                        title=f"Architectural smell: {getattr(getattr(smell, 'smell_type', None), 'value', 'unknown')}",
                        description=getattr(smell, "description", "") or "Architectural smell detected",
                        file_path=_relpath_posix(self.project_path, getattr(smell, "file_path", file_path)),
                        line_start=int(getattr(smell, "line_start", 1) or 1),
                        line_end=int(getattr(smell, "line_end", 1) or 1),
                        confidence=confidence,
                        evidence=evidence,
                        recommendations=list(getattr(smell, "recommendations", []) or []),
                        metadata={
                            "source": "decompose.smells",
                            "smell_type": getattr(getattr(smell, "smell_type", None), "value", None),
                            "symbol_name": getattr(smell, "symbol_name", None),
                            "metrics": getattr(smell, "metrics", None),
                        },
                    )
                    self.findings.append(finding)

        except Exception as e:
            # Do not fail audit for smells; just add a low-severity finding.
            self.findings.append(
                AuditFinding(
                    finding_id=self._get_next_finding_id(),
                    finding_type=AuditFindingType.QUALITY_ISSUE,
                    severity=AuditSeverity.LOW,
                    title="Smells analysis failed",
                    description=f"Could not run smells detection: {e}",
                    file_path=str(self.project_path),
                    line_start=1,
                    line_end=1,
                    confidence=0.5,
                    evidence=_build_evidence(
                        description=f"Smells analysis failed: {e}",
                        confidence=0.5,
                        file_path=self.project_path,
                        line_start=1,
                        line_end=1,
                        metadata={"exception_type": type(e).__name__, "source": "audit.smell_detector", "is_internal_error": True},
                    ),
                    recommendations=["Install/enable decompose dependencies and retry"],
                    metadata={"source": "audit.smell_detector", "is_internal_error": True},
                )
            )

    def _run_cross_reference_analysis(self) -> None:
        """Analyze cross-references between findings."""
        # Group findings by file for cross-referencing
        findings_by_file = {}
        for finding in self.findings:
            file_path = finding.file_path
            if file_path not in findings_by_file:
                findings_by_file[file_path] = []
            findings_by_file[file_path].append(finding)

        # Find related findings in the same file
        for file_path, file_findings in findings_by_file.items():
            if len(file_findings) > 1:
                for i, finding1 in enumerate(file_findings):
                    for finding2 in file_findings[i + 1 :]:
                        # Check if findings are related (overlapping lines or similar types)
                        if self._are_findings_related(finding1, finding2):
                            finding1.related_findings.append(finding2.finding_id)
                            finding2.related_findings.append(finding1.finding_id)

    def _are_findings_related(self, finding1: AuditFinding, finding2: AuditFinding) -> bool:
        """Check if two findings are related."""
        # Same file and overlapping lines
        if finding1.file_path == finding2.file_path:
            # Check for line overlap
            if (
                finding1.line_start <= finding2.line_end
                and finding2.line_start <= finding1.line_end
            ):
                return True

            # Check for proximity (within 10 lines)
            if abs(finding1.line_start - finding2.line_start) <= 10:
                return True

        # Similar finding types
        if finding1.finding_type == finding2.finding_type:
            return True

        return False

    def _get_clone_recommendations(self, clone_group) -> List[str]:
        """Get recommendations for a clone group."""
        recommendations = []

        if hasattr(clone_group, "extraction_strategy") and clone_group.extraction_strategy:
            strategy = clone_group.extraction_strategy.value
            if strategy == "extract_method":
                recommendations.append("Extract duplicate code into a shared method")
                recommendations.append("Consider using parameters for varying parts")
            elif strategy == "extract_class":
                recommendations.append("Extract duplicate code into a shared class")
                recommendations.append("Use inheritance or composition to share functionality")
            elif strategy == "decorator_pattern":
                recommendations.append("Consider using decorator pattern to eliminate duplication")
            else:
                recommendations.append("Refactor to eliminate code duplication")
        else:
            recommendations.append("Review duplicate code for refactoring opportunities")

        recommendations.append("Ensure consistent behavior across all instances")
        recommendations.append("Add unit tests before refactoring")

        return recommendations

    def _get_unused_code_recommendations(self, unused_finding) -> List[str]:
        """Get recommendations for unused code finding."""
        recommendations = []

        unused_type = unused_finding.unused_type.value

        if unused_type == "module_unreachable":
            recommendations.extend(
                [
                    "Remove unused module if no longer needed",
                    "Check if module should be added to entry points",
                    "Consider if module is used dynamically",
                ]
            )
        elif unused_type == "symbol_unused":
            recommendations.extend(
                [
                    "Remove unused symbol if no longer needed",
                    "Check if symbol is part of public API",
                    "Consider if symbol is used via reflection",
                ]
            )
        elif unused_type == "private_method_unused":
            recommendations.extend(
                ["Remove unused private method", "Check if method is called indirectly"]
            )
        elif unused_type == "tests_only":
            recommendations.extend(
                [
                    "Consider if test-only usage indicates missing functionality",
                    "Move to test utilities if appropriate",
                ]
            )
        elif unused_type == "uncertain_dynamic":
            recommendations.extend(
                [
                    "Review dynamic usage patterns",
                    "Add explicit usage if needed",
                    "Consider refactoring to reduce dynamic access",
                ]
            )
        else:
            recommendations.append("Review unused code for removal or refactoring")

        recommendations.append("Verify removal doesn't break functionality")

        return recommendations

    def _generate_statistics(self, analysis_time: float) -> AuditStatistics:
        """Generate statistics for audit results."""
        findings_by_severity = {}
        findings_by_type = {}
        confidence_distribution = {"high": 0, "medium": 0, "low": 0}

        for finding in self.findings:
            # Count by severity
            severity = finding.severity.value
            findings_by_severity[severity] = findings_by_severity.get(severity, 0) + 1

            # Count by type
            finding_type = finding.finding_type.value
            findings_by_type[finding_type] = findings_by_type.get(finding_type, 0) + 1

            # Count by confidence
            if finding.confidence >= 0.8:
                confidence_distribution["high"] += 1
            elif finding.confidence >= 0.5:
                confidence_distribution["medium"] += 1
            else:
                confidence_distribution["low"] += 1

        # Count analyzed files
        analyzed_files = len(set(f.file_path for f in self.findings))
        if self.index_result:
            analyzed_files = max(analyzed_files, self.index_result.files_processed)

        return AuditStatistics(
            total_findings=len(self.findings),
            findings_by_severity=findings_by_severity,
            findings_by_type=findings_by_type,
            files_analyzed=analyzed_files,
            analysis_time_seconds=analysis_time,
            confidence_distribution=confidence_distribution,
        )

    def _generate_specifications(self, audit_result: AuditResult) -> None:
        """Generate specification documents from audit results."""
        try:
            from .spec_generator import SpecGenerator

            # Generate basic Requirements.md with findings
            spec_generator = SpecGenerator()
            spec_content = spec_generator.generate_requirements_from_audit(audit_result)

            # Write to project directory
            spec_path = self.project_path / "AUDIT_REQUIREMENTS.md"
            with open(spec_path, "w", encoding="utf-8") as f:
                f.write(spec_content)

            # Add metadata about spec generation
            audit_result.analysis_metadata["spec_generated"] = True
            audit_result.analysis_metadata["spec_path"] = str(spec_path)

        except Exception as e:
            # Add finding about spec generation failure
            finding = AuditFinding(
                finding_id=self._get_next_finding_id(),
                finding_type=AuditFindingType.QUALITY_ISSUE,
                severity=AuditSeverity.LOW,
                title="Specification Generation Failed",
                description=f"Could not generate specification document: {str(e)}",
                file_path=str(self.project_path),
                line_start=1,
                line_end=1,
                confidence=1.0,
                evidence=_build_evidence(
                    description=f"Spec generation failed: {str(e)}",
                    confidence=1.0,
                    file_path=self.project_path,
                    line_start=1,
                    line_end=1,
                    metadata={"exception_type": type(e).__name__, "source": "audit.spec_generator", "is_internal_error": True},
                ),
                recommendations=[
                    "Check file permissions in project directory",
                    "Verify audit results are complete",
                ],
                metadata={"source": "audit.spec_generator", "is_internal_error": True},
            )
            self.findings.append(finding)

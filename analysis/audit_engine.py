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

import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json

from .index_builder import IndexBuilder, IndexBuildResult
from .index_store import IndexStore
from .index_query import IndexQuery
from .block_extractor import BlockExtractor
from .block_clone_detector import BlockCloneDetector
from .unused_code_detector import UnusedCodeDetector, UnusedCodeAnalysisResult
from .models import Evidence
from .audit_models import (
    AuditSeverity, AuditFindingType, AuditFinding, 
    AuditStatistics, AuditResult
)


class AuditEngine:
    """
    Unified audit engine that orchestrates all analysis components.
    
    Combines index building, duplicate detection, unused code analysis,
    and generates comprehensive reports with cross-references.
    """
    
    def __init__(self, project_path: Union[str, Path]):
        """Initialize the audit engine."""
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
        
        # Results storage
        self.index_result: Optional[IndexBuildResult] = None
        self.unused_result: Optional[UnusedCodeAnalysisResult] = None
        self.clone_groups: List[Any] = []
    
    def _get_next_finding_id(self) -> str:
        """Generate next finding ID."""
        self.finding_counter += 1
        return f"AUDIT-{self.finding_counter:04d}"
    
    def _get_index_db_path(self) -> Path:
        """Get the index database path for the project."""
        return self.project_path / '.intellirefactor' / 'index.db'
    
    def _ensure_index_directory(self) -> None:
        """Ensure the .intellirefactor directory exists."""
        intellirefactor_dir = self.project_path / '.intellirefactor'
        intellirefactor_dir.mkdir(exist_ok=True)
    
    def run_full_audit(self, 
                      include_index: bool = True,
                      include_duplicates: bool = True,
                      include_unused: bool = True,
                      generate_spec: bool = False,
                      incremental_index: bool = True,
                      min_confidence: float = 0.5,
                      include_patterns: Optional[List[str]] = None,
                      exclude_patterns: Optional[List[str]] = None) -> AuditResult:
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
        
        # Ensure directory structure
        self._ensure_index_directory()
        
        # Step 1: Index building and management
        if include_index:
            self._run_index_analysis(incremental_index)
        
        # Step 2: Duplicate code detection
        if include_duplicates:
            self._run_duplicate_analysis(include_patterns, exclude_patterns)
        
        # Step 3: Unused code detection
        if include_unused:
            self._run_unused_code_analysis(min_confidence, include_patterns, exclude_patterns)
        
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
                'analysis_time_seconds': analysis_time,
                'include_index': include_index,
                'include_duplicates': include_duplicates,
                'include_unused': include_unused,
                'min_confidence': min_confidence,
                'incremental_index': incremental_index
            }
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
                        evidence=Evidence(
                            description=f"Index build failed with error: {error}",
                            confidence=1.0,
                            code_snippets=[],
                            metadata={'error_type': 'index_build_failure'}
                        ),
                        recommendations=[
                            "Check file permissions and disk space",
                            "Verify Python syntax in all files",
                            "Review exclude patterns to skip problematic files"
                        ]
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
                    evidence=Evidence(
                        description=f"Index build performance: {self.index_result.build_time_seconds:.1f}s for {self.index_result.files_processed} files",
                        confidence=1.0,
                        code_snippets=[],
                        metadata={
                            'build_time': self.index_result.build_time_seconds,
                            'files_processed': self.index_result.files_processed
                        }
                    ),
                    recommendations=[
                        "Consider using more aggressive exclude patterns",
                        "Check for very large files that could be excluded",
                        "Use incremental updates for subsequent analyses"
                    ]
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
                evidence=Evidence(
                    description=f"Index analysis failed with exception: {str(e)}",
                    confidence=1.0,
                    code_snippets=[],
                    metadata={'exception_type': type(e).__name__}
                ),
                recommendations=[
                    "Check project structure and file permissions",
                    "Verify all Python files have valid syntax",
                    "Review system resources and disk space"
                ]
            )
            self.findings.append(finding)
    
    def _run_duplicate_analysis(self, include_patterns: Optional[List[str]], 
                               exclude_patterns: Optional[List[str]]) -> None:
        """Run duplicate code detection analysis."""
        try:
            import glob
            
            # Find Python files
            patterns = include_patterns or ['**/*.py']
            exclude = exclude_patterns or ['**/test_*.py', '**/*_test.py', '**/tests/**']
            
            python_files = []
            for pattern in patterns:
                python_files.extend(glob.glob(str(self.project_path / pattern), recursive=True))
            
            # Filter out excluded patterns
            for exclude_pattern in exclude:
                excluded_files = set(glob.glob(str(self.project_path / exclude_pattern), recursive=True))
                python_files = [f for f in python_files if f not in excluded_files]
            
            python_files = [Path(f) for f in python_files if Path(f).is_file()]
            
            if not python_files:
                return
            
            # Extract blocks from all files
            all_blocks = []
            for file_path in python_files:
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        source_code = f.read()
                    
                    blocks = self.block_extractor.extract_blocks(source_code, str(file_path))
                    all_blocks.extend(blocks)
                    
                except Exception as e:
                    # Create finding for file analysis failure
                    finding = AuditFinding(
                        finding_id=self._get_next_finding_id(),
                        finding_type=AuditFindingType.QUALITY_ISSUE,
                        severity=AuditSeverity.LOW,
                        title="File Analysis Failed",
                        description=f"Could not analyze file for duplicates: {str(e)}",
                        file_path=str(file_path),
                        line_start=1,
                        line_end=1,
                        confidence=0.9,
                        evidence=Evidence(
                            description=f"File analysis failed: {str(e)}",
                            confidence=0.9,
                            code_snippets=[],
                            metadata={'exception_type': type(e).__name__}
                        ),
                        recommendations=[
                            "Check file encoding and syntax",
                            "Verify file is valid Python code"
                        ]
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
                if group.clone_type.value == 'exact':
                    severity = AuditSeverity.HIGH
                elif group.clone_type.value == 'structural':
                    severity = AuditSeverity.MEDIUM
                else:
                    severity = AuditSeverity.LOW
                
                # Create finding for the clone group
                instance_descriptions = []
                for instance in group.instances:
                    instance_descriptions.append(f"{instance.file_path}:{instance.line_start}-{instance.line_end}")
                
                finding = AuditFinding(
                    finding_id=self._get_next_finding_id(),
                    finding_type=AuditFindingType.DUPLICATE_BLOCK,
                    severity=severity,
                    title=f"{group.clone_type.value.title()} Code Clone",
                    description=f"Found {len(group.instances)} duplicate code blocks",
                    file_path=group.instances[0].file_path,
                    line_start=group.instances[0].line_start,
                    line_end=group.instances[0].line_end,
                    confidence=group.similarity_score,
                    evidence=Evidence(
                        description=f"Duplicate {group.clone_type.value} blocks found with {group.similarity_score:.3f} similarity",
                        confidence=group.similarity_score,
                        code_snippets=[],
                        metadata={
                            'clone_type': group.clone_type.value,
                            'instance_count': len(group.instances),
                            'similarity_score': group.similarity_score,
                            'all_instances': instance_descriptions
                        }
                    ),
                    recommendations=self._get_clone_recommendations(group),
                    metadata={
                        'group_id': group.group_id,
                        'extraction_strategy': group.extraction_strategy.value if group.extraction_strategy else None,
                        'extraction_confidence': group.extraction_confidence
                    }
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
                evidence=Evidence(
                    description=f"Duplicate analysis failed: {str(e)}",
                    confidence=1.0,
                    code_snippets=[],
                    metadata={'exception_type': type(e).__name__}
                ),
                recommendations=[
                    "Check project structure and file accessibility",
                    "Verify Python syntax in all files"
                ]
            )
            self.findings.append(finding)
    
    def _run_unused_code_analysis(self, min_confidence: float,
                                 include_patterns: Optional[List[str]],
                                 exclude_patterns: Optional[List[str]]) -> None:
        """Run unused code detection analysis."""
        try:
            # Run unused code detection
            self.unused_result = self.unused_code_detector.detect_unused_code(
                entry_points=None,  # Auto-detect
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                min_confidence=min_confidence
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
                    file_path=unused_finding.file_path,
                    line_start=unused_finding.line_start,
                    line_end=unused_finding.line_end,
                    confidence=unused_finding.confidence,
                    evidence=unused_finding.evidence,
                    recommendations=self._get_unused_code_recommendations(unused_finding),
                    metadata={
                        'unused_type': unused_finding.unused_type.value,
                        'usage_references': len(unused_finding.usage_references),
                        'dynamic_indicators': len(unused_finding.dynamic_usage_indicators)
                    }
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
                evidence=Evidence(
                    description=f"Unused code analysis failed: {str(e)}",
                    confidence=1.0,
                    code_snippets=[],
                    metadata={'exception_type': type(e).__name__}
                ),
                recommendations=[
                    "Check project structure and entry points",
                    "Verify Python import paths are correct"
                ]
            )
            self.findings.append(finding)
    
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
                    for finding2 in file_findings[i+1:]:
                        # Check if findings are related (overlapping lines or similar types)
                        if self._are_findings_related(finding1, finding2):
                            finding1.related_findings.append(finding2.finding_id)
                            finding2.related_findings.append(finding1.finding_id)
    
    def _are_findings_related(self, finding1: AuditFinding, finding2: AuditFinding) -> bool:
        """Check if two findings are related."""
        # Same file and overlapping lines
        if finding1.file_path == finding2.file_path:
            # Check for line overlap
            if (finding1.line_start <= finding2.line_end and 
                finding2.line_start <= finding1.line_end):
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
        
        if hasattr(clone_group, 'extraction_strategy') and clone_group.extraction_strategy:
            strategy = clone_group.extraction_strategy.value
            if strategy == 'extract_method':
                recommendations.append("Extract duplicate code into a shared method")
                recommendations.append("Consider using parameters for varying parts")
            elif strategy == 'extract_class':
                recommendations.append("Extract duplicate code into a shared class")
                recommendations.append("Use inheritance or composition to share functionality")
            elif strategy == 'decorator_pattern':
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
        
        if unused_type == 'module_unreachable':
            recommendations.extend([
                "Remove unused module if no longer needed",
                "Check if module should be added to entry points",
                "Consider if module is used dynamically"
            ])
        elif unused_type == 'symbol_unused':
            recommendations.extend([
                "Remove unused symbol if no longer needed",
                "Check if symbol is part of public API",
                "Consider if symbol is used via reflection"
            ])
        elif unused_type == 'private_method_unused':
            recommendations.extend([
                "Remove unused private method",
                "Check if method is called indirectly"
            ])
        elif unused_type == 'tests_only':
            recommendations.extend([
                "Consider if test-only usage indicates missing functionality",
                "Move to test utilities if appropriate"
            ])
        elif unused_type == 'uncertain_dynamic':
            recommendations.extend([
                "Review dynamic usage patterns",
                "Add explicit usage if needed",
                "Consider refactoring to reduce dynamic access"
            ])
        else:
            recommendations.append("Review unused code for removal or refactoring")
        
        recommendations.append("Verify removal doesn't break functionality")
        
        return recommendations
    
    def _generate_statistics(self, analysis_time: float) -> AuditStatistics:
        """Generate statistics for audit results."""
        findings_by_severity = {}
        findings_by_type = {}
        confidence_distribution = {'high': 0, 'medium': 0, 'low': 0}
        
        for finding in self.findings:
            # Count by severity
            severity = finding.severity.value
            findings_by_severity[severity] = findings_by_severity.get(severity, 0) + 1
            
            # Count by type
            finding_type = finding.finding_type.value
            findings_by_type[finding_type] = findings_by_type.get(finding_type, 0) + 1
            
            # Count by confidence
            if finding.confidence >= 0.8:
                confidence_distribution['high'] += 1
            elif finding.confidence >= 0.5:
                confidence_distribution['medium'] += 1
            else:
                confidence_distribution['low'] += 1
        
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
            confidence_distribution=confidence_distribution
        )
    
    def _generate_specifications(self, audit_result: AuditResult) -> None:
        """Generate specification documents from audit results."""
        try:
            from .spec_generator import SpecGenerator
            
            # Generate basic Requirements.md with findings
            spec_generator = SpecGenerator()
            spec_content = spec_generator.generate_requirements_from_audit(audit_result)
            
            # Write to project directory
            spec_path = self.project_path / 'AUDIT_REQUIREMENTS.md'
            with open(spec_path, 'w', encoding='utf-8') as f:
                f.write(spec_content)
            
            # Add metadata about spec generation
            audit_result.analysis_metadata['spec_generated'] = True
            audit_result.analysis_metadata['spec_path'] = str(spec_path)
            
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
                evidence=Evidence(
                    description=f"Spec generation failed: {str(e)}",
                    confidence=1.0,
                    code_snippets=[],
                    metadata={'exception_type': type(e).__name__}
                ),
                recommendations=[
                    "Check file permissions in project directory",
                    "Verify audit results are complete"
                ]
            )
            self.findings.append(finding)
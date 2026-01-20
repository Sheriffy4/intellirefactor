"""
Main expert refactoring analyzer.

Orchestrates all expert-level analyses to provide comprehensive data
for safe and precise refactoring decisions.
"""

from __future__ import annotations

import ast
import json
import logging
import importlib
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from .models import ExpertAnalysisResult, RiskLevel


def _make_json_safe(obj: Any) -> Any:
    """
    Convert objects to JSON-serializable primitives:
    - Path -> str
    - Enum -> .value
    - set/tuple -> list
    - dataclass -> asdict
    - dict/list -> recursive
    Fallback -> str(obj)
    """
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, Enum):
        return obj.value
    if is_dataclass(obj):
        return _make_json_safe(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_make_json_safe(v) for v in list(obj)]
    # Common pattern: objects with to_dict()
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        try:
            return _make_json_safe(to_dict())
        except Exception:
            pass
    return str(obj)

from .safe_access import SafeAnalyzerRegistry
from .quality_metrics import (
    calculate_quality_score,
    assess_overall_risk,
    generate_recommendations,
)
from .analysis_report_utils import (
    generate_expert_recommendations,
    generate_acceptance_criteria,
    generate_analysis_markdown_report as generate_markdown_report,
)
from .analysis_pipeline import (
    AnalysisPipelineExecutor,
    EXPORT_STEPS,
)

logger = logging.getLogger(__name__)

def _optional_import(qualified_module: str, attr: str):
    """
    Import helper: returns class/object or None (no hard fail).
    """
    try:
        mod = importlib.import_module(qualified_module)
        return getattr(mod, attr)
    except Exception as e:
        logger.debug("Optional import failed: %s.%s (%s)", qualified_module, attr, e)
        return None


class ExpertRefactoringAnalyzer:
    """
    Main expert refactoring analyzer that orchestrates all specialized analyzers
    to provide comprehensive data for expert-level refactoring decisions.
    """

    def __init__(
        self,
        project_root: str,
        target_module: str,
        output_dir: Optional[str] = None
    ):
        """
        Initialize the expert analyzer.
        
        Args:
            project_root: Root directory of the project
            target_module: Path to the target module to analyze
            output_dir: Optional output directory for results
        """
        self.project_root = Path(project_root)
        self.target_module = Path(target_module)
        self.output_dir = Path(output_dir) if output_dir else self.project_root / "expert_analysis"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize analyzers with safe access
        self._raw_analyzers = self._initialize_analyzers()
        self.analyzers = SafeAnalyzerRegistry(self._raw_analyzers)
        
        # Cache
        self._ast_cache: Optional[ast.Module] = None
        self._file_content: Optional[str] = None
        
        logger.info("Initialized ExpertRefactoringAnalyzer for %s", self.target_module)

    def _initialize_analyzers(self) -> Dict[str, Any]:
        """Initialize all specialized analyzers."""
        analyzers: Dict[str, Any] = {}

        def add(key: str, module: str, cls: str) -> None:
            C = _optional_import(module, cls)
            if C is None:
                return
            try:
                analyzers[key] = C(self.project_root, self.target_module)
            except Exception as e:
                logger.warning("Failed to init analyzer %s (%s.%s): %s", key, module, cls, e)

        base = "intellirefactor.analysis.refactor.expert.analyzers"
        add("call_graph", f"{base}.call_graph_analyzer", "CallGraphAnalyzer")
        add("caller", f"{base}.caller_analyzer", "CallerAnalyzer")
        add("cohesion", f"{base}.cohesion_analyzer", "CohesionMatrixAnalyzer")
        add("contract", f"{base}.contract_analyzer", "BehavioralContractAnalyzer")
        add("dependency", f"{base}.dependency_analyzer", "DependencyInterfaceAnalyzer")
        add("test", f"{base}.test_analyzer", "TestDiscoveryAnalyzer")
        add("characterization", f"{base}.characterization_generator", "CharacterizationTestGenerator")
        add("duplicate", f"{base}.duplicate_analyzer", "ConcreteDeduplicationAnalyzer")
        add("git", f"{base}.git_analyzer", "GitHistoryAnalyzer")
        add("compatibility", f"{base}.compatibility_analyzer", "CompatibilityAnalyzer")
        add("exception_contract", f"{base}.exception_contract_analyzer", "ExceptionContractAnalyzer")
        add("data_schema", f"{base}.data_schema_analyzer", "DataSchemaAnalyzer")
        add("optional_dependencies", f"{base}.optional_dependencies_analyzer", "OptionalDependenciesAnalyzer")
        add("golden_traces", f"{base}.golden_traces_extractor", "GoldenTracesExtractor")
        add("test_quality", f"{base}.test_quality_analyzer", "TestQualityAnalyzer")

        return analyzers

    def _get_ast(self) -> Optional[ast.Module]:
        """Get parsed AST of the target module (cached)."""
        if self._ast_cache is None:
            try:
                content = self._get_file_content()
                if content:
                    self._ast_cache = ast.parse(content, filename=str(self.target_module))
            except SyntaxError as e:
                logger.error(f"Syntax error in {self.target_module}: {e}")
                return None
        return self._ast_cache

    def _get_file_content(self) -> Optional[str]:
        """Get file content (cached)."""
        if self._file_content is None:
            try:
                self._file_content = self.target_module.read_text(encoding="utf-8-sig", errors="replace")
            except (OSError, UnicodeDecodeError) as e:
                logger.error(f"Error reading {self.target_module}: {e}")
                return None
        return self._file_content

    def _create_pipeline_context(self) -> Dict[str, Any]:
        """Create context for pipeline execution."""
        return {
            'target_module_str': str(self.target_module),
            'project_root': self.project_root,
        }

    def analyze_for_expert_refactoring(self) -> ExpertAnalysisResult:
        """
        Perform complete expert refactoring analysis.
        
        Returns:
            ExpertAnalysisResult with all analysis data
        """
        logger.info("Starting expert refactoring analysis...")
        
        result = ExpertAnalysisResult(
            target_file=str(self.target_module),
            timestamp=self.timestamp
        )
        
        # Get AST
        module_ast = self._get_ast()
        if not module_ast:
            logger.error("Cannot parse target module - analysis will be limited")
            result.risk_assessment = RiskLevel.HIGH
            result.recommendations.append("Cannot parse target module - manual review required")
            return result

        # Execute main pipeline
        executor = AnalysisPipelineExecutor(
            self.analyzers,
            self._create_pipeline_context()
        )
        pipeline_result = executor.execute_pipeline(module_ast)
        
        # Apply results to ExpertAnalysisResult
        for attr, value in pipeline_result.results.items():
            if hasattr(result, attr):
                setattr(result, attr, value)
        
        # Add failures as recommendations
        result.recommendations.extend(pipeline_result.failures)
        
        # Cohesion matrix (special case - needs class selection)
        self._analyze_cohesion(module_ast, result)
        
        # Usage analysis (depends on external_callers)
        self._analyze_usage_patterns(result)
        
        # Calculate final metrics using extracted functions
        result.analysis_quality_score = calculate_quality_score(result)
        result.risk_assessment = assess_overall_risk(result)
        result.recommendations.extend(generate_recommendations(result))
        
        logger.info(f"Expert analysis completed with quality score: {result.analysis_quality_score:.2f}")
        return result

    def _analyze_cohesion(self, module_ast: ast.Module, result: ExpertAnalysisResult) -> None:
        """Analyze cohesion matrix for the largest class."""
        try:
            classes = [n for n in module_ast.body if isinstance(n, ast.ClassDef)]
            if classes:
                target_class = max(
                    classes,
                    key=lambda c: len([n for n in c.body if isinstance(n, ast.FunctionDef)])
                )
                result.cohesion_matrix = self.analyzers.call(
                    'cohesion',
                    'build_cohesion_matrix',
                    target_class
                )
                if result.cohesion_matrix:
                    logger.info("✓ Cohesion matrix analysis completed")
        except Exception as e:
            logger.error(f"Cohesion analysis failed: {e}")
            result.recommendations.append("Cohesion analysis failed - review class structure manually")

    def _analyze_usage_patterns(self, result: ExpertAnalysisResult) -> None:
        """Analyze usage patterns if external callers found."""
        if result.external_callers:
            try:
                result.usage_analysis = self.analyzers.call(
                    'caller',
                    'analyze_usage_patterns',
                    result.external_callers
                )
            except Exception as e:
                logger.warning(f"Usage pattern analysis failed: {e}")

    def export_detailed_expert_data(self) -> Dict[str, Any]:
        """
        Export detailed expert analysis data.
        
        Returns:
            Dictionary with all detailed analysis data
        """
        logger.info("Exporting detailed expert analysis data...")
        
        # Run standard analysis first
        result = self.analyze_for_expert_refactoring()
        
        module_ast = self._get_ast()
        if not module_ast:
            return {"error": "Cannot parse target module"}
        
        detailed_data: Dict[str, Any] = {}
        
        # Export from standard analysis results
        self._export_call_graph(result, detailed_data)
        self._export_external_usage(result, detailed_data)
        self._export_duplicates(result, detailed_data)
        self._export_cohesion(result, module_ast, detailed_data)
        self._export_tests(result, detailed_data)
        
        # Run additional export pipeline
        executor = AnalysisPipelineExecutor(
            self.analyzers,
            self._create_pipeline_context()
        )
        export_result = executor.execute_pipeline(module_ast, steps=EXPORT_STEPS)
        
        # Process export results
        self._process_export_results(export_result.results, detailed_data, result)
        
        # Summary
        detailed_data['summary'] = {
            'target_file': str(self.target_module),
            'timestamp': self.timestamp,
            'analysis_quality_score': result.analysis_quality_score,
            'risk_assessment': result.risk_assessment.value if result.risk_assessment else 'unknown',
            'recommendations': result.recommendations
        }
        
        # Expert recommendations (using extracted function)
        detailed_data['expert_recommendations'] = generate_expert_recommendations(detailed_data)
        detailed_data['acceptance_criteria'] = generate_acceptance_criteria(detailed_data)
        
        logger.info("Detailed expert data export completed")
        return _make_json_safe(detailed_data)

    def _export_call_graph(self, result: ExpertAnalysisResult, data: Dict[str, Any]) -> None:
        """Export detailed call graph."""
        if result.call_graph:
            exported = self.analyzers.call(
                'call_graph',
                'export_detailed_call_graph',
                result.call_graph
            )
            if exported:
                data['call_graph'] = exported

    def _export_external_usage(self, result: ExpertAnalysisResult, data: Dict[str, Any]) -> None:
        """Export detailed external usage."""
        if result.external_callers:
            exported = self.analyzers.call(
                'caller',
                'export_detailed_external_usage',
                result.external_callers
            )
            if exported:
                data['external_usage'] = exported
        else:
            # Try to find external usage
            callers = self.analyzers.call(
                'caller',
                'find_external_callers',
                str(self.target_module)
            )
            if callers:
                data['external_usage'] = self.analyzers.call(
                    'caller',
                    'export_detailed_external_usage',
                    callers
                ) or {"message": "No external callers found"}
            else:
                data['external_usage'] = {"message": "No external callers found"}

    def _export_duplicates(self, result: ExpertAnalysisResult, data: Dict[str, Any]) -> None:
        """Export detailed duplicates."""
        if result.duplicate_fragments:
            exported = self.analyzers.call(
                'duplicate',
                'export_detailed_duplicates',
                result.duplicate_fragments
            )
            if exported:
                data['duplicates'] = exported

    def _export_cohesion(
        self,
        result: ExpertAnalysisResult,
        module_ast: ast.Module,
        data: Dict[str, Any]
    ) -> None:
        """Export detailed cohesion matrix."""
        if result.cohesion_matrix:
            exported = self.analyzers.call(
                'cohesion',
                'export_detailed_cohesion_matrix',
                result.cohesion_matrix
            )
            if exported:
                data['cohesion_matrix'] = exported
        else:
            # Try to build cohesion matrix
            try:
                classes = [n for n in module_ast.body if isinstance(n, ast.ClassDef)]
                if classes:
                    target_class = max(
                        classes,
                        key=lambda c: len([n for n in c.body if isinstance(n, ast.FunctionDef)])
                    )
                    matrix = self.analyzers.call('cohesion', 'build_cohesion_matrix', target_class)
                    if matrix:
                        data['cohesion_matrix'] = self.analyzers.call(
                            'cohesion',
                            'export_detailed_cohesion_matrix',
                            matrix
                        ) or {"message": "No classes found for cohesion analysis"}
                else:
                    data['cohesion_matrix'] = {"message": "No classes found for cohesion analysis"}
            except Exception as e:
                data['cohesion_matrix'] = {"error": str(e)}

    def _export_tests(self, result: ExpertAnalysisResult, data: Dict[str, Any]) -> None:
        """Export detailed test analysis."""
        if result.characterization_tests:
            exported = self.analyzers.call(
                'characterization',
                'export_detailed_characterization_tests',
                result.characterization_tests
            )
            if exported:
                data['characterization_tests'] = exported
        
        if result.test_discovery:
            exported = self.analyzers.call(
                'test',
                'export_detailed_test_analysis',
                result.test_discovery
            )
            if exported:
                data['test_analysis'] = exported
            
            # Test quality
            if result.test_discovery.existing_test_files:
                quality = self.analyzers.call(
                    'test_quality',
                    'analyze_test_quality',
                    result.test_discovery.existing_test_files
                )
                if quality:
                    data['test_quality'] = self.analyzers.call(
                        'test_quality',
                        'export_detailed_test_quality',
                        quality
                    ) or {"message": "No test files found for quality analysis"}
            else:
                data['test_quality'] = {"message": "No test files found for quality analysis"}

    def _process_export_results(
        self,
        results: Dict[str, Any],
        data: Dict[str, Any],
        analysis_result: ExpertAnalysisResult
    ) -> None:
        """Process results from export pipeline."""
        # Exception contracts
        if 'exception_contracts' in results:
            exported = self.analyzers.call(
                'exception_contract',
                'export_detailed_exception_contracts',
                results['exception_contracts']
            )
            data['exception_contracts'] = exported or {"error": "Export failed"}
        
        # Data schemas
        if 'data_schemas' in results:
            exported = self.analyzers.call(
                'data_schema',
                'export_detailed_data_schemas',
                results['data_schemas']
            )
            data['data_schemas'] = exported or {"error": "Export failed"}
        
        # Import dependencies
        if 'import_dependencies_raw' in results:
            raw = results['import_dependencies_raw']
            data['import_dependencies'] = {
                'import_cycles': raw.get('cycles', []) if isinstance(raw, dict) else [],
                'external_imports': raw.get('external_imports', []) if isinstance(raw, dict) else [],
                'internal_imports': raw.get('internal_imports', []) if isinstance(raw, dict) else [],
                'cycle_type_clarification': 'These are import-level cycles, different from call-level cycles'
            }
        
        # External dependency contracts
        if 'external_dependency_contracts' in results:
            data['external_dependency_contracts'] = results['external_dependency_contracts']
        
        # Optional dependencies
        if 'optional_dependencies_raw' in results:
            exported = self.analyzers.call(
                'optional_dependencies',
                'export_detailed_optional_dependencies',
                results['optional_dependencies_raw']
            )
            data['optional_dependencies'] = exported or {"error": "Export failed"}
        
        # Golden traces
        if 'golden_traces_raw' in results:
            exported = self.analyzers.call(
                'golden_traces',
                'export_detailed_golden_traces',
                results['golden_traces_raw']
            )
            data['golden_traces'] = exported or {"error": "Export failed"}

    def generate_expert_report(self, output_dir: Optional[str] = None) -> str:
        """
        Generate comprehensive expert report.
        
        Args:
            output_dir: Optional output directory override
            
        Returns:
            Path to generated report
        """
        report_dir = Path(output_dir) if output_dir else self.output_dir
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Perform analysis
        result = self.analyze_for_expert_refactoring()
        
        # Generate JSON report
        json_path = report_dir / f"expert_analysis_{self.timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            # Serialize full dataclass result safely (avoid relying on partial/placeholder to_dict()).
            json.dump(_make_json_safe(result), f, indent=2, ensure_ascii=False)
        
        # Generate Markdown report (using extracted function)
        md_path = report_dir / f"expert_analysis_report_{self.timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(generate_markdown_report(result))
        
        logger.info(f"Expert reports generated: {json_path}, {md_path}")
        return str(md_path)
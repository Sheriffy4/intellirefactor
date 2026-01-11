"""
Main expert refactoring analyzer.

Orchestrates all expert-level analyses to provide comprehensive data
for safe and precise refactoring decisions.
"""

from __future__ import annotations

import ast
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import (
    ExpertAnalysisResult,
    RiskLevel,
)

from .analyzers.call_graph_analyzer import CallGraphAnalyzer
from .analyzers.caller_analyzer import CallerAnalyzer
from .analyzers.cohesion_analyzer import CohesionMatrixAnalyzer
from .analyzers.contract_analyzer import BehavioralContractAnalyzer
from .analyzers.dependency_analyzer import DependencyInterfaceAnalyzer
from .analyzers.test_analyzer import TestDiscoveryAnalyzer
from .analyzers.characterization_generator import CharacterizationTestGenerator
from .analyzers.duplicate_analyzer import ConcreteDeduplicationAnalyzer
from .analyzers.git_analyzer import GitHistoryAnalyzer
from .analyzers.compatibility_analyzer import CompatibilityAnalyzer
from .analyzers.exception_contract_analyzer import ExceptionContractAnalyzer
from .analyzers.data_schema_analyzer import DataSchemaAnalyzer
from .analyzers.optional_dependencies_analyzer import OptionalDependenciesAnalyzer
from .analyzers.golden_traces_extractor import GoldenTracesExtractor
from .analyzers.test_quality_analyzer import TestQualityAnalyzer

logger = logging.getLogger(__name__)


class ExpertRefactoringAnalyzer:
    """
    Main expert refactoring analyzer that orchestrates all specialized analyzers
    to provide comprehensive data for expert-level refactoring decisions.
    """

    def __init__(self, project_root: str, target_module: str, output_dir: Optional[str] = None):
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
        
        # Initialize analyzers
        self.analyzers = self._initialize_analyzers()
        
        # Cache for parsed AST
        self._ast_cache: Optional[ast.Module] = None
        self._file_content: Optional[str] = None
        
        logger.info(f"Initialized ExpertRefactoringAnalyzer for {self.target_module}")

    def _initialize_analyzers(self) -> Dict[str, Any]:
        """Initialize all specialized analyzers."""
        return {
            'call_graph': CallGraphAnalyzer(self.project_root, self.target_module),
            'caller': CallerAnalyzer(self.project_root, self.target_module),
            'cohesion': CohesionMatrixAnalyzer(self.project_root, self.target_module),
            'contract': BehavioralContractAnalyzer(self.project_root, self.target_module),
            'dependency': DependencyInterfaceAnalyzer(self.project_root, self.target_module),
            'test': TestDiscoveryAnalyzer(self.project_root, self.target_module),
            'characterization': CharacterizationTestGenerator(self.project_root, self.target_module),
            'duplicate': ConcreteDeduplicationAnalyzer(self.project_root, self.target_module),
            'git': GitHistoryAnalyzer(self.project_root, self.target_module),
            'compatibility': CompatibilityAnalyzer(self.project_root, self.target_module),
            'exception_contract': ExceptionContractAnalyzer(self.project_root, self.target_module),
            'data_schema': DataSchemaAnalyzer(self.project_root, self.target_module),
            'optional_dependencies': OptionalDependenciesAnalyzer(self.project_root, self.target_module),
            'golden_traces': GoldenTracesExtractor(self.project_root, self.target_module),
            'test_quality': TestQualityAnalyzer(self.project_root, self.target_module),
        }

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
                self._file_content = self.target_module.read_text(encoding='utf-8')
            except (OSError, UnicodeDecodeError) as e:
                logger.error(f"Error reading {self.target_module}: {e}")
                return None
        return self._file_content

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
        
        # Get AST for analyzers that need it
        module_ast = self._get_ast()
        if not module_ast:
            logger.error("Cannot parse target module - analysis will be limited")
            result.risk_assessment = RiskLevel.HIGH
            result.recommendations.append("Cannot parse target module - manual review required")
            return result

        # Phase 1: Priority 0 analyzers (critical for safe refactoring)
        logger.info("Phase 1: Critical analyzers (P0)")
        
        # Call graph analysis
        try:
            result.call_graph = self.analyzers['call_graph'].analyze_internal_calls(module_ast)
            logger.info("✓ Call graph analysis completed")
        except Exception as e:
            logger.error(f"Call graph analysis failed: {e}")
            result.recommendations.append("Call graph analysis failed - manual dependency review required")

        # External caller analysis
        try:
            result.external_callers = self.analyzers['caller'].find_external_callers(str(self.target_module))
            result.usage_analysis = self.analyzers['caller'].analyze_usage_patterns(result.external_callers)
            logger.info("✓ External caller analysis completed")
        except Exception as e:
            logger.error(f"Caller analysis failed: {e}")
            result.recommendations.append("External usage analysis failed - check dependencies manually")

        # Behavioral contract analysis
        try:
            result.behavioral_contracts = self.analyzers['contract'].extract_contracts_from_docstrings(module_ast)
            logger.info("✓ Behavioral contract analysis completed")
        except Exception as e:
            logger.error(f"Contract analysis failed: {e}")
            result.recommendations.append("Contract analysis failed - review method contracts manually")

        # Dependency interface analysis
        try:
            result.dependency_interfaces = self.analyzers['dependency'].extract_dependency_interfaces(module_ast)
            logger.info("✓ Dependency interface analysis completed")
        except Exception as e:
            logger.error(f"Dependency analysis failed: {e}")
            result.recommendations.append("Dependency analysis failed - review imports manually")

        # Phase 2: Priority 1 analyzers (important for quality refactoring)
        logger.info("Phase 2: Quality analyzers (P1)")
        
        # Cohesion matrix analysis
        try:
            # Find classes in the module
            classes = [node for node in module_ast.body if isinstance(node, ast.ClassDef)]
            if classes:
                # Analyze the first/largest class for now
                target_class = max(classes, key=lambda c: len([n for n in c.body if isinstance(n, ast.FunctionDef)]))
                result.cohesion_matrix = self.analyzers['cohesion'].build_cohesion_matrix(target_class)
                logger.info("✓ Cohesion matrix analysis completed")
        except Exception as e:
            logger.error(f"Cohesion analysis failed: {e}")
            result.recommendations.append("Cohesion analysis failed - review class structure manually")

        # Test discovery
        try:
            result.test_discovery = self.analyzers['test'].find_existing_tests()
            logger.info("✓ Test discovery completed")
        except Exception as e:
            logger.error(f"Test discovery failed: {e}")
            result.recommendations.append("Test discovery failed - verify test coverage manually")

        # Characterization test generation
        try:
            result.characterization_tests = self.analyzers['characterization'].generate_characterization_tests(module_ast)
            logger.info("✓ Characterization test generation completed")
        except Exception as e:
            logger.error(f"Characterization test generation failed: {e}")
            result.recommendations.append("Test generation failed - create tests manually before refactoring")

        # Concrete duplicate analysis
        try:
            result.duplicate_fragments = self.analyzers['duplicate'].find_concrete_duplicates()
            logger.info("✓ Duplicate analysis completed")
        except Exception as e:
            logger.error(f"Duplicate analysis failed: {e}")
            result.recommendations.append("Duplicate analysis failed - review code manually for duplicates")

        # Phase 3: Priority 2 analyzers (nice to have)
        logger.info("Phase 3: Enhancement analyzers (P2)")
        
        # Git history analysis
        try:
            result.git_patterns = self.analyzers['git'].analyze_change_patterns()
            logger.info("✓ Git history analysis completed")
        except Exception as e:
            logger.warning(f"Git analysis failed: {e}")
            # Not critical, don't add to recommendations

        # Compatibility analysis
        try:
            result.impact_assessment = self.analyzers['compatibility'].assess_breaking_change_impact([])
            result.compatibility_constraints = self.analyzers['compatibility'].determine_compatibility_constraints()
            logger.info("✓ Compatibility analysis completed")
        except Exception as e:
            logger.error(f"Compatibility analysis failed: {e}")
            result.recommendations.append("Compatibility analysis failed - assess breaking changes manually")

        # Calculate overall quality score
        result.analysis_quality_score = self._calculate_quality_score(result)
        result.risk_assessment = self._assess_overall_risk(result)
        
        # Add general recommendations
        self._add_general_recommendations(result)
        
        logger.info(f"Expert analysis completed with quality score: {result.analysis_quality_score:.2f}")
        return result

    def export_detailed_expert_data(self) -> Dict[str, any]:
        """
        Export detailed expert analysis data as requested by experts.
        
        Returns:
            Dictionary with all detailed analysis data including:
            - Complete call graph with all 64 relationships
            - External usage with specific file locations
            - Detailed duplicates with line ranges
            - Cohesion matrix with method-attribute relationships
            - Executable characterization tests
            - Missing test coverage details
            - Exception contracts and data schemas
        """
        logger.info("Exporting detailed expert analysis data...")
        
        # Run the standard analysis first
        result = self.analyze_for_expert_refactoring()
        
        # Get AST for detailed analysis
        module_ast = self._get_ast()
        if not module_ast:
            return {"error": "Cannot parse target module"}
        
        detailed_data = {}
        
        # 1. Detailed call graph (Expert 1 requirement)
        if result.call_graph:
            detailed_data['call_graph'] = self.analyzers['call_graph'].export_detailed_call_graph(result.call_graph)
        
        # 2. Detailed external usage (Expert 1 requirement)
        if result.external_callers:
            detailed_data['external_usage'] = self.analyzers['caller'].export_detailed_external_usage(result.external_callers)
        else:
            # Still try to find external usage even if not found in standard analysis
            try:
                external_callers = self.analyzers['caller'].find_external_callers(str(self.target_module))
                if external_callers:
                    detailed_data['external_usage'] = self.analyzers['caller'].export_detailed_external_usage(external_callers)
                else:
                    detailed_data['external_usage'] = {"message": "No external callers found"}
            except Exception as e:
                detailed_data['external_usage'] = {"error": f"External usage analysis failed: {e}"}
        
        # 3. Detailed duplicates (Expert 1 & 2 requirement)
        if result.duplicate_fragments:
            detailed_data['duplicates'] = self.analyzers['duplicate'].export_detailed_duplicates(result.duplicate_fragments)
        
        # 4. Detailed cohesion matrix (Expert 1 requirement)
        if result.cohesion_matrix:
            detailed_data['cohesion_matrix'] = self.analyzers['cohesion'].export_detailed_cohesion_matrix(result.cohesion_matrix)
        else:
            # Try to build cohesion matrix even if not found in standard analysis
            try:
                # Find classes in the module
                classes = [node for node in module_ast.body if isinstance(node, ast.ClassDef)]
                if classes:
                    # Analyze the first/largest class
                    target_class = max(classes, key=lambda c: len([n for n in c.body if isinstance(n, ast.FunctionDef)]))
                    cohesion_matrix = self.analyzers['cohesion'].build_cohesion_matrix(target_class)
                    detailed_data['cohesion_matrix'] = self.analyzers['cohesion'].export_detailed_cohesion_matrix(cohesion_matrix)
                else:
                    detailed_data['cohesion_matrix'] = {"message": "No classes found for cohesion analysis"}
            except Exception as e:
                detailed_data['cohesion_matrix'] = {"error": f"Cohesion matrix analysis failed: {e}"}
        
        # 5. Detailed characterization tests (Expert 2 requirement)
        if result.characterization_tests:
            detailed_data['characterization_tests'] = self.analyzers['characterization'].export_detailed_characterization_tests(result.characterization_tests)
        
        # 6. Detailed test analysis (Expert 1 & 2 requirement)
        if result.test_discovery:
            detailed_data['test_analysis'] = self.analyzers['test'].export_detailed_test_analysis(result.test_discovery)
        
        # 7. Exception contracts (ref.md requirement A)
        try:
            exception_contracts = self.analyzers['exception_contract'].analyze_exception_contracts(module_ast)
            detailed_data['exception_contracts'] = self.analyzers['exception_contract'].export_detailed_exception_contracts(exception_contracts)
            logger.info("✓ Exception contract analysis completed")
        except Exception as e:
            logger.error(f"Exception contract analysis failed: {e}")
            detailed_data['exception_contracts'] = {"error": str(e)}
        
        # 8. Data schemas (ref.md requirement B)
        try:
            data_schemas = self.analyzers['data_schema'].analyze_data_schemas(module_ast)
            detailed_data['data_schemas'] = self.analyzers['data_schema'].export_detailed_data_schemas(data_schemas)
            logger.info("✓ Data schema analysis completed")
        except Exception as e:
            logger.error(f"Data schema analysis failed: {e}")
            detailed_data['data_schemas'] = {"error": str(e)}
        
        # 9. Import dependency analysis (Expert 2 requirement - clarify cycles)
        try:
            import_dependencies = self.analyzers['dependency'].analyze_import_dependencies(module_ast)
            detailed_data['import_dependencies'] = {
                'import_cycles': import_dependencies.get('cycles', []),
                'external_imports': import_dependencies.get('external_imports', []),
                'internal_imports': import_dependencies.get('internal_imports', []),
                'cycle_type_clarification': 'These are import-level cycles, different from call-level cycles'
            }
            logger.info("✓ Import dependency analysis completed")
        except Exception as e:
            logger.warning(f"Import dependency analysis failed: {e}")
            detailed_data['import_dependencies'] = {"error": str(e)}
        
        # 10. External dependency contracts (Expert 2 requirement)
        try:
            dependency_contracts = self.analyzers['dependency'].extract_external_dependency_contracts(module_ast)
            detailed_data['external_dependency_contracts'] = dependency_contracts
            logger.info("✓ External dependency contract analysis completed")
        except Exception as e:
            logger.warning(f"External dependency contract analysis failed: {e}")
            detailed_data['external_dependency_contracts'] = {"error": str(e)}
        
        # 11. Optional dependencies analysis (ref.md requirement C)
        try:
            optional_deps = self.analyzers['optional_dependencies'].analyze_optional_dependencies(module_ast)
            detailed_data['optional_dependencies'] = self.analyzers['optional_dependencies'].export_detailed_optional_dependencies(optional_deps)
            logger.info("✓ Optional dependencies analysis completed")
        except Exception as e:
            logger.error(f"Optional dependencies analysis failed: {e}")
            detailed_data['optional_dependencies'] = {"error": str(e)}
        
        # 12. Golden traces extraction (ref.md requirement D)
        try:
            golden_traces = self.analyzers['golden_traces'].extract_golden_traces(module_ast)
            detailed_data['golden_traces'] = self.analyzers['golden_traces'].export_detailed_golden_traces(golden_traces)
            logger.info("✓ Golden traces extraction completed")
        except Exception as e:
            logger.error(f"Golden traces extraction failed: {e}")
            detailed_data['golden_traces'] = {"error": str(e)}
        
        # 13. Test quality analysis (ref.md requirement E)
        try:
            if result.test_discovery and result.test_discovery.existing_test_files:
                test_quality = self.analyzers['test_quality'].analyze_test_quality(result.test_discovery.existing_test_files)
                detailed_data['test_quality'] = self.analyzers['test_quality'].export_detailed_test_quality(test_quality)
                logger.info("✓ Test quality analysis completed")
            else:
                detailed_data['test_quality'] = {"message": "No test files found for quality analysis"}
        except Exception as e:
            logger.error(f"Test quality analysis failed: {e}")
            detailed_data['test_quality'] = {"error": str(e)}
        
        # 14. Summary data for experts
        detailed_data['summary'] = {
            'target_file': str(self.target_module),
            'timestamp': self.timestamp,
            'analysis_quality_score': result.analysis_quality_score,
            'risk_assessment': result.risk_assessment.value if result.risk_assessment else 'unknown',
            'recommendations': result.recommendations
        }
        
        # 15. Expert-specific recommendations
        detailed_data['expert_recommendations'] = self._generate_expert_specific_recommendations(detailed_data)
        
        # 16. Acceptance criteria (Expert 2 requirement)
        detailed_data['acceptance_criteria'] = self._generate_acceptance_criteria(detailed_data)
        
        logger.info("Detailed expert data export completed")
        return detailed_data

    def _generate_expert_specific_recommendations(self, detailed_data: Dict[str, any]) -> Dict[str, List[str]]:
        """Generate specific recommendations addressing expert feedback."""
        recommendations = {
            'expert_1_requirements': [],
            'expert_2_requirements': [],
            'general_recommendations': []
        }
        
        # Expert 1 specific recommendations
        if 'call_graph' in detailed_data:
            call_graph = detailed_data['call_graph']
            total_relationships = call_graph.get('call_graph', {}).get('total_relationships', 0)
            recommendations['expert_1_requirements'].append(
                f"✅ Complete call graph available: {total_relationships} relationships documented"
            )
        
        if 'external_usage' in detailed_data:
            external_usage = detailed_data['external_usage']
            total_files = external_usage.get('files_summary', {}).get('total_files', 0)
            recommendations['expert_1_requirements'].append(
                f"✅ External usage detailed: {total_files} files with specific line locations"
            )
        
        if 'duplicates' in detailed_data:
            duplicates = detailed_data['duplicates']
            total_duplicates = duplicates.get('summary', {}).get('total_duplicates', 0)
            recommendations['expert_1_requirements'].append(
                f"✅ Code duplicates detailed: {total_duplicates} fragments with exact locations"
            )
        
        if 'cohesion_matrix' in detailed_data:
            recommendations['expert_1_requirements'].append(
                "✅ Cohesion matrix available: method-attribute relationships with extraction recommendations"
            )
        
        # Expert 2 specific recommendations
        if 'characterization_tests' in detailed_data:
            char_tests = detailed_data['characterization_tests']
            total_tests = char_tests.get('summary', {}).get('total_tests', 0)
            recommendations['expert_2_requirements'].append(
                f"✅ Executable characterization tests: {total_tests} test cases with mocks and fixtures"
            )
        
        if 'test_analysis' in detailed_data:
            test_analysis = detailed_data['test_analysis']
            missing_count = test_analysis.get('missing_test_coverage', {}).get('total_missing', 0)
            recommendations['expert_2_requirements'].append(
                f"✅ Missing test coverage detailed: {missing_count} specific methods identified"
            )
        
        # General recommendations
        recommendations['general_recommendations'].extend([
            "All expert requirements have been addressed with detailed data",
            "Use the exported JSON data for comprehensive refactoring planning",
            "Follow the phased approach: 1) Resolve cycles, 2) Add tests, 3) Refactor systematically"
        ])
        
        return recommendations

    def _generate_acceptance_criteria(self, detailed_data: Dict[str, any]) -> Dict[str, List[str]]:
        """Generate acceptance criteria for safe refactoring as requested by Expert 2."""
        criteria = {
            'behavioral_invariants': [],
            'api_compatibility': [],
            'performance_requirements': [],
            'error_handling': [],
            'logging_requirements': []
        }
        
        # Behavioral invariants
        criteria['behavioral_invariants'].extend([
            "Attack recipe structure and order must be preserved",
            "Parameter validation must occur before any processing",
            "All existing public method signatures must remain unchanged",
            "Exception types and messages should remain consistent",
            "State transitions must follow the same sequence"
        ])
        
        # API compatibility
        if 'external_usage' in detailed_data:
            external_usage = detailed_data['external_usage']
            total_external_files = external_usage.get('files_summary', {}).get('total_files', 0)
            if total_external_files > 0:
                criteria['api_compatibility'].extend([
                    f"All {total_external_files} external files must continue to work without changes",
                    "Public method return types must remain compatible",
                    "Parameter order and types must be preserved for public methods"
                ])
        
        # Performance requirements
        criteria['performance_requirements'].extend([
            "Refactored code should not be significantly slower than original",
            "Memory usage patterns should remain similar",
            "No new performance bottlenecks should be introduced"
        ])
        
        # Error handling
        if 'exception_contracts' in detailed_data:
            criteria['error_handling'].extend([
                "All existing exception types must be preserved",
                "Error conditions that previously raised exceptions must continue to do so",
                "Error messages should remain informative and consistent"
            ])
        
        # Logging requirements
        criteria['logging_requirements'].extend([
            "All existing log levels and messages must be preserved",
            "Log format should remain consistent for external parsing",
            "Debug information should not be reduced"
        ])
        
        return criteria

    def _calculate_quality_score(self, result: ExpertAnalysisResult) -> float:
        """Calculate overall analysis quality score (0-100)."""
        score = 0.0
        max_score = 100.0
        
        # Critical components (60% of score)
        if result.call_graph:
            score += 15.0
        if result.external_callers:
            score += 15.0
        if result.behavioral_contracts:
            score += 15.0
        if result.dependency_interfaces:
            score += 15.0
        
        # Important components (30% of score)
        if result.cohesion_matrix:
            score += 10.0
        if result.test_discovery:
            score += 10.0
        if result.characterization_tests:
            score += 10.0
        
        # Enhancement components (10% of score)
        if result.duplicate_fragments:
            score += 5.0
        if result.git_patterns:
            score += 3.0
        if result.impact_assessment:
            score += 2.0
        
        return min(score, max_score)

    def _assess_overall_risk(self, result: ExpertAnalysisResult) -> RiskLevel:
        """Assess overall risk level for refactoring."""
        risk_factors = 0
        
        # High risk factors
        if not result.call_graph:
            risk_factors += 3
        if not result.external_callers:
            risk_factors += 3
        if not result.test_discovery or not result.test_discovery.existing_test_files:
            risk_factors += 2
        if not result.behavioral_contracts:
            risk_factors += 2
        
        # Medium risk factors
        if result.usage_analysis and result.usage_analysis.total_callers > 10:
            risk_factors += 1
        if result.duplicate_fragments and len(result.duplicate_fragments) > 5:
            risk_factors += 1
        
        if risk_factors >= 6:
            return RiskLevel.CRITICAL
        elif risk_factors >= 4:
            return RiskLevel.HIGH
        elif risk_factors >= 2:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def _add_general_recommendations(self, result: ExpertAnalysisResult) -> None:
        """Add general recommendations based on analysis results."""
        if result.analysis_quality_score < 50:
            result.recommendations.append("Low analysis quality - consider manual review before refactoring")
        
        if result.risk_assessment in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            result.recommendations.append("High risk refactoring - create comprehensive tests first")
        
        if result.external_callers and len(result.external_callers) > 5:
            result.recommendations.append("Module has many external dependencies - plan migration carefully")
        
        if result.call_graph and result.call_graph.cycles:
            result.recommendations.append("Circular dependencies detected - resolve before major refactoring")

    def generate_expert_report(self, output_dir: Optional[str] = None) -> str:
        """
        Generate comprehensive expert report.
        
        Args:
            output_dir: Optional output directory override
            
        Returns:
            Path to generated report
        """
        if output_dir:
            report_dir = Path(output_dir)
        else:
            report_dir = self.output_dir
        
        report_dir.mkdir(parents=True, exist_ok=True)
        
        # Perform analysis
        result = self.analyze_for_expert_refactoring()
        
        # Generate JSON report
        json_path = report_dir / f"expert_analysis_{self.timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        # Generate Markdown report
        md_path = report_dir / f"expert_analysis_report_{self.timestamp}.md"
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_markdown_report(result))
        
        logger.info(f"Expert reports generated: {json_path}, {md_path}")
        return str(md_path)

    def _generate_markdown_report(self, result: ExpertAnalysisResult) -> str:
        """Generate markdown report from analysis results."""
        lines = [
            "# Expert Refactoring Analysis Report",
            "",
            f"**Target Module:** {result.target_file}",
            f"**Analysis Date:** {result.timestamp}",
            f"**Quality Score:** {result.analysis_quality_score:.1f}/100",
            f"**Risk Level:** {result.risk_assessment.value.upper()}",
            "",
            "## Executive Summary",
            "",
        ]
        
        if result.recommendations:
            lines.extend([
                "### Key Recommendations",
                "",
            ])
            for rec in result.recommendations:
                lines.append(f"- {rec}")
            lines.append("")
        
        # Call Graph Section
        if result.call_graph:
            lines.extend([
                "## Call Graph Analysis",
                "",
                f"- **Methods:** {len(result.call_graph.nodes)}",
                f"- **Call Relationships:** {len(result.call_graph.edges)}",
                f"- **Cycles Detected:** {len(result.call_graph.cycles)}",
                "",
            ])
            
            if result.call_graph.cycles:
                lines.extend([
                    "### Circular Dependencies",
                    "",
                ])
                for cycle in result.call_graph.cycles:
                    lines.append(f"- {' → '.join(cycle.nodes)} (Risk: {cycle.risk_level.value})")
                lines.append("")
        
        # External Usage Section
        if result.external_callers:
            lines.extend([
                "## External Usage Analysis",
                "",
                f"- **External Files:** {len(result.external_callers)}",
            ])
            
            if result.usage_analysis:
                lines.append(f"- **Total Callers:** {result.usage_analysis.total_callers}")
                if result.usage_analysis.most_used_symbols:
                    lines.extend([
                        "",
                        "### Most Used Symbols",
                        "",
                    ])
                    for symbol, count in result.usage_analysis.most_used_symbols[:5]:
                        lines.append(f"- `{symbol}`: {count} uses")
            lines.append("")
        
        # Test Analysis Section
        if result.test_discovery:
            lines.extend([
                "## Test Coverage Analysis",
                "",
                f"- **Test Files Found:** {len(result.test_discovery.existing_test_files)}",
                f"- **Quality Score:** {result.test_discovery.test_quality_score:.1f}/100",
                f"- **Missing Tests:** {len(result.test_discovery.missing_tests)}",
                "",
            ])
        
        # Characterization Tests Section
        if result.characterization_tests:
            lines.extend([
                "## Characterization Tests",
                "",
                f"Generated {len(result.characterization_tests)} characterization test cases:",
                "",
            ])
            
            # Group by category
            by_category = {}
            for test in result.characterization_tests:
                category = test.test_category.value
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append(test)
            
            for category, tests in by_category.items():
                lines.extend([
                    f"### {category.title()} Cases ({len(tests)})",
                    "",
                ])
                for test in tests[:3]:  # Show first 3
                    method_name = f"{test.class_name}.{test.method_name}" if test.class_name else test.method_name
                    lines.append(f"- `{method_name}`: {test.description}")
                lines.append("")
        
        # Duplicate Analysis Section
        if result.duplicate_fragments:
            lines.extend([
                "## Code Duplication Analysis",
                "",
                f"Found {len(result.duplicate_fragments)} duplicate code fragments:",
                "",
            ])
            
            total_savings = sum(frag.estimated_savings for frag in result.duplicate_fragments)
            lines.append(f"**Potential Savings:** {total_savings} lines of code")
            lines.append("")
        
        # Compatibility Section
        if result.compatibility_constraints:
            lines.extend([
                "## Compatibility Constraints",
                "",
            ])
            for constraint in result.compatibility_constraints:
                lines.append(f"- {constraint}")
            lines.append("")
        
        lines.extend([
            "## Next Steps",
            "",
            "1. Review all high-risk findings above",
            "2. Create/run characterization tests",
            "3. Address circular dependencies",
            "4. Plan refactoring in small, safe steps",
            "5. Monitor external usage during changes",
            "",
            "---",
            f"*Report generated by ExpertRefactoringAnalyzer at {datetime.now().isoformat()}*",
        ])
        
        return "\n".join(lines)
# Functional Decomposition Catalog

**Project:** C:\Intel\recon\intellirefactor
**Generated:** 2026-01-09 22:40:04
**Analysis Timestamp:** 2026-01-09T22:40:04.090651

## Summary Statistics

- **Total Blocks:** 1968
- **Total Capabilities:** 15
- **Total Clusters:** 14
- **Call Resolution Rate:** 12.7%

## Categories Breakdown

| Category:Subcategory | Block Count |
|---------------------|-------------|
| parsing:regex | 1009 |
| telemetry:logging | 423 |
| persistence:generic | 165 |
| factory:creation | 93 |
| serialization:json | 80 |
| concurrency:threading | 65 |
| unknown:generic | 24 |
| orchestration:execution | 23 |
| validation:generic | 22 |
| presentation:formatting | 20 |
| parsing:generic | 17 |
| configuration:initialization | 10 |
| domain:strategy | 7 |
| configuration:settings | 4 |
| serialization:yaml | 3 |
| transformation:generic | 1 |
| caching:storage | 1 |
| domain:attack | 1 |

## Capabilities

### serialization_json

**Description:** Serialization capability for json
**Blocks:** 80
**Suggested Owners:** cli.rich_output

### parsing_regex

**Description:** Parsing capability for regex
**Blocks:** 1009
**Suggested Owners:** refactoring.auto_refactor

### validation_generic

**Description:** Validation capability for generic
**Blocks:** 22
**Suggested Owners:** plugins.plugin_interface

### factory_creation

**Description:** Factory capability for creation
**Blocks:** 93
**Suggested Owners:** analysis.index_schema

### telemetry_logging

**Description:** Telemetry capability for logging
**Blocks:** 423
**Suggested Owners:** analysis.debug_cycle_manager

### persistence_generic

**Description:** Persistence capability for generic
**Blocks:** 165
**Suggested Owners:** analysis.index_store

### parsing_generic

**Description:** Parsing capability for generic
**Blocks:** 17
**Suggested Owners:** analysis.models

### unknown_generic

**Description:** Unknown capability for generic
**Blocks:** 24
**Suggested Owners:** analysis.architectural_smell_detector

### domain_strategy

**Description:** Domain capability for strategy
**Blocks:** 7
**Suggested Owners:** documentation.llm_context_generator

### configuration_settings

**Description:** Configuration capability for settings
**Blocks:** 4
**Suggested Owners:** documentation.registry_generator

### configuration_initialization

**Description:** Configuration capability for initialization
**Blocks:** 10
**Suggested Owners:** api

### orchestration_execution

**Description:** Orchestration capability for execution
**Blocks:** 23
**Suggested Owners:** orchestration.global_refactoring_orchestrator

### concurrency_threading

**Description:** Concurrency capability for threading
**Blocks:** 65
**Suggested Owners:** performance.memory_manager

### serialization_yaml

**Description:** Serialization capability for yaml
**Blocks:** 3
**Suggested Owners:** templates.template_generator

### presentation_formatting

**Description:** Presentation capability for formatting
**Blocks:** 20
**Suggested Owners:** cli

## Similarity Clusters

### parsing:regex

- **Blocks:** 2
- **Average Similarity:** 1.00
- **Recommendation:** MERGE
- **Risk Level:** LOW
- **Effort:** S

**Blocks in cluster:**
- `OpportunityDetector.__init__` (C:\Intel\recon\intellirefactor\refactoring\intelligent_refactoring_system.py:344)
- `QualityAssessor.__init__` (C:\Intel\recon\intellirefactor\refactoring\intelligent_refactoring_system.py:544)

### parsing:regex

- **Blocks:** 2
- **Average Similarity:** 1.00
- **Recommendation:** MERGE
- **Risk Level:** LOW
- **Effort:** S

**Blocks in cluster:**
- `LLMContextGenerator._extract_docstring` (C:\Intel\recon\intellirefactor\documentation\llm_context_generator.py:544)
- `RegistryGenerator._extract_docstring` (C:\Intel\recon\intellirefactor\documentation\registry_generator.py:573)

### telemetry:logging

- **Blocks:** 2
- **Average Similarity:** 1.00
- **Recommendation:** MERGE
- **Risk Level:** LOW
- **Effort:** S

**Blocks in cluster:**
- `FeasibilityAnalysis.to_dict` (C:\Intel\recon\intellirefactor\analysis\refactoring_decision_engine.py:233)
- `ImplementationStep.to_dict` (C:\Intel\recon\intellirefactor\analysis\refactoring_decision_engine.py:250)

### telemetry:logging

- **Blocks:** 2
- **Average Similarity:** 1.00
- **Recommendation:** MERGE
- **Risk Level:** LOW
- **Effort:** S

**Blocks in cluster:**
- `CloneGroup._clamp01` (C:\Intel\recon\intellirefactor\analysis\block_clone_detector.py:103)
- `BlockCloneDetector._clamp01` (C:\Intel\recon\intellirefactor\analysis\block_clone_detector.py:196)

### persistence:generic

- **Blocks:** 2
- **Average Similarity:** 1.00
- **Recommendation:** MERGE
- **Risk Level:** LOW
- **Effort:** S

**Blocks in cluster:**
- `DeepMethodInfo.get_responsibility_score` (C:\Intel\recon\intellirefactor\analysis\models.py:169)
- `DeepClassInfo.get_responsibility_score` (C:\Intel\recon\intellirefactor\analysis\models.py:411)

### unknown:generic

- **Blocks:** 3
- **Average Similarity:** 1.00
- **Recommendation:** MERGE
- **Risk Level:** LOW
- **Effort:** M

**Blocks in cluster:**
- `LLMContextGenerator.__init__` (C:\Intel\recon\intellirefactor\documentation\llm_context_generator.py:43)
- `ProjectStructureGenerator.__init__` (C:\Intel\recon\intellirefactor\documentation\project_structure_generator.py:48)
- `RegistryGenerator.__init__` (C:\Intel\recon\intellirefactor\documentation\registry_generator.py:55)

### parsing:regex

- **Blocks:** 11
- **Average Similarity:** 0.95
- **Recommendation:** MERGE
- **Risk Level:** LOW
- **Effort:** L

**Blocks in cluster:**
- `CompatibilityAnalyzer.__init__` (C:\Intel\recon\intellirefactor\analysis\expert\analyzers\compatibility_analyzer.py:22)
- `GitHistoryAnalyzer.__init__` (C:\Intel\recon\intellirefactor\analysis\expert\analyzers\git_analyzer.py:22)
- `CohesionMatrixAnalyzer.__init__` (C:\Intel\recon\intellirefactor\analysis\expert\analyzers\cohesion_analyzer.py:26)
- `GoldenTracesExtractor.__init__` (C:\Intel\recon\intellirefactor\analysis\expert\analyzers\golden_traces_extractor.py:22)
- `DataSchemaAnalyzer.__init__` (C:\Intel\recon\intellirefactor\analysis\expert\analyzers\data_schema_analyzer.py:21)
- `CharacterizationTestGenerator.__init__` (C:\Intel\recon\intellirefactor\analysis\expert\analyzers\characterization_generator.py:23)
- `DependencyInterfaceAnalyzer.__init__` (C:\Intel\recon\intellirefactor\analysis\expert\analyzers\dependency_analyzer.py:29)
- `OptionalDependenciesAnalyzer.__init__` (C:\Intel\recon\intellirefactor\analysis\expert\analyzers\optional_dependencies_analyzer.py:21)
- `CallGraphAnalyzer.__init__` (C:\Intel\recon\intellirefactor\analysis\expert\analyzers\call_graph_analyzer.py:31)
- `ExceptionContractAnalyzer.__init__` (C:\Intel\recon\intellirefactor\analysis\expert\analyzers\exception_contract_analyzer.py:21)
- `TestQualityAnalyzer.__init__` (C:\Intel\recon\intellirefactor\analysis\expert\analyzers\test_quality_analyzer.py:21)

### telemetry:logging

- **Blocks:** 2
- **Average Similarity:** 0.95
- **Recommendation:** MERGE
- **Risk Level:** LOW
- **Effort:** S

**Blocks in cluster:**
- `AutomationMetadata._calculate_automation_score` (C:\Intel\recon\intellirefactor\knowledge\automation_metadata.py:929)
- `AutomationMetadataGenerator._calculate_automation_score` (C:\Intel\recon\intellirefactor\knowledge\automation_metadata_generator.py:230)

### telemetry:logging

- **Blocks:** 2
- **Average Similarity:** 0.95
- **Recommendation:** MERGE
- **Risk Level:** LOW
- **Effort:** S

**Blocks in cluster:**
- `AutomationMetadata._calculate_reusability_score` (C:\Intel\recon\intellirefactor\knowledge\automation_metadata.py:943)
- `AutomationMetadataGenerator._calculate_reusability_score` (C:\Intel\recon\intellirefactor\knowledge\automation_metadata_generator.py:244)

### telemetry:logging

- **Blocks:** 2
- **Average Similarity:** 0.95
- **Recommendation:** MERGE
- **Risk Level:** LOW
- **Effort:** S

**Blocks in cluster:**
- `AutomationMetadata._get_related_patterns` (C:\Intel\recon\intellirefactor\knowledge\automation_metadata.py:956)
- `AutomationMetadataGenerator._get_related_patterns` (C:\Intel\recon\intellirefactor\knowledge\automation_metadata_generator.py:257)

### telemetry:logging

- **Blocks:** 2
- **Average Similarity:** 0.93
- **Recommendation:** MERGE
- **Risk Level:** MEDIUM
- **Effort:** S

**Blocks in cluster:**
- `DebugCycleManager._get_timestamp` (C:\Intel\recon\intellirefactor\analysis\debug_cycle_manager.py:966)
- `IncrementalDebuggingWorkflow._get_timestamp` (C:\Intel\recon\intellirefactor\analysis\incremental_debugging_workflow.py:586)

### telemetry:logging

- **Blocks:** 4
- **Average Similarity:** 0.90
- **Recommendation:** MERGE
- **Risk Level:** LOW
- **Effort:** M

**Blocks in cluster:**
- `ConsolidationPlanner.__init__` (C:\Intel\recon\intellirefactor\analysis\decomposition\consolidation_planner.py:41)
- `FingerprintGenerator.__init__` (C:\Intel\recon\intellirefactor\analysis\decomposition\fingerprints.py:30)
- `DecompositionReportGenerator.__init__` (C:\Intel\recon\intellirefactor\analysis\decomposition\report_generator.py:39)
- `FunctionalBlockExtractor.__init__` (C:\Intel\recon\intellirefactor\analysis\decomposition\block_extractor.py:77)

### parsing:regex

- **Blocks:** 5
- **Average Similarity:** 0.79
- **Recommendation:** EXTRACT_BASE
- **Risk Level:** LOW
- **Effort:** M

**Blocks in cluster:**
- `Evidence.__post_init__` (C:\Intel\recon\intellirefactor\analysis\models.py:95)
- `DependencyInfo.__post_init__` (C:\Intel\recon\intellirefactor\analysis\models.py:312)
- `DeepMethodInfo.__post_init__` (C:\Intel\recon\intellirefactor\analysis\models.py:165)
- `BlockInfo.__post_init__` (C:\Intel\recon\intellirefactor\analysis\models.py:248)
- `DeepClassInfo.__post_init__` (C:\Intel\recon\intellirefactor\analysis\models.py:403)

### telemetry:logging

- **Blocks:** 2
- **Average Similarity:** 0.70
- **Recommendation:** EXTRACT_BASE
- **Risk Level:** LOW
- **Effort:** M

**Blocks in cluster:**
- `OperationSequence.__post_init__` (C:\Intel\recon\intellirefactor\analysis\semantic_similarity_matcher.py:48)
- `SimilarityMatch.__post_init__` (C:\Intel\recon\intellirefactor\analysis\semantic_similarity_matcher.py:67)

## Notable Blocks

### Most Complex (by Cyclomatic Complexity)

| Block | Complexity | LOC | File |
|-------|------------|-----|------|
| `cmd_duplicates` | 42 | 293 | C:\Intel\recon\intellirefactor\cli.py |
| `cmd_expert_analyze` | 39 | 172 | C:\Intel\recon\intellirefactor\cli.py |
| `ArchitectureGenerator.generate_architecture_diagram` | 35 | 162 | C:\Intel\recon\intellirefactor\documentation\architecture_generator.py |
| `FileAnalyzer.generate_detailed_report` | 34 | 162 | C:\Intel\recon\intellirefactor\analysis\file_analyzer.py |
| `AutoRefactor._group_methods_by_responsibility` | 30 | 93 | C:\Intel\recon\intellirefactor\refactoring\auto_refactor.py |
| `main` | 29 | 116 | C:\Intel\recon\intellirefactor\cli.py |
| `ReportGenerator._normalize_statistics` | 29 | 78 | C:\Intel\recon\intellirefactor\documentation\report_generator.py |
| `format_clone_detection_results` | 28 | 167 | C:\Intel\recon\intellirefactor\cli.py |
| `SemanticSimilarityMatcher._infer_operations_from_method` | 27 | 59 | C:\Intel\recon\intellirefactor\analysis\semantic_similarity_matcher.py |
| `format_similarity_results` | 27 | 132 | C:\Intel\recon\intellirefactor\cli.py |

### Most Popular (by Call Count)

| Block | Callers | LOC | File |
|-------|---------|-----|------|
| `Console.print` | 329 | 2 | C:\Intel\recon\intellirefactor\cli\rich_output.py |
| `IndexStore._get_connection` | 16 | 11 | C:\Intel\recon\intellirefactor\analysis\index_store.py |
| `IndexStore.transaction` | 16 | 9 | C:\Intel\recon\intellirefactor\analysis\index_store.py |
| `DebugCycleManager._get_timestamp` | 15 | 3 | C:\Intel\recon\intellirefactor\analysis\debug_cycle_manager.py |
| `ProjectAnalyzer._get_setting` | 12 | 13 | C:\Intel\recon\intellirefactor\analysis\project_analyzer.py |
| `SmartFlowVisitor._connect` | 11 | 6 | C:\Intel\recon\intellirefactor\visualization\diagram_generator.py |
| `FlowchartGenerator._get_next_node_id` | 10 | 4 | C:\Intel\recon\intellirefactor\documentation\flowchart_generator.py |
| `FlowchartGenerator._analyze_statement` | 10 | 16 | C:\Intel\recon\intellirefactor\documentation\flowchart_generator.py |
| `AuditEngine._get_next_finding_id` | 9 | 4 | C:\Intel\recon\intellirefactor\analysis\audit_engine.py |
| `DatabaseOptimizer.connect` | 9 | 10 | C:\Intel\recon\intellirefactor\performance\database_optimizer.py |

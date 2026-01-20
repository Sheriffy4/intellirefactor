# Документация модулей декомпозиции IntelliRefactor

Автоматически сгенерированная документация для модулей в директории `intellirefactor/analysis/decomposition`.

**Дата генерации:** 2026-01-13 09:07:48
**Система:** Decomposition Documentation Generator

## Обзор модулей

Данная документация содержит подробное описание каждого модуля системы декомпозиции, включая:
- Роль и назначение модуля
- Ключевые функции и возможности  
- Реализованные и отсутствующие функции
- Потенциальные риски и проблемы
- Пересечения с другими модулями
- Рекомендации по улучшению

---

## block_extractor.py

**Роль:** Analyzer (read-only analysis)
**Ключевые функции:**
- visit_ClassDef: Описание отсутствует
- visit_FunctionDef: Описание отсутствует
- visit_AsyncFunctionDef: Описание отсутствует
- extract_from_file: Extract functional blocks from a Python file.
- visit_FunctionDef: Описание отсутствует
- visit_AsyncFunctionDef: Описание отсутствует
- visit_ClassDef: Описание отсутствует
- visit_Lambda: Описание отсутствует
- visit_Call: Описание отсутствует
- visit_Import: Описание отсутствует
- visit_ImportFrom: Описание отсутствует
- visit_Constant: Описание отсутствует
- visit_Name: Описание отсутствует
- visit_FunctionDef: Описание отсутствует
- visit_AsyncFunctionDef: Описание отсутствует
- visit_ClassDef: Описание отсутствует
- visit_Lambda: Описание отсутствует
- visit_ListComp: Описание отсутствует
- visit_SetComp: Описание отсутствует
- visit_DictComp: Описание отсутствует
- visit_GeneratorExp: Описание отсутствует
- visit_Assign: Описание отсутствует
- visit_AnnAssign: Описание отсутствует
- visit_AugAssign: Описание отсутствует
- visit_For: Описание отсутствует
- visit_AsyncFor: Описание отсутствует
- visit_With: Описание отсутствует
- visit_AsyncWith: Описание отсутствует
- visit_ExceptHandler: Описание отсутствует
- visit_NamedExpr: Описание отсутствует
- collect: Описание отсутствует
- visit_FunctionDef: Описание отсутствует
- visit_AsyncFunctionDef: Описание отсутствует
- visit_ClassDef: Описание отсутствует
- visit_Lambda: Описание отсутствует
- visit_ListComp: Описание отсутствует
- visit_SetComp: Описание отсутствует
- visit_DictComp: Описание отсутствует
- visit_GeneratorExp: Описание отсутствует
- visit_If: Описание отсутствует
- visit_Try: Описание отсутствует
- visit_Assign: Описание отсутствует
- visit_AnnAssign: Описание отсутствует
- visit_With: Описание отсутствует
- visit_AsyncWith: Описание отсутствует
- fmt_arg: Описание отсутствует
- resolve_absolute_module: Описание отсутствует
- walk: Описание отсутствует
- Класс FunctionVisitor: AST visitor to extract functions with proper class context.

Uses stacks to track current class c...
  - visit_ClassDef: Описание отсутствует
  - visit_FunctionDef: Описание отсутствует
- Класс FunctionalBlockExtractor: Extracts FunctionalBlock instances from Python source files.

Uses direct AST parsing to extract ...
  - extract_from_file: Extract functional blocks from a Python file.

**Есть (реально реализовано):**
- Основная функциональность модуля
- Объектно-ориентированная структура
- Интеграция с внешними модулями
- Точки входа для использования
- Анализ и извлечение данных

**Нет / нестабильно / заглушка:**
- Документация для 47 публичных функций
- Обработка ошибок (не обнаружена)
- Модульные тесты (не обнаружены)

**Риски/опасности:**
- High number of external dependencies

**Пересечения/дубли с другими модулями:**
- Похожие функции с categorizer: __init__, __init__, _function_to_block~_categorize_single_block, _call_key~_categorize_single_block, _call_key~_categorize_by_name
- Общие зависимости с categorizer: 4 модулей
- Похожие функции с clustering: __init__, _process_function~_determine_recommendation, __init__, _call_key~_calculate_name_similarity, _calculate_cyclomatic~_calculate_name_similarity
- Похожие функции с consolidation_planner: __init__, __init__, _function_to_block~_find_canonical_block, _function_to_block~_find_files_importing_block, _extract_dependencies~_assess_plan_dependencies
- Общие зависимости с consolidation_planner: 4 модулей
- Похожие функции с decomposition_analyzer: __init__, __init__, visit_ClassDef~visit_FunctionDef, visit_ClassDef~visit_AsyncFunctionDef, visit_ClassDef
- Общие зависимости с decomposition_analyzer: 7 модулей
- Похожие функции с fingerprints: __init__, __init__, _extract_module_type_hints~generate_all_fingerprints, _call_key~_categorize_imports, _build_function_signature~_analyze_signature
- Общие зависимости с fingerprints: 6 модулей
- Похожие функции с functional_map: __init__, __init__, _extract_module_type_hints~_apply_block_import_hints, _extract_dependencies~_identify_capabilities, _build_function_signature~_build_call_graph_enhanced
- Общие зависимости с functional_map: 6 модулей
- Похожие функции с models: __init__~__post_init__, __init__~__post_init__, __init__~__post_init__, __init__~__post_init__, __init__~__post_init__
- Похожие функции с normalization: __init__, visit_ClassDef~visit_Call, visit_ClassDef~visit_Name, visit_ClassDef~visit_arg, visit_ClassDef~visit_Attribute
- Аналогичная роль с normalization: Analyzer (read-only analysis)
- Похожие функции с report_generator: __init__, _process_function~_generate_categories_section, _process_function~_generate_capabilities_section, _process_function~_generate_clusters_section, _process_function~_generate_top_blocks_section
- Общие зависимости с report_generator: 5 модулей
- Похожие функции с similarity: __init__, __init__, _function_to_block~_fuzzy_literals_similarity, _call_key~_pair_key, _build_function_signature~_parse_signature
- Общие зависимости с similarity: 4 модулей
- Похожие функции с utils: _scan_nodes~iter_toplevel_import_nodes, _attr_path~to_posix_path, _attr_path~normalize_module_path, _attr_path~to_posix_path, _attr_path~normalize_module_path
- Общие зависимости с utils: 4 модулей

**Что улучшить в первую очередь:**
- Документировать публичные функции
- Добавить модульные тесты
- Рассмотреть разделение на более мелкие модули

---

## categorizer.py

**Роль:** Executor (modifies files/data)
**Ключевые функции:**
- categorize_blocks: Categorize a list of functional blocks.
- Класс FunctionCategorizer: Categorizes functional blocks based on configurable rules.
  - categorize_blocks: Categorize a list of functional blocks.

**Есть (реально реализовано):**
- Основная функциональность модуля
- Объектно-ориентированная структура
- Интеграция с внешними модулями
- Точки входа для использования
- Модификация файлов и данных

**Нет / нестабильно / заглушка:**
- Обработка ошибок (не обнаружена)
- Модульные тесты (не обнаружены)

**Риски/опасности:**
- Критических рисков не выявлено

**Пересечения/дубли с другими модулями:**
- Похожие функции с block_extractor: __init__, __init__, __init__, __init__, __init__
- Общие зависимости с block_extractor: 4 модулей
- Похожие функции с clustering: __init__, _compile_patterns~_has_antonym_patterns, categorize_blocks~cluster_blocks, categorize_blocks~_cluster_category_blocks, _categorize_single_block~_calculate_name_similarity
- Аналогичная роль с clustering: Executor (modifies files/data)
- Похожие функции с consolidation_planner: __init__, _categorize_single_block~_find_canonical_block, _categorize_single_block~_find_files_importing_block, _generate_tags~_generate_consolidation_steps, _generate_tags~_generate_merge_steps
- Общие зависимости с consolidation_planner: 4 модулей
- Аналогичная роль с consolidation_planner: Executor (modifies files/data)
- Похожие функции с decomposition_analyzer: __init__, __init__, _compile_patterns~_collect_locals, _categorize_single_block~_choose_best_block, _categorize_single_block~_canonical_method_call_expr
- Общие зависимости с decomposition_analyzer: 6 модулей
- Аналогичная роль с decomposition_analyzer: Executor (modifies files/data)
- Похожие функции с fingerprints: __init__, _compile_patterns~_analyze_literal_patterns, _categorize_single_block~_categorize_imports, _categorize_by_name~_categorize_imports, _categorize_by_imports~_categorize_imports
- Общие зависимости с fingerprints: 5 модулей
- Аналогичная роль с fingerprints: Executor (modifies files/data)
- Похожие функции с functional_map: __init__, _compile_patterns~_collect_toplevel_class_names, _apply_categorization_rules~_discover_source_files, _apply_categorization_rules~_apply_block_import_hints, _categorize_by_name~_normalize_module_name
- Аналогичная роль с functional_map: Executor (modifies files/data)
- Похожие функции с models: __init__~__post_init__, __init__~__post_init__, __init__~__post_init__, _categorize_single_block~get_block, _categorize_by_name~class_name
- Общие зависимости с models: 4 модулей
- Аналогичная роль с models: Executor (modifies files/data)
- Похожие функции с normalization: __init__, _categorize_by_name~visit_Name
- Похожие функции с report_generator: __init__, _categorize_single_block~_capability_to_dict, _categorize_single_block~_calculate_total_benefit, _categorize_by_name~_capability_to_dict, _categorize_by_name~_calculate_total_benefit
- Общие зависимости с report_generator: 5 модулей
- Аналогичная роль с report_generator: Executor (modifies files/data)
- Похожие функции с similarity: __init__, categorize_blocks~find_similar_blocks, _categorize_by_name~_tokenize_name, _categorize_by_literals~_is_informative_literals, _is_likely_regex~_is_informative_ast
- Общие зависимости с similarity: 5 модулей
- Похожие функции с utils: _is_likely_regex~is_likely_regex, _categorize_by_path~to_posix_path, _categorize_by_path~normalize_module_path
- Общие зависимости с utils: 4 модулей
- Аналогичная роль с utils: Executor (modifies files/data)

**Что улучшить в первую очередь:**
- Добавить модульные тесты
- Добавить механизм резервного копирования
- Рассмотреть разделение на более мелкие модули

---

## clustering.py

**Роль:** Executor (modifies files/data)
**Ключевые функции:**
- cluster_blocks: Group functional blocks into similarity clusters.
- param_count: Описание отсутствует
- Класс FunctionalClusterer: Groups functional blocks into similarity clusters for consolidation.

Uses configurable threshold...
  - cluster_blocks: Group functional blocks into similarity clusters.

**Есть (реально реализовано):**
- Основная функциональность модуля
- Объектно-ориентированная структура
- Интеграция с внешними модулями
- Точки входа для использования
- Модификация файлов и данных

**Нет / нестабильно / заглушка:**
- Документация для 1 публичных функций
- Обработка ошибок (не обнаружена)
- Модульные тесты (не обнаружены)

**Риски/опасности:**
- Performs file system modifications

**Пересечения/дубли с другими модулями:**
- Похожие функции с block_extractor: __init__, __init__, __init__, __init__, __init__
- Похожие функции с categorizer: __init__, cluster_blocks~categorize_blocks, _cluster_category_blocks~categorize_blocks, _generate_proposed_target~_generate_tags, _generate_proposed_target~_generate_semantic_fingerprint
- Аналогичная роль с categorizer: Executor (modifies files/data)
- Похожие функции с consolidation_planner: __init__, _find_similarity_clusters~_find_canonical_block, _find_similarity_clusters~_find_files_importing_block, _create_similarity_cluster~_create_cluster_plan, _find_canonical_candidate~_find_canonical_block
- Аналогичная роль с consolidation_planner: Executor (modifies files/data)
- Похожие функции с decomposition_analyzer: __init__, __init__, _find_similarity_clusters~get_clusters, _find_similarity_clusters~_find_existing_unified_symbol_meta, _find_similarity_clusters~_find_def_node
- Общие зависимости с decomposition_analyzer: 4 модулей
- Аналогичная роль с decomposition_analyzer: Executor (modifies files/data)
- Похожие функции с fingerprints: __init__, _has_antonym_patterns~_analyze_literal_patterns, _calculate_name_similarity~_categorize_imports
- Общие зависимости с fingerprints: 4 модулей
- Аналогичная роль с fingerprints: Executor (modifies files/data)
- Похожие функции с functional_map: __init__, _signatures_compatible~build_file_symbol_table
- Общие зависимости с functional_map: 4 модулей
- Аналогичная роль с functional_map: Executor (modifies files/data)
- Похожие функции с models: __init__~__post_init__, __init__~__post_init__, __init__~__post_init__, _group_by_category~get_blocks_by_category, _signatures_compatible~is_actionable
- Аналогичная роль с models: Executor (modifies files/data)
- Похожие функции с normalization: __init__
- Похожие функции с report_generator: __init__, _cluster_category_blocks~_cluster_to_dict, _cluster_with_threshold~_cluster_to_dict, _determine_recommendation~_generate_categories_section, _determine_recommendation~_generate_capabilities_section
- Аналогичная роль с report_generator: Executor (modifies files/data)
- Похожие функции с similarity: __init__, cluster_blocks~find_similar_blocks, _cluster_category_blocks~find_similar_blocks, _determine_recommendation~_dependency_similarity, _signatures_compatible~_signature_similarity
- Общие зависимости с similarity: 4 модулей
- Аналогичная роль с utils: Executor (modifies files/data)

**Что улучшить в первую очередь:**
- Документировать публичные функции
- Добавить модульные тесты
- Добавить механизм резервного копирования
- Рассмотреть разделение на более мелкие модули

---

## consolidation_planner.py

**Роль:** Executor (modifies files/data)
**Ключевые функции:**
- create_consolidation_plans: Create consolidation plans for all clusters.
- Класс ConsolidationPlanner: Generates consolidation plans for similarity clusters.

Safe consolidation pattern:
1) ADD_NEW_MO...
  - create_consolidation_plans: Create consolidation plans for all clusters.

**Есть (реально реализовано):**
- Основная функциональность модуля
- Объектно-ориентированная структура
- Интеграция с внешними модулями
- Точки входа для использования
- Модификация файлов и данных

**Нет / нестабильно / заглушка:**
- Обработка ошибок (не обнаружена)
- Модульные тесты (не обнаружены)

**Риски/опасности:**
- Performs file system modifications

**Пересечения/дубли с другими модулями:**
- Похожие функции с block_extractor: __init__, __init__, __init__, __init__, __init__
- Общие зависимости с block_extractor: 4 модулей
- Похожие функции с categorizer: __init__, _generate_consolidation_steps~_generate_tags, _generate_consolidation_steps~_generate_semantic_fingerprint, _generate_merge_steps~_generate_tags, _generate_merge_steps~_generate_semantic_fingerprint
- Общие зависимости с categorizer: 4 модулей
- Аналогичная роль с categorizer: Executor (modifies files/data)
- Похожие функции с clustering: __init__, _create_cluster_plan~_create_similarity_cluster, _reserve_unique_target~_generate_proposed_target, _parse_proposed_target~_generate_proposed_target, _generate_consolidation_steps~_generate_proposed_target
- Аналогичная роль с clustering: Executor (modifies files/data)
- Похожие функции с decomposition_analyzer: __init__, __init__, create_consolidation_plans~get_plans, _reserve_unique_target~_resolve_target_module_path, _reserve_unique_target~_resolve_block_file_path
- Общие зависимости с decomposition_analyzer: 4 модулей
- Аналогичная роль с decomposition_analyzer: Executor (modifies files/data)
- Похожие функции с fingerprints: __init__
- Общие зависимости с fingerprints: 4 модулей
- Аналогичная роль с fingerprints: Executor (modifies files/data)
- Похожие функции с functional_map: __init__, _reserve_unique_target~_resolve_call_enhanced, _reserve_unique_target~_resolve_by_context, _reserve_unique_target~_resolve_ambiguous_call, _parse_proposed_target~_path_to_module_name
- Аналогичная роль с functional_map: Executor (modifies files/data)
- Похожие функции с models: __init__~__post_init__, __init__~__post_init__, __init__~__post_init__, _find_canonical_block~get_block, _find_files_importing_block~get_block
- Аналогичная роль с models: Executor (modifies files/data)
- Похожие функции с normalization: __init__
- Похожие функции с report_generator: __init__, create_consolidation_plans~_generate_detailed_plans, _generate_consolidation_steps~_generate_categories_section, _generate_consolidation_steps~_generate_capabilities_section, _generate_consolidation_steps~_generate_clusters_section
- Общие зависимости с report_generator: 4 модулей
- Аналогичная роль с report_generator: Executor (modifies files/data)
- Похожие функции с similarity: __init__, _parse_proposed_target~_pair_key, _parse_proposed_target~_parse_signature, _generate_consolidation_steps~_generate_candidate_pairs, _generate_consolidation_steps~_get_cached_signature
- Общие зависимости с similarity: 4 модулей
- Аналогичная роль с utils: Executor (modifies files/data)

**Что улучшить в первую очередь:**
- Добавить модульные тесты
- Добавить механизм резервного копирования
- Рассмотреть разделение на более мелкие модули

---

## decomposition_analyzer.py

**Роль:** Executor (modifies files/data)
**Ключевые функции:**
- analyze_project: Описание отсутствует
- get_functional_map: Описание отсутствует
- get_clusters: Описание отсутствует
- get_plans: Описание отсутствует
- get_cluster_details: Описание отсутствует
- get_top_opportunities: Описание отсутствует
- export_results: Описание отсутствует
- is_wrapped: Описание отсутствует
- pick: Описание отсутствует
- add_target: Описание отсутствует
- visit_FunctionDef: Описание отсутствует
- visit_AsyncFunctionDef: Описание отсутствует
- visit_ClassDef: Описание отсутствует
- visit_Lambda: Описание отсутствует
- visit_Name: Описание отсутствует
- visit_FunctionDef: Описание отсутствует
- visit_AsyncFunctionDef: Описание отсутствует
- visit_ClassDef: Описание отсутствует
- visit_Lambda: Описание отсутствует
- visit_Global: Описание отсутствует
- visit_Nonlocal: Описание отсутствует
- visit_Name: Описание отсутствует
- visit_Assign: Описание отсутствует
- visit_AnnAssign: Описание отсутствует
- visit_AugAssign: Описание отсутствует
- visit_For: Описание отсутствует
- visit_AsyncFor: Описание отсутствует
- visit_With: Описание отсутствует
- visit_AsyncWith: Описание отсутствует
- visit_ExceptHandler: Описание отсутствует
- visit_comprehension: Описание отсутствует
- visit: Описание отсутствует
- visit_Expr: Описание отсутствует
- visit_arg: Описание отсутствует
- visit_Name: Описание отсутствует
- visit_Call: Описание отсутствует
- visit_FunctionDef: Описание отсутствует
- visit_AsyncFunctionDef: Описание отсутствует
- visit_ClassDef: Описание отсутствует
- visit_Lambda: Описание отсутствует
- Класс DecompositionAnalyzer: Main analyzer for functional decomposition and consolidation.
- Класс V: Описание отсутствует
  - visit_FunctionDef: Описание отсутствует
  - visit_AsyncFunctionDef: Описание отсутствует
  - visit_ClassDef: Описание отсутствует
- Класс V: Описание отсутствует
  - visit_FunctionDef: Описание отсутствует
  - visit_AsyncFunctionDef: Описание отсутствует
  - visit_ClassDef: Описание отсутствует
- Класс Normalizer: Описание отсутствует
  - visit: Описание отсутствует

**Есть (реально реализовано):**
- Основная функциональность модуля
- Объектно-ориентированная структура
- Интеграция с внешними модулями
- Точки входа для использования
- Генерация выходных артефактов
- Модификация файлов и данных

**Нет / нестабильно / заглушка:**
- Документация для 40 публичных функций
- Документация для 3 публичных классов
- Обработка ошибок (не обнаружена)
- Модульные тесты (не обнаружены)

**Риски/опасности:**
- Uses potentially risky module: shutil
- Performs file system modifications
- High number of external dependencies

**Пересечения/дубли с другими модулями:**
- Похожие функции с block_extractor: __init__, __init__, __init__, __init__, __init__
- Общие зависимости с block_extractor: 7 модулей
- Похожие функции с categorizer: __init__, _has_top_level_line~_has_persistence_context, _has_top_level_line~_has_serialization_context, _has_top_level_line~_has_telemetry_evidence, _apply_consolidation~_apply_categorization_rules
- Общие зависимости с categorizer: 6 модулей
- Аналогичная роль с categorizer: Executor (modifies files/data)
- Похожие функции с clustering: __init__, _has_top_level_line~_has_dunder_methods, _has_top_level_line~_has_antonym_patterns, _has_top_level_line~_has_variant_mix, _has_top_level_line~_has_modal_mix
- Общие зависимости с clustering: 4 модулей
- Аналогичная роль с clustering: Executor (modifies files/data)
- Похожие функции с consolidation_planner: __init__, get_plans~create_consolidation_plans, get_top_opportunities~_assess_plan_dependencies, _choose_best_block~_find_canonical_block, _choose_best_block~_find_files_importing_block
- Общие зависимости с consolidation_planner: 4 модулей
- Аналогичная роль с consolidation_planner: Executor (modifies files/data)
- Похожие функции с fingerprints: __init__, _normalized_callable_fingerprint~generate_token_fingerprint, _normalized_callable_fingerprint~generate_semantic_fingerprint, _canonical_method_call_expr~_categorize_imports, _build_call_arguments~generate_all_fingerprints
- Общие зависимости с fingerprints: 9 модулей
- Аналогичная роль с fingerprints: Executor (modifies files/data)
- Похожие функции с functional_map: __init__, get_functional_map~build_functional_map, get_functional_map~update_functional_map, get_top_opportunities~_identify_capabilities, _apply_consolidation~_apply_block_import_hints
- Общие зависимости с functional_map: 7 модулей
- Аналогичная роль с functional_map: Executor (modifies files/data)
- Похожие функции с models: __init__~__post_init__, __init__~__post_init__, __init__~__post_init__, get_functional_map~get_similarity, get_functional_map~get_block
- Общие зависимости с models: 4 модулей
- Аналогичная роль с models: Executor (modifies files/data)
- Похожие функции с normalization: __init__, get_functional_map~_map, _decorator_name~visit_Name, _free_names~_preserve_names, _file_module_name~visit_Name
- Похожие функции с report_generator: __init__, get_plans~_generate_detailed_plans, _apply_consolidation~_generate_categories_section, _apply_consolidation~_generate_capabilities_section, _apply_consolidation~_generate_clusters_section
- Общие зависимости с report_generator: 7 модулей
- Аналогичная роль с report_generator: Executor (modifies files/data)
- Похожие функции с similarity: __init__, _detect_package_root~_dependency_similarity, _parse_file~_pair_key, _parse_file~_parse_signature, _decorator_name~_dependency_similarity
- Общие зависимости с similarity: 6 модулей
- Похожие функции с utils: _choose_canonical_block_id~make_block_id, _choose_canonical_block_id~make_hash_id, _resolve_target_module_path~to_posix_path, _resolve_target_module_path~normalize_module_path, _resolve_block_file_path~to_posix_path
- Общие зависимости с utils: 6 модулей
- Аналогичная роль с utils: Executor (modifies files/data)

**Что улучшить в первую очередь:**
- Документировать публичные функции
- Добавить модульные тесты
- Добавить механизм резервного копирования
- Рассмотреть разделение на более мелкие модули

---

## fingerprints.py

**Роль:** Executor (modifies files/data)
**Ключевые функции:**
- generate_all_fingerprints: Generate all fingerprints and store them in the block.
- generate_ast_hash: Generate normalized AST hash from source.
- generate_ast_hash_from_node: Описание отсутствует
- generate_token_fingerprint: Generate token-based fingerprint using multiset of tokens.

Fixes:
- stable hashing via sha256
- ...
- generate_semantic_fingerprint: Generate semantic fingerprint based on block characteristics.
- Класс FingerprintGenerator: Generates multiple types of fingerprints for functional blocks.
  - generate_all_fingerprints: Generate all fingerprints and store them in the block.
  - generate_ast_hash: Generate normalized AST hash from source.

**Есть (реально реализовано):**
- Основная функциональность модуля
- Объектно-ориентированная структура
- Интеграция с внешними модулями
- Точки входа для использования
- Модификация файлов и данных

**Нет / нестабильно / заглушка:**
- Документация для 1 публичных функций
- Обработка ошибок (не обнаружена)
- Модульные тесты (не обнаружены)

**Риски/опасности:**
- High number of external dependencies

**Пересечения/дубли с другими модулями:**
- Похожие функции с block_extractor: __init__, __init__, __init__, __init__, __init__
- Общие зависимости с block_extractor: 6 модулей
- Похожие функции с categorizer: __init__, generate_token_fingerprint~_generate_semantic_fingerprint, generate_semantic_fingerprint~_generate_semantic_fingerprint, _categorize_imports~_categorize_single_block, _categorize_imports~_categorize_by_name
- Общие зависимости с categorizer: 5 модулей
- Аналогичная роль с categorizer: Executor (modifies files/data)
- Похожие функции с clustering: __init__, _categorize_imports~_calculate_name_similarity, _analyze_literal_patterns~_has_antonym_patterns
- Общие зависимости с clustering: 4 модулей
- Аналогичная роль с clustering: Executor (modifies files/data)
- Похожие функции с consolidation_planner: __init__
- Общие зависимости с consolidation_planner: 4 модулей
- Аналогичная роль с consolidation_planner: Executor (modifies files/data)
- Похожие функции с decomposition_analyzer: __init__, __init__, generate_all_fingerprints~_build_call_arguments, generate_ast_hash_from_node~_find_def_node, generate_token_fingerprint~_normalized_callable_fingerprint
- Общие зависимости с decomposition_analyzer: 9 модулей
- Аналогичная роль с decomposition_analyzer: Executor (modifies files/data)
- Похожие функции с functional_map: __init__, generate_all_fingerprints~_apply_block_import_hints
- Общие зависимости с functional_map: 6 модулей
- Аналогичная роль с functional_map: Executor (modifies files/data)
- Похожие функции с models: __init__~__post_init__, __init__~__post_init__, __init__~__post_init__
- Аналогичная роль с models: Executor (modifies files/data)
- Похожие функции с normalization: __init__, generate_ast_hash~normalize_for_hash
- Похожие функции с report_generator: __init__, generate_all_fingerprints~generate_all_reports, generate_all_fingerprints~generate_json_report, generate_all_fingerprints~generate_catalog_markdown, generate_all_fingerprints~generate_plan_markdown
- Общие зависимости с report_generator: 4 модулей
- Аналогичная роль с report_generator: Executor (modifies files/data)
- Похожие функции с similarity: __init__, _analyze_signature~_parse_signature, _analyze_signature~_get_cached_signature, _analyze_signature~_is_informative_signature
- Общие зависимости с similarity: 4 модулей
- Общие зависимости с utils: 4 модулей
- Аналогичная роль с utils: Executor (modifies files/data)

**Что улучшить в первую очередь:**
- Документировать публичные функции
- Добавить модульные тесты
- Добавить механизм резервного копирования

---

## functional_map.py

**Роль:** Executor (modifies files/data)
**Ключевые функции:**
- build_file_symbol_table: Build symbol table for a file from its AST.

Args:
    tree: AST module
    current_module: Curre...
- scan_nodes: Описание отсутствует
- build_functional_map: Build complete functional map for a project.
- update_functional_map: Описание отсутствует
- Класс FunctionalMapBuilder: Orchestrates the building of a complete functional map for a project.
  - build_functional_map: Build complete functional map for a project.

**Есть (реально реализовано):**
- Основная функциональность модуля
- Объектно-ориентированная структура
- Интеграция с внешними модулями
- Точки входа для использования
- Модификация файлов и данных

**Нет / нестабильно / заглушка:**
- Документация для 2 публичных функций
- Обработка ошибок (не обнаружена)
- Модульные тесты (не обнаружены)

**Риски/опасности:**
- High number of external dependencies

**Пересечения/дубли с другими модулями:**
- Похожие функции с block_extractor: _safe_resolve_path~_attr_path, _safe_resolve_path~_attr_path, _collect_toplevel_class_names~_collect_target, _collect_toplevel_class_names~_collect_subscript_args, _collect_toplevel_class_names~_collect_union_binop
- Общие зависимости с block_extractor: 6 модулей
- Похожие функции с categorizer: _safe_resolve_path~_categorize_by_path, _collect_toplevel_class_names~_compile_patterns, __init__, _normalize_module_name~_categorize_by_name, _discover_source_files~_apply_categorization_rules
- Аналогичная роль с categorizer: Executor (modifies files/data)
- Похожие функции с clustering: build_file_symbol_table~_signatures_compatible, __init__
- Общие зависимости с clustering: 4 модулей
- Аналогичная роль с clustering: Executor (modifies files/data)
- Похожие функции с consolidation_planner: __init__, _path_to_module_name~_parse_proposed_target, _resolve_call_enhanced~_reserve_unique_target, _resolve_by_context~_reserve_unique_target, _resolve_ambiguous_call~_reserve_unique_target
- Аналогичная роль с consolidation_planner: Executor (modifies files/data)
- Похожие функции с decomposition_analyzer: _safe_resolve_path~_resolve_target_module_path, _safe_resolve_path~_resolve_block_file_path, _safe_resolve_path~_module_dotted_from_filepath, _safe_resolve_path~_safe_exact_fingerprint, _collect_toplevel_class_names~_free_names
- Общие зависимости с decomposition_analyzer: 7 модулей
- Аналогичная роль с decomposition_analyzer: Executor (modifies files/data)
- Похожие функции с fingerprints: __init__, _apply_block_import_hints~generate_all_fingerprints
- Общие зависимости с fingerprints: 6 модулей
- Аналогичная роль с fingerprints: Executor (modifies files/data)
- Похожие функции с models: _safe_resolve_path~_normalize_project_path, build_file_symbol_table~is_actionable, __init__~__post_init__, __init__~__post_init__, __init__~__post_init__
- Аналогичная роль с models: Executor (modifies files/data)
- Похожие функции с normalization: _collect_toplevel_class_names~_preserve_names, __init__, _normalize_module_name~visit_Name, build_functional_map~_map, _path_to_module_name~visit_Name
- Похожие функции с report_generator: __init__
- Общие зависимости с report_generator: 5 модулей
- Аналогичная роль с report_generator: Executor (modifies files/data)
- Похожие функции с similarity: __init__, _normalize_module_name~_tokenize_name, _path_to_module_name~_pair_key, _path_to_module_name~_parse_signature, _path_to_module_name~_tokenize_name
- Похожие функции с utils: _safe_resolve_path~to_posix_path, _safe_resolve_path~normalize_module_path, scan_nodes~iter_toplevel_import_nodes
- Аналогичная роль с utils: Executor (modifies files/data)

**Что улучшить в первую очередь:**
- Документировать публичные функции
- Добавить модульные тесты
- Добавить механизм резервного копирования
- Рассмотреть разделение на более мелкие модули

---

## models.py

**Роль:** Executor (modifies files/data)
**Ключевые функции:**
- as_reason: Normalize various reason representations into UnresolvedReason.
- is_actionable: Описание отсутствует
- is_internal: Описание отсутствует
- resolve_call: Resolve a raw call using the symbol table.

Args:
    raw_call: Raw call string like 'dumps' or '...
- is_method: Описание отсутствует
- class_name: Описание отсутствует
- method_name: Описание отсутствует
- block_count: Описание отсутствует
- block_count: Описание отсутствует
- get_similarity: Описание отсутствует
- step_count: Описание отсутствует
- recompute_stats: Описание отсутствует
- get_block: Описание отсутствует
- get_blocks_by_category: Описание отсутствует
- get_blocks_by_module: Описание отсутствует
- default: Описание отсутствует
- Класс RecommendationType: Types of consolidation recommendations.
- Класс RiskLevel: Risk levels for refactoring operations.
- Класс UnresolvedReason: Reasons why a call couldn't be resolved (normalized, lowercase values).
  - is_actionable: Описание отсутствует
  - is_internal: Описание отсутствует
- Класс FileSymbolTable: Per-file symbol table for import alias resolution.

Tracks module aliases and symbol aliases to e...
  - resolve_call: Resolve a raw call using the symbol table.

Args:
    raw_call: Raw call stri...
- Класс EffortClass: Effort classification for refactoring operations.
- Класс PatchStepKind: Types of patch operations.
- Класс ApplicationMode: Modes for applying changes.
- Класс FunctionalBlock: Atomic unit of analysis at method/function level.

Represents a single function or method with al...
  - is_method: Описание отсутствует
  - class_name: Описание отсутствует
- Класс Capability: Group of blocks that solve the same task (even in different modules).
  - block_count: Описание отсутствует
- Класс SimilarityCluster: Cluster of similar blocks within capability/category.
  - block_count: Описание отсутствует
  - get_similarity: Описание отсутствует
- Класс PatchStep: Minimal applicable step that can be rolled back and validated separately.
- Класс CanonicalizationPlan: Plan for canonicalizing a cluster.
  - step_count: Описание отсутствует
- Класс ProjectFunctionalMap: Complete functional map of the project built from atomic blocks.
  - recompute_stats: Описание отсутствует
  - get_block: Описание отсутствует
- Класс DecompositionConfig: Configuration for functional decomposition analysis.
  - default: Описание отсутствует

**Есть (реально реализовано):**
- Основная функциональность модуля
- Объектно-ориентированная структура
- Интеграция с внешними модулями
- Точки входа для использования
- Модификация файлов и данных

**Нет / нестабильно / заглушка:**
- Документация для 14 публичных функций
- Обработка ошибок (не обнаружена)
- Модульные тесты (не обнаружены)

**Риски/опасности:**
- Критических рисков не выявлено

**Пересечения/дубли с другими модулями:**
- Похожие функции с block_extractor: resolve_call~visit_Call, resolve_call~_is_cast_call, resolve_call~_is_cast_call, resolve_call~resolve_absolute_module, _normalize_project_path~_attr_path
- Похожие функции с categorizer: _normalize_project_path~_categorize_by_path, __post_init__~__init__, class_name~_categorize_by_name, method_name~_categorize_by_name, __post_init__~__init__
- Общие зависимости с categorizer: 4 модулей
- Аналогичная роль с categorizer: Executor (modifies files/data)
- Похожие функции с clustering: is_actionable~_signatures_compatible, __post_init__~__init__, block_count~param_count, block_count~param_count, get_similarity~_calculate_name_similarity
- Аналогичная роль с clustering: Executor (modifies files/data)
- Похожие функции с consolidation_planner: __post_init__~__init__, __post_init__~__init__, __post_init__~__init__, get_block~_find_canonical_block, get_block~_find_files_importing_block
- Аналогичная роль с consolidation_planner: Executor (modifies files/data)
- Похожие функции с decomposition_analyzer: is_actionable~is_wrapped, is_internal~is_wrapped, resolve_call~_is_noise_call, resolve_call~visit_Call, _normalize_project_path~_normalized_callable_fingerprint
- Общие зависимости с decomposition_analyzer: 4 модулей
- Аналогичная роль с decomposition_analyzer: Executor (modifies files/data)
- Похожие функции с fingerprints: __post_init__~__init__, __post_init__~__init__, __post_init__~__init__
- Аналогичная роль с fingerprints: Executor (modifies files/data)
- Похожие функции с functional_map: is_actionable~build_file_symbol_table, resolve_call~_is_external_call, resolve_call~_is_dynamic_attribute_call, resolve_call~_resolve_ambiguous_call, _normalize_project_path~_safe_resolve_path
- Аналогичная роль с functional_map: Executor (modifies files/data)
- Похожие функции с normalization: resolve_call~visit_Call, __post_init__~__init__, class_name~visit_Name, method_name~visit_Name, __post_init__~__init__
- Похожие функции с report_generator: __post_init__~__init__, __post_init__~__init__, __post_init__~__init__, default~_json_default
- Общие зависимости с report_generator: 5 модулей
- Аналогичная роль с report_generator: Executor (modifies files/data)
- Похожие функции с similarity: is_actionable~is_informative, is_internal~_jaccard_no_signal, is_internal~is_informative, __post_init__~__init__, is_method~is_informative
- Похожие функции с utils: is_actionable~is_likely_regex, is_internal~is_likely_regex, _normalize_project_path~to_posix_path, _normalize_project_path~normalize_module_path, is_method~is_likely_regex
- Общие зависимости с utils: 4 модулей
- Аналогичная роль с utils: Executor (modifies files/data)

**Что улучшить в первую очередь:**
- Документировать публичные функции
- Добавить модульные тесты
- Добавить механизм резервного копирования
- Рассмотреть разделение на более мелкие модули

---

## normalization.py

**Роль:** Analyzer (read-only analysis)
**Ключевые функции:**
- normalize_for_hash: Описание отсутствует
- visit_Call: Описание отсутствует
- visit_Name: Описание отсутствует
- visit_arg: Описание отсутствует
- visit_Attribute: Описание отсутствует
- visit_FunctionDef: Описание отсутствует
- visit_AsyncFunctionDef: Описание отсутствует
- visit_ClassDef: Описание отсутствует
- visit_Constant: Описание отсутствует
- Класс APIPreservingASTNormalizer: Описание отсутствует

**Есть (реально реализовано):**
- Основная функциональность модуля
- Объектно-ориентированная структура
- Интеграция с внешними модулями
- Точки входа для использования
- Анализ и извлечение данных

**Нет / нестабильно / заглушка:**
- Документация для 9 публичных функций
- Документация для 1 публичных классов
- Обработка ошибок (не обнаружена)
- Модульные тесты (не обнаружены)

**Риски/опасности:**
- Критических рисков не выявлено

**Пересечения/дубли с другими модулями:**
- Похожие функции с block_extractor: __init__, __init__, __init__, __init__, __init__
- Аналогичная роль с block_extractor: Analyzer (read-only analysis)
- Похожие функции с categorizer: __init__, visit_Name~_categorize_by_name
- Похожие функции с clustering: __init__
- Похожие функции с consolidation_planner: __init__
- Похожие функции с decomposition_analyzer: __init__, __init__, _preserve_names~_free_names, _map~get_functional_map, visit_Call~_is_noise_call
- Похожие функции с fingerprints: normalize_for_hash~generate_ast_hash, __init__
- Похожие функции с functional_map: __init__, _preserve_names~_collect_toplevel_class_names, _map~build_functional_map, _map~update_functional_map, visit_Call~_is_external_call
- Похожие функции с models: __init__~__post_init__, __init__~__post_init__, __init__~__post_init__, visit_Call~resolve_call, visit_Name~class_name
- Похожие функции с report_generator: __init__
- Похожие функции с similarity: __init__, visit_Name~_tokenize_name
- Похожие функции с utils: normalize_for_hash~normalize_module_path

**Что улучшить в первую очередь:**
- Документировать публичные функции
- Добавить модульные тесты
- Рассмотреть разделение на более мелкие модули

---

## report_generator.py

**Роль:** Executor (modifies files/data)
**Ключевые функции:**
- generate_all_reports: Generate all report formats.
- generate_json_report: Generate machine-readable JSON report.
- generate_catalog_markdown: Generate human-readable catalog in Markdown format.
- generate_plan_markdown: Generate consolidation plan in Markdown format.
- generate_mermaid_diagram: Generate Mermaid diagram showing functional relationships.
- generate_summary_report: Generate executive summary report.
- Класс DecompositionReportGenerator: Generates comprehensive reports for functional decomposition analysis.
  - generate_all_reports: Generate all report formats.
  - generate_json_report: Generate machine-readable JSON report.

**Есть (реально реализовано):**
- Основная функциональность модуля
- Объектно-ориентированная структура
- Интеграция с внешними модулями
- Точки входа для использования
- Генерация выходных артефактов
- Модификация файлов и данных

**Нет / нестабильно / заглушка:**
- Обработка ошибок (не обнаружена)
- Модульные тесты (не обнаружены)

**Риски/опасности:**
- Критических рисков не выявлено

**Пересечения/дубли с другими модулями:**
- Похожие функции с block_extractor: __init__, __init__, __init__, __init__, __init__
- Общие зависимости с block_extractor: 5 модулей
- Похожие функции с categorizer: __init__, generate_all_reports~_categorize_by_imports, _capability_to_dict~_categorize_single_block, _capability_to_dict~_categorize_by_name, _capability_to_dict~_categorize_by_imports
- Общие зависимости с categorizer: 5 модулей
- Аналогичная роль с categorizer: Executor (modifies files/data)
- Похожие функции с clustering: __init__, _capability_to_dict~_calculate_name_similarity, _cluster_to_dict~_cluster_category_blocks, _cluster_to_dict~_cluster_with_threshold, _generate_categories_section~_determine_recommendation
- Аналогичная роль с clustering: Executor (modifies files/data)
- Похожие функции с consolidation_planner: __init__, _plan_to_dict~_plan_priority_key, _generate_categories_section~_generate_consolidation_steps, _generate_categories_section~_generate_merge_steps, _generate_categories_section~_generate_extract_base_steps
- Общие зависимости с consolidation_planner: 4 модулей
- Аналогичная роль с consolidation_planner: Executor (modifies files/data)
- Похожие функции с decomposition_analyzer: __init__, __init__, _capability_to_dict~_canonical_method_call_expr, _capability_to_dict~_calculate_cluster_benefit, _generate_categories_section~_apply_consolidation
- Общие зависимости с decomposition_analyzer: 7 модулей
- Аналогичная роль с decomposition_analyzer: Executor (modifies files/data)
- Похожие функции с fingerprints: __init__, generate_all_reports~generate_all_fingerprints, generate_all_reports~generate_ast_hash, generate_all_reports~generate_ast_hash_from_node, generate_all_reports~generate_token_fingerprint
- Общие зависимости с fingerprints: 4 модулей
- Аналогичная роль с fingerprints: Executor (modifies files/data)
- Похожие функции с functional_map: __init__
- Общие зависимости с functional_map: 5 модулей
- Аналогичная роль с functional_map: Executor (modifies files/data)
- Похожие функции с models: __init__~__post_init__, __init__~__post_init__, __init__~__post_init__, _json_default~default
- Общие зависимости с models: 5 модулей
- Аналогичная роль с models: Executor (modifies files/data)
- Похожие функции с normalization: __init__
- Похожие функции с similarity: __init__, _generate_categories_section~_generate_candidate_pairs, _generate_categories_section~_get_cached_signature, _generate_categories_section~_get_cached_name_tokens, _generate_categories_section~_get_cached_deps_set
- Общие зависимости с similarity: 4 модулей
- Аналогичная роль с utils: Executor (modifies files/data)

**Что улучшить в первую очередь:**
- Добавить модульные тесты
- Добавить механизм резервного копирования
- Рассмотреть разделение на более мелкие модули

---

## similarity.py

**Роль:** Mixed (both analysis and modification)
**Ключевые функции:**
- calculate_similarity: Calculate overall similarity score between two functional blocks (0..1).
- calculate_similarity_matrix: Calculate similarity matrix.

Return format: {(min_id, max_id): similarity}, one entry per pair.
- find_similar_blocks: Find pairs of blocks with similarity above threshold.
- is_informative: Описание отсутствует
- add_bucket_pairs: Описание отсутствует
- Класс SimilarityCalculator: Calculates similarity scores between functional blocks using multiple channels.

Weighted similar...
  - calculate_similarity: Calculate overall similarity score between two functional blocks (0..1).
  - calculate_similarity_matrix: Calculate similarity matrix.

Return format: {(min_id, max_id): similarity}, ...

**Есть (реально реализовано):**
- Основная функциональность модуля
- Объектно-ориентированная структура
- Интеграция с внешними модулями
- Точки входа для использования

**Нет / нестабильно / заглушка:**
- Документация для 2 публичных функций
- Обработка ошибок (не обнаружена)
- Модульные тесты (не обнаружены)

**Риски/опасности:**
- Критических рисков не выявлено

**Пересечения/дубли с другими модулями:**
- Похожие функции с block_extractor: _pair_key~_call_key, _pair_key~_call_key, __init__, __init__, __init__
- Общие зависимости с block_extractor: 4 модулей
- Похожие функции с categorizer: __init__, find_similar_blocks~categorize_blocks, _tokenize_name~_categorize_by_name, _generate_candidate_pairs~_generate_tags, _generate_candidate_pairs~_generate_semantic_fingerprint
- Общие зависимости с categorizer: 5 модулей
- Похожие функции с clustering: __init__, calculate_similarity~_calculate_name_similarity, find_similar_blocks~cluster_blocks, find_similar_blocks~_cluster_category_blocks, _ast_shape_similarity~_assess_risk_level
- Общие зависимости с clustering: 4 модулей
- Похожие функции с consolidation_planner: _pair_key~_parse_proposed_target, _pair_key~_plan_priority_key, __init__, _ast_shape_similarity~_assess_plan_dependencies, _parse_signature~_parse_proposed_target
- Общие зависимости с consolidation_planner: 4 модулей
- Похожие функции с decomposition_analyzer: _pair_key~_parse_file, __init__, __init__, _signature_similarity~_signature_shape, _dependency_similarity~_detect_package_root
- Общие зависимости с decomposition_analyzer: 6 модулей
- Похожие функции с fingerprints: __init__, _parse_signature~_analyze_signature, _get_cached_signature~_analyze_signature, _is_informative_signature~_analyze_signature
- Общие зависимости с fingerprints: 4 модулей
- Похожие функции с functional_map: _pair_key~_path_to_module_name, __init__, _parse_signature~_path_to_module_name, _tokenize_name~_normalize_module_name, _tokenize_name~_path_to_module_name
- Похожие функции с models: __init__~__post_init__, __init__~__post_init__, __init__~__post_init__, calculate_similarity~get_similarity, _ast_shape_similarity~get_similarity
- Похожие функции с normalization: __init__, _tokenize_name~visit_Name
- Похожие функции с report_generator: __init__, calculate_similarity_matrix~_generate_priority_matrix, _generate_candidate_pairs~_generate_categories_section, _generate_candidate_pairs~_generate_capabilities_section, _generate_candidate_pairs~_generate_clusters_section
- Общие зависимости с report_generator: 4 модулей
- Похожие функции с utils: is_informative~is_likely_regex

**Что улучшить в первую очередь:**
- Документировать публичные функции
- Добавить модульные тесты
- Рассмотреть разделение на более мелкие модули

---

## utils.py

**Роль:** Executor (modifies files/data)
**Ключевые функции:**
- iter_toplevel_import_nodes: Yield nodes that may contain imports, including nested Try/If blocks.

FIX: recursion (previous i...
- is_likely_regex: Strict check if string is likely a regex pattern (avoid false positives).
- to_posix_path: Описание отсутствует
- normalize_module_path: Remove known project roots from module paths.
Example:
  normalize_module_path("intellirefactor.a...
- make_block_id: Unified stable ID strategy:
  {module_or_relpath}:{qualname}:{lineno}

- Prefer module when relia...
- make_hash_id: Stable short hash id (sha256).

**Есть (реально реализовано):**
- Основная функциональность модуля
- Интеграция с внешними модулями
- Модификация файлов и данных

**Нет / нестабильно / заглушка:**
- Документация для 1 публичных функций
- Обработка ошибок (не обнаружена)
- Модульные тесты (не обнаружены)

**Риски/опасности:**
- Критических рисков не выявлено

**Пересечения/дубли с другими модулями:**
- Похожие функции с block_extractor: iter_toplevel_import_nodes~_scan_nodes, to_posix_path~_attr_path, to_posix_path~_attr_path, normalize_module_path~_attr_path, normalize_module_path~_attr_path
- Общие зависимости с block_extractor: 4 модулей
- Похожие функции с categorizer: is_likely_regex~_is_likely_regex, to_posix_path~_categorize_by_path, normalize_module_path~_categorize_by_path
- Общие зависимости с categorizer: 4 модулей
- Аналогичная роль с categorizer: Executor (modifies files/data)
- Аналогичная роль с clustering: Executor (modifies files/data)
- Аналогичная роль с consolidation_planner: Executor (modifies files/data)
- Похожие функции с decomposition_analyzer: is_likely_regex~is_wrapped, to_posix_path~_resolve_target_module_path, to_posix_path~_resolve_block_file_path, to_posix_path~_module_dotted_from_filepath, normalize_module_path~_resolve_target_module_path
- Общие зависимости с decomposition_analyzer: 6 модулей
- Аналогичная роль с decomposition_analyzer: Executor (modifies files/data)
- Общие зависимости с fingerprints: 4 модулей
- Аналогичная роль с fingerprints: Executor (modifies files/data)
- Похожие функции с functional_map: iter_toplevel_import_nodes~scan_nodes, to_posix_path~_safe_resolve_path, normalize_module_path~_safe_resolve_path
- Аналогичная роль с functional_map: Executor (modifies files/data)
- Похожие функции с models: is_likely_regex~is_actionable, is_likely_regex~is_internal, is_likely_regex~is_method, to_posix_path~_normalize_project_path, normalize_module_path~_normalize_project_path
- Общие зависимости с models: 4 модулей
- Аналогичная роль с models: Executor (modifies files/data)
- Похожие функции с normalization: normalize_module_path~normalize_for_hash
- Аналогичная роль с report_generator: Executor (modifies files/data)
- Похожие функции с similarity: is_likely_regex~is_informative

**Что улучшить в первую очередь:**
- Документировать публичные функции
- Добавить модульные тесты
- Добавить механизм резервного копирования

---


# Экспертные вопросы по модулям

Результаты оценки каждого модуля по 10 универсальным вопросам:

## block_extractor.py

**1. Тип модуля (анализатор/исполнитель):** Тип: Analyzer (read-only analysis)

**2. Точки входа (CLI/API/классы/функции):** Точки входа: Class: FunctionVisitor, Class: FunctionalBlockExtractor

**3. Создаваемые артефакты:** Артефакты: Не создает файловых артефактов

**4. Поддержка dry-run режима:** Dry-run: Не обнаружено явной поддержки

**5. Форматы вывода:** Форматы вывода: Не определены

**6. Функции безопасности:** Безопасность: Механизмы безопасности не обнаружены

**7. Правила фильтрации:** Фильтрация: Не обнаружено явных правил фильтрации

**8. Дублирование функциональности:** Дублирование: 20 потенциальных пересечений обнаружено

**9. Оценка стабильности:** Стабильность: Средняя (есть области для улучшения)

**10. Частые причины сбоев:** Частые сбои: Файловые операции, Множественные зависимости

---

## categorizer.py

**1. Тип модуля (анализатор/исполнитель):** Тип: Executor (modifies files/data)

**2. Точки входа (CLI/API/классы/функции):** Точки входа: Class: FunctionCategorizer

**3. Создаваемые артефакты:** Артефакты: Не создает файловых артефактов

**4. Поддержка dry-run режима:** Dry-run: Не обнаружено явной поддержки

**5. Форматы вывода:** Форматы вывода: Не определены

**6. Функции безопасности:** Безопасность: Валидация данных

**7. Правила фильтрации:** Фильтрация: Не обнаружено явных правил фильтрации

**8. Дублирование функциональности:** Дублирование: 27 потенциальных пересечений обнаружено

**9. Оценка стабильности:** Стабильность: Высокая (хорошо документирован и структурирован)

**10. Частые причины сбоев:** Частые сбои: Низкий риск сбоев

---

## clustering.py

**1. Тип модуля (анализатор/исполнитель):** Тип: Executor (modifies files/data)

**2. Точки входа (CLI/API/классы/функции):** Точки входа: Class: FunctionalClusterer

**3. Создаваемые артефакты:** Артефакты: Не создает файловых артефактов

**4. Поддержка dry-run режима:** Dry-run: Не обнаружено явной поддержки

**5. Форматы вывода:** Форматы вывода: Не определены

**6. Функции безопасности:** Безопасность: Механизмы безопасности не обнаружены

**7. Правила фильтрации:** Фильтрация: Не обнаружено явных правил фильтрации

**8. Дублирование функциональности:** Дублирование: 22 потенциальных пересечений обнаружено

**9. Оценка стабильности:** Стабильность: Высокая (хорошо документирован и структурирован)

**10. Частые причины сбоев:** Частые сбои: Низкий риск сбоев

---

## consolidation_planner.py

**1. Тип модуля (анализатор/исполнитель):** Тип: Executor (modifies files/data)

**2. Точки входа (CLI/API/классы/функции):** Точки входа: Class: ConsolidationPlanner

**3. Создаваемые артефакты:** Артефакты: Не создает файловых артефактов

**4. Поддержка dry-run режима:** Dry-run: Не обнаружено явной поддержки

**5. Форматы вывода:** Форматы вывода: Не определены

**6. Функции безопасности:** Безопасность: Механизмы безопасности не обнаружены

**7. Правила фильтрации:** Фильтрация: Не обнаружено явных правил фильтрации

**8. Дублирование функциональности:** Дублирование: 24 потенциальных пересечений обнаружено

**9. Оценка стабильности:** Стабильность: Высокая (хорошо документирован и структурирован)

**10. Частые причины сбоев:** Частые сбои: Файловые операции

---

## decomposition_analyzer.py

**1. Тип модуля (анализатор/исполнитель):** Тип: Executor (modifies files/data)

**2. Точки входа (CLI/API/классы/функции):** Точки входа: Class: DecompositionAnalyzer, Class: V, Class: V

**3. Создаваемые артефакты:** Артефакты: Output files

**4. Поддержка dry-run режима:** Dry-run: Не обнаружено явной поддержки

**5. Форматы вывода:** Форматы вывода: JSON

**6. Функции безопасности:** Безопасность: Резервное копирование, Валидация данных

**7. Правила фильтрации:** Фильтрация: Не обнаружено явных правил фильтрации

**8. Дублирование функциональности:** Дублирование: 29 потенциальных пересечений обнаружено

**9. Оценка стабильности:** Стабильность: Низкая (множественные риски и проблемы)

**10. Частые причины сбоев:** Частые сбои: Системные операции, Файловые операции, Множественные зависимости

---

## fingerprints.py

**1. Тип модуля (анализатор/исполнитель):** Тип: Executor (modifies files/data)

**2. Точки входа (CLI/API/классы/функции):** Точки входа: Class: FingerprintGenerator

**3. Создаваемые артефакты:** Артефакты: Не создает файловых артефактов

**4. Поддержка dry-run режима:** Dry-run: Не обнаружено явной поддержки

**5. Форматы вывода:** Форматы вывода: Не определены

**6. Функции безопасности:** Безопасность: Механизмы безопасности не обнаружены

**7. Правила фильтрации:** Фильтрация: Не обнаружено явных правил фильтрации

**8. Дублирование функциональности:** Дублирование: 27 потенциальных пересечений обнаружено

**9. Оценка стабильности:** Стабильность: Высокая (хорошо документирован и структурирован)

**10. Частые причины сбоев:** Частые сбои: Множественные зависимости

---

## functional_map.py

**1. Тип модуля (анализатор/исполнитель):** Тип: Executor (modifies files/data)

**2. Точки входа (CLI/API/классы/функции):** Точки входа: Class: FunctionalMapBuilder

**3. Создаваемые артефакты:** Артефакты: Не создает файловых артефактов

**4. Поддержка dry-run режима:** Dry-run: Не обнаружено явной поддержки

**5. Форматы вывода:** Форматы вывода: Не определены

**6. Функции безопасности:** Безопасность: Механизмы безопасности не обнаружены

**7. Правила фильтрации:** Фильтрация: Не обнаружено явных правил фильтрации

**8. Дублирование функциональности:** Дублирование: 24 потенциальных пересечений обнаружено

**9. Оценка стабильности:** Стабильность: Средняя (есть области для улучшения)

**10. Частые причины сбоев:** Частые сбои: Файловые операции, Множественные зависимости

---

## models.py

**1. Тип модуля (анализатор/исполнитель):** Тип: Executor (modifies files/data)

**2. Точки входа (CLI/API/классы/функции):** Точки входа: Class: RecommendationType, Class: RiskLevel, Class: UnresolvedReason

**3. Создаваемые артефакты:** Артефакты: Не создает файловых артефактов

**4. Поддержка dry-run режима:** Dry-run: Не обнаружено явной поддержки

**5. Форматы вывода:** Форматы вывода: Не определены

**6. Функции безопасности:** Безопасность: Механизмы безопасности не обнаружены

**7. Правила фильтрации:** Фильтрация: Не обнаружено явных правил фильтрации

**8. Дублирование функциональности:** Дублирование: 23 потенциальных пересечений обнаружено

**9. Оценка стабильности:** Стабильность: Высокая (хорошо документирован и структурирован)

**10. Частые причины сбоев:** Частые сбои: Низкий риск сбоев

---

## normalization.py

**1. Тип модуля (анализатор/исполнитель):** Тип: Analyzer (read-only analysis)

**2. Точки входа (CLI/API/классы/функции):** Точки входа: Class: APIPreservingASTNormalizer

**3. Создаваемые артефакты:** Артефакты: Не создает файловых артефактов

**4. Поддержка dry-run режима:** Dry-run: Не обнаружено явной поддержки

**5. Форматы вывода:** Форматы вывода: Не определены

**6. Функции безопасности:** Безопасность: Механизмы безопасности не обнаружены

**7. Правила фильтрации:** Фильтрация: Не обнаружено явных правил фильтрации

**8. Дублирование функциональности:** Дублирование: 12 потенциальных пересечений обнаружено

**9. Оценка стабильности:** Стабильность: Высокая (хорошо документирован и структурирован)

**10. Частые причины сбоев:** Частые сбои: Низкий риск сбоев

---

## report_generator.py

**1. Тип модуля (анализатор/исполнитель):** Тип: Executor (modifies files/data)

**2. Точки входа (CLI/API/классы/функции):** Точки входа: Class: DecompositionReportGenerator

**3. Создаваемые артефакты:** Артефакты: Report files

**4. Поддержка dry-run режима:** Dry-run: Не обнаружено явной поддержки

**5. Форматы вывода:** Форматы вывода: JSON

**6. Функции безопасности:** Безопасность: Механизмы безопасности не обнаружены

**7. Правила фильтрации:** Фильтрация: Не обнаружено явных правил фильтрации

**8. Дублирование функциональности:** Дублирование: 26 потенциальных пересечений обнаружено

**9. Оценка стабильности:** Стабильность: Высокая (хорошо документирован и структурирован)

**10. Частые причины сбоев:** Частые сбои: Низкий риск сбоев

---

## similarity.py

**1. Тип модуля (анализатор/исполнитель):** Тип: Mixed (both analysis and modification)

**2. Точки входа (CLI/API/классы/функции):** Точки входа: Class: SimilarityCalculator

**3. Создаваемые артефакты:** Артефакты: Не создает файловых артефактов

**4. Поддержка dry-run режима:** Dry-run: Не обнаружено явной поддержки

**5. Форматы вывода:** Форматы вывода: Не определены

**6. Функции безопасности:** Безопасность: Механизмы безопасности не обнаружены

**7. Правила фильтрации:** Фильтрация: Не обнаружено явных правил фильтрации

**8. Дублирование функциональности:** Дублирование: 18 потенциальных пересечений обнаружено

**9. Оценка стабильности:** Стабильность: Высокая (хорошо документирован и структурирован)

**10. Частые причины сбоев:** Частые сбои: Низкий риск сбоев

---

## utils.py

**1. Тип модуля (анализатор/исполнитель):** Тип: Executor (modifies files/data)

**2. Точки входа (CLI/API/классы/функции):** Точки входа: Не обнаружены явные точки входа

**3. Создаваемые артефакты:** Артефакты: Не создает файловых артефактов

**4. Поддержка dry-run режима:** Dry-run: Не обнаружено явной поддержки

**5. Форматы вывода:** Форматы вывода: Не определены

**6. Функции безопасности:** Безопасность: Механизмы безопасности не обнаружены

**7. Правила фильтрации:** Фильтрация: Не обнаружено явных правил фильтрации

**8. Дублирование функциональности:** Дублирование: 20 потенциальных пересечений обнаружено

**9. Оценка стабильности:** Стабильность: Высокая (хорошо документирован и структурирован)

**10. Частые причины сбоев:** Частые сбои: Низкий риск сбоев

---


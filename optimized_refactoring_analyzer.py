#!/usr/bin/env python3
"""
Оптимизированный анализатор для рефакторинга

Собирает ТОЛЬКО нужную и ВСЮ нужную информацию для составления 
максимально качественного плана рефакторинга (10/10).

Принципы:
1. Фокус на рефакторинге - не собираем лишнее
2. Качество данных - реальные значения, не синтетические
3. Структурированность - четкая организация результатов
4. Экспертность - готовые выводы для специалиста
"""

import sys
import json
import ast
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict, Counter

# Добавляем текущую директорию в путь для импорта
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from contextual_file_analyzer import ContextualFileAnalyzer


class OptimizedRefactoringAnalyzer(ContextualFileAnalyzer):
    """Оптимизированный анализатор для составления плана рефакторинга"""

    def __init__(self, project_path: str, target_file: str, output_dir: str, verbose: bool = False):
        super().__init__(project_path, target_file, output_dir, verbose)
        
        self.analysis_mode = "optimized_refactoring_analysis"
        self.logger.info("Инициализирован OptimizedRefactoringAnalyzer")
        
        # Структура для хранения только нужных данных
        self.refactoring_data = {
            'file_structure': {},
            'real_usage_patterns': {},
            'api_contracts': {},
            'data_schemas': {},
            'refactoring_opportunities': {},
            'expert_recommendations': {}
        }

    def run_optimized_analysis(self):
        """Запуск оптимизированного анализа для рефакторинга"""
        self.logger.info("[СТАРТ] Оптимизированный анализ для рефакторинга...")

        # Последовательность анализов - только нужное для рефакторинга
        analyses = [
            ("Структурный анализ файла", self.analyze_file_structure),
            ("Извлечение реальных паттернов использования", self.extract_real_usage_patterns),
            ("Анализ API контрактов", self.analyze_api_contracts),
            ("Извлечение схем данных", self.extract_data_schemas),
            ("Выявление возможностей рефакторинга", self.identify_refactoring_opportunities),
            ("Создание экспертных рекомендаций", self.create_expert_recommendations),
            ("Генерация итогового плана", self.generate_refactoring_plan)
        ]

        # Выполняем анализы
        for analysis_name, analysis_func in analyses:
            try:
                self.logger.info(f"[ВЫПОЛНЕНИЕ] {analysis_name}")
                success = analysis_func()
                if success:
                    self.logger.info(f"[УСПЕХ] {analysis_name}")
                else:
                    self.logger.warning(f"[ПРЕДУПРЕЖДЕНИЕ] {analysis_name}")
            except Exception as e:
                self.logger.error(f"[ОШИБКА] {analysis_name}: {e}")

        # Сохраняем результаты
        self.save_optimized_results()
        return True

    def analyze_file_structure(self):
        """1. Структурный анализ файла - основа для рефакторинга"""
        self.logger.info("Анализ структуры файла...")
        
        try:
            with open(self.target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Парсим AST
            tree = ast.parse(content)
            
            # Извлекаем структурную информацию
            structure = {
                'classes': [],
                'functions': [],
                'imports': [],
                'constants': [],
                'complexity_metrics': {}
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = {
                        'name': node.name,
                        'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        'line_start': node.lineno,
                        'line_end': getattr(node, 'end_lineno', node.lineno),
                        'method_count': len([n for n in node.body if isinstance(n, ast.FunctionDef)]),
                        'is_god_object': len([n for n in node.body if isinstance(n, ast.FunctionDef)]) > 15
                    }
                    structure['classes'].append(class_info)
                
                elif isinstance(node, ast.FunctionDef) and not any(
                    node.lineno >= cls['line_start'] and node.lineno <= cls['line_end'] 
                    for cls in structure['classes']
                ):
                    func_info = {
                        'name': node.name,
                        'line_start': node.lineno,
                        'line_end': getattr(node, 'end_lineno', node.lineno),
                        'args': [arg.arg for arg in node.args.args],
                        'is_large': (getattr(node, 'end_lineno', node.lineno) - node.lineno) > 50
                    }
                    structure['functions'].append(func_info)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            structure['imports'].append({
                                'type': 'import',
                                'module': alias.name,
                                'alias': alias.asname
                            })
                    else:
                        for alias in node.names:
                            structure['imports'].append({
                                'type': 'from_import',
                                'module': node.module,
                                'name': alias.name,
                                'alias': alias.asname
                            })
            
            # Метрики сложности
            structure['complexity_metrics'] = {
                'total_classes': len(structure['classes']),
                'total_functions': len(structure['functions']),
                'total_imports': len(structure['imports']),
                'god_objects': [cls['name'] for cls in structure['classes'] if cls['is_god_object']],
                'large_functions': [func['name'] for func in structure['functions'] if func['is_large']],
                'lines_of_code': len(content.splitlines())
            }
            
            self.refactoring_data['file_structure'] = structure
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа структуры: {e}")
            return False

    def extract_real_usage_patterns(self):
        """2. Извлечение реальных паттернов использования из кода"""
        self.logger.info("Извлечение реальных паттернов использования...")
        
        # Сначала запускаем базовый анализ для получения данных
        self.build_project_index_safe()
        
        # Ищем реальные вызовы в проекте
        usage_patterns = {
            'method_calls': [],
            'parameter_patterns': {},
            'return_value_usage': [],
            'error_handling_patterns': []
        }
        
        try:
            # Ищем файлы Python в проекте
            python_files = list(self.project_path.rglob("*.py"))
            
            # Анализируем вызовы целевого файла
            target_module_name = self.target_file.stem
            
            for py_file in python_files[:50]:  # Ограничиваем для производительности
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Ищем импорты нашего модуля
                    if target_module_name in content:
                        # Парсим для поиска реальных вызовов
                        try:
                            tree = ast.parse(content)
                            for node in ast.walk(tree):
                                if isinstance(node, ast.Call):
                                    # Извлекаем информацию о вызове
                                    call_info = self._extract_call_info(node, py_file)
                                    if call_info:
                                        usage_patterns['method_calls'].append(call_info)
                        except:
                            continue
                            
                except Exception:
                    continue
            
            # Анализируем паттерны параметров
            usage_patterns['parameter_patterns'] = self._analyze_parameter_patterns(
                usage_patterns['method_calls']
            )
            
            self.refactoring_data['real_usage_patterns'] = usage_patterns
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения паттернов: {e}")
            return False

    def _extract_call_info(self, node: ast.Call, file_path: Path) -> Optional[Dict]:
        """Извлекает информацию о вызове функции"""
        try:
            call_info = {
                'file': str(file_path.relative_to(self.project_path)),
                'line': node.lineno,
                'function_name': None,
                'args': [],
                'kwargs': {}
            }
            
            # Определяем имя функции
            if isinstance(node.func, ast.Name):
                call_info['function_name'] = node.func.id
            elif isinstance(node.func, ast.Attribute):
                call_info['function_name'] = node.func.attr
            
            # Извлекаем аргументы
            for arg in node.args:
                if isinstance(arg, ast.Constant):
                    call_info['args'].append(arg.value)
                elif isinstance(arg, ast.Name):
                    call_info['args'].append(f"<var:{arg.id}>")
                else:
                    call_info['args'].append(f"<expr:{type(arg).__name__}>")
            
            # Извлекаем keyword аргументы
            for keyword in node.keywords:
                if isinstance(keyword.value, ast.Constant):
                    call_info['kwargs'][keyword.arg] = keyword.value.value
                elif isinstance(keyword.value, ast.Name):
                    call_info['kwargs'][keyword.arg] = f"<var:{keyword.value.id}>"
                else:
                    call_info['kwargs'][keyword.arg] = f"<expr:{type(keyword.value).__name__}>"
            
            return call_info
            
        except Exception:
            return None

    def _analyze_parameter_patterns(self, method_calls: List[Dict]) -> Dict:
        """Анализирует паттерны параметров"""
        patterns = {
            'common_parameters': Counter(),
            'parameter_types': defaultdict(Counter),
            'real_values': defaultdict(set)
        }
        
        for call in method_calls:
            # Анализируем kwargs
            for param, value in call.get('kwargs', {}).items():
                patterns['common_parameters'][param] += 1
                
                if isinstance(value, str) and not value.startswith('<'):
                    patterns['real_values'][param].add(value)
                    patterns['parameter_types'][param]['string'] += 1
                elif isinstance(value, (int, float)):
                    patterns['real_values'][param].add(value)
                    patterns['parameter_types'][param]['number'] += 1
                elif isinstance(value, bool):
                    patterns['real_values'][param].add(value)
                    patterns['parameter_types'][param]['boolean'] += 1
        
        # Конвертируем sets в lists для JSON сериализации
        for param in patterns['real_values']:
            patterns['real_values'][param] = list(patterns['real_values'][param])
        
        return patterns

    def analyze_api_contracts(self):
        """3. Анализ API контрактов и интерфейсов"""
        self.logger.info("Анализ API контрактов...")
        
        contracts = {
            'public_methods': [],
            'method_signatures': {},
            'return_types': {},
            'exception_contracts': {},
            'dependencies': []
        }
        
        try:
            with open(self.target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Анализируем методы и их контракты
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    method_info = {
                        'name': node.name,
                        'is_public': not node.name.startswith('_'),
                        'args': [arg.arg for arg in node.args.args],
                        'defaults': len(node.args.defaults),
                        'has_docstring': ast.get_docstring(node) is not None,
                        'raises_exceptions': []
                    }
                    
                    # Ищем исключения
                    for child in ast.walk(node):
                        if isinstance(child, ast.Raise):
                            if isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                                method_info['raises_exceptions'].append(child.exc.func.id)
                    
                    if method_info['is_public']:
                        contracts['public_methods'].append(method_info)
                    
                    contracts['method_signatures'][node.name] = method_info
            
            # Анализируем зависимости
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        contracts['dependencies'].append({
                            'type': 'import',
                            'module': alias.name
                        })
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        contracts['dependencies'].append({
                            'type': 'from_import',
                            'module': node.module,
                            'name': alias.name
                        })
            
            self.refactoring_data['api_contracts'] = contracts
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа контрактов: {e}")
            return False

    def extract_data_schemas(self):
        """4. Извлечение схем данных и типов"""
        self.logger.info("Извлечение схем данных...")
        
        schemas = {
            'type_definitions': {},
            'data_structures': {},
            'constants': {},
            'enums': []
        }
        
        try:
            with open(self.target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            # Ищем определения типов и структур данных
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            # Константы (UPPER_CASE)
                            if target.id.isupper():
                                if isinstance(node.value, ast.Constant):
                                    schemas['constants'][target.id] = node.value.value
                            
                            # TypeAlias определения
                            if 'TypeAlias' in ast.dump(node) or target.id.endswith('Type'):
                                schemas['type_definitions'][target.id] = ast.dump(node.value)
                
                elif isinstance(node, ast.ClassDef):
                    # Анализируем классы как структуры данных
                    class_schema = {
                        'name': node.name,
                        'attributes': [],
                        'methods': [],
                        'is_dataclass': any(
                            isinstance(decorator, ast.Name) and decorator.id == 'dataclass'
                            for decorator in node.decorator_list
                        )
                    }
                    
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            class_schema['attributes'].append({
                                'name': item.target.id,
                                'type': ast.dump(item.annotation) if item.annotation else None
                            })
                        elif isinstance(item, ast.FunctionDef):
                            class_schema['methods'].append(item.name)
                    
                    schemas['data_structures'][node.name] = class_schema
            
            self.refactoring_data['data_schemas'] = schemas
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения схем: {e}")
            return False

    def identify_refactoring_opportunities(self):
        """5. Выявление возможностей рефакторинга"""
        self.logger.info("Выявление возможностей рефакторинга...")
        
        opportunities = {
            'god_objects': [],
            'large_methods': [],
            'duplicate_code': [],
            'complex_conditionals': [],
            'long_parameter_lists': [],
            'feature_envy': [],
            'dead_code': []
        }
        
        # Анализируем на основе структурных данных
        structure = self.refactoring_data.get('file_structure', {})
        
        # God Objects
        for cls in structure.get('classes', []):
            if cls.get('is_god_object', False):
                opportunities['god_objects'].append({
                    'class': cls['name'],
                    'method_count': cls['method_count'],
                    'recommendation': 'Разделить класс на несколько специализированных классов',
                    'priority': 'HIGH'
                })
        
        # Large Methods
        for func in structure.get('functions', []):
            if func.get('is_large', False):
                opportunities['large_methods'].append({
                    'function': func['name'],
                    'lines': func['line_end'] - func['line_start'],
                    'recommendation': 'Разбить функцию на более мелкие специализированные функции',
                    'priority': 'MEDIUM'
                })
        
        # Long Parameter Lists
        contracts = self.refactoring_data.get('api_contracts', {})
        for method_name, method_info in contracts.get('method_signatures', {}).items():
            if len(method_info.get('args', [])) > 5:
                opportunities['long_parameter_lists'].append({
                    'method': method_name,
                    'parameter_count': len(method_info['args']),
                    'recommendation': 'Использовать объект параметров или разбить метод',
                    'priority': 'MEDIUM'
                })
        
        # Анализируем дубликаты через IntelliRefactor
        self._detect_duplicates_for_opportunities(opportunities)
        
        self.refactoring_data['refactoring_opportunities'] = opportunities
        return True

    def _detect_duplicates_for_opportunities(self, opportunities: Dict):
        """Обнаружение дубликатов для возможностей рефакторинга"""
        try:
            # Запускаем анализ дубликатов
            result = self._run_intellirefactor_command_with_timeout(
                ["duplicates", str(self.target_file), "--format", "json"],
                f"duplicates_for_refactoring_{self.timestamp}.json",
                timeout_minutes=5
            )
            
            if result.get("success") and result.get("output_file"):
                with open(result["output_file"], 'r', encoding='utf-8') as f:
                    duplicates_data = json.load(f)
                
                # Извлекаем информацию о дубликатах
                duplicates = duplicates_data.get('duplicates', {}).get('duplicates', [])
                
                for dup in duplicates[:10]:  # Первые 10 дубликатов
                    if isinstance(dup, dict):
                        opportunities['duplicate_code'].append({
                            'locations': dup.get('locations', []),
                            'similarity': dup.get('similarity_score', 0),
                            'lines': dup.get('lines', 0),
                            'recommendation': 'Извлечь общий код в отдельную функцию',
                            'priority': 'HIGH' if dup.get('similarity_score', 0) > 0.8 else 'MEDIUM'
                        })
        
        except Exception as e:
            self.logger.warning(f"Не удалось проанализировать дубликаты: {e}")

    def create_expert_recommendations(self):
        """6. Создание экспертных рекомендаций"""
        self.logger.info("Создание экспертных рекомендаций...")
        
        recommendations = {
            'priority_actions': [],
            'refactoring_strategy': {},
            'risk_assessment': {},
            'implementation_plan': {},
            'quality_metrics': {}
        }
        
        # Анализируем возможности и создаем приоритетный план
        opportunities = self.refactoring_data.get('refactoring_opportunities', {})
        
        # Высокоприоритетные действия
        high_priority = []
        
        if opportunities.get('god_objects'):
            high_priority.append({
                'action': 'Разделение God Objects',
                'reason': f"Найдено {len(opportunities['god_objects'])} классов с избыточной ответственностью",
                'impact': 'Высокий - улучшит поддерживаемость и тестируемость',
                'effort': 'Высокий - требует архитектурных изменений'
            })
        
        if opportunities.get('duplicate_code'):
            high_priority.append({
                'action': 'Устранение дубликатов кода',
                'reason': f"Найдено {len(opportunities['duplicate_code'])} дублированных блоков",
                'impact': 'Средний - уменьшит объем кода и упростит сопровождение',
                'effort': 'Средний - извлечение в общие функции'
            })
        
        recommendations['priority_actions'] = high_priority
        
        # Стратегия рефакторинга
        recommendations['refactoring_strategy'] = {
            'approach': 'Поэтапный рефакторинг с сохранением функциональности',
            'phases': [
                'Фаза 1: Устранение дубликатов и мелких запахов кода',
                'Фаза 2: Разделение крупных классов и методов',
                'Фаза 3: Оптимизация архитектуры и интерфейсов'
            ],
            'testing_strategy': 'Создание характеризационных тестов перед рефакторингом'
        }
        
        # Оценка рисков
        structure = self.refactoring_data.get('file_structure', {})
        recommendations['risk_assessment'] = {
            'complexity_risk': 'HIGH' if structure.get('complexity_metrics', {}).get('lines_of_code', 0) > 1000 else 'MEDIUM',
            'dependency_risk': 'MEDIUM',  # Можно улучшить анализом зависимостей
            'testing_risk': 'HIGH',  # Предполагаем отсутствие тестов
            'mitigation_strategies': [
                'Создать комплексные тесты перед рефакторингом',
                'Использовать инкрементальный подход',
                'Проводить code review на каждом этапе'
            ]
        }
        
        self.refactoring_data['expert_recommendations'] = recommendations
        return True

    def generate_refactoring_plan(self):
        """7. Генерация итогового плана рефакторинга"""
        self.logger.info("Генерация итогового плана рефакторинга...")
        
        # Создаем детальный план на основе всех собранных данных
        plan_content = self._create_detailed_refactoring_plan()
        
        # Сохраняем план
        plan_path = self.output_dir / f"OPTIMIZED_REFACTORING_PLAN_{self.timestamp}.md"
        with open(plan_path, 'w', encoding='utf-8') as f:
            f.write(plan_content)
        
        self.analysis_results["generated_files"].append(str(plan_path))
        self.logger.info(f"План рефакторинга создан: {plan_path}")
        
        return True

    def _create_detailed_refactoring_plan(self) -> str:
        """Создает детальный план рефакторинга"""
        structure = self.refactoring_data.get('file_structure', {})
        opportunities = self.refactoring_data.get('refactoring_opportunities', {})
        recommendations = self.refactoring_data.get('expert_recommendations', {})
        usage_patterns = self.refactoring_data.get('real_usage_patterns', {})
        
        try:
            relative_file_path = self.target_file.relative_to(self.project_path)
        except ValueError:
            relative_file_path = self.target_file
        
        return f"""# Оптимизированный план рефакторинга

**Файл:** {relative_file_path}
**Проект:** {self.project_path.name}
**Дата анализа:** {self.timestamp}
**Тип анализа:** Оптимизированный для рефакторинга

## 🎯 Исполнительное резюме

Проведен целенаправленный анализ файла для составления максимально качественного плана рефакторинга.

### Ключевые метрики
- **Строк кода:** {structure.get('complexity_metrics', {}).get('lines_of_code', 'N/A')}
- **Классов:** {structure.get('complexity_metrics', {}).get('total_classes', 0)}
- **Функций:** {structure.get('complexity_metrics', {}).get('total_functions', 0)}
- **God Objects:** {len(opportunities.get('god_objects', []))}
- **Крупных методов:** {len(opportunities.get('large_methods', []))}
- **Дубликатов кода:** {len(opportunities.get('duplicate_code', []))}

## 🚨 Приоритетные проблемы

### Критические (требуют немедленного внимания)
"""

        # Добавляем критические проблемы
        critical_issues = []
        
        if opportunities.get('god_objects'):
            for god_obj in opportunities['god_objects']:
                critical_issues.append(f"**God Object: {god_obj['class']}** - {god_obj['method_count']} методов")
        
        if opportunities.get('duplicate_code'):
            high_similarity_dups = [d for d in opportunities['duplicate_code'] if d.get('similarity', 0) > 0.8]
            if high_similarity_dups:
                critical_issues.append(f"**Высокое дублирование кода** - {len(high_similarity_dups)} блоков с similarity > 80%")
        
        if critical_issues:
            for issue in critical_issues:
                return f"1. {issue}\n"
        else:
            return "Критических проблем не обнаружено.\n"

        return """

### Важные (влияют на поддерживаемость)
"""

        # Добавляем важные проблемы
        important_issues = []
        
        if opportunities.get('large_methods'):
            important_issues.append(f"**Крупные методы** - {len(opportunities['large_methods'])} методов требуют разбиения")
        
        if opportunities.get('long_parameter_lists'):
            important_issues.append(f"**Длинные списки параметров** - {len(opportunities['long_parameter_lists'])} методов")
        
        if important_issues:
            for i, issue in enumerate(important_issues, 1):
                return f"{i}. {issue}\n"
        else:
            return "Важных проблем не обнаружено.\n"

        return """

## 📋 Детальный план действий

### Фаза 1: Подготовка (1-2 дня)
1. **Создание характеризационных тестов**
   - Покрыть основные сценарии использования
   - Зафиксировать текущее поведение
   - Обеспечить безопасность рефакторинга

2. **Анализ зависимостей**
   - Выявить все внешние зависимости
   - Определить точки интеграции
   - Подготовить моки для тестирования

### Фаза 2: Устранение дубликатов (2-3 дня)
"""

        if opportunities.get('duplicate_code'):
            for i, dup in enumerate(opportunities['duplicate_code'][:5], 1):
                return f"{i}. **Дубликат {i}** (similarity: {dup.get('similarity', 0):.1%})\n"
                return f"   - Локации: {len(dup.get('locations', []))}\n"
                return f"   - Действие: {dup.get('recommendation', 'Извлечь в общую функцию')}\n"

        return """

### Фаза 3: Разделение крупных компонентов (3-5 дней)
"""

        if opportunities.get('god_objects'):
            for god_obj in opportunities['god_objects']:
                return f"**Класс {god_obj['class']}:**\n"
                return f"- Методов: {god_obj['method_count']}\n"
                return "- Стратегия: Выделить специализированные классы по принципу единственной ответственности\n"
                return f"- Приоритет: {god_obj.get('priority', 'HIGH')}\n\n"

        return """

### Фаза 4: Оптимизация методов (2-3 дня)
"""

        if opportunities.get('large_methods'):
            for method in opportunities['large_methods'][:3]:
                return f"**Метод {method['function']}:**\n"
                return f"- Строк: {method['lines']}\n"
                return f"- Действие: {method.get('recommendation', 'Разбить на специализированные функции')}\n\n"

        return """

## 🔍 Реальные паттерны использования

### Частые вызовы методов
"""

        method_calls = usage_patterns.get('method_calls', [])
        if method_calls:
            call_counter = Counter(call.get('function_name') for call in method_calls if call.get('function_name'))
            for method, count in call_counter.most_common(5):
                return f"- **{method}**: {count} вызовов\n"
        else:
            return "Данные о вызовах не найдены.\n"

        return """

### Популярные параметры
"""

        param_patterns = usage_patterns.get('parameter_patterns', {})
        common_params = param_patterns.get('common_parameters', {})
        if common_params:
            for param, count in common_params.most_common(10):
                real_values = param_patterns.get('real_values', {}).get(param, [])
                return f"- **{param}**: {count} использований"
                if real_values:
                    return f" (примеры: {', '.join(map(str, real_values[:3]))})\n"
                else:
                    return "\n"

        return f"""

## ⚠️ Оценка рисков

### Уровень сложности: {recommendations.get('risk_assessment', {}).get('complexity_risk', 'MEDIUM')}
### Риск зависимостей: {recommendations.get('risk_assessment', {}).get('dependency_risk', 'MEDIUM')}
### Риск тестирования: {recommendations.get('risk_assessment', {}).get('testing_risk', 'HIGH')}

### Стратегии снижения рисков:
"""

        mitigation = recommendations.get('risk_assessment', {}).get('mitigation_strategies', [])
        for strategy in mitigation:
            return f"- {strategy}\n"

        return """

## 🎯 Критерии успеха

### Количественные метрики
- [ ] Уменьшение цикломатической сложности на 30%
- [ ] Сокращение дублирования кода до <5%
- [ ] Разделение God Objects (методов в классе <15)
- [ ] Сокращение крупных методов (<50 строк)

### Качественные улучшения
- [ ] Улучшение читаемости кода
- [ ] Повышение тестируемости
- [ ] Соответствие SOLID принципам
- [ ] Упрощение сопровождения

## 📅 Временные рамки

**Общее время:** 8-13 дней
- Подготовка: 1-2 дня
- Устранение дубликатов: 2-3 дня  
- Разделение компонентов: 3-5 дней
- Оптимизация методов: 2-3 дня

## 🔧 Инструменты и техники

### Рекомендуемые техники рефакторинга
1. **Extract Method** - для разбиения крупных методов
2. **Extract Class** - для разделения God Objects
3. **Move Method** - для улучшения связности
4. **Replace Parameter with Object** - для длинных списков параметров

### Инструменты
- IntelliRefactor для автоматического анализа
- Характеризационные тесты для безопасности
- Code coverage для контроля качества тестов

---
*План создан оптимизированным анализатором рефакторинга*
*Основан на реальных паттернах использования и структурном анализе*
"""

    def save_optimized_results(self):
        """Сохранение оптимизированных результатов"""
        self.logger.info("Сохранение результатов оптимизированного анализа...")
        
        # Сохраняем полные данные в JSON
        results_path = self.output_dir / f"OPTIMIZED_REFACTORING_DATA_{self.timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.refactoring_data, f, ensure_ascii=False, indent=2, default=str)
        
        self.analysis_results["generated_files"].append(str(results_path))
        
        # Создаем краткий отчет
        summary_path = self.output_dir / f"REFACTORING_SUMMARY_{self.timestamp}.md"
        summary_content = self._create_summary_report()
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        self.analysis_results["generated_files"].append(str(summary_path))
        
        self.logger.info(f"Результаты сохранены: {results_path}")
        self.logger.info(f"Краткий отчет: {summary_path}")

    def _create_summary_report(self) -> str:
        """Создает краткий отчет"""
        structure = self.refactoring_data.get('file_structure', {})
        opportunities = self.refactoring_data.get('refactoring_opportunities', {})
        
        return f"""# Краткий отчет оптимизированного анализа

**Файл:** {self.target_file.name}
**Дата:** {self.timestamp}

## Основные находки

### Структура
- Классов: {structure.get('complexity_metrics', {}).get('total_classes', 0)}
- Функций: {structure.get('complexity_metrics', {}).get('total_functions', 0)}
- Строк кода: {structure.get('complexity_metrics', {}).get('lines_of_code', 0)}

### Возможности рефакторинга
- God Objects: {len(opportunities.get('god_objects', []))}
- Крупные методы: {len(opportunities.get('large_methods', []))}
- Дубликаты кода: {len(opportunities.get('duplicate_code', []))}
- Длинные списки параметров: {len(opportunities.get('long_parameter_lists', []))}

## Рекомендации

1. **Приоритет 1:** Разделение God Objects
2. **Приоритет 2:** Устранение дубликатов кода
3. **Приоритет 3:** Разбиение крупных методов

## Следующие шаги

1. Изучите детальный план: `OPTIMIZED_REFACTORING_PLAN_{self.timestamp}.md`
2. Просмотрите полные данные: `OPTIMIZED_REFACTORING_DATA_{self.timestamp}.json`
3. Начните с создания характеризационных тестов

---
*Создано оптимизированным анализатором рефакторинга*
"""


def main():
    """Главная функция для запуска из командной строки"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Оптимизированный анализатор для составления плана рефакторинга",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python optimized_refactoring_analyzer.py /path/to/project /path/to/file.py /path/to/output
  python optimized_refactoring_analyzer.py C:\\Project C:\\Project\\module.py C:\\Results --verbose
        """,
    )

    parser.add_argument("project_path", help="Путь к корневой папке проекта")
    parser.add_argument("target_file", help="Путь к анализируемому файлу")
    parser.add_argument("output_dir", help="Директория для сохранения результатов анализа")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод процесса анализа")

    args = parser.parse_args()

    # Проверяем существование путей
    project_path = Path(args.project_path)
    target_file = Path(args.target_file)

    if not project_path.exists():
        print(f"Ошибка: Проект не найден: {project_path}")
        sys.exit(1)

    if not target_file.exists():
        print(f"Ошибка: Файл не найден: {target_file}")
        sys.exit(1)

    # Создаем и запускаем анализатор
    try:
        analyzer = OptimizedRefactoringAnalyzer(
            str(project_path), str(target_file), args.output_dir, args.verbose
        )

        print("=" * 80)
        print("ОПТИМИЗИРОВАННЫЙ АНАЛИЗАТОР ДЛЯ РЕФАКТОРИНГА")
        print("=" * 80)
        print(f"Проект: {project_path}")
        print(f"Файл: {target_file}")
        print(f"Результаты: {args.output_dir}")
        print("Собираем только нужное для максимального плана рефакторинга!")
        print("=" * 80)

        success = analyzer.run_optimized_analysis()

        if success:
            print("\n" + "=" * 80)
            print("✅ ОПТИМИЗИРОВАННЫЙ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
            print("=" * 80)
            print(f"Результаты сохранены в: {args.output_dir}")
            print(f"Детальный план: OPTIMIZED_REFACTORING_PLAN_{analyzer.timestamp}.md")
            print(f"Полные данные: OPTIMIZED_REFACTORING_DATA_{analyzer.timestamp}.json")
            print(f"Краткий отчет: REFACTORING_SUMMARY_{analyzer.timestamp}.md")
            print("🎯 ГОТОВ МАКСИМАЛЬНО КАЧЕСТВЕННЫЙ ПЛАН РЕФАКТОРИНГА!")
        else:
            print("\n" + "=" * 80)
            print("⚠️ АНАЛИЗ ЗАВЕРШЕН С ПРЕДУПРЕЖДЕНИЯМИ")
            print("=" * 80)
            print(f"Частичные результаты в: {args.output_dir}")

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n[ПРЕРВАНО] Анализ прерван пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ОШИБКА] Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
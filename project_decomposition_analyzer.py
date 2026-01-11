#!/usr/bin/env python3
"""
Project Decomposition Analyzer - Анализатор архитектурной декомпозиции проекта

Анализирует весь проект для выявления:
1. Функциональных дубликатов между модулями
2. Возможностей объединения/разделения компонентов
3. Устаревших/неиспользуемых модулей
4. Архитектурных проблем (God Objects, Feature Envy)
5. Оптимальной структуры проекта

Создает:
- Карту функциональности всего проекта
- Матрицу дублирования функций
- Диаграмму зависимостей
- План архитектурной реорганизации
"""

import sys
import os
import json
import ast
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging

# Добавляем текущую директорию в путь для импорта
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from contextual_file_analyzer import ContextualFileAnalyzer


class ProjectDecompositionAnalyzer(ContextualFileAnalyzer):
    """Анализатор архитектурной декомпозиции проекта"""

    def __init__(self, project_path: str, output_dir: str, verbose: bool = False):
        # Создаем фиктивный target_file для совместимости с базовым классом
        dummy_target = Path(project_path) / "__dummy__.py"
        super().__init__(project_path, str(dummy_target), output_dir, verbose)
        
        self.analysis_mode = "project_decomposition_analysis"
        self.logger.info("Инициализирован ProjectDecompositionAnalyzer")
        
        # Структуры данных для анализа
        self.project_map = {
            'modules': {},           # Карта всех модулей
            'functions': {},         # Все функции в проекте
            'classes': {},           # Все классы в проекте
            'dependencies': {},      # Граф зависимостей
            'duplicates': {},        # Функциональные дубликаты
            'god_objects': {},       # God Objects
            'dead_code': {},         # Мертвый код
            'feature_clusters': {}   # Кластеры функциональности
        }
        
        # Настройки анализа
        self.max_files_to_analyze = 200  # Ограничение для больших проектов
        self.similarity_threshold = 0.7   # Порог схожести для дубликатов
        self.god_object_threshold = 15    # Порог для God Object

    def run_project_decomposition(self):
        """Запуск полного анализа декомпозиции проекта"""
        self.logger.info("[СТАРТ] Анализ архитектурной декомпозиции проекта...")

        analyses = [
            ("Сканирование структуры проекта", self.scan_project_structure),
            ("Анализ функциональности модулей", self.analyze_module_functionality),
            ("Построение графа зависимостей", self.build_dependency_graph),
            ("Поиск функциональных дубликатов", self.find_functional_duplicates),
            ("Выявление God Objects", self.identify_god_objects),
            ("Обнаружение мертвого кода", self.detect_dead_code),
            ("Кластеризация функциональности", self.cluster_functionality),
            ("Создание плана декомпозиции", self.create_decomposition_plan),
            ("Генерация диаграмм и отчетов", self.generate_visualizations)
        ]

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
        self.save_decomposition_results()
        return True

    def scan_project_structure(self):
        """1. Сканирование структуры проекта"""
        self.logger.info("Сканирование структуры проекта...")
        
        try:
            python_files = list(self.project_path.rglob("*.py"))
            
            # Ограничиваем количество файлов для анализа
            if len(python_files) > self.max_files_to_analyze:
                self.logger.warning(f"Найдено {len(python_files)} файлов, анализируем первые {self.max_files_to_analyze}")
                python_files = python_files[:self.max_files_to_analyze]
            
            for py_file in python_files:
                try:
                    # Пропускаем служебные файлы
                    if self._should_skip_file(py_file):
                        continue
                    
                    relative_path = py_file.relative_to(self.project_path)
                    module_info = self._analyze_single_module(py_file)
                    
                    if module_info:
                        self.project_map['modules'][str(relative_path)] = module_info
                        
                except Exception as e:
                    self.logger.warning(f"Ошибка анализа файла {py_file}: {e}")
                    continue
            
            self.logger.info(f"Проанализировано {len(self.project_map['modules'])} модулей")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка сканирования проекта: {e}")
            return False

    def _should_skip_file(self, file_path: Path) -> bool:
        """Определяет, нужно ли пропустить файл"""
        skip_patterns = [
            '__pycache__', '.git', '.pytest_cache', '.venv', 'venv',
            'node_modules', 'build', 'dist', '.egg-info',
            'test_', '_test', 'tests/', '/test/', 'backup', 'old'
        ]
        
        file_str = str(file_path).lower()
        return any(pattern in file_str for pattern in skip_patterns)

    def _analyze_single_module(self, file_path: Path) -> Optional[Dict]:
        """Анализирует отдельный модуль"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            
            module_info = {
                'path': str(file_path),
                'size_lines': len(content.splitlines()),
                'size_bytes': len(content.encode('utf-8')),
                'classes': [],
                'functions': [],
                'imports': [],
                'constants': [],
                'last_modified': file_path.stat().st_mtime,
                'functionality_keywords': set()
            }
            
            # Анализируем AST
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_info = self._analyze_class(node, content)
                    module_info['classes'].append(class_info)
                    
                elif isinstance(node, ast.FunctionDef):
                    # Только функции верхнего уровня
                    if not any(node.lineno >= cls['line_start'] and node.lineno <= cls['line_end'] 
                             for cls in module_info['classes']):
                        func_info = self._analyze_function(node, content)
                        module_info['functions'].append(func_info)
                
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_info = self._analyze_import(node)
                    module_info['imports'].extend(import_info)
                
                elif isinstance(node, ast.Assign):
                    const_info = self._analyze_constants(node)
                    if const_info:
                        module_info['constants'].extend(const_info)
            
            # Извлекаем ключевые слова функциональности
            module_info['functionality_keywords'] = self._extract_functionality_keywords(content)
            
            return module_info
            
        except Exception as e:
            self.logger.warning(f"Не удалось проанализировать {file_path}: {e}")
            return None

    def _analyze_class(self, node: ast.ClassDef, content: str) -> Dict:
        """Анализирует класс"""
        methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
        
        class_info = {
            'name': node.name,
            'line_start': node.lineno,
            'line_end': getattr(node, 'end_lineno', node.lineno),
            'methods': [method.name for method in methods],
            'method_count': len(methods),
            'is_god_object': len(methods) > self.god_object_threshold,
            'docstring': ast.get_docstring(node),
            'functionality_keywords': set()
        }
        
        # Извлекаем функциональность из имени класса и методов
        class_text = f"{node.name} {' '.join(method.name for method in methods)}"
        class_info['functionality_keywords'] = self._extract_functionality_keywords(class_text)
        
        return class_info

    def _analyze_function(self, node: ast.FunctionDef, content: str) -> Dict:
        """Анализирует функцию"""
        func_info = {
            'name': node.name,
            'line_start': node.lineno,
            'line_end': getattr(node, 'end_lineno', node.lineno),
            'args': [arg.arg for arg in node.args.args],
            'arg_count': len(node.args.args),
            'is_large': (getattr(node, 'end_lineno', node.lineno) - node.lineno) > 50,
            'docstring': ast.get_docstring(node),
            'functionality_keywords': set()
        }
        
        # Извлекаем функциональность из имени и docstring
        func_text = f"{node.name} {func_info.get('docstring', '')}"
        func_info['functionality_keywords'] = self._extract_functionality_keywords(func_text)
        
        return func_info

    def _analyze_import(self, node) -> List[Dict]:
        """Анализирует импорты"""
        imports = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    'type': 'import',
                    'module': alias.name,
                    'alias': alias.asname,
                    'is_external': not alias.name.startswith('.')
                })
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imports.append({
                    'type': 'from_import',
                    'module': node.module,
                    'name': alias.name,
                    'alias': alias.asname,
                    'is_external': not (node.module and node.module.startswith('.'))
                })
        
        return imports

    def _analyze_constants(self, node: ast.Assign) -> List[Dict]:
        """Анализирует константы"""
        constants = []
        
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id.isupper():
                const_info = {
                    'name': target.id,
                    'line': node.lineno,
                    'type': type(node.value).__name__
                }
                
                if isinstance(node.value, ast.Constant):
                    const_info['value'] = node.value.value
                
                constants.append(const_info)
        
        return constants

    def _extract_functionality_keywords(self, text: str) -> Set[str]:
        """Извлекает ключевые слова функциональности"""
        if not text:
            return set()
        
        # Функциональные ключевые слова
        functionality_patterns = {
            'parsing': ['parse', 'parser', 'parsing', 'decode', 'extract'],
            'validation': ['validate', 'validator', 'validation', 'check', 'verify'],
            'normalization': ['normalize', 'normalizer', 'normalization', 'clean', 'sanitize'],
            'execution': ['execute', 'executor', 'execution', 'run', 'perform'],
            'handling': ['handle', 'handler', 'handling', 'process', 'processor'],
            'logging': ['log', 'logger', 'logging', 'telemetry', 'metrics'],
            'storage': ['store', 'storage', 'save', 'persist', 'database'],
            'network': ['network', 'socket', 'connection', 'request', 'response'],
            'security': ['security', 'auth', 'authentication', 'authorization', 'crypto'],
            'configuration': ['config', 'configuration', 'settings', 'options'],
            'testing': ['test', 'testing', 'mock', 'fixture', 'assert'],
            'utility': ['util', 'utility', 'helper', 'common', 'shared']
        }
        
        text_lower = text.lower()
        found_keywords = set()
        
        for category, keywords in functionality_patterns.items():
            for keyword in keywords:
                if keyword in text_lower:
                    found_keywords.add(category)
                    break
        
        return found_keywords

    def analyze_module_functionality(self):
        """2. Анализ функциональности модулей"""
        self.logger.info("Анализ функциональности модулей...")
        
        try:
            # Создаем карту функций и классов
            for module_path, module_info in self.project_map['modules'].items():
                # Функции модуля
                for func in module_info['functions']:
                    func_key = f"{module_path}::{func['name']}"
                    self.project_map['functions'][func_key] = {
                        'module': module_path,
                        'name': func['name'],
                        'type': 'function',
                        'functionality': func['functionality_keywords'],
                        'size': func['line_end'] - func['line_start'],
                        'args': func['args']
                    }
                
                # Классы модуля
                for cls in module_info['classes']:
                    cls_key = f"{module_path}::{cls['name']}"
                    self.project_map['classes'][cls_key] = {
                        'module': module_path,
                        'name': cls['name'],
                        'type': 'class',
                        'functionality': cls['functionality_keywords'],
                        'method_count': cls['method_count'],
                        'is_god_object': cls['is_god_object'],
                        'methods': cls['methods']
                    }
            
            self.logger.info(f"Найдено {len(self.project_map['functions'])} функций и {len(self.project_map['classes'])} классов")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа функциональности: {e}")
            return False

    def build_dependency_graph(self):
        """3. Построение графа зависимостей"""
        self.logger.info("Построение графа зависимостей...")
        
        try:
            for module_path, module_info in self.project_map['modules'].items():
                dependencies = []
                
                for import_info in module_info['imports']:
                    if not import_info['is_external']:
                        # Внутренние зависимости проекта
                        dep_module = self._resolve_internal_import(import_info, module_path)
                        if dep_module:
                            dependencies.append(dep_module)
                
                self.project_map['dependencies'][module_path] = dependencies
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка построения графа зависимостей: {e}")
            return False

    def _resolve_internal_import(self, import_info: Dict, current_module: str) -> Optional[str]:
        """Разрешает внутренний импорт в путь модуля"""
        # Упрощенная логика разрешения импортов
        if import_info['type'] == 'from_import' and import_info['module']:
            if import_info['module'].startswith('.'):
                # Относительный импорт
                current_dir = Path(current_module).parent
                relative_path = import_info['module'].lstrip('.')
                resolved_path = current_dir / f"{relative_path}.py"
                return str(resolved_path) if resolved_path.exists() else None
            else:
                # Абсолютный импорт внутри проекта
                module_path = f"{import_info['module'].replace('.', '/')}.py"
                return module_path if module_path in self.project_map['modules'] else None
        
        return None

    def find_functional_duplicates(self):
        """4. Поиск функциональных дубликатов"""
        self.logger.info("Поиск функциональных дубликатов...")
        
        try:
            # Группируем по функциональности
            functionality_groups = defaultdict(list)
            
            # Группируем функции
            for func_key, func_info in self.project_map['functions'].items():
                for functionality in func_info['functionality']:
                    functionality_groups[functionality].append({
                        'key': func_key,
                        'type': 'function',
                        'info': func_info
                    })
            
            # Группируем классы
            for cls_key, cls_info in self.project_map['classes'].items():
                for functionality in cls_info['functionality']:
                    functionality_groups[functionality].append({
                        'key': cls_key,
                        'type': 'class',
                        'info': cls_info
                    })
            
            # Находим дубликаты
            duplicates = {}
            for functionality, items in functionality_groups.items():
                if len(items) > 1:
                    # Анализируем схожесть
                    duplicate_groups = self._analyze_similarity(items)
                    if duplicate_groups:
                        duplicates[functionality] = duplicate_groups
            
            self.project_map['duplicates'] = duplicates
            self.logger.info(f"Найдено {len(duplicates)} групп функциональных дубликатов")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка поиска дубликатов: {e}")
            return False

    def _analyze_similarity(self, items: List[Dict]) -> List[Dict]:
        """Анализирует схожесть элементов"""
        duplicate_groups = []
        
        # Простая эвристика схожести на основе имен
        for i, item1 in enumerate(items):
            for j, item2 in enumerate(items[i+1:], i+1):
                similarity = self._calculate_name_similarity(
                    item1['info']['name'], 
                    item2['info']['name']
                )
                
                if similarity > self.similarity_threshold:
                    duplicate_groups.append({
                        'items': [item1, item2],
                        'similarity': similarity,
                        'recommendation': self._get_duplicate_recommendation(item1, item2)
                    })
        
        return duplicate_groups

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Вычисляет схожесть имен (упрощенная версия)"""
        # Простая метрика на основе общих слов
        words1 = set(re.findall(r'[A-Z][a-z]*|[a-z]+', name1))
        words2 = set(re.findall(r'[A-Z][a-z]*|[a-z]+', name2))
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0

    def _get_duplicate_recommendation(self, item1: Dict, item2: Dict) -> str:
        """Получает рекомендацию по устранению дубликата"""
        if item1['type'] == 'function' and item2['type'] == 'function':
            return "Рассмотрите объединение функций или извлечение общей логики"
        elif item1['type'] == 'class' and item2['type'] == 'class':
            return "Рассмотрите объединение классов или создание общего базового класса"
        else:
            return "Рассмотрите унификацию подходов к решению задачи"

    def identify_god_objects(self):
        """5. Выявление God Objects"""
        self.logger.info("Выявление God Objects...")
        
        try:
            god_objects = {}
            
            for cls_key, cls_info in self.project_map['classes'].items():
                if cls_info['is_god_object']:
                    # Анализируем методы для группировки по функциональности
                    method_groups = self._group_methods_by_functionality(cls_info)
                    
                    god_objects[cls_key] = {
                        'class_info': cls_info,
                        'method_groups': method_groups,
                        'decomposition_suggestions': self._suggest_class_decomposition(cls_info, method_groups)
                    }
            
            self.project_map['god_objects'] = god_objects
            self.logger.info(f"Найдено {len(god_objects)} God Objects")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка выявления God Objects: {e}")
            return False

    def _group_methods_by_functionality(self, cls_info: Dict) -> Dict:
        """Группирует методы класса по функциональности"""
        method_groups = defaultdict(list)
        
        for method_name in cls_info['methods']:
            # Определяем функциональность метода по имени
            functionality = self._extract_functionality_keywords(method_name)
            
            if functionality:
                for func in functionality:
                    method_groups[func].append(method_name)
            else:
                method_groups['other'].append(method_name)
        
        return dict(method_groups)

    def _suggest_class_decomposition(self, cls_info: Dict, method_groups: Dict) -> List[Dict]:
        """Предлагает варианты декомпозиции класса"""
        suggestions = []
        
        for functionality, methods in method_groups.items():
            if len(methods) >= 3:  # Достаточно методов для отдельного класса
                suggestions.append({
                    'new_class_name': f"{cls_info['name']}{functionality.title()}",
                    'functionality': functionality,
                    'methods': methods,
                    'rationale': f"Выделить {functionality} в отдельный класс"
                })
        
        return suggestions

    def detect_dead_code(self):
        """6. Обнаружение мертвого кода"""
        self.logger.info("Обнаружение мертвого кода...")
        
        try:
            # Простая эвристика: модули, которые никто не импортирует
            imported_modules = set()
            
            for module_info in self.project_map['modules'].values():
                for import_info in module_info['imports']:
                    if not import_info['is_external']:
                        imported_modules.add(import_info.get('module', ''))
            
            dead_modules = []
            for module_path in self.project_map['modules'].keys():
                module_name = Path(module_path).stem
                if module_name not in imported_modules and not module_path.endswith('__init__.py'):
                    # Проверяем, не является ли это entry point
                    if not self._is_entry_point(module_path):
                        dead_modules.append(module_path)
            
            self.project_map['dead_code'] = {
                'potentially_dead_modules': dead_modules,
                'analysis_note': 'Требует дополнительной проверки - могут быть entry points'
            }
            
            self.logger.info(f"Найдено {len(dead_modules)} потенциально мертвых модулей")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка обнаружения мертвого кода: {e}")
            return False

    def _is_entry_point(self, module_path: str) -> bool:
        """Проверяет, является ли модуль точкой входа"""
        entry_point_patterns = [
            'main.py', 'cli.py', 'app.py', 'server.py', 'run.py',
            '__main__.py', 'manage.py', 'setup.py'
        ]
        
        module_name = Path(module_path).name
        return any(pattern in module_name for pattern in entry_point_patterns)

    def cluster_functionality(self):
        """7. Кластеризация функциональности"""
        self.logger.info("Кластеризация функциональности...")
        
        try:
            # Собираем всю функциональность
            all_functionality = defaultdict(list)
            
            for module_path, module_info in self.project_map['modules'].items():
                module_functionality = set()
                
                # Собираем функциональность из классов и функций
                for cls in module_info['classes']:
                    module_functionality.update(cls['functionality_keywords'])
                
                for func in module_info['functions']:
                    module_functionality.update(func['functionality_keywords'])
                
                # Группируем модули по функциональности
                for functionality in module_functionality:
                    all_functionality[functionality].append(module_path)
            
            # Создаем кластеры
            clusters = {}
            for functionality, modules in all_functionality.items():
                if len(modules) > 1:
                    clusters[functionality] = {
                        'modules': modules,
                        'count': len(modules),
                        'consolidation_opportunity': len(modules) > 3
                    }
            
            self.project_map['feature_clusters'] = clusters
            self.logger.info(f"Создано {len(clusters)} кластеров функциональности")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка кластеризации: {e}")
            return False

    def create_decomposition_plan(self):
        """8. Создание плана декомпозиции"""
        self.logger.info("Создание плана декомпозиции...")
        
        try:
            plan_content = self._generate_decomposition_plan()
            
            plan_path = self.output_dir / f"PROJECT_DECOMPOSITION_PLAN_{self.timestamp}.md"
            with open(plan_path, 'w', encoding='utf-8') as f:
                f.write(plan_content)
            
            self.analysis_results["generated_files"].append(str(plan_path))
            self.logger.info(f"План декомпозиции создан: {plan_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка создания плана: {e}")
            return False

    def _generate_decomposition_plan(self) -> str:
        """Генерирует план декомпозиции проекта"""
        god_objects = self.project_map.get('god_objects', {})
        duplicates = self.project_map.get('duplicates', {})
        clusters = self.project_map.get('feature_clusters', {})
        dead_code = self.project_map.get('dead_code', {})
        
        return f"""# План архитектурной декомпозиции проекта

**Проект:** {self.project_path.name}
**Дата анализа:** {self.timestamp}
**Проанализировано модулей:** {len(self.project_map['modules'])}

## [РЕЗЮМЕ] Исполнительное резюме

Проведен анализ архитектурной декомпозиции проекта для выявления возможностей реорганизации и улучшения структуры.

### Ключевые находки
- **God Objects:** {len(god_objects)} классов требуют декомпозиции
- **Функциональные дубликаты:** {len(duplicates)} групп дублирующейся функциональности
- **Кластеры функциональности:** {len(clusters)} возможностей консолидации
- **Потенциально мертвый код:** {len(dead_code.get('potentially_dead_modules', []))} модулей

## [КРИТИЧНО] Критические проблемы архитектуры

### God Objects (требуют немедленной декомпозиции)
"""

        # Добавляем информацию о God Objects
        if god_objects:
            for cls_key, god_info in god_objects.items():
                cls_info = god_info['class_info']
                suggestions = god_info['decomposition_suggestions']
                
                return f"""
**{cls_info['name']}** ({cls_info['module']})
- Методов: {cls_info['method_count']}
- Функциональные группы: {len(god_info['method_groups'])}

Предлагаемая декомпозиция:
"""
                for suggestion in suggestions:
                    return f"- **{suggestion['new_class_name']}**: {suggestion['rationale']} ({len(suggestion['methods'])} методов)\n"

        return f"""

### Функциональные дубликаты
"""

        if duplicates:
            for functionality, duplicate_groups in duplicates.items():
                return f"\n**{functionality.title()}** функциональность:\n"
                for group in duplicate_groups:
                    items = group['items']
                    return f"- Дубликат: {items[0]['info']['name']} ↔ {items[1]['info']['name']} (схожесть: {group['similarity']:.1%})\n"
                    return f"  Рекомендация: {group['recommendation']}\n"

        return f"""

## [КЛАСТЕРЫ] Кластеры функциональности

### Возможности консолидации
"""

        if clusters:
            for functionality, cluster_info in clusters.items():
                if cluster_info['consolidation_opportunity']:
                    return f"\n**{functionality.title()}** ({cluster_info['count']} модулей):\n"
                    for module in cluster_info['modules'][:5]:  # Первые 5
                        return f"- {module}\n"
                    if len(cluster_info['modules']) > 5:
                        return f"- ... и еще {len(cluster_info['modules']) - 5} модулей\n"
                    return f"[РЕКОМЕНДАЦИЯ] Рассмотрите создание единого {functionality} пакета\n"

        return f"""

## [ОЧИСТКА] Очистка проекта

### Потенциально неиспользуемые модули
"""

        dead_modules = dead_code.get('potentially_dead_modules', [])
        if dead_modules:
            for module in dead_modules[:10]:  # Первые 10
                return f"- {module}\n"
            if len(dead_modules) > 10:
                return f"- ... и еще {len(dead_modules) - 10} модулей\n"
            return f"\n[ВНИМАНИЕ] {dead_code.get('analysis_note', '')}\n"

        return f"""

## [ПЛАН] План реорганизации

### Фаза 1: Декомпозиция God Objects (2-4 недели)
"""

        if god_objects:
            phase1_priority = sorted(god_objects.items(), 
                                   key=lambda x: x[1]['class_info']['method_count'], 
                                   reverse=True)[:3]
            
            for i, (cls_key, god_info) in enumerate(phase1_priority, 1):
                cls_info = god_info['class_info']
                return f"{i}. **{cls_info['name']}** - разделить на {len(god_info['decomposition_suggestions'])} специализированных класса\n"

        return f"""

### Фаза 2: Устранение дубликатов (1-2 недели)
"""

        if duplicates:
            high_priority_dups = []
            for functionality, groups in duplicates.items():
                for group in groups:
                    if group['similarity'] > 0.8:
                        high_priority_dups.append((functionality, group))
            
            for i, (functionality, group) in enumerate(high_priority_dups[:5], 1):
                items = group['items']
                return f"{i}. Объединить {functionality} логику: {items[0]['info']['name']} + {items[1]['info']['name']}\n"

        return f"""

### Фаза 3: Консолидация функциональности (1-3 недели)
"""

        if clusters:
            consolidation_candidates = [(func, info) for func, info in clusters.items() 
                                      if info['consolidation_opportunity']][:3]
            
            for i, (functionality, cluster_info) in enumerate(consolidation_candidates, 1):
                return f"{i}. Создать {functionality} пакет из {cluster_info['count']} модулей\n"

        return f"""

### Фаза 4: Очистка проекта (1 неделя)
"""

        if dead_modules:
            return f"1. Проверить и удалить {len(dead_modules)} потенциально неиспользуемых модулей\n"
            return f"2. Провести анализ покрытия кода для подтверждения\n"
            return f"3. Обновить документацию и зависимости\n"

        return f"""

## [РЕЗУЛЬТАТЫ] Ожидаемые результаты

### Количественные улучшения
- Сокращение God Objects: {len(god_objects)} -> 0
- Устранение дубликатов: {len(duplicates)} групп
- Консолидация модулей: {sum(info['count'] for info in clusters.values() if info['consolidation_opportunity'])} -> {len([info for info in clusters.values() if info['consolidation_opportunity']])} пакетов
- Очистка кода: -{len(dead_modules)} неиспользуемых модулей

### Качественные улучшения
- Четкое разделение ответственности
- Улучшенная поддерживаемость
- Упрощенное тестирование
- Лучшая навигация по коду

## [РИСКИ] Риски и митигация

### Высокие риски
- **Нарушение существующих интерфейсов** -> Создать характеризационные тесты
- **Сложность миграции** -> Поэтапный подход с обратной совместимостью

### Средние риски
- **Временные затраты** -> Приоритизация по критичности
- **Регрессии** -> Автоматизированное тестирование

## [ВРЕМЯ] Временные рамки

**Общее время:** 5-10 недель
- Фаза 1 (God Objects): 2-4 недели
- Фаза 2 (Дубликаты): 1-2 недели  
- Фаза 3 (Консолидация): 1-3 недели
- Фаза 4 (Очистка): 1 неделя

---
*План создан анализатором архитектурной декомпозиции проекта*
*Основан на структурном анализе {len(self.project_map['modules'])} модулей*
"""

    def generate_visualizations(self):
        """9. Генерация диаграмм и отчетов"""
        self.logger.info("Генерация диаграмм и отчетов...")
        
        try:
            # Создаем диаграмму зависимостей в формате Mermaid
            mermaid_content = self._generate_dependency_diagram()
            mermaid_path = self.output_dir / f"project_dependencies_{self.timestamp}.mmd"
            
            with open(mermaid_path, 'w', encoding='utf-8') as f:
                f.write(mermaid_content)
            
            self.analysis_results["generated_files"].append(str(mermaid_path))
            
            # Создаем матрицу функциональности
            matrix_content = self._generate_functionality_matrix()
            matrix_path = self.output_dir / f"functionality_matrix_{self.timestamp}.md"
            
            with open(matrix_path, 'w', encoding='utf-8') as f:
                f.write(matrix_content)
            
            self.analysis_results["generated_files"].append(str(matrix_path))
            
            self.logger.info("Диаграммы и отчеты созданы")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка генерации визуализаций: {e}")
            return False

    def _generate_dependency_diagram(self) -> str:
        """Генерирует диаграмму зависимостей в формате Mermaid"""
        mermaid = ["graph TD"]
        
        # Добавляем узлы и связи
        for module, dependencies in self.project_map['dependencies'].items():
            module_name = Path(module).stem
            
            for dep in dependencies:
                dep_name = Path(dep).stem if dep else "unknown"
                mermaid.append(f"    {module_name} --> {dep_name}")
        
        # Выделяем God Objects
        for cls_key, god_info in self.project_map.get('god_objects', {}).items():
            cls_info = god_info['class_info']
            module_name = Path(cls_info['module']).stem
            mermaid.append(f"    {module_name}:::godObject")
        
        # Стили
        mermaid.extend([
            "",
            "    classDef godObject fill:#ff9999,stroke:#ff0000,stroke-width:2px"
        ])
        
        return "\n".join(mermaid)

    def _generate_functionality_matrix(self) -> str:
        """Генерирует матрицу функциональности"""
        # Собираем все функциональности
        all_functionalities = set()
        module_functionalities = {}
        
        for module_path, module_info in self.project_map['modules'].items():
            module_func = set()
            
            for cls in module_info['classes']:
                module_func.update(cls['functionality_keywords'])
            
            for func in module_info['functions']:
                module_func.update(func['functionality_keywords'])
            
            module_functionalities[module_path] = module_func
            all_functionalities.update(module_func)
        
        # Создаем матрицу
        matrix = ["# Матрица функциональности проекта\n"]
        matrix.append("| Модуль | " + " | ".join(sorted(all_functionalities)) + " |")
        matrix.append("|" + "---|" * (len(all_functionalities) + 1))
        
        for module_path, functionalities in module_functionalities.items():
            module_name = Path(module_path).name
            row = [module_name]
            
            for func in sorted(all_functionalities):
                row.append("✓" if func in functionalities else " ")
            
            matrix.append("| " + " | ".join(row) + " |")
        
        return "\n".join(matrix)

    def save_decomposition_results(self):
        """Сохранение результатов анализа декомпозиции"""
        self.logger.info("Сохранение результатов анализа декомпозиции...")
        
        # Сохраняем полные данные в JSON
        results_path = self.output_dir / f"PROJECT_DECOMPOSITION_DATA_{self.timestamp}.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(self.project_map, f, ensure_ascii=False, indent=2, default=str)
        
        self.analysis_results["generated_files"].append(str(results_path))
        
        # Создаем краткий отчет
        summary_path = self.output_dir / f"DECOMPOSITION_SUMMARY_{self.timestamp}.md"
        summary_content = self._create_decomposition_summary()
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)
        
        self.analysis_results["generated_files"].append(str(summary_path))
        
        self.logger.info(f"Результаты сохранены: {results_path}")
        self.logger.info(f"Краткий отчет: {summary_path}")

    def _create_decomposition_summary(self) -> str:
        """Создает краткий отчет по декомпозиции"""
        god_objects = self.project_map.get('god_objects', {})
        duplicates = self.project_map.get('duplicates', {})
        clusters = self.project_map.get('feature_clusters', {})
        
        return f"""# Краткий отчет по декомпозиции проекта

**Проект:** {self.project_path.name}
**Дата:** {self.timestamp}

## Основные находки

### Структура проекта
- Модулей: {len(self.project_map['modules'])}
- Функций: {len(self.project_map['functions'])}
- Классов: {len(self.project_map['classes'])}

### Архитектурные проблемы
- God Objects: {len(god_objects)}
- Функциональные дубликаты: {len(duplicates)} групп
- Возможности консолидации: {len([c for c in clusters.values() if c['consolidation_opportunity']])}

## Приоритетные действия

1. **Декомпозиция God Objects** - {len(god_objects)} классов
2. **Устранение дубликатов** - {len(duplicates)} групп
3. **Консолидация функциональности** - {len(clusters)} кластеров

## Следующие шаги

1. Изучите детальный план: `PROJECT_DECOMPOSITION_PLAN_{self.timestamp}.md`
2. Просмотрите диаграмму зависимостей: `project_dependencies_{self.timestamp}.mmd`
3. Изучите матрицу функциональности: `functionality_matrix_{self.timestamp}.md`

---
*Создано анализатором архитектурной декомпозиции проекта*
"""


def main():
    """Главная функция для запуска из командной строки"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Анализатор архитектурной декомпозиции проекта",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python project_decomposition_analyzer.py /path/to/project /path/to/output
  python project_decomposition_analyzer.py C:\\Project C:\\Results --verbose
        """,
    )

    parser.add_argument("project_path", help="Путь к корневой папке проекта")
    parser.add_argument("output_dir", help="Директория для сохранения результатов анализа")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод процесса анализа")

    args = parser.parse_args()

    # Проверяем существование путей
    project_path = Path(args.project_path)

    if not project_path.exists():
        print(f"Ошибка: Проект не найден: {project_path}")
        sys.exit(1)

    if not project_path.is_dir():
        print(f"Ошибка: Путь к проекту должен быть папкой: {project_path}")
        sys.exit(1)

    # Создаем и запускаем анализатор
    try:
        analyzer = ProjectDecompositionAnalyzer(
            str(project_path), args.output_dir, args.verbose
        )

        print("=" * 80)
        print("АНАЛИЗАТОР АРХИТЕКТУРНОЙ ДЕКОМПОЗИЦИИ ПРОЕКТА")
        print("=" * 80)
        print(f"Проект: {project_path}")
        print(f"Результаты: {args.output_dir}")
        print("Анализ структуры проекта для оптимальной реорганизации!")
        print("=" * 80)

        success = analyzer.run_project_decomposition()

        if success:
            print("\n" + "=" * 80)
            print("[УСПЕХ] АНАЛИЗ ДЕКОМПОЗИЦИИ ЗАВЕРШЕН УСПЕШНО!")
            print("=" * 80)
            print(f"Результаты сохранены в: {args.output_dir}")
            print(f"План декомпозиции: PROJECT_DECOMPOSITION_PLAN_{analyzer.timestamp}.md")
            print(f"Полные данные: PROJECT_DECOMPOSITION_DATA_{analyzer.timestamp}.json")
            print(f"Краткий отчет: DECOMPOSITION_SUMMARY_{analyzer.timestamp}.md")
            print("[ГОТОВО] ПЛАН АРХИТЕКТУРНОЙ РЕОРГАНИЗАЦИИ!")
        else:
            print("\n" + "=" * 80)
            print("[ПРЕДУПРЕЖДЕНИЕ] АНАЛИЗ ЗАВЕРШЕН С ПРЕДУПРЕЖДЕНИЯМИ")
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
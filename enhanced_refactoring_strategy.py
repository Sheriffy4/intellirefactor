#!/usr/bin/env python3
"""
Улучшенная стратегия рефакторинга для IntelliRefactor.

Этот модуль создает более агрессивную и качественную стратегию рефакторинга,
основанную на семантическом анализе кода и выделении четких ответственностей.
"""

from pathlib import Path
from typing import Dict, List, Any
import ast
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class MethodAnalysis:
    """Детальный анализ метода."""
    name: str
    line_start: int
    line_end: int
    size_lines: int
    complexity_score: int
    is_private: bool
    is_dunder: bool
    calls_methods: List[str]
    uses_attributes: List[str]
    semantic_keywords: List[str]
    responsibility_score: Dict[str, float]

@dataclass
class ComponentPlan:
    """План компонента для извлечения."""
    name: str
    description: str
    methods: List[str]
    interface_methods: List[str]
    estimated_lines: int
    cohesion_score: float
    dependencies: List[str]

class EnhancedRefactoringStrategy:
    """Улучшенная стратегия рефакторинга с семантическим анализом."""
    
    def __init__(self):
        # Семантические паттерны для группировки методов
        self.semantic_patterns = {
            'dispatch_orchestration': {
                'keywords': ['dispatch', 'orchestrat', 'coordinate', 'route', 'execute'],
                'patterns': [r'dispatch.*', r'.*_internal', r'.*_wrapper'],
                'description': 'Оркестрация и маршрутизация атак',
                'weight': 1.0
            },
            'strategy_resolution': {
                'keywords': ['strategy', 'resolve', 'parse', 'combo', 'recipe'],
                'patterns': [r'.*strategy.*', r'resolve.*', r'parse.*', r'.*combo.*'],
                'description': 'Разрешение стратегий и рецептов',
                'weight': 0.9
            },
            'parameter_management': {
                'keywords': ['param', 'normalize', 'validate', 'map', 'filter'],
                'patterns': [r'.*param.*', r'normalize.*', r'validate.*', r'map.*'],
                'description': 'Управление параметрами атак',
                'weight': 0.8
            },
            'tls_protocol_handling': {
                'keywords': ['tls', 'sni', 'cipher', 'extension', 'hostname', 'clienthello'],
                'patterns': [r'.*sni.*', r'.*tls.*', r'.*cipher.*', r'.*hostname.*'],
                'description': 'Обработка TLS протокола и SNI',
                'weight': 0.9
            },
            'attack_execution': {
                'keywords': ['attack', 'execute', 'primitive', 'advanced', 'technique'],
                'patterns': [r'.*attack.*', r'execute.*', r'.*primitive.*'],
                'description': 'Выполнение атак и техник',
                'weight': 0.8
            },
            'logging_monitoring': {
                'keywords': ['log', 'monitor', 'trace', 'correlation', 'metadata'],
                'patterns': [r'.*log.*', r'.*correlation.*', r'.*metadata.*'],
                'description': 'Логирование и мониторинг',
                'weight': 0.7
            },
            'data_processing': {
                'keywords': ['parse', 'extract', 'find', 'position', 'offset'],
                'patterns': [r'.*parse.*', r'find.*', r'extract.*', r'.*position.*'],
                'description': 'Обработка и парсинг данных',
                'weight': 0.6
            },
            'utility_helpers': {
                'keywords': ['helper', 'util', 'support', 'create', 'generate'],
                'patterns': [r'.*helper.*', r'create.*', r'generate.*', r'.*util.*'],
                'description': 'Вспомогательные функции',
                'weight': 0.5
            }
        }
        
        # Минимальные требования для создания компонента
        self.min_methods_per_component = 2  # Снижаем до 2
        self.min_lines_per_component = 30   # Снижаем до 30
        self.min_cohesion_score = 0.1       # Снижаем до 0.1
        
    def analyze_method(self, method_node: ast.FunctionDef, content: str) -> MethodAnalysis:
        """Детальный анализ метода."""
        
        lines = content.splitlines()
        method_lines = lines[method_node.lineno-1:getattr(method_node, 'end_lineno', method_node.lineno+10)]
        method_content = '\n'.join(method_lines)
        
        # Извлекаем вызовы методов
        calls = re.findall(r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', method_content)
        
        # Извлекаем использование атрибутов
        attributes = re.findall(r'self\.([a-zA-Z_][a-zA-Z0-9_]*)', method_content)
        attributes = [attr for attr in attributes if attr not in calls]  # Исключаем вызовы методов
        
        # Извлекаем семантические ключевые слова
        semantic_keywords = self._extract_semantic_keywords(method_content, method_node.name)
        
        # Вычисляем оценки ответственности
        responsibility_score = self._calculate_responsibility_scores(method_node.name, method_content)
        
        return MethodAnalysis(
            name=method_node.name,
            line_start=method_node.lineno,
            line_end=getattr(method_node, 'end_lineno', method_node.lineno + 10),
            size_lines=len(method_lines),
            complexity_score=self._calculate_complexity(method_content),
            is_private=method_node.name.startswith('_'),
            is_dunder=method_node.name.startswith('__') and method_node.name.endswith('__'),
            calls_methods=list(set(calls)),
            uses_attributes=list(set(attributes)),
            semantic_keywords=semantic_keywords,
            responsibility_score=responsibility_score
        )
    
    def _extract_semantic_keywords(self, content: str, method_name: str) -> List[str]:
        """Извлекает семантические ключевые слова из метода."""
        keywords = []
        content_lower = content.lower()
        name_lower = method_name.lower()
        
        # Ключевые слова из имени метода
        for word in re.findall(r'[a-zA-Z]+', name_lower):
            if len(word) > 2:
                keywords.append(word)
        
        # Ключевые слова из комментариев и строк
        comments = re.findall(r'#\s*(.+)', content)
        strings = re.findall(r'["\']([^"\']+)["\']', content)
        
        for text in comments + strings:
            for word in re.findall(r'[a-zA-Z]+', text.lower()):
                if len(word) > 3:
                    keywords.append(word)
        
        return list(set(keywords))
    
    def _calculate_responsibility_scores(self, method_name: str, content: str) -> Dict[str, float]:
        """Вычисляет оценки ответственности для каждой категории."""
        scores = {}
        content_lower = content.lower()
        name_lower = method_name.lower()
        
        for category, config in self.semantic_patterns.items():
            score = 0.0
            
            # Оценка по ключевым словам
            for keyword in config['keywords']:
                if keyword in name_lower:
                    score += 2.0  # Высокий вес для имени метода
                if keyword in content_lower:
                    score += 1.0
            
            # Оценка по паттернам
            for pattern in config['patterns']:
                if re.search(pattern, name_lower):
                    score += 1.5
            
            # Нормализация с учетом веса категории
            scores[category] = score * config['weight']
        
        return scores
    
    def _calculate_complexity(self, content: str) -> int:
        """Вычисляет сложность метода."""
        complexity = 1
        
        # Условные конструкции
        complexity += content.count('if ')
        complexity += content.count('elif ')
        complexity += content.count('else:')
        
        # Циклы
        complexity += content.count('for ')
        complexity += content.count('while ')
        
        # Обработка исключений
        complexity += content.count('try:')
        complexity += content.count('except')
        complexity += content.count('finally:')
        
        # Контекстные менеджеры
        complexity += content.count('with ')
        
        # Логические операторы
        complexity += content.count(' and ')
        complexity += content.count(' or ')
        
        return complexity
    
    def group_methods_by_responsibility(self, methods: List[MethodAnalysis]) -> Dict[str, List[str]]:
        """Группирует методы по ответственностям с использованием семантического анализа."""
        
        groups = {category: [] for category in self.semantic_patterns.keys()}
        unassigned = []
        
        for method in methods:
            # Находим категорию с максимальной оценкой
            if method.responsibility_score:
                best_category = max(method.responsibility_score.items(), key=lambda x: x[1])
                
                if best_category[1] > 0.5:  # Снижаем минимальный порог
                    groups[best_category[0]].append(method.name)
                else:
                    unassigned.append(method.name)
            else:
                unassigned.append(method.name)
        
        # Обрабатываем неназначенные методы
        if unassigned:
            groups['utility_helpers'].extend(unassigned)
        
        # Удаляем пустые группы
        return {k: v for k, v in groups.items() if v}
    
    def calculate_cohesion_score(self, method_names: List[str], methods: List[MethodAnalysis]) -> float:
        """Вычисляет оценку связности группы методов."""
        
        if len(method_names) < 2:
            return 0.0
        
        group_methods = [m for m in methods if m.name in method_names]
        
        # Анализируем взаимные вызовы
        call_connections = 0
        total_possible_connections = len(group_methods) * (len(group_methods) - 1)
        
        for method in group_methods:
            for other_method in group_methods:
                if other_method.name in method.calls_methods:
                    call_connections += 1
        
        call_cohesion = call_connections / max(total_possible_connections, 1)
        
        # Анализируем общие атрибуты
        all_attributes = set()
        for method in group_methods:
            all_attributes.update(method.uses_attributes)
        
        if all_attributes:
            shared_attributes = 0
            for attr in all_attributes:
                using_methods = sum(1 for m in group_methods if attr in m.uses_attributes)
                if using_methods > 1:
                    shared_attributes += using_methods - 1
            
            attr_cohesion = shared_attributes / (len(all_attributes) * len(group_methods))
        else:
            attr_cohesion = 0.0
        
        # Анализируем семантическую близость
        semantic_cohesion = self._calculate_semantic_cohesion(group_methods)
        
        # Итоговая оценка связности
        return (call_cohesion * 0.4 + attr_cohesion * 0.3 + semantic_cohesion * 0.3)
    
    def _calculate_semantic_cohesion(self, methods: List[MethodAnalysis]) -> float:
        """Вычисляет семантическую связность методов."""
        
        if len(methods) < 2:
            return 0.0
        
        # Собираем все семантические ключевые слова
        all_keywords = set()
        for method in methods:
            all_keywords.update(method.semantic_keywords)
        
        if not all_keywords:
            return 0.0
        
        # Вычисляем пересечения ключевых слов
        shared_keywords = 0
        for keyword in all_keywords:
            methods_with_keyword = sum(1 for m in methods if keyword in m.semantic_keywords)
            if methods_with_keyword > 1:
                shared_keywords += methods_with_keyword - 1
        
        return shared_keywords / (len(all_keywords) * len(methods))
    
    def create_component_plans(self, groups: Dict[str, List[str]], methods: List[MethodAnalysis]) -> List[ComponentPlan]:
        """Создает планы компонентов на основе групп методов."""
        
        component_plans = []
        
        for group_name, method_names in groups.items():
            if len(method_names) < self.min_methods_per_component:
                continue
            
            group_methods = [m for m in methods if m.name in method_names]
            total_lines = sum(m.size_lines for m in group_methods)
            
            if total_lines < self.min_lines_per_component:
                continue
            
            cohesion_score = self.calculate_cohesion_score(method_names, methods)
            
            if cohesion_score < self.min_cohesion_score:
                continue
            
            # Определяем публичные методы (интерфейс)
            interface_methods = [m.name for m in group_methods if not m.is_private and not m.is_dunder]
            
            # Анализируем зависимости
            dependencies = set()
            for method in group_methods:
                for call in method.calls_methods:
                    if call not in method_names:  # Внешний вызов
                        dependencies.add(call)
            
            component_plan = ComponentPlan(
                name=self._generate_component_name(group_name),
                description=self.semantic_patterns[group_name]['description'],
                methods=method_names,
                interface_methods=interface_methods,
                estimated_lines=total_lines,
                cohesion_score=cohesion_score,
                dependencies=list(dependencies)
            )
            
            component_plans.append(component_plan)
        
        return component_plans
    
    def _generate_component_name(self, group_name: str) -> str:
        """Генерирует имя компонента на основе группы."""
        name_mapping = {
            'dispatch_orchestration': 'AttackOrchestrator',
            'strategy_resolution': 'StrategyResolver',
            'parameter_management': 'ParameterManager',
            'tls_protocol_handling': 'TlsProtocolHandler',
            'attack_execution': 'AttackExecutor',
            'logging_monitoring': 'LoggingMonitor',
            'data_processing': 'DataProcessor',
            'utility_helpers': 'UtilityHelper'
        }
        return name_mapping.get(group_name, f'{group_name.title()}Service')
    
    def generate_enhanced_refactoring_config(self, file_path: Path) -> Dict[str, Any]:
        """Генерирует улучшенную конфигурацию рефакторинга."""
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        content = file_path.read_text(encoding='utf-8')
        tree = ast.parse(content)
        
        # Находим класс AttackDispatcher
        target_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "AttackDispatcher":
                target_class = node
                break
        
        if not target_class:
            raise ValueError("AttackDispatcher class not found")
        
        # Анализируем все методы
        methods = []
        for item in target_class.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_analysis = self.analyze_method(item, content)
                methods.append(method_analysis)
        
        # Группируем методы
        groups = self.group_methods_by_responsibility(methods)
        
        # Создаем планы компонентов
        component_plans = self.create_component_plans(groups, methods)
        
        # Генерируем конфигурацию для IntelliRefactor
        enhanced_config = {
            'god_class_threshold': 5,  # Более агрессивный порог
            'min_methods_for_extraction': 3,  # Минимум 3 метода на компонент
            'extract_private_methods': True,
            'cohesion_cluster_other': True,
            'cohesion_similarity_threshold': 0.25,  # Более низкий порог для большего извлечения
            
            # Улучшенные ключевые слова ответственности
            'responsibility_keywords': {
                component.name.lower().replace('service', '').replace('handler', '').replace('manager', ''): 
                self._extract_keywords_from_methods(component.methods, methods)
                for component in component_plans
            },
            
            # Метаданные для улучшенного рефакторинга
            'enhanced_refactoring': {
                'total_methods_analyzed': len(methods),
                'proposed_components': len(component_plans),
                'expected_extraction_rate': sum(len(cp.methods) for cp in component_plans) / len(methods),
                'component_plans': [
                    {
                        'name': cp.name,
                        'description': cp.description,
                        'methods': cp.methods,
                        'interface_methods': cp.interface_methods,
                        'estimated_lines': cp.estimated_lines,
                        'cohesion_score': cp.cohesion_score,
                        'dependencies': cp.dependencies
                    }
                    for cp in component_plans
                ]
            }
        }
        
        return enhanced_config
    
    def _extract_keywords_from_methods(self, method_names: List[str], all_methods: List[MethodAnalysis]) -> List[str]:
        """Извлекает ключевые слова из группы методов."""
        keywords = set()
        
        for method_name in method_names:
            method = next((m for m in all_methods if m.name == method_name), None)
            if method:
                # Добавляем слова из имени метода
                name_words = re.findall(r'[a-zA-Z]+', method_name.lower())
                keywords.update(word for word in name_words if len(word) > 2)
                
                # Добавляем семантические ключевые слова
                keywords.update(method.semantic_keywords[:5])  # Топ-5 ключевых слов
        
        return list(keywords)[:10]  # Ограничиваем 10 ключевыми словами

def main():
    """Демонстрация улучшенной стратегии рефакторинга."""
    
    strategy = EnhancedRefactoringStrategy()
    file_path = Path('core/bypass/engine/attack_dispatcher.py')
    
    if not file_path.exists():
        print(f"❌ Файл не найден: {file_path}")
        return
    
    try:
        config = strategy.generate_enhanced_refactoring_config(file_path)
        
        print("🚀 УЛУЧШЕННАЯ СТРАТЕГИЯ РЕФАКТОРИНГА")
        print("=" * 50)
        
        enhanced = config['enhanced_refactoring']
        print("📊 Анализ:")
        print(f"  Всего методов: {enhanced['total_methods_analyzed']}")
        print(f"  Предлагаемых компонентов: {enhanced['proposed_components']}")
        print(f"  Ожидаемый процент извлечения: {enhanced['expected_extraction_rate']:.1%}")
        
        print("\n🏗️ Планы компонентов:")
        for plan in enhanced['component_plans']:
            print(f"\n📦 {plan['name']}")
            print(f"   {plan['description']}")
            print(f"   Методов: {len(plan['methods'])} (строк: {plan['estimated_lines']})")
            print(f"   Связность: {plan['cohesion_score']:.2f}")
            print(f"   Интерфейс: {len(plan['interface_methods'])} публичных методов")
            if plan['dependencies']:
                print(f"   Зависимости: {len(plan['dependencies'])} внешних вызовов")
        
        # Сохраняем конфигурацию
        config_file = Path('enhanced_refactoring_config.json')
        import json
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Конфигурация сохранена в: {config_file}")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")

if __name__ == "__main__":
    main()
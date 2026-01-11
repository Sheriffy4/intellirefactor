#!/usr/bin/env python3
"""
Улучшенная система группировки методов для IntelliRefactor.

Этот модуль предоставляет более качественную группировку методов на основе
семантического анализа и анализа связности, что приводит к более эффективному
рефакторингу с большим процентом извлечения кода.
"""

from __future__ import annotations

import ast
import re
import logging
from typing import Dict, List, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EnhancedMethodInfo:
    """Расширенная информация о методе для улучшенного анализа."""
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
    responsibility_scores: Dict[str, float]
    content: str

class EnhancedMethodGrouping:
    """Улучшенная система группировки методов."""
    
    def __init__(self):
        # Семантические паттерны для более точной группировки
        self.semantic_patterns = {
            'orchestration': {
                'keywords': ['dispatch', 'orchestrat', 'coordinate', 'route', 'execute', 'internal'],
                'patterns': [r'dispatch.*', r'.*_internal', r'.*_wrapper', r'orchestrat.*'],
                'description': 'Оркестрация и координация выполнения',
                'weight': 1.0,
                'component_name': 'ExecutionOrchestrator'
            },
            'strategy_resolution': {
                'keywords': ['strategy', 'resolve', 'parse', 'combo', 'recipe', 'combination'],
                'patterns': [r'.*strategy.*', r'resolve.*', r'parse.*standard.*', r'.*combo.*', r'.*recipe.*'],
                'description': 'Разрешение стратегий и рецептов атак',
                'weight': 0.9,
                'component_name': 'StrategyResolver'
            },
            'parameter_processing': {
                'keywords': ['param', 'normalize', 'validate', 'map', 'filter', 'critical'],
                'patterns': [r'.*param.*', r'normalize.*', r'validate.*', r'map.*', r'.*critical.*'],
                'description': 'Обработка и валидация параметров',
                'weight': 0.8,
                'component_name': 'ParameterProcessor'
            },
            'protocol_handling': {
                'keywords': ['tls', 'sni', 'cipher', 'extension', 'hostname', 'clienthello', 'parse'],
                'patterns': [r'.*sni.*', r'.*tls.*', r'.*cipher.*', r'.*hostname.*', r'.*extension.*'],
                'description': 'Обработка сетевых протоколов',
                'weight': 0.9,
                'component_name': 'ProtocolHandler'
            },
            'attack_execution': {
                'keywords': ['attack', 'execute', 'primitive', 'advanced', 'technique', 'disorder'],
                'patterns': [r'.*attack.*', r'execute.*', r'.*primitive.*', r'.*advanced.*', r'.*disorder.*'],
                'description': 'Выполнение атак и техник обхода',
                'weight': 0.8,
                'component_name': 'AttackExecutor'
            },
            'logging_monitoring': {
                'keywords': ['log', 'monitor', 'trace', 'correlation', 'metadata', 'operation'],
                'patterns': [r'.*log.*', r'.*correlation.*', r'.*metadata.*', r'.*operation.*'],
                'description': 'Логирование и мониторинг операций',
                'weight': 0.7,
                'component_name': 'OperationLogger'
            },
            'data_processing': {
                'keywords': ['find', 'extract', 'position', 'offset', 'parse', 'legacy'],
                'patterns': [r'find.*', r'extract.*', r'.*position.*', r'.*offset.*', r'.*legacy.*'],
                'description': 'Обработка и извлечение данных',
                'weight': 0.6,
                'component_name': 'DataProcessor'
            },
            'utility_support': {
                'keywords': ['helper', 'util', 'support', 'create', 'generate', 'valid'],
                'patterns': [r'.*helper.*', r'create.*', r'generate.*', r'.*util.*', r'.*valid.*'],
                'description': 'Вспомогательные функции и утилиты',
                'weight': 0.5,
                'component_name': 'UtilityService'
            }
        }
        
        # Настройки для группировки
        self.min_methods_per_group = 2
        self.min_responsibility_score = 0.3
        self.cohesion_weight = 0.4
        self.semantic_weight = 0.6
    
    def analyze_method_enhanced(self, method_node: ast.FunctionDef, content: str) -> EnhancedMethodInfo:
        """Расширенный анализ метода."""
        
        lines = content.splitlines()
        start_line = method_node.lineno - 1
        end_line = getattr(method_node, 'end_lineno', method_node.lineno + 10) - 1
        
        method_lines = lines[start_line:end_line + 1]
        method_content = '\n'.join(method_lines)
        
        # Извлекаем вызовы методов
        calls = re.findall(r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', method_content)
        
        # Извлекаем использование атрибутов
        attributes = re.findall(r'self\.([a-zA-Z_][a-zA-Z0-9_]*)', method_content)
        attributes = [attr for attr in set(attributes) if attr not in calls]
        
        # Извлекаем семантические ключевые слова
        semantic_keywords = self._extract_semantic_keywords(method_content, method_node.name)
        
        # Вычисляем оценки ответственности
        responsibility_scores = self._calculate_responsibility_scores(method_node.name, method_content)
        
        return EnhancedMethodInfo(
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
            responsibility_scores=responsibility_scores,
            content=method_content
        )
    
    def _extract_semantic_keywords(self, content: str, method_name: str) -> List[str]:
        """Извлекает семантические ключевые слова."""
        keywords = set()
        content_lower = content.lower()
        
        # Ключевые слова из имени метода
        name_words = re.findall(r'[a-zA-Z]+', method_name.lower())
        keywords.update(word for word in name_words if len(word) > 2)
        
        # Ключевые слова из комментариев
        comments = re.findall(r'#\s*(.+)', content)
        for comment in comments:
            words = re.findall(r'[a-zA-Z]+', comment.lower())
            keywords.update(word for word in words if len(word) > 3)
        
        # Ключевые слова из строковых литералов
        strings = re.findall(r'["\']([^"\']{4,})["\']', content)
        for string in strings:
            words = re.findall(r'[a-zA-Z]+', string.lower())
            keywords.update(word for word in words if len(word) > 3)
        
        # Ключевые слова из имен переменных
        variables = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)\s*=', content)
        for var in variables:
            if not var.isupper():  # Исключаем константы
                var_words = re.findall(r'[a-zA-Z]+', var.lower())
                keywords.update(word for word in var_words if len(word) > 2)
        
        return list(keywords)
    
    def _calculate_responsibility_scores(self, method_name: str, content: str) -> Dict[str, float]:
        """Вычисляет оценки ответственности для каждой категории."""
        scores = {}
        content_lower = content.lower()
        name_lower = method_name.lower()
        
        for category, config in self.semantic_patterns.items():
            score = 0.0
            
            # Высокий вес для совпадений в имени метода
            for keyword in config['keywords']:
                if keyword in name_lower:
                    score += 3.0
                # Частичные совпадения в содержимом
                content_matches = len(re.findall(rf'\b{keyword}\b', content_lower))
                score += content_matches * 0.5
            
            # Оценка по регулярным выражениям
            for pattern in config['patterns']:
                if re.search(pattern, name_lower):
                    score += 2.0
                # Поиск паттернов в содержимом
                content_pattern_matches = len(re.findall(pattern, content_lower))
                score += content_pattern_matches * 0.3
            
            # Нормализация с учетом веса категории
            scores[category] = score * config['weight']
        
        return scores
    
    def _calculate_complexity(self, content: str) -> int:
        """Вычисляет цикломатическую сложность."""
        complexity = 1
        
        # Условные конструкции
        complexity += len(re.findall(r'\bif\b', content))
        complexity += len(re.findall(r'\belif\b', content))
        
        # Циклы
        complexity += len(re.findall(r'\bfor\b', content))
        complexity += len(re.findall(r'\bwhile\b', content))
        
        # Обработка исключений
        complexity += len(re.findall(r'\btry\b', content))
        complexity += len(re.findall(r'\bexcept\b', content))
        
        # Логические операторы
        complexity += len(re.findall(r'\band\b', content))
        complexity += len(re.findall(r'\bor\b', content))
        
        return complexity
    
    def group_methods_enhanced(self, methods: List[EnhancedMethodInfo]) -> Dict[str, List[str]]:
        """Улучшенная группировка методов по ответственностям."""
        
        # Исключаем dunder методы из группировки
        groupable_methods = [m for m in methods if not m.is_dunder]
        
        groups = {}
        assigned_methods = set()
        
        # Первый проход: назначаем методы с четкими ответственностями
        for method in groupable_methods:
            if method.name in assigned_methods:
                continue
                
            best_category = None
            best_score = 0.0
            
            for category, score in method.responsibility_scores.items():
                if score > best_score and score >= self.min_responsibility_score:
                    best_score = score
                    best_category = category
            
            if best_category:
                if best_category not in groups:
                    groups[best_category] = []
                groups[best_category].append(method.name)
                assigned_methods.add(method.name)
        
        # Второй проход: группируем оставшиеся методы по связности
        unassigned = [m for m in groupable_methods if m.name not in assigned_methods]
        
        if unassigned:
            cohesion_groups = self._group_by_cohesion(unassigned)
            
            for i, cohesion_group in enumerate(cohesion_groups):
                if len(cohesion_group) >= self.min_methods_per_group:
                    group_name = f'cohesion_group_{i + 1}'
                    groups[group_name] = [m.name for m in cohesion_group]
                    assigned_methods.update(m.name for m in cohesion_group)
        
        # Третий проход: оставшиеся методы в утилиты
        still_unassigned = [m for m in groupable_methods if m.name not in assigned_methods]
        if still_unassigned:
            if 'utility_support' not in groups:
                groups['utility_support'] = []
            groups['utility_support'].extend(m.name for m in still_unassigned)
        
        # Фильтруем группы по минимальному размеру
        filtered_groups = {}
        for group_name, method_names in groups.items():
            if len(method_names) >= self.min_methods_per_group:
                filtered_groups[group_name] = method_names
        
        return filtered_groups
    
    def _group_by_cohesion(self, methods: List[EnhancedMethodInfo]) -> List[List[EnhancedMethodInfo]]:
        """Группирует методы по связности."""
        
        if len(methods) < 2:
            return [methods] if methods else []
        
        # Вычисляем матрицу связности
        cohesion_matrix = {}
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i != j:
                    cohesion = self._calculate_method_cohesion(method1, method2)
                    cohesion_matrix[(i, j)] = cohesion
        
        # Кластеризация методов
        clusters = []
        used_indices = set()
        
        for i, method in enumerate(methods):
            if i in used_indices:
                continue
                
            cluster = [method]
            cluster_indices = {i}
            
            # Находим связанные методы
            for j, other_method in enumerate(methods):
                if j != i and j not in used_indices:
                    avg_cohesion = sum(
                        cohesion_matrix.get((k, j), 0) for k in cluster_indices
                    ) / len(cluster_indices)
                    
                    if avg_cohesion > 0.2:  # Порог связности
                        cluster.append(other_method)
                        cluster_indices.add(j)
            
            if len(cluster) >= self.min_methods_per_group:
                clusters.append(cluster)
                used_indices.update(cluster_indices)
        
        return clusters
    
    def _calculate_method_cohesion(self, method1: EnhancedMethodInfo, method2: EnhancedMethodInfo) -> float:
        """Вычисляет связность между двумя методами."""
        
        # Связность по вызовам методов
        call_cohesion = 0.0
        if method2.name in method1.calls_methods or method1.name in method2.calls_methods:
            call_cohesion = 1.0
        
        # Связность по общим атрибутам
        common_attributes = set(method1.uses_attributes) & set(method2.uses_attributes)
        attr_cohesion = len(common_attributes) / max(
            len(set(method1.uses_attributes) | set(method2.uses_attributes)), 1
        )
        
        # Семантическая связность
        common_keywords = set(method1.semantic_keywords) & set(method2.semantic_keywords)
        semantic_cohesion = len(common_keywords) / max(
            len(set(method1.semantic_keywords) | set(method2.semantic_keywords)), 1
        )
        
        # Связность по ответственностям
        responsibility_cohesion = 0.0
        for category in self.semantic_patterns.keys():
            score1 = method1.responsibility_scores.get(category, 0)
            score2 = method2.responsibility_scores.get(category, 0)
            if score1 > 0 and score2 > 0:
                responsibility_cohesion = max(responsibility_cohesion, min(score1, score2) / max(score1, score2))
        
        # Итоговая связность
        total_cohesion = (
            call_cohesion * 0.4 +
            attr_cohesion * 0.2 +
            semantic_cohesion * 0.2 +
            responsibility_cohesion * 0.2
        )
        
        return total_cohesion
    
    def generate_enhanced_responsibility_keywords(self, groups: Dict[str, List[str]], 
                                                methods: List[EnhancedMethodInfo]) -> Dict[str, List[str]]:
        """Генерирует улучшенные ключевые слова ответственности для IntelliRefactor."""
        
        enhanced_keywords = {}
        
        for group_name, method_names in groups.items():
            group_methods = [m for m in methods if m.name in method_names]
            
            # Собираем ключевые слова из методов группы
            keywords = set()
            
            for method in group_methods:
                # Ключевые слова из имени метода
                name_words = re.findall(r'[a-zA-Z]+', method.name.lower())
                keywords.update(word for word in name_words if len(word) > 2)
                
                # Топ семантические ключевые слова
                keywords.update(method.semantic_keywords[:3])
            
            # Добавляем ключевые слова из семантических паттернов
            if group_name in self.semantic_patterns:
                keywords.update(self.semantic_patterns[group_name]['keywords'])
            
            # Генерируем имя компонента
            component_name = self._generate_component_name(group_name, group_methods)
            enhanced_keywords[component_name.lower()] = list(keywords)[:15]  # Ограничиваем 15 словами
        
        return enhanced_keywords
    
    def _generate_component_name(self, group_name: str, methods: List[EnhancedMethodInfo]) -> str:
        """Генерирует имя компонента."""
        
        if group_name in self.semantic_patterns:
            return self.semantic_patterns[group_name]['component_name']
        
        # Для групп связности генерируем имя на основе доминирующих ключевых слов
        all_keywords = []
        for method in methods:
            all_keywords.extend(method.semantic_keywords)
        
        if all_keywords:
            # Находим наиболее частые ключевые слова
            keyword_counts = {}
            for keyword in all_keywords:
                keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
            
            top_keyword = max(keyword_counts.items(), key=lambda x: x[1])[0]
            return f"{top_keyword.title()}Service"
        
        return f"Component{group_name.title()}"

def integrate_enhanced_grouping_with_auto_refactor():
    """Интегрирует улучшенную группировку с AutoRefactor."""
    
    enhanced_grouping = EnhancedMethodGrouping()
    
    def enhanced_group_methods_by_responsibility(self, class_node: ast.ClassDef, module_level_names: Set[str]):
        """Замена для метода _group_methods_by_responsibility в AutoRefactor."""
        
        # Анализируем все методы с улучшенным анализом
        enhanced_methods = []
        content = getattr(self, '_cached_content', '')
        
        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                enhanced_method = enhanced_grouping.analyze_method_enhanced(node, content)
                enhanced_methods.append(enhanced_method)
        
        # Группируем методы с улучшенной логикой
        enhanced_groups = enhanced_grouping.group_methods_enhanced(enhanced_methods)
        
        # Генерируем улучшенные ключевые слова
        enhanced_keywords = enhanced_grouping.generate_enhanced_responsibility_keywords(
            enhanced_groups, enhanced_methods
        )
        
        # Обновляем ключевые слова ответственности
        self.responsibility_keywords.update(enhanced_keywords)
        
        logger.info(f"Enhanced grouping: {len(enhanced_groups)} groups, "
                   f"{sum(len(methods) for methods in enhanced_groups.values())} methods")
        
        # Преобразуем обратно в формат AutoRefactor
        # ... (здесь нужна дополнительная логика для интеграции)
        
        return enhanced_groups
    
    return enhanced_group_methods_by_responsibility

# Экспорт для использования в других модулях
__all__ = ['EnhancedMethodGrouping', 'EnhancedMethodInfo', 'integrate_enhanced_grouping_with_auto_refactor']
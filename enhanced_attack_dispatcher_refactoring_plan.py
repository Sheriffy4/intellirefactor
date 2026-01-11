#!/usr/bin/env python3
"""
Улучшенный план рефакторинга AttackDispatcher.

Анализ показывает, что текущий рефакторинг извлек только 2 метода (2.4% уменьшение).
Это недостаточно для такого большого класса (72KB, ~48 методов).

Предлагается более агрессивный рефакторинг с выделением 7-8 компонентов.
"""

from pathlib import Path
from typing import Dict, List, Any
import ast
import re

class EnhancedAttackDispatcherAnalyzer:
    """Анализатор для создания улучшенного плана рефакторинга."""
    
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.content = file_path.read_text(encoding='utf-8')
        self.tree = ast.parse(self.content)
        self.methods = self._extract_methods()
        
    def _extract_methods(self) -> List[Dict[str, Any]]:
        """Извлекает все методы из класса AttackDispatcher."""
        methods = []
        
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and node.name == "AttackDispatcher":
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_info = {
                            'name': item.name,
                            'line_start': item.lineno,
                            'line_end': getattr(item, 'end_lineno', item.lineno + 10),
                            'is_private': item.name.startswith('_'),
                            'is_dunder': item.name.startswith('__') and item.name.endswith('__'),
                            'docstring': ast.get_docstring(item),
                            'decorators': [ast.unparse(d) for d in item.decorator_list],
                        }
                        
                        # Анализируем содержимое метода
                        lines = self.content.splitlines()
                        method_lines = lines[item.lineno-1:getattr(item, 'end_lineno', item.lineno+10)]
                        method_content = '\n'.join(method_lines)
                        
                        method_info.update({
                            'size_lines': len(method_lines),
                            'content': method_content,
                            'calls_other_methods': self._find_method_calls(method_content),
                            'complexity_score': self._calculate_complexity(method_content),
                        })
                        
                        methods.append(method_info)
                        
        return methods
    
    def _find_method_calls(self, content: str) -> List[str]:
        """Находит вызовы других методов."""
        calls = []
        # Ищем self.method_name()
        pattern = r'self\.([a-zA-Z_][a-zA-Z0-9_]*)\s*\('
        matches = re.findall(pattern, content)
        return list(set(matches))
    
    def _calculate_complexity(self, content: str) -> int:
        """Простая оценка сложности метода."""
        complexity = 1  # базовая сложность
        
        # Увеличиваем за условные конструкции
        complexity += content.count('if ')
        complexity += content.count('elif ')
        complexity += content.count('for ')
        complexity += content.count('while ')
        complexity += content.count('try:')
        complexity += content.count('except')
        complexity += content.count('with ')
        
        return complexity
    
    def analyze_responsibilities(self) -> Dict[str, List[str]]:
        """Анализирует ответственности и группирует методы."""
        
        # Более детальная группировка на основе анализа кода
        groups = {
            # Основная диспетчеризация
            'dispatch_core': [],
            
            # Разрешение стратегий
            'strategy_resolution': [],
            
            # Парсинг и валидация параметров  
            'parameter_processing': [],
            
            # Работа с TLS и SNI
            'tls_sni_processing': [],
            
            # Работа с рецептами атак
            'recipe_management': [],
            
            # Комбинированные атаки
            'combo_attacks': [],
            
            # Логирование и мониторинг
            'logging_monitoring': [],
            
            # Утилиты и вспомогательные функции
            'utilities': [],
        }
        
        for method in self.methods:
            name = method['name']
            content = method['content'].lower()
            docstring = (method['docstring'] or '').lower()
            
            # Анализируем по ключевым словам и паттернам
            if any(word in name.lower() for word in ['dispatch', 'internal']) and not name.startswith('_log'):
                groups['dispatch_core'].append(name)
                
            elif any(word in name.lower() for word in ['strategy', 'resolve', 'parse_standard']):
                groups['strategy_resolution'].append(name)
                
            elif any(word in name.lower() for word in ['param', 'normalize', 'validate', 'map_recipe']):
                groups['parameter_processing'].append(name)
                
            elif any(word in name.lower() for word in ['sni', 'tls', 'cipher', 'hostname', 'extension']):
                groups['tls_sni_processing'].append(name)
                
            elif any(word in name.lower() for word in ['recipe', 'resolve_recipe']):
                groups['recipe_management'].append(name)
                
            elif any(word in name.lower() for word in ['combo', 'combination', 'integrated']):
                groups['combo_attacks'].append(name)
                
            elif any(word in name.lower() for word in ['log', '_log', 'correlation']):
                groups['logging_monitoring'].append(name)
                
            else:
                groups['utilities'].append(name)
        
        # Удаляем пустые группы
        return {k: v for k, v in groups.items() if v}
    
    def generate_refactoring_plan(self) -> Dict[str, Any]:
        """Генерирует детальный план рефакторинга."""
        
        responsibilities = self.analyze_responsibilities()
        
        plan = {
            'original_file_size': len(self.content),
            'total_methods': len(self.methods),
            'proposed_components': {},
            'facade_methods': [],
            'estimated_size_reduction': 0,
        }
        
        # Анализируем каждую группу
        for group_name, method_names in responsibilities.items():
            if len(method_names) >= 2:  # Только группы с 2+ методами
                methods_info = [m for m in self.methods if m['name'] in method_names]
                
                total_lines = sum(m['size_lines'] for m in methods_info)
                avg_complexity = sum(m['complexity_score'] for m in methods_info) / len(methods_info)
                
                component_info = {
                    'methods': method_names,
                    'method_count': len(method_names),
                    'total_lines': total_lines,
                    'avg_complexity': avg_complexity,
                    'description': self._get_component_description(group_name),
                    'interface_methods': [m for m in method_names if not m.startswith('_')],
                }
                
                plan['proposed_components'][group_name] = component_info
                plan['estimated_size_reduction'] += total_lines * 0.8  # 80% методов уйдет в компоненты
            else:
                # Методы, которые останутся в фасаде
                plan['facade_methods'].extend(method_names)
        
        # Добавляем основные методы в фасад
        core_facade_methods = ['__init__', 'dispatch_attack', 'get_attack_info', 'list_available_attacks']
        plan['facade_methods'].extend(core_facade_methods)
        plan['facade_methods'] = list(set(plan['facade_methods']))
        
        return plan
    
    def _get_component_description(self, group_name: str) -> str:
        """Возвращает описание компонента."""
        descriptions = {
            'dispatch_core': 'Основная логика диспетчеризации атак',
            'strategy_resolution': 'Разрешение и парсинг стратегий zapret-style',
            'parameter_processing': 'Обработка, нормализация и валидация параметров',
            'tls_sni_processing': 'Парсинг TLS ClientHello и извлечение SNI',
            'recipe_management': 'Управление рецептами атак и их компонентами',
            'combo_attacks': 'Обработка комбинированных и интегрированных атак',
            'logging_monitoring': 'Логирование операций и мониторинг выполнения',
            'utilities': 'Вспомогательные функции и утилиты',
        }
        return descriptions.get(group_name, f'Компонент {group_name}')
    
    def print_analysis_report(self):
        """Выводит детальный отчет анализа."""
        plan = self.generate_refactoring_plan()
        
        print("🔍 АНАЛИЗ ATTACK_DISPATCHER ДЛЯ УЛУЧШЕННОГО РЕФАКТОРИНГА")
        print("=" * 70)
        
        print(f"📊 Общая статистика:")
        print(f"  Размер файла: {plan['original_file_size']:,} байт")
        print(f"  Всего методов: {plan['total_methods']}")
        print(f"  Предлагаемых компонентов: {len(plan['proposed_components'])}")
        print(f"  Методов останется в фасаде: {len(plan['facade_methods'])}")
        print(f"  Ожидаемое уменьшение размера: ~{plan['estimated_size_reduction']:.0f} строк")
        
        print(f"\n🏗️ ПРЕДЛАГАЕМЫЕ КОМПОНЕНТЫ:")
        print("-" * 50)
        
        for comp_name, info in plan['proposed_components'].items():
            print(f"\n📦 {comp_name.upper()}")
            print(f"   Описание: {info['description']}")
            print(f"   Методов: {info['method_count']} (строк: {info['total_lines']})")
            print(f"   Средняя сложность: {info['avg_complexity']:.1f}")
            print(f"   Публичные методы: {len(info['interface_methods'])}")
            
            print(f"   Методы:")
            for method in info['methods']:
                method_info = next(m for m in self.methods if m['name'] == method)
                visibility = "🔒" if method.startswith('_') else "🔓"
                print(f"     {visibility} {method} ({method_info['size_lines']} строк, сложность: {method_info['complexity_score']})")
        
        print(f"\n🏛️ МЕТОДЫ ФАСАДА:")
        print("-" * 30)
        facade_methods = [m for m in self.methods if m['name'] in plan['facade_methods']]
        for method in facade_methods:
            visibility = "🔒" if method['name'].startswith('_') else "🔓"
            print(f"  {visibility} {method['name']} ({method['size_lines']} строк)")
        
        print(f"\n📈 СРАВНЕНИЕ С ТЕКУЩИМ РЕФАКТОРИНГОМ:")
        print("-" * 40)
        print(f"  Текущий: 2 компонента, 2 метода извлечено (2.4% уменьшение)")
        print(f"  Предлагаемый: {len(plan['proposed_components'])} компонентов, ~{sum(info['method_count'] for info in plan['proposed_components'].values())} методов")
        print(f"  Ожидаемое улучшение: ~{(plan['estimated_size_reduction'] / plan['original_file_size'] * 100):.1f}% уменьшение размера")
        
        return plan

def main():
    """Запускает анализ AttackDispatcher."""
    
    file_path = Path('core/bypass/engine/attack_dispatcher.py')
    
    if not file_path.exists():
        print(f"❌ Файл не найден: {file_path}")
        return
    
    analyzer = EnhancedAttackDispatcherAnalyzer(file_path)
    plan = analyzer.print_analysis_report()
    
    print(f"\n💡 РЕКОМЕНДАЦИИ:")
    print("-" * 20)
    print("1. Создать 7-8 специализированных компонентов вместо 2")
    print("2. Извлечь ~35-40 методов вместо 2")
    print("3. Уменьшить размер основного файла на 40-50%")
    print("4. Использовать семантическую группировку методов")
    print("5. Создать четкие интерфейсы для каждого компонента")
    
    return plan

if __name__ == "__main__":
    main()
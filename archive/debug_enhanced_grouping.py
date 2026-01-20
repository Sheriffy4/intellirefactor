#!/usr/bin/env python3
"""
Отладка улучшенной группировки методов.
"""

from pathlib import Path
from intellirefactor.refactoring.auto_refactor import AutoRefactor

def debug_enhanced_grouping():
    """Отлаживает улучшенную группировку."""
    
    print("🔍 ОТЛАДКА УЛУЧШЕННОЙ ГРУППИРОВКИ")
    print("=" * 50)
    
    config = {
        'god_class_threshold': 3,
        'min_methods_for_extraction': 2,
        'disable_contextual_analysis': True,
        'skip_methods_with_dangerous_patterns': False,
        'skip_methods_with_bare_self_usage': False,
        'skip_methods_with_module_level_deps': False,
        'extract_private_methods': True,
    }
    
    refactor = AutoRefactor(config)
    filepath = Path('core/bypass/engine/attack_dispatcher.py')
    
    print(f"📁 Анализируем: {filepath}")
    
    # Анализируем с отладкой
    plan = refactor.analyze_god_object(filepath)
    
    print("\n📊 РЕЗУЛЬТАТЫ АНАЛИЗА:")
    print(f"🎯 Целевой класс: '{plan.target_class_name}'")
    print(f"🔧 Компонентов: {len(plan.extracted_components)}")
    print(f"⚡ Трансформаций: {len(plan.transformations)}")
    
    # Проверяем внутренние данные
    if hasattr(plan, '_method_groups'):
        print("\n🔍 ВНУТРЕННИЕ ГРУППЫ МЕТОДОВ:")
        print(f"Всего групп: {len(plan._method_groups)}")
        
        for group_name, methods in plan._method_groups.items():
            print(f"\n📦 Группа '{group_name}':")
            print(f"  Всего методов: {len(methods)}")
            
            # Анализируем методы
            public_methods = [m for m in methods if not m.name.startswith('_')]
            private_methods = [m for m in methods if m.name.startswith('_') and not m.name.startswith('__')]
            extractable_public = [m for m in public_methods if len(m.dangerous_reasons) == 0]
            
            print(f"  Публичных: {len(public_methods)}")
            print(f"  Приватных: {len(private_methods)}")
            print(f"  Извлекаемых публичных: {len(extractable_public)}")
            
            if public_methods:
                print(f"  Публичные методы: {[m.name for m in public_methods[:5]]}")
            
            if extractable_public:
                print(f"  Извлекаемые: {[m.name for m in extractable_public[:3]]}")
            elif public_methods:
                print("  Проблемы с публичными методами:")
                for method in public_methods[:3]:
                    if method.dangerous_reasons:
                        print(f"    {method.name}: {list(method.dangerous_reasons)}")
                    else:
                        print(f"    {method.name}: нет опасных причин, но не извлекается")
    
    else:
        print("❌ Нет данных о группах методов")
    
    # Проверяем, почему не создаются компоненты
    print("\n🤔 АНАЛИЗ ПРОБЛЕМЫ:")
    
    if not plan.target_class_name:
        print("❌ Не найден целевой класс")
    elif not hasattr(plan, '_method_groups') or not plan._method_groups:
        print("❌ Не созданы группы методов")
    else:
        print("✅ Группы методов созданы")
        
        # Проверяем каждую группу на соответствие критериям
        for group_name, methods in plan._method_groups.items():
            extractable_public = [m for m in methods if not m.name.startswith('_') and len(m.dangerous_reasons) == 0]
            min_required = refactor.min_methods_for_extraction
            
            print(f"  Группа {group_name}: {len(extractable_public)} извлекаемых (нужно >= {min_required})")
            
            if len(extractable_public) >= min_required:
                print("    ✅ Группа соответствует критериям")
            else:
                print("    ❌ Группа НЕ соответствует критериям")

if __name__ == "__main__":
    debug_enhanced_grouping()
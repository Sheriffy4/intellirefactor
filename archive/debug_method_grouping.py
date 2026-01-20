#!/usr/bin/env python3
"""
Детальная отладка группировки методов в AttackDispatcher
"""

import sys
import ast
from pathlib import Path
from typing import Dict, List, Set

# Add intellirefactor to path
sys.path.insert(0, str(Path(__file__).parent / 'intellirefactor'))

from intellirefactor.refactoring.auto_refactor import AutoRefactor, analyze_method

def debug_method_grouping():
    """Детальная отладка группировки методов."""
    
    file_path = Path("core/bypass/engine/attack_dispatcher.py")
    
    print(f"🔍 Анализируем группировку методов в: {file_path}")
    
    try:
        # Читаем файл
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Парсим AST
        tree = ast.parse(content)
        
        # Находим AttackDispatcher класс
        attack_dispatcher_class = None
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == "AttackDispatcher":
                attack_dispatcher_class = node
                break
        
        if not attack_dispatcher_class:
            print("❌ AttackDispatcher класс не найден!")
            return
        
        print(f"✅ Найден класс AttackDispatcher с {len([n for n in attack_dispatcher_class.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))])} методами")
        
        # Создаем AutoRefactor
        refactor = AutoRefactor()
        
        print("📋 Ключевые слова ответственности:")
        for group, keywords in refactor.responsibility_keywords.items():
            print(f"   {group}: {keywords}")
        
        # Анализируем методы
        module_level_names: Set[str] = set()
        
        # Собираем имена на уровне модуля
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                module_level_names.add(node.name)
            elif isinstance(node, ast.ClassDef):
                module_level_names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        module_level_names.add(target.id)
        
        print("\n📝 Анализируем методы:")
        
        public_methods = []
        for node in attack_dispatcher_class.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith("__"):  # Исключаем dunder методы
                    info = analyze_method(
                        node,
                        module_level_names=module_level_names,
                        allow_bare_self=True,
                        allow_dangerous=True,
                        allow_module_level_deps=True,
                        decorated_extract_allowed=True,
                    )
                    
                    if not node.name.startswith("_"):  # Только публичные методы
                        public_methods.append((node.name, info))
                        print(f"   📌 {node.name} (публичный)")
                    else:
                        print(f"   🔒 {node.name} (приватный)")
        
        print(f"\n🎯 Группировка {len(public_methods)} публичных методов:")
        
        # Группируем методы по ключевым словам
        groups: Dict[str, List[str]] = {k: [] for k in refactor.responsibility_keywords}
        other_methods = []
        
        for method_name, method_info in public_methods:
            name_lower = method_name.lower()
            scores: Dict[str, int] = {}
            
            print(f"\n   🔍 Анализируем метод '{method_name}':")
            
            for group, words in refactor.responsibility_keywords.items():
                score = sum(1 for w in words if w in name_lower)
                if score:
                    scores[group] = score
                    print(f"      ✅ {group}: {score} совпадений ({[w for w in words if w in name_lower]})")
                else:
                    print(f"      ❌ {group}: 0 совпадений")
            
            if scores:
                best = max(scores, key=lambda k: scores[k])
                groups[best].append(method_name)
                print(f"      🎯 Назначен в группу: {best} (счет: {scores[best]})")
            else:
                other_methods.append(method_name)
                print("      ❓ Не назначен ни в одну группу")
        
        print("\n📊 Результаты группировки:")
        for group, methods in groups.items():
            if methods:
                print(f"   {group}: {len(methods)} методов - {methods}")
        
        if other_methods:
            print(f"   other: {len(other_methods)} методов - {other_methods}")
        
        # Проверяем минимальное количество методов для извлечения
        print(f"\n⚖️  Минимальное количество методов для извлечения: {refactor.min_methods_for_extraction}")
        
        extractable_groups = []
        for group, methods in groups.items():
            if len(methods) >= refactor.min_methods_for_extraction:
                extractable_groups.append((group, len(methods)))
                print(f"   ✅ {group}: {len(methods)} методов (достаточно для извлечения)")
            elif methods:
                print(f"   ❌ {group}: {len(methods)} методов (недостаточно для извлечения)")
        
        print(f"\n🎯 Итого групп для извлечения: {len(extractable_groups)}")
        
        if not extractable_groups:
            print("❌ Ни одна группа не имеет достаточно методов для извлечения!")
            print("💡 Возможные решения:")
            print("   1. Уменьшить min_methods_for_extraction")
            print("   2. Добавить больше ключевых слов ответственности")
            print("   3. Использовать кластеризацию по когезии для 'other' методов")
        else:
            print("✅ Найдены группы для извлечения!")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_method_grouping()
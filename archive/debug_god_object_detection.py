#!/usr/bin/env python3
"""
Отладочный скрипт для проверки обнаружения God Object в attack_dispatcher.py
"""

import sys
import ast
from pathlib import Path

# Add intellirefactor to path
sys.path.insert(0, str(Path(__file__).parent / 'intellirefactor'))

def debug_god_object_detection():
    """Отладка обнаружения God Object."""
    
    file_path = Path("core/bypass/engine/attack_dispatcher.py")
    
    print(f"🔍 Анализируем файл: {file_path}")
    print(f"📁 Размер файла: {file_path.stat().st_size} байт")
    
    try:
        # Читаем файл
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        print(f"📄 Содержимое прочитано: {len(content)} символов")
        
        # Парсим AST
        try:
            tree = ast.parse(content)
            print("✅ AST успешно создан")
        except SyntaxError as e:
            print(f"❌ Ошибка синтаксиса при парсинге AST: {e}")
            return
        
        # Ищем классы верхнего уровня
        classes = []
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                classes.append((node.name, len(methods)))
                print(f"🏛️  Класс '{node.name}': {len(methods)} методов")
        
        if not classes:
            print("❌ Классы не найдены!")
            return
        
        # Находим самый большой класс
        largest_class = max(classes, key=lambda x: x[1])
        print(f"🎯 Самый большой класс: '{largest_class[0]}' с {largest_class[1]} методами")
        
        # Проверяем порог
        god_class_threshold = 10  # по умолчанию
        print(f"⚖️  Порог God Object: {god_class_threshold}")
        
        if largest_class[1] >= god_class_threshold:
            print(f"✅ '{largest_class[0]}' является God Object ({largest_class[1]} >= {god_class_threshold})")
        else:
            print(f"❌ '{largest_class[0]}' НЕ является God Object ({largest_class[1]} < {god_class_threshold})")
        
        # Теперь проверим с помощью AutoRefactor
        print("\n" + "="*60)
        print("🔧 Проверка с помощью AutoRefactor:")
        
        from intellirefactor.refactoring.auto_refactor import AutoRefactor
        
        refactor = AutoRefactor()
        print(f"⚙️  god_class_threshold в AutoRefactor: {refactor.god_class_threshold}")
        
        plan = refactor.analyze_god_object(file_path)
        
        print("📋 План рефакторинга:")
        print(f"   - target_class_name: {plan.target_class_name}")
        print(f"   - transformations: {len(plan.transformations)}")
        print(f"   - extracted_components: {len(plan.extracted_components)}")
        print(f"   - risk_level: {plan.risk_level}")
        
        if plan.transformations:
            print("✅ AutoRefactor обнаружил God Object!")
        else:
            print("❌ AutoRefactor НЕ обнаружил God Object!")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_god_object_detection()
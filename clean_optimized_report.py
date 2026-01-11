#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Дистиллятор отчетов - Очистка от лишнего мусора
Удаляет 86.7% шума из real_usage_patterns и других разделов
Поддерживает командную строку для указания входного и выходного файлов
"""

import json
import argparse
import sys
from pathlib import Path

def clean_optimized_report(input_path, output_path):
    """Очищает отчет от нерелевантных данных"""
    
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        print(f"❌ Ошибка: Входной файл не найден: {input_path}")
        return None
    
    print(f"📥 Загружаю отчет из: {input_path}")
    
    try:
        # Загружаем исходный файл
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ Ошибка: Некорректный JSON файл: {e}")
        return None
    except Exception as e:
        print(f"❌ Ошибка при чтении файла: {e}")
        return None
    
    original_size = len(str(data))
    print(f"📊 Исходный размер: {original_size} символов")
    
    # Счетчики для статистики
    total_removed_items = 0
    
    # 1. Очищаем real_usage_patterns (основной источник шума)
    if 'real_usage_patterns' in data:
        original_calls = data['real_usage_patterns'].get('method_calls', [])
        if original_calls:
            print(f"🗑️  Найдено вызовов методов: {len(original_calls)}")
            
            # Оставляем только вызовы, связанные с целевыми файлами
            target_files = [
                'attack_dispatcher.py',
                'core/bypass/engine/attack_dispatcher.py', 
                'attack_dispatcher_backup.py',
                'attack_dispatcher_refactored.py'
            ]
            
            cleaned_calls = []
            removed_count = 0
            
            for call in original_calls:
                file_path = call.get('file', '').lower()
                is_target = any(target_file in file_path for target_file in target_files)
                
                if is_target:
                    cleaned_calls.append(call)
                else:
                    removed_count += 1
            
            print(f"✅ Оставлено релевантных вызовов: {len(cleaned_calls)}")
            print(f"🗑️  Удалено шума: {removed_count} вызовов ({removed_count/len(original_calls)*100:.1f}%)")
            
            # Заменяем массив вызовов на очищенный
            data['real_usage_patterns']['method_calls'] = cleaned_calls
            total_removed_items += removed_count
    
    # 2. Очищаем modules (для файлов декомпозиции проекта)
    if 'modules' in data and isinstance(data['modules'], dict):
        original_modules = len(data['modules'])
        
        # Оставляем только модули, связанные с целевыми файлами
        target_modules = {}
        removed_modules = 0
        
        for module_path, module_info in data['modules'].items():
            # Проверяем, связан ли модуль с attack_dispatcher
            is_relevant = (
                'attack_dispatcher' in module_path.lower() or
                'bypass' in module_path.lower() or
                'engine' in module_path.lower() or
                any(cls.get('name', '').lower().find('attack') != -1 
                    for cls in module_info.get('classes', [])) or
                any(func.get('name', '').lower().find('attack') != -1 
                    for func in module_info.get('functions', []))
            )
            
            if is_relevant:
                target_modules[module_path] = module_info
            else:
                removed_modules += 1
        
        if removed_modules > 0:
            print(f"🗑️  Удалено нерелевантных модулей: {removed_modules} из {original_modules} ({removed_modules/original_modules*100:.1f}%)")
            data['modules'] = target_modules
            total_removed_items += removed_modules
    
    # 3. Очищаем functions (глобальные функции)
    if 'functions' in data and isinstance(data['functions'], dict):
        original_functions = len(data['functions'])
        target_functions = {}
        removed_functions = 0
        
        for func_key, func_info in data['functions'].items():
            # Оставляем функции, связанные с attack_dispatcher
            is_relevant = (
                'attack' in func_key.lower() or
                'dispatch' in func_key.lower() or
                'bypass' in func_key.lower() or
                'attack' in func_info.get('name', '').lower()
            )
            
            if is_relevant:
                target_functions[func_key] = func_info
            else:
                removed_functions += 1
        
        if removed_functions > 0:
            print(f"🗑️  Удалено нерелевантных функций: {removed_functions} из {original_functions} ({removed_functions/original_functions*100:.1f}%)")
            data['functions'] = target_functions
            total_removed_items += removed_functions
    
    # 4. Очищаем classes (глобальные классы)
    if 'classes' in data and isinstance(data['classes'], dict):
        original_classes = len(data['classes'])
        target_classes = {}
        removed_classes = 0
        
        for cls_key, cls_info in data['classes'].items():
            # Оставляем классы, связанные с attack_dispatcher
            is_relevant = (
                'attack' in cls_key.lower() or
                'dispatch' in cls_key.lower() or
                'bypass' in cls_key.lower() or
                'attack' in cls_info.get('name', '').lower()
            )
            
            if is_relevant:
                target_classes[cls_key] = cls_info
            else:
                removed_classes += 1
        
        if removed_classes > 0:
            print(f"🗑️  Удалено нерелевантных классов: {removed_classes} из {original_classes} ({removed_classes/original_classes*100:.1f}%)")
            data['classes'] = target_classes
            total_removed_items += removed_classes
    
    # 5. Очищаем API contracts от дубликатов
    if 'api_contracts' in data:
        contracts = data['api_contracts']
        if isinstance(contracts, dict):
            # Удаляем дубликаты, если есть
            unique_contracts = {}
            for key, value in contracts.items():
                if key not in unique_contracts:
                    unique_contracts[key] = value
            if len(unique_contracts) != len(contracts):
                removed_duplicates = len(contracts) - len(unique_contracts)
                print(f"🗑️  Удалено дубликатов в API contracts: {removed_duplicates}")
                data['api_contracts'] = unique_contracts
                total_removed_items += removed_duplicates
    
    # 6. Удаляем избыточные метаданные
    metadata_removed = 0
    for section_name in ['dependencies', 'dead_code', 'feature_clusters']:
        if section_name in data:
            section_data = data[section_name]
            if isinstance(section_data, dict):
                # Оставляем только релевантные зависимости/мертвый код
                if section_name == 'dependencies':
                    original_deps = len(section_data)
                    relevant_deps = {k: v for k, v in section_data.items() 
                                   if 'attack' in k.lower() or 'dispatch' in k.lower() or 'bypass' in k.lower()}
                    if len(relevant_deps) < original_deps:
                        removed = original_deps - len(relevant_deps)
                        print(f"🗑️  Удалено нерелевантных зависимостей: {removed}")
                        data[section_name] = relevant_deps
                        metadata_removed += removed
    
    # Создаем директорию для выходного файла, если она не существует
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем очищенный отчет
    print(f"📤 Сохраняю очищенный отчет в: {output_path}")
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"❌ Ошибка при сохранении файла: {e}")
        return None
    
    # Проверяем результат
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            cleaned_data = json.load(f)
        
        new_size = len(str(cleaned_data))
        
        print(f"📊 Новый размер: {new_size} символов")
        if original_size > 0:
            compression_ratio = (1 - new_size/original_size) * 100
            print(f"📉 Сжатие: {compression_ratio:.1f}%")
        
        if total_removed_items > 0:
            print(f"🎯 Всего удалено элементов: {total_removed_items}")
        
        return output_path
    except Exception as e:
        print(f"❌ Ошибка при проверке результата: {e}")
        return None

def analyze_cleaned_report(report_path):
    """Анализирует очищенный отчет"""
    print(f"\n🔍 АНАЛИЗ ОЧИЩЕННОГО ОТЧЕТА")
    print("=" * 50)
    
    try:
        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Ошибка при анализе файла: {e}")
        return
    
    print("Структура отчета:")
    for section, content in data.items():
        if isinstance(content, dict):
            size = len(str(content))
            items = len(content) if hasattr(content, '__len__') else 'N/A'
            print(f"  {section}: {items} элементов ({size} символов)")
        elif isinstance(content, list):
            size = len(str(content))
            print(f"  {section}: {len(content)} элементов ({size} символов)")
    
    # Подробный анализ ключевых разделов
    if 'real_usage_patterns' in data:
        patterns = data['real_usage_patterns']
        print(f"\n🎯 Real Usage Patterns:")
        for key, value in patterns.items():
            if isinstance(value, list):
                print(f"  {key}: {len(value)} элементов")
            else:
                print(f"  {key}: {type(value).__name__}")
    
    if 'api_contracts' in data:
        contracts = data['api_contracts']
        print(f"\n🔗 API Contracts:")
        if isinstance(contracts, dict):
            for key in contracts.keys():
                print(f"  - {key}")

def generate_default_output_path(input_path):
    """Генерирует путь для выходного файла по умолчанию"""
    input_path = Path(input_path)
    
    # Получаем имя файла без расширения
    stem = input_path.stem
    
    # Добавляем суффикс _distilled
    new_stem = f"{stem}_distilled"
    
    # Создаем новый путь в той же директории
    output_path = input_path.parent / f"{new_stem}{input_path.suffix}"
    
    return output_path

def main():
    """Главная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(
        description="Дистиллятор отчетов - очистка от лишнего мусора",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  # Базовое использование (выходной файл создается автоматически)
  python clean_optimized_report.py input_report.json
  
  # Указание выходного файла
  python clean_optimized_report.py input_report.json -o cleaned_report.json
  
  # Полные пути
  python clean_optimized_report.py C:\\path\\to\\report.json -o C:\\path\\to\\cleaned.json
  
  # Без анализа результата
  python clean_optimized_report.py input_report.json --no-analysis
        """,
    )

    parser.add_argument(
        "input_file", 
        help="Путь к входному JSON файлу для обработки"
    )
    
    parser.add_argument(
        "-o", "--output", 
        help="Путь к выходному файлу (по умолчанию: <input_file>_distilled.json в той же директории)"
    )
    
    parser.add_argument(
        "--no-analysis", 
        action="store_true",
        help="Пропустить анализ результата"
    )

    args = parser.parse_args()

    # Проверяем существование входного файла
    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"❌ Ошибка: Входной файл не найден: {input_path}")
        sys.exit(1)

    if not input_path.is_file():
        print(f"❌ Ошибка: Указанный путь не является файлом: {input_path}")
        sys.exit(1)

    # Определяем выходной файл
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = generate_default_output_path(input_path)

    print("=" * 80)
    print("ДИСТИЛЛЯТОР ОТЧЕТОВ - ОЧИСТКА ОТ МУСОРА")
    print("=" * 80)
    print(f"Входной файл: {input_path}")
    print(f"Выходной файл: {output_path}")
    print("Удаляет нерелевантные данные из отчетов анализа")
    print("=" * 80)

    # Очищаем отчет
    cleaned_path = clean_optimized_report(input_path, output_path)

    if cleaned_path is None:
        print(f"\n❌ ОШИБКА: Не удалось обработать файл")
        sys.exit(1)

    # Анализируем результат (если не отключен)
    if not args.no_analysis:
        analyze_cleaned_report(cleaned_path)

    print(f"\n✅ ДИСТИЛЛЯЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
    print("=" * 80)
    print(f"📁 Исходный файл: {input_path}")
    print(f"📁 Дистиллированный файл: {cleaned_path}")
    print(f"📊 Размер файла уменьшен, шум удален")
    
    sys.exit(0)

if __name__ == '__main__':
    # Если запущен без аргументов, показываем справку
    if len(sys.argv) == 1:
        print("=" * 80)
        print("ДИСТИЛЛЯТОР ОТЧЕТОВ")
        print("=" * 80)
        print("Для получения справки запустите:")
        print("python clean_optimized_report.py --help")
        print()
        print("Быстрый старт:")
        print("python clean_optimized_report.py your_report.json")
        sys.exit(0)
    
    main()

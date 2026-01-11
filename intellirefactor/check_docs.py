#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Проверка обновленной документации
"""

import os

def check_docs():
    print("=== ПРОВЕРКА ОБНОВЛЕННОЙ ДОКУМЕНТАЦИИ ===")
    print()
    
    # Проверяем основные файлы
    files_to_check = [
        'readme.md',
        'docs/АРХИТЕКТУРА_МОДУЛЕЙ_INTELLIREFACTOR.md',
        'docs/ПОДРОБНОЕ_ОПИСАНИЕ_STRUCTURED_ULTIMATE_ANALYZER.md'
    ]
    
    for filename in files_to_check:
        try:
            if os.path.exists(filename):
                with open(filename, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                lines = len(content.splitlines())
                has_version = '2.0.0' in content
                has_title = any(title in content for title in [
                    '# IntelliRefactor', 
                    '# Архитектура', 
                    '# Подробное описание'
                ])
                
                print(f"{filename}:")
                print(f"  - Строк: {lines}")
                print(f"  - Версия 2.0.0: {'ДА' if has_version else 'НЕТ'}")
                print(f"  - Заголовок: {'ДА' if has_title else 'НЕТ'}")
                print()
            else:
                print(f"{filename}: ФАЙЛ НЕ НАЙДЕН")
                print()
        except Exception as e:
            print(f"{filename}: ОШИБКА - {e}")
            print()

if __name__ == "__main__":
    check_docs()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для обновления документации IntelliRefactor
"""

import shutil
import os
from pathlib import Path

def update_documentation():
    """Обновляет документацию проекта"""
    
    # Текущая директория
    current_dir = Path.cwd()
    print(f"Текущая директория: {current_dir}")
    
    # Обновляем README
    readme_src = current_dir / "readme_updated.md"
    readme_dst = current_dir / "readme.md"
    
    if readme_src.exists():
        shutil.copy2(readme_src, readme_dst)
        print(f"✅ Обновлен: {readme_dst}")
    else:
        print(f"❌ Не найден: {readme_src}")
    
    # Обновляем архитектурную документацию
    docs_dir = current_dir / "docs"
    if docs_dir.exists():
        arch_src = docs_dir / "АРХИТЕКТУРА_МОДУЛЕЙ_INTELLIREFACTOR_UPDATED.md"
        arch_dst = docs_dir / "АРХИТЕКТУРА_МОДУЛЕЙ_INTELLIREFACTOR.md"
        
        if arch_src.exists():
            shutil.copy2(arch_src, arch_dst)
            print(f"✅ Обновлен: {arch_dst}")
        else:
            print(f"❌ Не найден: {arch_src}")
        
        # Обновляем описание Structured Ultimate Analyzer
        struct_src = docs_dir / "ПОДРОБНОЕ_ОПИСАНИЕ_STRUCTURED_ULTIMATE_ANALYZER_UPDATED.md"
        struct_dst = docs_dir / "ПОДРОБНОЕ_ОПИСАНИЕ_STRUCTURED_ULTIMATE_ANALYZER.md"
        
        if struct_src.exists():
            shutil.copy2(struct_src, struct_dst)
            print(f"✅ Обновлен: {struct_dst}")
        else:
            print(f"❌ Не найден: {struct_src}")
    else:
        print(f"❌ Директория docs не найдена: {docs_dir}")

if __name__ == "__main__":
    update_documentation()

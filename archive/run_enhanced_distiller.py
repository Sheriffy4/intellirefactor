#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Запуск улучшенного дистиллятора контекста.

Этот скрипт запускает enhanced_context_distiller.py для извлечения
критически важных поведенческих контрактов из проекта.
"""

import sys
import subprocess
from pathlib import Path

def main():
    # Пути
    project_root = Path.cwd()
    target_file = "core/bypass/engine/attack_dispatcher.py"
    original_json = "expert_analysis_output/expert_analysis_detailed_20260109_132347.json"
    output_file = "expert_analysis_output/enhanced_behavioral_contracts.json"
    
    # Дополнительные файлы для анализа (как просил эксперт)
    additional_sources = [
        "core/bypass/engine/parameter_normalizer.py",
        "core/attacks/attack_registry.py", 
        "core/attacks/metadata.py",
        "core/bypass/engine/unified_attack_dispatcher.py",
        "core/bypass/engine/combo_builder.py"
    ]
    
    # Проверяем существование файлов
    if not (project_root / original_json).exists():
        print(f"❌ Original JSON not found: {original_json}")
        print("Please run expert analysis first to generate the detailed JSON.")
        return 1
    
    if not (project_root / target_file).exists():
        print(f"❌ Target file not found: {target_file}")
        return 1
    
    # Фильтруем дополнительные источники (только существующие)
    existing_sources = []
    for source in additional_sources:
        if (project_root / source).exists():
            existing_sources.append(source)
            print(f"✅ Found additional source: {source}")
        else:
            print(f"⚠️  Additional source not found (skipping): {source}")
    
    # Команда для запуска
    cmd = [
        sys.executable,
        "enhanced_context_distiller.py",
        "--project-root", str(project_root),
        "--target-file", target_file,
        "--original-json", original_json,
        "--output", output_file
    ]
    
    if existing_sources:
        cmd.extend(["--additional-sources"] + existing_sources)
    
    print("\n🚀 Running enhanced context distiller...")
    print(f"📁 Project root: {project_root}")
    print(f"🎯 Target file: {target_file}")
    print(f"📊 Original JSON: {original_json}")
    print(f"💾 Output: {output_file}")
    print(f"📚 Additional sources: {len(existing_sources)}")
    
    try:
        # Запускаем дистиллятор
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print("\n" + result.stdout)
        
        if result.stderr:
            print("Warnings/Errors:")
            print(result.stderr)
        
        # Проверяем результат
        if (project_root / output_file).exists():
            print("\n🎉 Enhanced distillation completed successfully!")
            print(f"📄 Report saved to: {output_file}")
            
            # Показываем размер файла
            size_kb = (project_root / output_file).stat().st_size / 1024
            print(f"📏 Report size: {size_kb:.1f} KB")
            
            print("\n📋 Next steps:")
            print(f"1. Review the enhanced behavioral contracts in {output_file}")
            print("2. Use the discovered call-sites for characterization tests")
            print("3. Plan refactoring based on real usage patterns")
            print("4. Create fixtures from discovered usage examples")
            
        else:
            print(f"❌ Output file not created: {output_file}")
            return 1
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Enhanced distiller failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())
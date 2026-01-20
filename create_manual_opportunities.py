#!/usr/bin/env python3
"""
Создание ручных возможностей рефакторинга для attack_dispatcher.py
"""

import json
from pathlib import Path

def create_attack_dispatcher_opportunities():
    """Создает возможности рефакторинга для attack_dispatcher.py на основе анализа"""
    
    opportunities = [
        {
            "id": "extract_validation_methods_attack_dispatcher_20260108",
            "type": "extract_class",
            "priority": 9,
            "description": "Extract validation methods from AttackDispatcher into ValidationHelper",
            "target_files": ["core\\bypass\\engine\\attack_dispatcher.py"],
            "estimated_impact": {
                "complexity_reduction": 0.3,
                "maintainability_improvement": 0.6,
                "automation_potential": 0.7
            },
            "prerequisites": [
                "Identify validation methods",
                "Create ValidationHelper class",
                "Update imports and calls"
            ],
            "automation_confidence": 0.7,
            "risk_level": "medium",
            "refactoring_strategy": "extract_methods",
            "target_methods": [
                "_validate_critical_attacks",
                "_normalize_parameters", 
                "validate_attack_parameters",
                "_normalize_attack_type"
            ]
        },
        {
            "id": "extract_logging_methods_attack_dispatcher_20260108",
            "type": "extract_class", 
            "priority": 8,
            "description": "Extract logging methods from AttackDispatcher into LoggingHelper",
            "target_files": ["core\\bypass\\engine\\attack_dispatcher.py"],
            "estimated_impact": {
                "complexity_reduction": 0.2,
                "maintainability_improvement": 0.5,
                "automation_potential": 0.8
            },
            "prerequisites": [
                "Identify logging methods",
                "Create LoggingHelper class", 
                "Update method calls"
            ],
            "automation_confidence": 0.8,
            "risk_level": "low",
            "refactoring_strategy": "extract_methods",
            "target_methods": [
                "_log_dispatch_start",
                "_log_dispatch_success", 
                "_log_dispatch_error",
                "_log_segment_details",
                "_log_operations_for_validation"
            ]
        },
        {
            "id": "split_large_methods_attack_dispatcher_20260108",
            "type": "method_decomposition",
            "priority": 7,
            "description": "Split large methods in AttackDispatcher into smaller functions",
            "target_files": ["core\\bypass\\engine\\attack_dispatcher.py"],
            "estimated_impact": {
                "complexity_reduction": 0.4,
                "maintainability_improvement": 0.6,
                "automation_potential": 0.6
            },
            "prerequisites": [
                "Identify logical blocks in large methods",
                "Extract helper methods",
                "Preserve method signatures"
            ],
            "automation_confidence": 0.6,
            "risk_level": "medium",
            "refactoring_strategy": "extract_method",
            "target_methods": [
                "_dispatch_single_attack",  # 106 строк
                "_dispatch_strategy",       # 66 строк  
                "dispatch_attack"           # 53 строки
            ]
        }
    ]
    
    # Сохраняем возможности
    output_file = Path("attack_dispatcher_opportunities.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(opportunities, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Создано {len(opportunities)} возможностей рефакторинга в {output_file}")
    
    for i, opp in enumerate(opportunities, 1):
        print(f"  {i}. {opp['type']}: {opp['description']}")
    
    return output_file

def test_apply_opportunities(opportunities_file):
    """Тестирует применение созданных возможностей"""
    import subprocess
    import shutil
    
    print(f"\n=== Тест применения возможностей из {opportunities_file} ===")
    
    # Создаем копию файла для тестирования
    original_file = Path("core/bypass/engine/attack_dispatcher.py")
    test_file = Path("test_attack_dispatcher_manual.py")
    
    try:
        shutil.copy2(original_file, test_file)
        print(f"Создана тестовая копия: {test_file}")
        
        # Пытаемся применить возможности
        cmd = [
            "python", "-m", "intellirefactor", 
            "apply", 
            str(opportunities_file),
            "--target", str(test_file)
        ]
        
        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            timeout=120,
            cwd="."
        )
        
        print(f"Команда: {' '.join(cmd)}")
        print(f"Код возврата: {result.returncode}")
        
        if result.stdout.strip():
            print("Вывод:")
            print(result.stdout[:500] + "..." if len(result.stdout) > 500 else result.stdout)
        
        if result.stderr.strip():
            print("Ошибки:")
            print(result.stderr[:300] + "..." if len(result.stderr) > 300 else result.stderr)
        
        # Проверяем изменения
        if test_file.exists():
            original_size = original_file.stat().st_size
            test_size = test_file.stat().st_size
            print(f"Размер оригинала: {original_size} байт")
            print(f"Размер после рефакторинга: {test_size} байт")
            
            if abs(original_size - test_size) > 50:
                print("✅ Файл был изменен рефакторингом")
                return True
            else:
                print("⚠️ Файл не изменился или изменения минимальны")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Команда превысила таймаут")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False
    finally:
        if test_file.exists():
            test_file.unlink()
            print(f"Удален тестовый файл: {test_file}")

def main():
    """Основная функция"""
    print("Создание и тестирование ручных возможностей рефакторинга\n")
    
    opportunities_file = create_attack_dispatcher_opportunities()
    success = test_apply_opportunities(opportunities_file)
    
    print("\n=== Результат ===")
    if success:
        print("🎉 Ручные возможности рефакторинга работают!")
    else:
        print("💥 Ручные возможности не сработали")
        print("Возможно, IntelliRefactor не поддерживает ручные возможности в таком формате")
    
    return 0 if success else 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
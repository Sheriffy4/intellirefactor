#!/usr/bin/env python3
"""
Диагностический скрипт для отладки проблем с автоматизированным анализатором

Проверяет все компоненты системы и выявляет проблемы
"""

import os
import sys
import subprocess
from pathlib import Path
import json


def check_python():
    """Проверяет Python"""
    print("🐍 Проверка Python:")
    print(f"   Версия: {sys.version}")
    print(f"   Исполняемый файл: {sys.executable}")
    print(f"   Рабочая директория: {os.getcwd()}")
    return True


def check_files():
    """Проверяет наличие файлов"""
    print("\n📁 Проверка файлов:")

    required_files = [
        "automated_intellirefactor_analyzer.py",
        "intellirefactor",
        "intellirefactor/__init__.py",
        "intellirefactor/cli.py",
        "intellirefactor/api.py",
    ]

    all_good = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"   ✅ {file_path}")
        else:
            print(f"   ❌ {file_path} - НЕ НАЙДЕН")
            all_good = False

    return all_good


def test_intellirefactor_import():
    """Тестирует импорт intellirefactor"""
    print("\n📦 Тестирование импорта intellirefactor:")

    try:
        # Добавляем текущую директорию в путь
        current_dir = Path(__file__).parent
        if str(current_dir) not in sys.path:
            sys.path.insert(0, str(current_dir))

        import intellirefactor

        print("   ✅ intellirefactor импортирован успешно")
        print(f"   📍 Путь: {intellirefactor.__file__}")
        return True
    except ImportError as e:
        print(f"   ❌ Ошибка импорта intellirefactor: {e}")
        return False
    except Exception as e:
        print(f"   ❌ Неожиданная ошибка: {e}")
        return False


def test_intellirefactor_cli():
    """Тестирует CLI intellirefactor"""
    print("\n🖥️ Тестирование CLI intellirefactor:")

    try:
        # Тестируем help команду
        result = subprocess.run(
            [sys.executable, "-m", "intellirefactor", "--help"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            print("   ✅ CLI работает корректно")
            print("   📄 Вывод (первые 3 строки):")
            lines = result.stdout.split("\n")[:3]
            for line in lines:
                if line.strip():
                    print(f"      {line}")
            return True
        else:
            print(f"   ❌ CLI вернул код ошибки: {result.returncode}")
            print(f"   📄 Ошибка: {result.stderr}")
            return False

    except subprocess.TimeoutExpired:
        print("   ❌ CLI не отвечает (таймаут)")
        return False
    except Exception as e:
        print(f"   ❌ Ошибка запуска CLI: {e}")
        return False


def test_simple_analysis():
    """Тестирует простой анализ"""
    print("\n🧪 Тестирование простого анализа:")

    try:
        # Создаем простой тестовый файл
        test_file = Path("debug_test.py")
        test_content = '''def hello():
    """Простая функция"""
    return "Hello, World!"

def unused():
    """Неиспользуемая функция"""
    return "Never called"

if __name__ == "__main__":
    print(hello())
'''

        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)

        print(f"   📝 Создан тестовый файл: {test_file}")

        # Запускаем простой анализ
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "intellirefactor",
                "analyze",
                str(test_file),
                "--format",
                "json",
            ],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=60,
        )

        # Удаляем тестовый файл
        test_file.unlink()

        if result.returncode == 0:
            print("   ✅ Простой анализ выполнен успешно")

            # Пробуем парсить JSON
            try:
                if result.stdout.strip():
                    data = json.loads(result.stdout)
                    print("   📊 JSON результат получен и валиден")
                else:
                    print("   ⚠️ Пустой результат анализа")
            except json.JSONDecodeError:
                print("   ⚠️ Результат не является валидным JSON")

            return True
        else:
            print(f"   ❌ Анализ завершился с ошибкой: {result.returncode}")
            print(f"   📄 Stderr: {result.stderr}")
            print(f"   📄 Stdout: {result.stdout}")
            return False

    except Exception as e:
        print(f"   ❌ Ошибка тестирования: {e}")
        # Удаляем тестовый файл если он остался
        if test_file.exists():
            test_file.unlink()
        return False


def test_automated_analyzer():
    """Тестирует автоматизированный анализатор"""
    print("\n🤖 Тестирование автоматизированного анализатора:")

    try:
        # Создаем тестовый файл
        test_file = Path("analyzer_test.py")
        test_content = '''def test_function():
    """Тестовая функция"""
    x = 1 + 1
    return x
'''

        with open(test_file, "w", encoding="utf-8") as f:
            f.write(test_content)

        # Создаем выходную директорию
        output_dir = Path("debug_output")
        output_dir.mkdir(exist_ok=True)

        print(f"   📝 Создан тестовый файл: {test_file}")
        print(f"   📁 Выходная директория: {output_dir}")

        # Запускаем автоматизированный анализатор
        result = subprocess.run(
            [
                sys.executable,
                "automated_intellirefactor_analyzer.py",
                str(test_file),
                str(output_dir),
            ],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Очистка
        test_file.unlink()
        if output_dir.exists():
            import shutil

            shutil.rmtree(output_dir)

        if result.returncode == 0:
            print("   ✅ Автоматизированный анализатор работает")
            return True
        else:
            print(f"   ❌ Анализатор завершился с ошибкой: {result.returncode}")
            print(f"   📄 Stderr: {result.stderr}")
            if result.stdout:
                print(f"   📄 Stdout (первые 500 символов): {result.stdout[:500]}")
            return False

    except Exception as e:
        print(f"   ❌ Ошибка тестирования анализатора: {e}")
        return False


def main():
    """Главная функция диагностики"""
    print("🔍 ДИАГНОСТИКА АВТОМАТИЗИРОВАННОГО АНАЛИЗАТОРА INTELLIREFACTOR")
    print("=" * 70)

    tests = [
        ("Python", check_python),
        ("Файлы", check_files),
        ("Импорт IntelliRefactor", test_intellirefactor_import),
        ("CLI IntelliRefactor", test_intellirefactor_cli),
        ("Простой анализ", test_simple_analysis),
        ("Автоматизированный анализатор", test_automated_analyzer),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"   💥 Критическая ошибка в тесте {test_name}: {e}")
            results.append((test_name, False))

    # Итоги
    print("\n" + "=" * 70)
    print("📋 ИТОГИ ДИАГНОСТИКИ")
    print("=" * 70)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✅ ПРОШЕЛ" if result else "❌ ПРОВАЛЕН"
        print(f"{test_name:30} {status}")

    print(f"\nОбщий результат: {passed}/{total} тестов прошли успешно")

    if passed == total:
        print("\n🎉 Все тесты прошли! Система готова к работе.")
        print("\n💡 Рекомендации:")
        print("   - Попробуйте запустить: python quick_analyze.py")
        print("   - Или используйте: automated_analyzer.bat")
    else:
        print("\n⚠️ Обнаружены проблемы. Рекомендации по устранению:")

        for test_name, result in results:
            if not result:
                if "Файлы" in test_name:
                    print("   - Убедитесь, что все файлы IntelliRefactor на месте")
                    print("   - Проверьте, что вы запускаете скрипт из правильной директории")
                elif "Импорт" in test_name:
                    print("   - Проверьте структуру директории intellirefactor")
                    print("   - Убедитесь, что __init__.py файлы существуют")
                elif "CLI" in test_name:
                    print("   - Возможно, IntelliRefactor установлен неправильно")
                    print("   - Проверьте зависимости Python")
                elif "анализ" in test_name:
                    print("   - Проверьте конфигурацию IntelliRefactor")
                    print("   - Убедитесь, что нет конфликтов зависимостей")

    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        input("\nНажмите Enter для выхода...")
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n⏹️ Диагностика прервана")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Критическая ошибка диагностики: {e}")
        sys.exit(1)

#!/usr/bin/env python3
"""
Скрипт установки и настройки автоматизированной системы анализа IntelliRefactor

Проверяет зависимости, настраивает окружение и создает ярлыки для удобного использования
"""

import sys
import subprocess
import shutil
from pathlib import Path
import json


class AutomatedAnalyzerSetup:
    """Установщик автоматизированной системы анализа"""

    def __init__(self):
        self.script_dir = Path(__file__).resolve().parent
        self.errors = []
        self.warnings = []

    def check_python_version(self):
        """Проверяет версию Python"""
        print("🐍 Проверка версии Python...")

        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 7):
            self.errors.append(f"Требуется Python 3.7+, найден {version.major}.{version.minor}")
            return False

        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True

    def check_intellirefactor(self):
        """Проверяет наличие IntelliRefactor"""
        print("🔍 Проверка IntelliRefactor...")

        intellirefactor_dir = self.script_dir / "intellirefactor"
        if not intellirefactor_dir.exists():
            self.errors.append("Директория intellirefactor не найдена")
            return False

        # Проверяем основные модули
        required_modules = [
            "intellirefactor/__init__.py",
            "intellirefactor/api.py",
            "intellirefactor/cli.py",
            "intellirefactor/config.py",
        ]

        missing_modules = []
        for module in required_modules:
            if not (self.script_dir / module).exists():
                missing_modules.append(module)

        if missing_modules:
            self.errors.append(f"Отсутствуют модули IntelliRefactor: {', '.join(missing_modules)}")
            return False

        print("✅ IntelliRefactor найден и готов к использованию")
        return True

    def check_required_files(self):
        """Проверяет наличие необходимых файлов"""
        print("📁 Проверка файлов системы...")

        required_files = [
            "automated_intellirefactor_analyzer.py",
            "automated_analyzer.bat",
            "quick_analyze.py",
            "test_analyzer.py",
            "README_AUTOMATED_ANALYZER.md",
        ]

        missing_files = []
        for file_name in required_files:
            if not (self.script_dir / file_name).exists():
                missing_files.append(file_name)

        if missing_files:
            self.errors.append(f"Отсутствуют файлы: {', '.join(missing_files)}")
            return False

        print("✅ Все необходимые файлы найдены")
        return True

    def check_dependencies(self):
        """Проверяет Python зависимости"""
        print("📦 Проверка зависимостей...")

        # Основные зависимости
        required_packages = [
            "pathlib",
            "json",
            "subprocess",
            "argparse",
            "logging",
            "datetime",
            "typing",
        ]

        # Дополнительные зависимости для GUI
        optional_packages = [("tkinter", "GUI интерфейс"), ("threading", "Многопоточность")]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            self.errors.append(f"Отсутствуют пакеты: {', '.join(missing_packages)}")
            return False

        # Проверяем дополнительные пакеты
        for package, description in optional_packages:
            try:
                __import__(package)
                print(f"✅ {package} ({description}) - доступен")
            except ImportError:
                self.warnings.append(f"{package} ({description}) - недоступен")

        print("✅ Основные зависимости установлены")
        return True

    def test_intellirefactor_cli(self):
        """Тестирует CLI IntelliRefactor"""
        print("🧪 Тестирование CLI IntelliRefactor...")

        try:
            # Пробуем запустить help команду
            result = subprocess.run(
                [sys.executable, "-m", "intellirefactor", "--help"],
                cwd=self.script_dir,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode == 0:
                print("✅ CLI IntelliRefactor работает корректно")
                return True
            else:
                self.errors.append(f"CLI IntelliRefactor не работает: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.errors.append("CLI IntelliRefactor не отвечает (таймаут)")
            return False
        except Exception as e:
            self.errors.append(f"Ошибка тестирования CLI: {e}")
            return False

    def create_shortcuts(self):
        """Создает ярлыки для удобного запуска"""
        print("🔗 Создание ярлыков...")

        try:
            # Создаем директорию для ярлыков
            shortcuts_dir = self.script_dir / "shortcuts"
            shortcuts_dir.mkdir(exist_ok=True)

            # Ярлык для GUI
            gui_shortcut = shortcuts_dir / "GUI_Analyzer.bat"
            gui_content = f"""@echo off
cd /d "{self.script_dir}"
python quick_analyze.py
pause
"""
            with open(gui_shortcut, "w", encoding="utf-8") as f:
                f.write(gui_content)

            # Ярлык для командной строки
            cli_shortcut = shortcuts_dir / "CLI_Analyzer.bat"
            cli_content = f"""@echo off
cd /d "{self.script_dir}"
automated_analyzer.bat
"""
            with open(cli_shortcut, "w", encoding="utf-8") as f:
                f.write(cli_content)

            # Ярлык для тестирования
            test_shortcut = shortcuts_dir / "Test_Analyzer.bat"
            test_content = f"""@echo off
cd /d "{self.script_dir}"
python test_analyzer.py
pause
"""
            with open(test_shortcut, "w", encoding="utf-8") as f:
                f.write(test_content)

            # Ярлык для README
            readme_shortcut = shortcuts_dir / "Open_README.bat"
            readme_content = f"""@echo off
start "" "{self.script_dir / 'README_AUTOMATED_ANALYZER.md'}"
"""
            with open(readme_shortcut, "w", encoding="utf-8") as f:
                f.write(readme_content)

            print(f"✅ Ярлыки созданы в: {shortcuts_dir}")
            return True

        except Exception as e:
            self.warnings.append(f"Не удалось создать ярлыки: {e}")
            return False

    def create_config_template(self):
        """Создает шаблон конфигурации"""
        print("⚙️ Создание шаблона конфигурации...")

        try:
            config_template = {
                "analyzer_settings": {
                    "default_output_dir": "./analysis_results",
                    "verbose_by_default": False,
                    "auto_open_results": True,
                    "max_analysis_timeout": 600,
                },
                "intellirefactor_config": {
                    "safety_level": "moderate",
                    "include_patterns": ["**/*.py"],
                    "exclude_patterns": [
                        "**/__pycache__/**",
                        "**/.*",
                        "**/test_*.py",
                        "**/*_test.py",
                        "**/tests/**",
                    ],
                },
                "output_formats": {
                    "generate_json": True,
                    "generate_markdown": True,
                    "generate_html": False,
                    "generate_visualizations": True,
                },
            }

            config_file = self.script_dir / "analyzer_config.json"
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(config_template, f, indent=2, ensure_ascii=False)

            print(f"✅ Шаблон конфигурации создан: {config_file}")
            return True

        except Exception as e:
            self.warnings.append(f"Не удалось создать конфигурацию: {e}")
            return False

    def run_quick_test(self):
        """Запускает быстрый тест системы"""
        print("🚀 Запуск быстрого теста...")

        try:
            # Создаем простой тестовый файл
            test_file = self.script_dir / "quick_test.py"
            test_content = '''def hello_world():
    """Простая тестовая функция"""
    print("Hello, World!")
    return "success"

def unused_function():
    """Неиспользуемая функция для теста"""
    return "never called"

if __name__ == "__main__":
    hello_world()
'''
            with open(test_file, "w", encoding="utf-8") as f:
                f.write(test_content)

            # Запускаем анализ
            result = subprocess.run(
                [
                    sys.executable,
                    "automated_intellirefactor_analyzer.py",
                    str(test_file),
                    "./quick_test_results",
                ],
                cwd=self.script_dir,
                capture_output=True,
                text=True,
                timeout=120,
            )

            # Удаляем тестовый файл
            test_file.unlink()

            if result.returncode == 0:
                print("✅ Быстрый тест прошел успешно")

                # Удаляем результаты теста
                test_results_dir = self.script_dir / "quick_test_results"
                if test_results_dir.exists():
                    shutil.rmtree(test_results_dir)

                return True
            else:
                self.warnings.append(f"Быстрый тест завершился с ошибками: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            self.warnings.append("Быстрый тест превысил таймаут")
            return False
        except Exception as e:
            self.warnings.append(f"Ошибка быстрого теста: {e}")
            return False

    def print_summary(self):
        """Выводит итоговую информацию"""
        print("\n" + "=" * 70)
        print("📋 ИТОГИ УСТАНОВКИ")
        print("=" * 70)

        if not self.errors:
            print("🎉 Установка завершена успешно!")
            print("\n✅ Система готова к использованию")

            print("\n🚀 СПОСОБЫ ЗАПУСКА:")
            print("1. GUI интерфейс:     python quick_analyze.py")
            print("2. Командная строка:  automated_analyzer.bat")
            print(
                "3. Прямой вызов:      python automated_intellirefactor_analyzer.py <path> <output>"
            )
            print("4. Тестирование:      python test_analyzer.py")

            shortcuts_dir = self.script_dir / "shortcuts"
            if shortcuts_dir.exists():
                print(f"\n🔗 Ярлыки созданы в: {shortcuts_dir}")
                print("   - GUI_Analyzer.bat - графический интерфейс")
                print("   - CLI_Analyzer.bat - командная строка")
                print("   - Test_Analyzer.bat - тестирование")
                print("   - Open_README.bat - документация")

            print(f"\n📚 Документация: {self.script_dir / 'README_AUTOMATED_ANALYZER.md'}")

        else:
            print("❌ Установка завершилась с ошибками:")
            for error in self.errors:
                print(f"   - {error}")

        if self.warnings:
            print("\n⚠️ Предупреждения:")
            for warning in self.warnings:
                print(f"   - {warning}")

        print("\n" + "=" * 70)

    def run_setup(self):
        """Запускает процесс установки"""
        print("🔧 УСТАНОВКА АВТОМАТИЗИРОВАННОЙ СИСТЕМЫ АНАЛИЗА INTELLIREFACTOR")
        print("=" * 70)

        # Проверки
        checks = [
            ("Версия Python", self.check_python_version),
            ("IntelliRefactor", self.check_intellirefactor),
            ("Файлы системы", self.check_required_files),
            ("Зависимости", self.check_dependencies),
            ("CLI IntelliRefactor", self.test_intellirefactor_cli),
        ]

        for check_name, check_func in checks:
            if not check_func():
                print(f"\n❌ Критическая ошибка в проверке: {check_name}")
                self.print_summary()
                return False

        # Дополнительные настройки
        print("\n🔧 Настройка системы...")
        self.create_shortcuts()
        self.create_config_template()

        # Быстрый тест
        print("\n🧪 Финальное тестирование...")
        self.run_quick_test()

        # Итоги
        self.print_summary()

        return len(self.errors) == 0


def main():
    """Главная функция"""
    try:
        setup = AutomatedAnalyzerSetup()
        success = setup.run_setup()

        if success:
            # Предлагаем запустить тест
            try:
                choice = input("\n🧪 Запустить полный тест системы? (y/N): ")
                if choice.lower() in ["y", "yes"]:
                    print("\n🚀 Запуск полного теста...")
                    subprocess.run([sys.executable, "test_analyzer.py"], cwd=Path(__file__).parent)
            except KeyboardInterrupt:
                print("\n⏹️ Тест отменен")

            return 0
        else:
            return 1

    except KeyboardInterrupt:
        print("\n⏹️ Установка прервана пользователем")
        return 130
    except Exception as e:
        print(f"\n💥 Критическая ошибка установки: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

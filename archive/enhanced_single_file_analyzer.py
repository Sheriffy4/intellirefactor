#!/usr/bin/env python3
"""
Улучшенный анализатор для отдельных файлов
Применяет максимум анализов к одному файлу для качественного рефакторинга
"""

import sys
from pathlib import Path

# Добавляем текущую директорию в путь для импорта
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from automated_intellirefactor_analyzer import AutomatedIntelliRefactorAnalyzer


class EnhancedSingleFileAnalyzer(AutomatedIntelliRefactorAnalyzer):
    """Улучшенный анализатор для максимально качественного анализа отдельных файлов"""

    def run_full_analysis(self):
        """Запуск полного анализа с максимумом возможных анализов для файла"""
        self.logger.info("[СТАРТ] Запуск улучшенного анализа отдельного файла...")
        self.logger.info(f"Цель: {self.target_path}")
        self.logger.info(f"Выходная директория: {self.output_dir}")

        # Проверяем наличие intellirefactor
        intellirefactor_path = Path(__file__).parent / "intellirefactor"
        if not intellirefactor_path.exists():
            self.logger.error("[ОШИБКА] Директория intellirefactor не найдена!")
            return False

        # Список анализов для отдельного файла (расширенный)
        analyses = [
            # Базовые анализы
            ("Базовый анализ", self.run_basic_analysis),
            ("Расширенный анализ", self.run_enhanced_analysis_for_file),
            # Анализы качества кода (применимы к файлу)
            ("Обнаружение дубликатов в файле", self.detect_file_duplicates),
            ("Обнаружение неиспользуемого кода в файле", self.detect_unused_code_in_file),
            ("Обнаружение архитектурных запахов в файле", self.detect_file_architectural_smells),
            # Анализы структуры (применимы к файлу)
            ("Анализ сложности", self.analyze_complexity),
            ("Анализ зависимостей", self.analyze_dependencies),
            ("Анализ метрик", self.analyze_metrics),
            # Генерация решений и рекомендаций
            ("Генерация решений по рефакторингу файла", self.generate_file_refactoring_decisions),
            ("Создание файла требований", self.generate_file_requirements),
            # Документация и визуализация
            ("Генерация документации", self.generate_documentation),
            ("Генерация визуализаций", self.generate_visualizations),
            ("Генерация детальных диаграмм", self.generate_detailed_visualizations),
        ]

        # Выполняем все анализы
        for analysis_name, analysis_func in analyses:
            try:
                self.logger.info(f"[ВЫПОЛНЕНИЕ] {analysis_name}")
                success = analysis_func()
                if not success:
                    self.logger.warning(
                        f"[ПРЕДУПРЕЖДЕНИЕ] {analysis_name} завершился с предупреждениями"
                    )
            except Exception as e:
                self.logger.error(f"[ОШИБКА] Ошибка в {analysis_name}: {e}")

        # Генерируем итоговый отчет
        self.generate_enhanced_summary_report()

        # Подсчитываем статистику
        total_analyses = len(self.analysis_results["completed_analyses"]) + len(
            self.analysis_results["failed_analyses"]
        )
        success_rate = (
            len(self.analysis_results["completed_analyses"]) / total_analyses * 100
            if total_analyses > 0
            else 0
        )

        self.logger.info("[ЗАВЕРШЕНИЕ] Анализ завершен!")
        self.logger.info(
            f"[СТАТИСТИКА] {len(self.analysis_results['completed_analyses'])}/{total_analyses} анализов выполнено успешно"
        )
        self.logger.info(f"[ФАЙЛЫ] Создано файлов: {len(self.analysis_results['generated_files'])}")

        return success_rate > 50  # Считаем успешным, если больше 50% анализов прошло

    def run_enhanced_analysis_for_file(self):
        """Расширенный анализ для отдельного файла"""
        self.logger.info("[РАСШИРЕННЫЙ] Запуск расширенного анализа файла...")

        command = [
            "analyze-enhanced",
            str(self.target_path),
            "--format",
            "markdown",
            "--include-metrics",
            "--include-opportunities",
            "--include-safety",
            "--single-file-mode",  # Специальный режим для файлов
        ]

        result = self._run_intellirefactor_command(
            command, f"enhanced_file_analysis_{self.timestamp}.md"
        )

        self._save_analysis_result("Расширенный анализ файла", result)
        return result["success"]

    def detect_file_duplicates(self):
        """Обнаружение дубликатов внутри файла"""
        self.logger.info("[ДУБЛИКАТЫ] Поиск дубликатов внутри файла...")

        # Блочные дубликаты внутри файла
        command = [
            "duplicates",
            "blocks",
            str(self.target_path),
            "--format",
            "json",
            "--show-code",
            "--intra-file-only",  # Только внутри файла
        ]

        result = self._run_intellirefactor_command(
            command, f"file_duplicate_blocks_{self.timestamp}.json"
        )

        self._save_analysis_result("Обнаружение дубликатов блоков в файле", result)

        # Дубликаты методов внутри файла
        command = [
            "duplicates",
            "methods",
            str(self.target_path),
            "--format",
            "json",
            "--show-signatures",
            "--intra-file-only",
        ]

        result = self._run_intellirefactor_command(
            command, f"file_duplicate_methods_{self.timestamp}.json"
        )

        self._save_analysis_result("Обнаружение дубликатов методов в файле", result)
        return result["success"]

    def detect_unused_code_in_file(self):
        """Обнаружение неиспользуемого кода внутри файла"""
        self.logger.info("[НЕИСПОЛЬЗУЕМЫЙ] Поиск неиспользуемого кода в файле...")

        command = [
            "unused",
            "detect",
            str(self.target_path),
            "--level",
            "file",  # Уровень файла
            "--format",
            "json",
            "--show-evidence",
            "--show-usage",
            "--include-private",  # Включаем приватные методы
        ]

        result = self._run_intellirefactor_command(
            command, f"file_unused_code_{self.timestamp}.json"
        )

        self._save_analysis_result("Обнаружение неиспользуемого кода в файле", result)
        return result["success"]

    def detect_file_architectural_smells(self):
        """Обнаружение архитектурных запахов в файле"""
        self.logger.info("[ЗАПАХИ] Поиск архитектурных запахов в файле...")

        command = [
            "smells",
            "detect",
            str(self.target_path),
            "--format",
            "json",
            "--show-evidence",
            "--show-recommendations",
            "--file-level-only",  # Только на уровне файла
        ]

        result = self._run_intellirefactor_command(
            command, f"file_architectural_smells_{self.timestamp}.json"
        )

        self._save_analysis_result("Обнаружение архитектурных запахов в файле", result)
        return result["success"]

    def analyze_complexity(self):
        """Анализ сложности кода в файле"""
        self.logger.info("[СЛОЖНОСТЬ] Анализ сложности кода...")

        command = [
            "metrics",
            "complexity",
            str(self.target_path),
            "--format",
            "json",
            "--include-cyclomatic",
            "--include-cognitive",
            "--include-halstead",
        ]

        result = self._run_intellirefactor_command(
            command, f"complexity_analysis_{self.timestamp}.json"
        )

        self._save_analysis_result("Анализ сложности", result)
        return result["success"]

    def analyze_dependencies(self):
        """Анализ зависимостей файла"""
        self.logger.info("[ЗАВИСИМОСТИ] Анализ зависимостей файла...")

        command = [
            "dependencies",
            "analyze",
            str(self.target_path),
            "--format",
            "json",
            "--show-imports",
            "--show-usage",
            "--show-external",
        ]

        result = self._run_intellirefactor_command(
            command, f"dependencies_analysis_{self.timestamp}.json"
        )

        self._save_analysis_result("Анализ зависимостей", result)
        return result["success"]

    def analyze_metrics(self):
        """Анализ метрик кода"""
        self.logger.info("[МЕТРИКИ] Анализ метрик кода...")

        command = ["metrics", "analyze", str(self.target_path), "--format", "json", "--include-all"]

        result = self._run_intellirefactor_command(command, f"code_metrics_{self.timestamp}.json")

        self._save_analysis_result("Анализ метрик", result)
        return result["success"]

    def generate_file_refactoring_decisions(self):
        """Генерация решений по рефакторингу для файла"""
        self.logger.info("[РЕШЕНИЯ] Генерация решений по рефакторингу файла...")

        command = [
            "decide",
            "analyze",
            str(self.target_path),
            "--format",
            "json",
            "--export-decisions",
            str(self.output_dir / f"file_refactoring_decisions_{self.timestamp}.json"),
            "--prioritize",
            "--include-impact",
        ]

        result = self._run_intellirefactor_command(
            command, f"file_decision_analysis_{self.timestamp}.json"
        )

        self._save_analysis_result("Генерация решений по рефакторингу файла", result)
        return result["success"]

    def generate_file_requirements(self):
        """Создание файла требований для отдельного файла"""
        self.logger.info("[ТРЕБОВАНИЯ] Создание файла требований...")

        command = [
            "audit",
            str(self.target_path),
            "--format",
            "json",
            "--emit-spec",
            "--spec-output",
            str(self.output_dir / f"FILE_REQUIREMENTS_{self.timestamp}.md"),
            "--emit-json",
            "--json-output",
            str(self.output_dir / f"file_audit_{self.timestamp}.json"),
            "--single-file-mode",
        ]

        result = self._run_intellirefactor_command(command, f"file_audit_{self.timestamp}.json")

        self._save_analysis_result("Создание файла требований", result)
        return result["success"]

    def generate_detailed_visualizations(self):
        """Генерация детальных визуализаций для файла"""
        self.logger.info("[ВИЗУАЛИЗАЦИЯ] Генерация детальных визуализаций...")

        # Диаграмма классов
        command = [
            "visualize",
            "class",
            str(self.target_path),
            "--format",
            "mermaid",
            "--include-methods",
            "--include-attributes",
        ]

        result = self._run_intellirefactor_command(command, f"class_diagram_{self.timestamp}.mmd")

        self._save_analysis_result("Генерация диаграммы классов", result)

        # Граф вызовов
        command = [
            "visualize",
            "calls",
            str(self.target_path),
            "--format",
            "mermaid",
            "--include-external",
        ]

        result = self._run_intellirefactor_command(command, f"call_graph_{self.timestamp}.mmd")

        self._save_analysis_result("Генерация графа вызовов", result)
        return result["success"]

    def generate_enhanced_summary_report(self):
        """Генерация улучшенного итогового отчета для файла"""
        self.logger.info("[ОТЧЕТ] Генерация улучшенного итогового отчета...")

        # Подсчитываем все созданные файлы в выходной директории
        all_files = []
        if self.output_dir.exists():
            for file_path in self.output_dir.iterdir():
                if (
                    file_path.is_file()
                    and file_path.name != f"ENHANCED_FILE_REPORT_{self.timestamp}.md"
                ):
                    all_files.append(str(file_path))

        self.analysis_results["generated_files"] = list(
            set(self.analysis_results["generated_files"] + all_files)
        )

        # Подсчитываем статистику
        total_analyses = len(self.analysis_results["completed_analyses"]) + len(
            self.analysis_results["failed_analyses"]
        )
        success_rate = (
            len(self.analysis_results["completed_analyses"]) / total_analyses * 100
            if total_analyses > 0
            else 0
        )

        # Создаем улучшенный отчет
        report_content = f"""# Отчет улучшенного анализа отдельного файла

## Общая информация
- **Анализируемый файл:** {self.target_path}
- **Выходная директория:** {self.output_dir}
- **Время анализа:** {self.timestamp}
- **Тип анализа:** Максимально качественный анализ отдельного файла

## Статистика выполнения
- **Всего анализов:** {total_analyses}
- **Успешно выполнено:** {len(self.analysis_results['completed_analyses'])}
- **Завершено с ошибками:** {len(self.analysis_results['failed_analyses'])}
- **Процент успеха:** {success_rate:.1f}%
- **Создано файлов:** {len(self.analysis_results['generated_files'])}

## Выполненные анализы
"""

        for analysis in self.analysis_results["completed_analyses"]:
            report_content += f"- ✅ {analysis}\n"

        if self.analysis_results["failed_analyses"]:
            report_content += "\n## Анализы с ошибками\n"
            for failed in self.analysis_results["failed_analyses"]:
                report_content += f"- ❌ {failed['name']}: {failed['error'][:100]}...\n"

        report_content += f"""
## Рекомендации по рефакторингу

### 1. Анализ качества кода
- Проверьте файл `file_architectural_smells_{self.timestamp}.json` на архитектурные проблемы
- Изучите `complexity_analysis_{self.timestamp}.json` для оптимизации сложности
- Просмотрите `file_duplicate_blocks_{self.timestamp}.json` для устранения дубликатов

### 2. Структурные улучшения
- Используйте `file_refactoring_decisions_{self.timestamp}.json` для приоритизации изменений
- Проанализируйте `dependencies_analysis_{self.timestamp}.json` для оптимизации зависимостей
- Изучите `file_unused_code_{self.timestamp}.json` для очистки кода

### 3. Документация и визуализация
- Используйте созданные диаграммы для понимания структуры
- Обратитесь к `FILE_REQUIREMENTS_{self.timestamp}.md` для требований к рефакторингу

## Передача разработчику

Для качественного рефакторинга передайте разработчику:

1. **Исходный файл:** {self.target_path}
2. **Файл требований:** FILE_REQUIREMENTS_{self.timestamp}.md
3. **Решения по рефакторингу:** file_refactoring_decisions_{self.timestamp}.json
4. **Анализ зависимостей:** dependencies_analysis_{self.timestamp}.json
5. **Диаграммы:** class_diagram_{self.timestamp}.mmd, call_graph_{self.timestamp}.mmd

---
*Отчет создан улучшенной системой анализа отдельных файлов*
"""

        # Сохраняем отчет
        report_path = self.output_dir / f"ENHANCED_FILE_REPORT_{self.timestamp}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        self.analysis_results["generated_files"].append(str(report_path))
        self.logger.info(f"[ОТЧЕТ] Улучшенный отчет создан: {report_path}")

        return True


def main():
    """Запуск улучшенного анализатора для отдельного файла"""
    print("=" * 80)
    print("УЛУЧШЕННЫЙ АНАЛИЗАТОР ДЛЯ ОТДЕЛЬНЫХ ФАЙЛОВ")
    print("Максимально качественный анализ для рефакторинга")
    print("=" * 80)

    # Параметры по умолчанию
    target_path = r"C:\Intel\recon\core\bypass\engine\attack_dispatcher.py"
    output_dir = r"C:\Intel\recon\enhanced_file_analysis"

    print(f"\n[ЦЕЛЬ] {target_path}")
    print(f"[РЕЗУЛЬТАТЫ] {output_dir}")

    # Создаем улучшенный анализатор
    try:
        analyzer = EnhancedSingleFileAnalyzer(target_path, output_dir, verbose=True)

        print("\n[ИНФОРМАЦИЯ] Улучшенный анализ включает:")
        print("  - Все базовые анализы")
        print("  - Поиск дубликатов внутри файла")
        print("  - Анализ неиспользуемого кода")
        print("  - Обнаружение архитектурных запахов")
        print("  - Анализ сложности и метрик")
        print("  - Анализ зависимостей")
        print("  - Генерацию решений по рефакторингу")
        print("  - Создание файла требований")
        print("  - Детальные визуализации")

        # Запускаем полный анализ
        success = analyzer.run_full_analysis()

        if success:
            print("\n" + "=" * 80)
            print("✅ УЛУЧШЕННЫЙ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
            print("=" * 80)

            print(f"\n📋 Создан файл требований: FILE_REQUIREMENTS_{analyzer.timestamp}.md")
            print(
                f"🔧 Решения по рефакторингу: file_refactoring_decisions_{analyzer.timestamp}.json"
            )
            print(f"📊 Итоговый отчет: ENHANCED_FILE_REPORT_{analyzer.timestamp}.md")

        else:
            print("\n" + "=" * 80)
            print("⚠️ АНАЛИЗ ЗАВЕРШЕН С ПРЕДУПРЕЖДЕНИЯМИ")
            print("=" * 80)

    except Exception as e:
        print(f"\n[КРИТИЧЕСКАЯ ОШИБКА] {e}")
        return False

    print(f"\n[ЗАВЕРШЕНИЕ] Проверьте результаты в: {output_dir}")
    return success


if __name__ == "__main__":
    success = main()

    # Пауза для просмотра результатов
    input("\nНажмите Enter для завершения...")

    sys.exit(0 if success else 1)

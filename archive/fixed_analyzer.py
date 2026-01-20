#!/usr/bin/env python3
"""
Исправленная версия автоматизированного анализатора IntelliRefactor
Версия: 1.1.0 - Исправлены ошибки обработки команд документации
"""

import sys
from pathlib import Path

# Добавляем текущую директорию в путь для импорта
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from automated_intellirefactor_analyzer import AutomatedIntelliRefactorAnalyzer


def main():
    """Запуск исправленного анализатора"""
    print("=" * 80)
    print("ИСПРАВЛЕННЫЙ АВТОМАТИЗИРОВАННЫЙ АНАЛИЗАТОР INTELLIREFACTOR")
    print("Версия 1.1.0 - Исправления обработки команд документации")
    print("=" * 80)

    # Параметры по умолчанию из предыдущего анализа
    target_path = r"C:\Intel\recon\core\bypass\engine\attack_dispatcher.py"
    output_dir = r"C:\Intel\recon\analysis_results"

    print(f"\n[ЦЕЛЬ] {target_path}")
    print(f"[РЕЗУЛЬТАТЫ] {output_dir}")

    # Создаем анализатор
    try:
        analyzer = AutomatedIntelliRefactorAnalyzer(target_path, output_dir, verbose=True)

        print("\n[СТАРТ] Запуск исправленного анализа...")
        print("[ИНФОРМАЦИЯ] Исправления:")
        print("  - Улучшена обработка ошибок кодировки в командах документации")
        print("  - Добавлена проверка созданных файлов по логам IntelliRefactor")
        print("  - Исправлен подсчет всех созданных файлов")
        print("  - Добавлена транслитерация имен файлов результатов")

        # Запускаем полный анализ
        success = analyzer.run_full_analysis()

        if success:
            print("\n" + "=" * 80)
            print("[УСПЕХ] АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
            print("=" * 80)

            # Выводим статистику
            stats = analyzer.analysis_results.get("statistics", {})
            print("\n[СТАТИСТИКА]")
            print(f"  Всего анализов: {stats.get('total_analyses', 0)}")
            print(f"  Успешно: {stats.get('successful_analyses', 0)}")
            print(f"  С ошибками: {stats.get('failed_analyses', 0)}")
            print(f"  Процент успеха: {stats.get('success_rate', 0):.1f}%")
            print(f"  Создано файлов: {stats.get('generated_files_count', 0)}")

            # Показываем созданные файлы
            print("\n[СОЗДАННЫЕ ФАЙЛЫ]")
            for file_path in analyzer.analysis_results.get("generated_files", []):
                rel_path = Path(file_path).relative_to(output_dir)
                print(f"  - {rel_path}")

            print(f"\n[ОТЧЕТ] Итоговый отчет: {output_dir}/SUMMARY_REPORT_{analyzer.timestamp}.md")

        else:
            print("\n" + "=" * 80)
            print("[ОШИБКА] АНАЛИЗ ЗАВЕРШЕН С ОШИБКАМИ")
            print("=" * 80)

            # Показываем ошибки
            for failed in analyzer.analysis_results.get("failed_analyses", []):
                print(f"\n[ОШИБКА] {failed['name']}")
                print(f"  Команда: {failed['command']}")
                print(f"  Ошибка: {failed['error'][:200]}...")

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

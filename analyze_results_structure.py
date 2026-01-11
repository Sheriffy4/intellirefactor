#!/usr/bin/env python3
"""
Анализатор структуры результатов максимально полного анализа

Анализирует созданные файлы и предлагает улучшения структуры.
"""

import os
import json
from pathlib import Path
from datetime import datetime


def analyze_results_directory(results_dir):
    """Анализирует директорию с результатами"""
    results_path = Path(results_dir)

    if not results_path.exists():
        print(f"❌ Директория не найдена: {results_dir}")
        return

    print("=" * 80)
    print("АНАЛИЗ СТРУКТУРЫ РЕЗУЛЬТАТОВ")
    print("=" * 80)
    print(f"Директория: {results_path}")
    print()

    # Собираем все файлы
    all_files = []
    total_size = 0

    for file_path in results_path.iterdir():
        if file_path.is_file():
            size = file_path.stat().st_size
            all_files.append(
                {
                    "name": file_path.name,
                    "path": file_path,
                    "size": size,
                    "size_mb": size / (1024 * 1024),
                    "extension": file_path.suffix.lower(),
                }
            )
            total_size += size

    # Сортируем по размеру
    all_files.sort(key=lambda x: x["size"], reverse=True)

    print(f"📊 ОБЩАЯ СТАТИСТИКА:")
    print(f"Всего файлов: {len(all_files)}")
    print(f"Общий размер: {total_size / (1024 * 1024):.1f} МБ")
    print()

    # Анализируем большие файлы
    print("🔍 БОЛЬШИЕ ФАЙЛЫ (>1 МБ):")
    large_files = [f for f in all_files if f["size_mb"] > 1]

    if large_files:
        for file_info in large_files:
            print(f"  📄 {file_info['name']}: {file_info['size_mb']:.1f} МБ")
    else:
        print("  Нет файлов больше 1 МБ")
    print()

    # Категоризируем файлы
    categories = {
        "Основные документы": [],
        "JSON анализы": [],
        "Markdown отчеты": [],
        "Логи": [],
        "Другие": [],
    }

    main_docs = ["Requirements.md", "Design.md", "Implementation.md"]

    for file_info in all_files:
        name = file_info["name"]
        ext = file_info["extension"]

        if name in main_docs:
            categories["Основные документы"].append(file_info)
        elif ext == ".json":
            categories["JSON анализы"].append(file_info)
        elif ext == ".md":
            categories["Markdown отчеты"].append(file_info)
        elif ext == ".log":
            categories["Логи"].append(file_info)
        else:
            categories["Другие"].append(file_info)

    # Выводим категории
    for category, files in categories.items():
        if files:
            print(f"📁 {category.upper()}:")
            for file_info in files:
                size_str = (
                    f"{file_info['size_mb']:.1f} МБ"
                    if file_info["size_mb"] > 0.1
                    else f"{file_info['size']} байт"
                )
                print(f"  • {file_info['name']} ({size_str})")
            print()

    # Анализируем проблемы
    print("⚠️ ВЫЯВЛЕННЫЕ ПРОБЛЕМЫ:")
    problems = []

    # Проблема 1: Очень большие JSON файлы
    huge_json = [f for f in all_files if f["extension"] == ".json" and f["size_mb"] > 50]
    if huge_json:
        problems.append(f"Очень большие JSON файлы: {', '.join([f['name'] for f in huge_json])}")

    # Проблема 2: Отсутствие главного файла
    main_files = [
        f
        for f in all_files
        if "ULTIMATE" in f["name"] or "MAIN" in f["name"] or "INDEX" in f["name"]
    ]
    if not main_files:
        problems.append("Отсутствует главный индексный файл")

    # Проблема 3: Много временных файлов
    temp_files = [
        f for f in all_files if "temp" in f["name"].lower() or "attempt" in f["name"].lower()
    ]
    if len(temp_files) > 5:
        problems.append(f"Много временных файлов: {len(temp_files)}")

    if problems:
        for i, problem in enumerate(problems, 1):
            print(f"  {i}. {problem}")
    else:
        print("  Проблем не выявлено")
    print()

    # Рекомендации
    print("💡 РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ:")
    recommendations = []

    if huge_json:
        recommendations.append("Сжать или разделить большие JSON файлы")
        recommendations.append("Создать сводки вместо полных данных")

    if not main_files:
        recommendations.append("Создать главный индексный файл со ссылками на все результаты")

    recommendations.append("Структурировать файлы по папкам (docs/, json/, reports/)")
    recommendations.append("Создать навигационную систему между файлами")

    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    print()

    return {
        "total_files": len(all_files),
        "total_size_mb": total_size / (1024 * 1024),
        "large_files": large_files,
        "categories": categories,
        "problems": problems,
        "recommendations": recommendations,
    }


def main():
    """Главная функция"""
    import sys

    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        # Ищем последнюю директорию analysis_results
        current_dir = Path.cwd()
        analysis_dirs = list(current_dir.glob("**/analysis_results"))

        if analysis_dirs:
            results_dir = str(analysis_dirs[0])
            print(f"Найдена директория: {results_dir}")
        else:
            results_dir = input("Введите путь к директории с результатами анализа: ")

    analyze_results_directory(results_dir)


if __name__ == "__main__":
    main()

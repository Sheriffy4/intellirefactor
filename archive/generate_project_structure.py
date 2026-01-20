#!/usr/bin/env python3
"""
Генератор документации структуры проекта.
Создает PROJECT_STRUCTURE.md с описанием всех директорий, entry points и конфигурационных файлов.
"""

import logging
from pathlib import Path

from core.refactoring.structure_analyzer import ProjectStructureAnalyzer, StructureDocumenter

# Настройка логирования
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def main():
    """Основная функция для генерации документации структуры проекта."""
    try:
        logger.info("Starting project structure documentation generation")

        # Инициализируем анализатор и документатор
        root_path = Path.cwd()
        analyzer = ProjectStructureAnalyzer(root_path)
        documenter = StructureDocumenter(root_path)

        # Анализируем структуру проекта
        logger.info("Analyzing project structure...")
        structure = analyzer.analyze_structure()

        # Выводим статистику
        logger.info(f"Found {len(structure.directories)} directories")
        logger.info(f"Found {len(structure.entry_points)} entry points")
        logger.info(f"Found {len(structure.config_files)} configuration files")

        # Создаем документацию
        logger.info("Creating PROJECT_STRUCTURE.md...")
        documenter.create_structure_doc(structure)

        logger.info("Project structure documentation generated successfully!")

        # Выводим краткую сводку
        print("\n=== Сводка анализа структуры проекта ===")
        print(f"Директорий проанализировано: {len(structure.directories)}")
        print(f"Entry points найдено: {len(structure.entry_points)}")
        print(f"Конфигурационных файлов: {len(structure.config_files)}")
        print("Документ создан: PROJECT_STRUCTURE.md")

        if structure.entry_points:
            print("\nОсновные entry points:")
            for ep in sorted(structure.entry_points)[:5]:  # Показываем первые 5
                print(f"  - {ep}")

        if structure.config_files:
            print("\nОсновные конфигурационные файлы:")
            for cf in sorted(structure.config_files)[:5]:  # Показываем первые 5
                print(f"  - {cf}")

    except Exception as e:
        logger.error(f"Error generating project structure documentation: {e}")
        raise


if __name__ == "__main__":
    main()

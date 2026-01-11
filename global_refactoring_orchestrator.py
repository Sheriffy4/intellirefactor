#!/usr/bin/env python3
"""
Global Refactoring Orchestrator

Главный скрипт для выполнения всех этапов глобального рефакторинга проекта.
Объединяет все этапы в один исполняемый процесс с логированием и обработкой ошибок.

Этапы:
1. Очистка мусора (файлы логов, временные файлы, дебаг скрипты)
2. Анализ структуры проекта (создание PROJECT_STRUCTURE.md)
3. Создание реестра модулей (создание MODULE_REGISTRY.md)
4. Создание LLM контекста (создание LLM_CONTEXT.md)
"""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any

# Импорты компонентов рефакторинга
try:
    from core.refactoring.file_scanner import FileScanner
    from core.refactoring.safe_remover import SafeRemover
    from core.refactoring.structure_analyzer import ProjectStructureAnalyzer, StructureDocumenter
    from core.refactoring.module_scanner import ModuleScanner
    from core.refactoring.module_registry_builder import ModuleRegistryBuilder
    from core.refactoring.module_categorizer import ModuleCategorizer
    from core.refactoring.llm_context_generator import LLMContextGenerator
except ImportError as e:
    print(f"Ошибка импорта модулей рефакторинга: {e}")
    print("Убедитесь что все модули рефакторинга созданы и доступны")
    sys.exit(1)


@dataclass
class StageResult:
    """Результат выполнения этапа рефакторинга."""

    stage_name: str
    success: bool
    duration: float
    message: str
    details: Dict[str, Any] = None
    error: Optional[Exception] = None


@dataclass
class RefactoringReport:
    """Итоговый отчет о рефакторинге."""

    start_time: datetime
    end_time: datetime
    total_duration: float
    stages: List[StageResult]
    overall_success: bool

    @property
    def successful_stages(self) -> int:
        return len([s for s in self.stages if s.success])

    @property
    def failed_stages(self) -> int:
        return len([s for s in self.stages if not s.success])


class GlobalRefactoringOrchestrator:
    """Оркестратор для выполнения всех этапов глобального рефакторинга."""

    def __init__(self, project_root: Path = None, dry_run: bool = False):
        """
        Инициализация оркестратора.

        Args:
            project_root: Корневая директория проекта
            dry_run: Режим "сухого прогона" - показать что будет сделано без выполнения
        """
        self.project_root = project_root or Path.cwd()
        self.dry_run = dry_run
        self.logger = self._setup_logging()

        # Инициализируем компоненты
        self.file_scanner = FileScanner(self.project_root)
        self.safe_remover = SafeRemover(self.project_root)
        self.structure_analyzer = ProjectStructureAnalyzer(self.project_root)
        self.structure_documenter = StructureDocumenter(self.project_root)
        self.module_scanner = ModuleScanner(self.project_root)
        self.module_registry_builder = ModuleRegistryBuilder(self.project_root)
        self.llm_context_generator = LLMContextGenerator(self.project_root)

        self.logger.info(f"Инициализирован оркестратор рефакторинга для {self.project_root}")
        if self.dry_run:
            self.logger.info("РЕЖИМ СУХОГО ПРОГОНА - изменения не будут применены")

    def _setup_logging(self) -> logging.Logger:
        """Настройка логирования."""
        logger = logging.getLogger("refactoring_orchestrator")
        logger.setLevel(logging.INFO)

        # Создаем форматтер
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

        # Консольный хендлер
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # Файловый хендлер
        log_file = self.project_root / f"refactoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def run_all_stages(self) -> RefactoringReport:
        """
        Выполняет все этапы рефакторинга.

        Returns:
            RefactoringReport с результатами всех этапов
        """
        start_time = datetime.now()
        self.logger.info("=" * 60)
        self.logger.info("НАЧАЛО ГЛОБАЛЬНОГО РЕФАКТОРИНГА")
        self.logger.info("=" * 60)

        stages = []

        # Этап 1: Очистка мусора
        stage1 = self._run_stage_1_cleanup()
        stages.append(stage1)

        # Этап 2: Анализ структуры проекта
        stage2 = self._run_stage_2_structure()
        stages.append(stage2)

        # Этап 3: Создание реестра модулей
        stage3 = self._run_stage_3_modules()
        stages.append(stage3)

        # Этап 4: Создание LLM контекста
        stage4 = self._run_stage_4_llm_context()
        stages.append(stage4)

        end_time = datetime.now()
        total_duration = (end_time - start_time).total_seconds()
        overall_success = all(stage.success for stage in stages)

        report = RefactoringReport(
            start_time=start_time,
            end_time=end_time,
            total_duration=total_duration,
            stages=stages,
            overall_success=overall_success,
        )

        self._print_final_report(report)
        return report

    def run_single_stage(self, stage_number: int) -> StageResult:
        """
        Выполняет отдельный этап рефакторинга.

        Args:
            stage_number: Номер этапа (1-4)

        Returns:
            StageResult с результатом выполнения
        """
        self.logger.info(f"Запуск этапа {stage_number}")

        if stage_number == 1:
            return self._run_stage_1_cleanup()
        elif stage_number == 2:
            return self._run_stage_2_structure()
        elif stage_number == 3:
            return self._run_stage_3_modules()
        elif stage_number == 4:
            return self._run_stage_4_llm_context()
        else:
            return StageResult(
                stage_name=f"Этап {stage_number}",
                success=False,
                duration=0.0,
                message=f"Неизвестный номер этапа: {stage_number}",
                error=ValueError(f"Этап {stage_number} не существует"),
            )

    def _run_stage_1_cleanup(self) -> StageResult:
        """Этап 1: Очистка мусора."""
        stage_name = "Этап 1: Очистка мусора"
        start_time = time.time()

        try:
            self.logger.info("🧹 " + stage_name)

            # Сканируем файлы мусора
            self.logger.info("Сканирование файлов мусора...")
            garbage_files = self.file_scanner.scan_project()

            if not garbage_files:
                return StageResult(
                    stage_name=stage_name,
                    success=True,
                    duration=time.time() - start_time,
                    message="Файлы мусора не найдены",
                    details={"garbage_files_count": 0},
                )

            self.logger.info(f"Найдено {len(garbage_files)} файлов мусора")

            # Показываем что будет сделано
            categorized = self.file_scanner.get_files_by_category(garbage_files)
            for category, files in categorized.items():
                if files:
                    self.logger.info(f"  {category.value}: {len(files)} файлов")

            if self.dry_run:
                return StageResult(
                    stage_name=stage_name,
                    success=True,
                    duration=time.time() - start_time,
                    message=f"[DRY RUN] Найдено {len(garbage_files)} файлов для удаления",
                    details={"garbage_files_count": len(garbage_files), "dry_run": True},
                )

            # Перемещаем файлы
            self.logger.info("Перемещение файлов в _to_delete/...")
            removal_report = self.safe_remover.move_files_to_delete(garbage_files)

            # Сохраняем отчет
            report_path = self.safe_remover.save_report(removal_report)
            self.logger.info(f"Отчет сохранен: {report_path}")

            return StageResult(
                stage_name=stage_name,
                success=True,
                duration=time.time() - start_time,
                message=f"Перемещено {len(removal_report.moved_files)} файлов, освобождено {self._format_size(removal_report.total_size_freed)}",
                details={
                    "moved_files": len(removal_report.moved_files),
                    "failed_moves": len(removal_report.failed_moves),
                    "size_freed": removal_report.total_size_freed,
                    "report_path": str(report_path),
                },
            )

        except Exception as e:
            self.logger.error(f"Ошибка в {stage_name}: {e}")
            return StageResult(
                stage_name=stage_name,
                success=False,
                duration=time.time() - start_time,
                message=f"Ошибка: {str(e)}",
                error=e,
            )

    def _run_stage_2_structure(self) -> StageResult:
        """Этап 2: Анализ структуры проекта."""
        stage_name = "Этап 2: Анализ структуры проекта"
        start_time = time.time()

        try:
            self.logger.info("📁 " + stage_name)

            # Анализируем структуру
            self.logger.info("Анализ структуры проекта...")
            project_structure = self.structure_analyzer.analyze_structure()

            self.logger.info(f"Найдено директорий: {len(project_structure.directories)}")
            self.logger.info(f"Найдено entry points: {len(project_structure.entry_points)}")
            self.logger.info(
                f"Найдено конфигурационных файлов: {len(project_structure.config_files)}"
            )

            if self.dry_run:
                return StageResult(
                    stage_name=stage_name,
                    success=True,
                    duration=time.time() - start_time,
                    message=f"[DRY RUN] Проанализировано {len(project_structure.directories)} директорий",
                    details={
                        "directories_count": len(project_structure.directories),
                        "entry_points_count": len(project_structure.entry_points),
                        "config_files_count": len(project_structure.config_files),
                        "dry_run": True,
                    },
                )

            # Создаем документацию
            self.logger.info("Создание PROJECT_STRUCTURE.md...")
            doc_path = self.structure_documenter.create_structure_doc(project_structure)
            self.logger.info(f"Документация создана: {doc_path}")

            return StageResult(
                stage_name=stage_name,
                success=True,
                duration=time.time() - start_time,
                message=f"Проанализировано {len(project_structure.directories)} директорий, создан PROJECT_STRUCTURE.md",
                details={
                    "directories_count": len(project_structure.directories),
                    "entry_points_count": len(project_structure.entry_points),
                    "config_files_count": len(project_structure.config_files),
                    "doc_path": str(doc_path),
                },
            )

        except Exception as e:
            self.logger.error(f"Ошибка в {stage_name}: {e}")
            return StageResult(
                stage_name=stage_name,
                success=False,
                duration=time.time() - start_time,
                message=f"Ошибка: {str(e)}",
                error=e,
            )

    def _run_stage_3_modules(self) -> StageResult:
        """Этап 3: Создание реестра модулей."""
        stage_name = "Этап 3: Создание реестра модулей"
        start_time = time.time()

        try:
            self.logger.info("🐍 " + stage_name)

            # Сканируем модули
            self.logger.info("Сканирование Python модулей...")
            modules = self.module_scanner.scan_modules()

            if not modules:
                return StageResult(
                    stage_name=stage_name,
                    success=True,
                    duration=time.time() - start_time,
                    message="Python модули не найдены",
                    details={"modules_count": 0},
                )

            self.logger.info(f"Найдено {len(modules)} модулей")

            # Категоризируем модули
            self.logger.info("Категоризация модулей...")
            categorizer = ModuleCategorizer()
            categorized_modules = []

            for module in modules:
                category = categorizer.categorize_module(module)
                module.category = category
                categorized_modules.append(module)

            # Показываем статистику по категориям
            categories = {}
            for module in categorized_modules:
                categories[module.category] = categories.get(module.category, 0) + 1

            for category, count in categories.items():
                self.logger.info(f"  {category}: {count} модулей")

            if self.dry_run:
                return StageResult(
                    stage_name=stage_name,
                    success=True,
                    duration=time.time() - start_time,
                    message=f"[DRY RUN] Найдено {len(modules)} модулей в {len(categories)} категориях",
                    details={
                        "modules_count": len(modules),
                        "categories": categories,
                        "dry_run": True,
                    },
                )

            # Создаем реестр
            self.logger.info("Создание MODULE_REGISTRY.md...")
            registry_path = self.module_registry_builder.build_registry(categorized_modules)
            self.logger.info(f"Реестр создан: {registry_path}")

            return StageResult(
                stage_name=stage_name,
                success=True,
                duration=time.time() - start_time,
                message=f"Проанализировано {len(modules)} модулей в {len(categories)} категориях, создан MODULE_REGISTRY.md",
                details={
                    "modules_count": len(modules),
                    "categories": categories,
                    "registry_path": str(registry_path),
                },
            )

        except Exception as e:
            self.logger.error(f"Ошибка в {stage_name}: {e}")
            return StageResult(
                stage_name=stage_name,
                success=False,
                duration=time.time() - start_time,
                message=f"Ошибка: {str(e)}",
                error=e,
            )

    def _run_stage_4_llm_context(self) -> StageResult:
        """Этап 4: Создание LLM контекста."""
        stage_name = "Этап 4: Создание LLM контекста"
        start_time = time.time()

        try:
            self.logger.info("🤖 " + stage_name)

            if self.dry_run:
                return StageResult(
                    stage_name=stage_name,
                    success=True,
                    duration=time.time() - start_time,
                    message="[DRY RUN] LLM_CONTEXT.md будет создан",
                    details={"dry_run": True},
                )

            # Создаем LLM контекст
            self.logger.info("Создание LLM_CONTEXT.md...")
            context_path = self.llm_context_generator.generate_context()
            self.logger.info(f"LLM контекст создан: {context_path}")

            return StageResult(
                stage_name=stage_name,
                success=True,
                duration=time.time() - start_time,
                message="Создан LLM_CONTEXT.md с правилами работы с проектом",
                details={"context_path": str(context_path)},
            )

        except Exception as e:
            self.logger.error(f"Ошибка в {stage_name}: {e}")
            return StageResult(
                stage_name=stage_name,
                success=False,
                duration=time.time() - start_time,
                message=f"Ошибка: {str(e)}",
                error=e,
            )

    def _print_final_report(self, report: RefactoringReport) -> None:
        """Выводит итоговый отчет о рефакторинге."""
        self.logger.info("=" * 60)
        self.logger.info("ИТОГОВЫЙ ОТЧЕТ РЕФАКТОРИНГА")
        self.logger.info("=" * 60)

        self.logger.info(f"Время начала: {report.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Время окончания: {report.end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        self.logger.info(f"Общее время: {self._format_duration(report.total_duration)}")
        self.logger.info(
            f"Общий результат: {'✅ УСПЕХ' if report.overall_success else '❌ ОШИБКА'}"
        )
        self.logger.info(f"Успешных этапов: {report.successful_stages}/{len(report.stages)}")

        self.logger.info("\nДетали по этапам:")
        for i, stage in enumerate(report.stages, 1):
            status = "✅" if stage.success else "❌"
            self.logger.info(f"{i}. {status} {stage.stage_name}")
            self.logger.info(f"   Время: {self._format_duration(stage.duration)}")
            self.logger.info(f"   Результат: {stage.message}")

            if stage.details:
                for key, value in stage.details.items():
                    if key != "dry_run":
                        self.logger.info(f"   {key}: {value}")

            if stage.error:
                self.logger.error(f"   Ошибка: {stage.error}")

        if report.overall_success:
            self.logger.info("\n🎉 Глобальный рефакторинг завершен успешно!")
            self.logger.info("Созданные документы:")
            self.logger.info("- PROJECT_STRUCTURE.md - структура проекта")
            self.logger.info("- MODULE_REGISTRY.md - реестр всех модулей")
            self.logger.info("- LLM_CONTEXT.md - правила работы с проектом для LLM")
        else:
            self.logger.error("\n❌ Рефакторинг завершен с ошибками")
            self.logger.error("Проверьте логи для получения подробной информации")

    def _format_size(self, size_bytes: int) -> str:
        """Форматирует размер в человекочитаемый вид."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"

    def _format_duration(self, seconds: float) -> str:
        """Форматирует длительность в человекочитаемый вид."""
        if seconds < 60:
            return f"{seconds:.1f}с"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}м"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}ч"


def main():
    """Главная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description="Оркестратор глобального рефакторинга проекта recon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python global_refactoring_orchestrator.py                    # Запустить все этапы
  python global_refactoring_orchestrator.py --stage 1         # Запустить только этап 1
  python global_refactoring_orchestrator.py --dry-run         # Показать что будет сделано
  python global_refactoring_orchestrator.py --project-root /path/to/project  # Указать путь к проекту
        """,
    )

    parser.add_argument(
        "--stage",
        type=int,
        choices=[1, 2, 3, 4],
        help="Номер этапа для выполнения (1-4). Если не указан, выполняются все этапы",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Режим сухого прогона - показать что будет сделано без выполнения",
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        help="Путь к корневой директории проекта (по умолчанию текущая директория)",
    )

    parser.add_argument("--verbose", action="store_true", help="Подробный вывод логов")

    args = parser.parse_args()

    # Настраиваем уровень логирования
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    try:
        # Создаем оркестратор
        orchestrator = GlobalRefactoringOrchestrator(
            project_root=args.project_root, dry_run=args.dry_run
        )

        # Выполняем рефакторинг
        if args.stage:
            result = orchestrator.run_single_stage(args.stage)
            success = result.success
        else:
            report = orchestrator.run_all_stages()
            success = report.overall_success

        # Возвращаем код выхода
        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n❌ Рефакторинг прерван пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

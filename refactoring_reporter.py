#!/usr/bin/env python3
"""
Refactoring Final Reporter - Thin wrapper for intellirefactor.orchestration.refactoring_reporter

This is a deprecated wrapper. Please use intellirefactor.orchestration.RefineryReporter directly.
"""

import warnings
from intellirefactor.orchestration.refactoring_reporter import *

warnings.warn(
    "refactoring_reporter.py is deprecated; use intellirefactor.orchestration.RefineryReporter",
    DeprecationWarning,
    stacklevel=2,
)


@dataclass
class RefactoringStats:
    """Статистика рефакторинга."""

    garbage_files_found: int = 0
    garbage_files_moved: int = 0
    size_freed_bytes: int = 0
    directories_analyzed: int = 0
    entry_points_found: int = 0
    config_files_found: int = 0
    modules_analyzed: int = 0
    categories_found: int = 0
    documents_created: int = 0


class RefactoringReporter:
    """Генератор итогового отчета о рефакторинге."""

    def __init__(self, project_root: Path = None):
        """
        Инициализация репортера.

        Args:
            project_root: Корневая директория проекта
        """
        self.project_root = project_root or Path.cwd()
        self.stats = RefactoringStats()

    def generate_report(self) -> Dict[str, Any]:
        """
        Генерирует итоговый отчет о рефакторинге.

        Returns:
            Словарь с данными отчета
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "stats": {},
            "documents": {},
            "cleanup": {},
            "validation": {},
            "summary": {},
        }

        # Анализируем результаты очистки
        self._analyze_cleanup_results(report)

        # Анализируем созданные документы
        self._analyze_created_documents(report)

        # Анализируем структуру проекта
        self._analyze_project_structure(report)

        # Анализируем реестр модулей
        self._analyze_module_registry(report)

        # Создаем сводку
        self._create_summary(report)

        return report

    def _analyze_cleanup_results(self, report: Dict[str, Any]) -> None:
        """Анализирует результаты очистки мусора."""
        to_delete_path = self.project_root / "_to_delete"

        cleanup_data = {
            "to_delete_exists": to_delete_path.exists(),
            "moved_files": 0,
            "categories": {},
            "total_size": 0,
        }

        if to_delete_path.exists():
            # Подсчитываем перемещенные файлы
            moved_files = list(to_delete_path.rglob("*"))
            moved_files = [f for f in moved_files if f.is_file()]
            cleanup_data["moved_files"] = len(moved_files)

            # Подсчитываем размер
            total_size = sum(f.stat().st_size for f in moved_files if f.exists())
            cleanup_data["total_size"] = total_size
            self.stats.size_freed_bytes = total_size

            # Анализируем категории по папкам
            categories = {}
            for file_path in moved_files:
                # Определяем категорию по родительской папке
                parent = file_path.parent.name
                if parent != "_to_delete":
                    categories[parent] = categories.get(parent, 0) + 1

            cleanup_data["categories"] = categories
            self.stats.garbage_files_moved = len(moved_files)

        report["cleanup"] = cleanup_data

    def _analyze_created_documents(self, report: Dict[str, Any]) -> None:
        """Анализирует созданные документы."""
        expected_docs = ["PROJECT_STRUCTURE.md", "MODULE_REGISTRY.md", "LLM_CONTEXT.md"]

        documents_data = {"expected": len(expected_docs), "created": 0, "details": {}}

        for doc_name in expected_docs:
            doc_path = self.project_root / doc_name
            doc_info = {"exists": doc_path.exists(), "size": 0, "lines": 0, "created_time": None}

            if doc_path.exists():
                documents_data["created"] += 1
                doc_info["size"] = doc_path.stat().st_size

                try:
                    content = doc_path.read_text(encoding="utf-8")
                    doc_info["lines"] = len(content.splitlines())
                except Exception:
                    doc_info["lines"] = 0

                # Время создания
                doc_info["created_time"] = datetime.fromtimestamp(
                    doc_path.stat().st_mtime
                ).isoformat()

            documents_data["details"][doc_name] = doc_info

        self.stats.documents_created = documents_data["created"]
        report["documents"] = documents_data

    def _analyze_project_structure(self, report: Dict[str, Any]) -> None:
        """Анализирует PROJECT_STRUCTURE.md."""
        doc_path = self.project_root / "PROJECT_STRUCTURE.md"

        structure_data = {
            "exists": doc_path.exists(),
            "directories_documented": 0,
            "entry_points_found": 0,
            "config_files_found": 0,
        }

        if doc_path.exists():
            try:
                content = doc_path.read_text(encoding="utf-8")

                # Подсчитываем директории (строки с ###)
                directories = len(re.findall(r"^###\s+", content, re.MULTILINE))
                structure_data["directories_documented"] = directories
                self.stats.directories_analyzed = directories

                # Подсчитываем entry points
                entry_points = len(re.findall(r"entry.?point", content, re.IGNORECASE))
                structure_data["entry_points_found"] = entry_points
                self.stats.entry_points_found = entry_points

                # Подсчитываем конфигурационные файлы
                config_files = len(re.findall(r"\.(json|yaml|yml|ini|conf|toml)", content))
                structure_data["config_files_found"] = config_files
                self.stats.config_files_found = config_files

            except Exception as e:
                structure_data["error"] = str(e)

        report["structure"] = structure_data

    def _analyze_module_registry(self, report: Dict[str, Any]) -> None:
        """Анализирует MODULE_REGISTRY.md."""
        doc_path = self.project_root / "MODULE_REGISTRY.md"

        registry_data = {
            "exists": doc_path.exists(),
            "modules_documented": 0,
            "categories_found": 0,
            "categories": {},
        }

        if doc_path.exists():
            try:
                content = doc_path.read_text(encoding="utf-8")

                # Подсчитываем модули (более гибкие паттерны)
                module_patterns = [
                    r"^##\s+.*\.py",  # ## module.py
                    r"^\*\*.*\.py\*\*",  # **module.py**
                    r"###\s+.*\.py",  # ### module.py
                ]

                modules = 0
                for pattern in module_patterns:
                    matches = re.findall(pattern, content, re.MULTILINE)
                    modules += len(matches)

                registry_data["modules_documented"] = modules
                self.stats.modules_analyzed = modules

                # Подсчитываем категории (строки с #, но не заголовок документа)
                categories = re.findall(r"^#\s+(.+)", content, re.MULTILINE)
                # Исключаем заголовок документа и общие заголовки
                categories = [
                    cat
                    for cat in categories
                    if not any(
                        word in cat.lower()
                        for word in ["module registry", "реестр модулей", "overview", "обзор"]
                    )
                ]
                registry_data["categories_found"] = len(categories)
                self.stats.categories_found = len(categories)

                # Детали по категориям
                for category in categories:
                    # Подсчитываем модули в каждой категории
                    category_pattern = rf"^#\s+{re.escape(category)}.*?(?=^#|\Z)"
                    category_section = re.search(
                        category_pattern, content, re.MULTILINE | re.DOTALL
                    )
                    if category_section:
                        category_content = category_section.group(0)
                        module_count = 0
                        for pattern in module_patterns:
                            matches = re.findall(pattern, category_content, re.MULTILINE)
                            module_count += len(matches)
                        registry_data["categories"][category] = module_count

            except Exception as e:
                registry_data["error"] = str(e)

        report["registry"] = registry_data

    def _create_summary(self, report: Dict[str, Any]) -> None:
        """Создает итоговую сводку."""
        summary = {
            "overall_success": True,
            "completed_stages": [],
            "issues": [],
            "recommendations": [],
        }

        # Проверяем этап очистки
        if report["cleanup"]["to_delete_exists"]:
            summary["completed_stages"].append("Очистка мусора")
            if report["cleanup"]["moved_files"] == 0:
                summary["issues"].append("Файлы мусора не найдены или не перемещены")
        else:
            summary["issues"].append("Этап очистки не выполнен")
            summary["overall_success"] = False

        # Проверяем создание документов
        docs_created = report["documents"]["created"]
        docs_expected = report["documents"]["expected"]

        if docs_created == docs_expected:
            summary["completed_stages"].append("Создание документации")
        else:
            summary["issues"].append(f"Создано {docs_created} из {docs_expected} документов")
            summary["overall_success"] = False

        # Проверяем анализ структуры
        if report.get("structure", {}).get("exists"):
            summary["completed_stages"].append("Анализ структуры проекта")
        else:
            summary["issues"].append("PROJECT_STRUCTURE.md не создан")
            summary["overall_success"] = False

        # Проверяем реестр модулей
        if report.get("registry", {}).get("exists"):
            summary["completed_stages"].append("Создание реестра модулей")
        else:
            summary["issues"].append("MODULE_REGISTRY.md не создан")
            summary["overall_success"] = False

        # Рекомендации
        if self.stats.modules_analyzed > 100:
            summary["recommendations"].append("Рассмотрите возможность разделения больших модулей")

        if self.stats.categories_found < 5:
            summary["recommendations"].append("Возможно стоит пересмотреть категоризацию модулей")

        if report["cleanup"]["moved_files"] > 200:
            summary["recommendations"].append("Рассмотрите настройку автоматической очистки")

        report["summary"] = summary
        report["stats"] = asdict(self.stats)

    def save_report(self, report: Dict[str, Any], filename: str = None) -> Path:
        """
        Сохраняет отчет в файл.

        Args:
            report: Данные отчета
            filename: Имя файла (по умолчанию генерируется автоматически)

        Returns:
            Путь к сохраненному файлу
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"refactoring_report_{timestamp}.json"

        report_path = self.project_root / filename

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return report_path

    def print_report(self, report: Dict[str, Any]) -> None:
        """Выводит отчет в консоль."""
        print("=" * 60)
        print("ИТОГОВЫЙ ОТЧЕТ ГЛОБАЛЬНОГО РЕФАКТОРИНГА")
        print("=" * 60)

        # Основная информация
        print(f"Проект: {report['project_root']}")
        print(f"Время создания отчета: {report['timestamp']}")
        print()

        # Статистика
        stats = report["stats"]
        print("📊 СТАТИСТИКА:")
        print(f"  Файлов мусора перемещено: {stats['garbage_files_moved']}")
        print(f"  Освобождено места: {self._format_size(stats['size_freed_bytes'])}")
        print(f"  Директорий проанализировано: {stats['directories_analyzed']}")
        print(f"  Entry points найдено: {stats['entry_points_found']}")
        print(f"  Конфигурационных файлов: {stats['config_files_found']}")
        print(f"  Модулей проанализировано: {stats['modules_analyzed']}")
        print(f"  Категорий функционала: {stats['categories_found']}")
        print(f"  Документов создано: {stats['documents_created']}")
        print()

        # Созданные документы
        print("📄 СОЗДАННЫЕ ДОКУМЕНТЫ:")
        for doc_name, doc_info in report["documents"]["details"].items():
            status = "✅" if doc_info["exists"] else "❌"
            print(f"  {status} {doc_name}")
            if doc_info["exists"]:
                print(f"      Размер: {self._format_size(doc_info['size'])}")
                print(f"      Строк: {doc_info['lines']}")
        print()

        # Категории модулей
        if report.get("registry", {}).get("categories"):
            print("🏷️  КАТЕГОРИИ МОДУЛЕЙ:")
            for category, count in report["registry"]["categories"].items():
                print(f"  {category}: {count} модулей")
            print()

        # Результаты очистки
        if report["cleanup"]["categories"]:
            print("🧹 ОЧИСТКА ПО КАТЕГОРИЯМ:")
            for category, count in report["cleanup"]["categories"].items():
                print(f"  {category}: {count} файлов")
            print()

        # Итоговая сводка
        summary = report["summary"]
        print("📋 СВОДКА:")
        print(f"  Общий результат: {'✅ УСПЕХ' if summary['overall_success'] else '❌ ПРОБЛЕМЫ'}")
        print(f"  Завершенные этапы: {len(summary['completed_stages'])}")
        for stage in summary["completed_stages"]:
            print(f"    ✅ {stage}")

        if summary["issues"]:
            print(f"  Найденные проблемы: {len(summary['issues'])}")
            for issue in summary["issues"]:
                print(f"    ❌ {issue}")

        if summary["recommendations"]:
            print(f"  Рекомендации: {len(summary['recommendations'])}")
            for rec in summary["recommendations"]:
                print(f"    💡 {rec}")

        print("=" * 60)

    def _format_size(self, size_bytes: int) -> str:
        """Форматирует размер в человекочитаемый вид."""
        for unit in ["B", "KB", "MB", "GB"]:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"


def main():
    """Главная функция для запуска репортера."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Генератор итогового отчета о глобальном рефакторинге"
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        help="Путь к корневой директории проекта (по умолчанию текущая директория)",
    )

    parser.add_argument("--save", action="store_true", help="Сохранить отчет в JSON файл")

    parser.add_argument("--output", type=str, help="Имя файла для сохранения отчета")

    args = parser.parse_args()

    try:
        reporter = RefactoringReporter(project_root=args.project_root)
        report = reporter.generate_report()

        # Выводим отчет в консоль
        reporter.print_report(report)

        # Сохраняем в файл если нужно
        if args.save:
            report_path = reporter.save_report(report, args.output)
            print(f"\n💾 Отчет сохранен: {report_path}")

    except Exception as e:
        print(f"❌ Ошибка создания отчета: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

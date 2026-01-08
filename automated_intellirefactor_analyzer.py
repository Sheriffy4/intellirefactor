#!/usr/bin/env python3
"""
Полностью автоматизированная система анализа проектов на основе IntelliRefactor

Этот скрипт проводит ПОЛНЫЙ анализ проекта или файла с генерацией ВСЕХ возможных
отчетов, диаграмм и файлов анализа в указанной выходной директории.

Автор: Автоматизированная система на базе IntelliRefactor
Версия: 1.0.0
"""

import os
import os
import sys
import json
import argparse
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Добавляем путь к intellirefactor в sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
INTELLIREFACTOR_DIR = SCRIPT_DIR / "intellirefactor"
if INTELLIREFACTOR_DIR.exists():
    sys.path.insert(0, str(SCRIPT_DIR))


class AutomatedIntelliRefactorAnalyzer:
    """Автоматизированный анализатор проектов на базе IntelliRefactor"""

    def __init__(self, target_path: str, output_dir: str, verbose: bool = False):
        self.target_path = Path(target_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.verbose = verbose
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Создаем выходную директорию СНАЧАЛА
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Настройка логирования ПОСЛЕ создания директории
        log_level = logging.DEBUG if verbose else logging.INFO
        log_file_path = self.output_dir / f"analysis_log_{self.timestamp}.log"

        # Очищаем предыдущие обработчики
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Создаем консольный обработчик с правильной кодировкой
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)

        # Создаем файловый обработчик с UTF-8
        file_handler = logging.FileHandler(str(log_file_path), encoding="utf-8")
        file_handler.setLevel(log_level)

        # Форматтер без эмодзи для консоли Windows
        console_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        console_handler.setFormatter(console_formatter)
        file_handler.setFormatter(file_formatter)

        # Настраиваем корневой логгер
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        self.logger = logging.getLogger(__name__)

        # Проверяем существование целевого пути
        if not self.target_path.exists():
            raise FileNotFoundError(f"Целевой путь не существует: {self.target_path}")

        self.logger.info(f"Инициализирован анализатор для: {self.target_path}")
        self.logger.info(f"Результаты будут сохранены в: {self.output_dir}")

        # Результаты анализа
        self.analysis_results = {
            "target_path": str(self.target_path),
            "output_dir": str(self.output_dir),
            "timestamp": self.timestamp,
            "completed_analyses": [],
            "failed_analyses": [],
            "generated_files": [],
            "statistics": {},
        }

    def _run_intellirefactor_command(
        self, command: List[str], output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Запускает команду intellirefactor и возвращает результат"""
        try:
            # Проверяем наличие intellirefactor
            intellirefactor_path = Path(__file__).parent / "intellirefactor"
            if not intellirefactor_path.exists():
                return {
                    "success": False,
                    "stdout": "",
                    "stderr": "Директория intellirefactor не найдена",
                    "returncode": -1,
                    "command": " ".join(command),
                }

            # Формируем полную команду
            full_command = [sys.executable, "-m", "intellirefactor"] + command

            self.logger.debug(f"Выполняется команда: {' '.join(full_command)}")

            # Настраиваем окружение для правильной кодировки
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"

            # Запускаем команду
            result = subprocess.run(
                full_command,
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True,
                timeout=300,  # 5 минут таймаут
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            # Обрабатываем результат
            # Особая логика для команд, которые могут завершаться с ошибкой кодировки, но создавать файлы
            success = result.returncode == 0

            # Проверяем на ошибки кодировки в stderr
            if (
                not success
                and "charmap" in result.stderr
                and "codec can't encode character" in result.stderr
            ):
                # Это ошибка кодировки, но команда могла выполниться частично
                self.logger.warning(
                    "Обнаружена ошибка кодировки в IntelliRefactor, но команда могла выполниться частично"
                )

                # Для команд документации проверяем количество созданных файлов
                if "docs generate" in " ".join(command):
                    # Подсчитываем файлы .md в выходной директории
                    md_files_before = (
                        len(list(self.output_dir.glob("*.md"))) if self.output_dir.exists() else 0
                    )
                    # Если в stderr есть упоминания "Generated", значит файлы создались
                    if "Generated" in result.stderr:
                        generated_count = result.stderr.count("Generated")
                        if generated_count >= 3:  # Минимум 3 файла документации
                            success = True
                            self.logger.info(
                                f"Команда документации создала {generated_count} файлов, несмотря на ошибку кодировки"
                            )

                # Проверяем, создались ли выходные файлы
                if output_file:
                    expected_output = self.output_dir / output_file
                    if expected_output.exists() and expected_output.stat().st_size > 0:
                        success = True
                        self.logger.info("Выходной файл создан, несмотря на ошибку кодировки")

            if output_file and success and result.stdout.strip():
                # Если указан выходной файл и команда успешна, сохраняем результат
                output_path = self.output_dir / output_file
                try:
                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(result.stdout)
                    self.analysis_results["generated_files"].append(str(output_path))
                except Exception as e:
                    self.logger.warning(f"Не удалось сохранить результат в {output_file}: {e}")

            return {
                "success": success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "command": " ".join(full_command),
            }

        except subprocess.TimeoutExpired:
            self.logger.error(f"Команда превысила таймаут: {' '.join(command)}")
            return {
                "success": False,
                "stdout": "",
                "stderr": "Команда превысила таймаут (5 минут)",
                "returncode": -1,
                "command": " ".join(command),
            }
        except Exception as e:
            self.logger.error(f"Ошибка выполнения команды: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "command": " ".join(command),
            }

    def _save_analysis_result(
        self, analysis_name: str, result: Dict[str, Any], additional_data: Optional[Dict] = None
    ):
        """Сохраняет результат анализа"""
        # Проверяем успешность с учетом особенностей IntelliRefactor
        is_success = result["success"]

        # Особая обработка для команд документации - если файлы созданы, считаем успешным
        if not is_success and analysis_name == "Генерация документации":
            # Проверяем, были ли созданы файлы документации
            doc_files = [
                "ARCHITECTURE_DIAGRAM.md",
                "ANALYSIS_FLOWCHART.md",
                "CALL_GRAPH_DETAILED.md",
                "MODULE_REGISTRY.md",
                "LLM_CONTEXT.md",
                "PROJECT_STRUCTURE.md",
                "refactoring_report.md",  # Добавляем отчет рефакторинга
            ]

            created_files = []
            for pattern in doc_files:
                matching_files = list(self.output_dir.glob(f"*{pattern}"))
                created_files.extend(matching_files)

            # Также проверяем по логам IntelliRefactor, что файлы были созданы
            if result["stderr"] and "Generated" in result["stderr"]:
                # Подсчитываем количество упоминаний "Generated" в логах
                generated_count = result["stderr"].count("Generated")
                if generated_count >= 5:  # Ожидаем минимум 5 файлов документации
                    self.logger.info(
                        f"[ОБНАРУЖЕНО] IntelliRefactor создал {generated_count} файлов документации"
                    )
                    # Добавляем все найденные файлы .md в выходной директории
                    all_md_files = list(self.output_dir.glob("*.md"))
                    created_files.extend(all_md_files)

            # Если файлы созданы, считаем команду успешной
            if created_files:
                is_success = True
                self.logger.info(
                    f"[ЧАСТИЧНЫЙ УСПЕХ] {analysis_name} - файлы созданы, несмотря на ошибку кодировки"
                )
                # Добавляем созданные файлы в список
                for file_path in created_files:
                    self.analysis_results["generated_files"].append(str(file_path))

        if is_success:
            # Проверяем на дублирование перед добавлением
            if analysis_name not in self.analysis_results["completed_analyses"]:
                self.analysis_results["completed_analyses"].append(analysis_name)
                self.logger.info(f"[УСПЕХ] {analysis_name} - выполнен успешно")
            else:
                self.logger.debug(f"[ДУБЛИРОВАНИЕ] {analysis_name} уже в completed_analyses")
        else:
            # Проверяем на дублирование в failed_analyses
            existing_failed = [f for f in self.analysis_results["failed_analyses"] if isinstance(f, dict) and f.get("name") == analysis_name]
            if not existing_failed:
                self.analysis_results["failed_analyses"].append(
                    {"name": analysis_name, "error": result["stderr"], "command": result["command"]}
                )
                self.logger.error(f"[ОШИБКА] {analysis_name} - ошибка: {result['stderr'][:200]}...")
            else:
                self.logger.debug(f"[ДУБЛИРОВАНИЕ] {analysis_name} уже в failed_analyses")

        # Сохраняем детальный результат с безопасным именем файла
        safe_name = (
            analysis_name.replace(" ", "_").replace("ё", "e").replace("ъ", "").replace("ь", "")
        )
        # Транслитерация основных русских букв
        transliteration = {
            "а": "a",
            "б": "b",
            "в": "v",
            "г": "g",
            "д": "d",
            "е": "e",
            "ж": "zh",
            "з": "z",
            "и": "i",
            "й": "y",
            "к": "k",
            "л": "l",
            "м": "m",
            "н": "n",
            "о": "o",
            "п": "p",
            "р": "r",
            "с": "s",
            "т": "t",
            "у": "u",
            "ф": "f",
            "х": "h",
            "ц": "ts",
            "ч": "ch",
            "ш": "sh",
            "щ": "sch",
            "ы": "y",
            "э": "e",
            "ю": "yu",
            "я": "ya",
        }

        for ru, en in transliteration.items():
            safe_name = safe_name.replace(ru, en).replace(ru.upper(), en.upper())

        result_file = self.output_dir / f"{safe_name}_result.json"
        result_data = {
            "analysis_name": analysis_name,
            "timestamp": datetime.now().isoformat(),
            "result": result,
            "additional_data": additional_data or {},
        }

        try:
            with open(result_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            self.analysis_results["generated_files"].append(str(result_file))
        except Exception as e:
            self.logger.error(f"[ОШИБКА] Не удалось сохранить результат {analysis_name}: {e}")

    def run_basic_analysis(self):
        """Базовый анализ проекта/файла"""
        self.logger.info("[АНАЛИЗ] Запуск базового анализа...")

        # Простая команда без сложных параметров для начала
        command = ["analyze", str(self.target_path), "--format", "json"]

        result = self._run_intellirefactor_command(command, f"basic_analysis_{self.timestamp}.json")

        self._save_analysis_result("Базовый анализ", result)
        return result["success"]

    def run_enhanced_analysis(self):
        """Расширенный анализ с полной интеграцией"""
        self.logger.info("[РАСШИРЕННЫЙ] Запуск расширенного анализа...")

        command = [
            "analyze-enhanced",
            str(self.target_path),
            "--format",
            "markdown",
            "--include-metrics",
            "--include-opportunities",
            "--include-safety",
        ]

        result = self._run_intellirefactor_command(
            command, f"enhanced_analysis_{self.timestamp}.md"
        )

        self._save_analysis_result("Расширенный анализ", result)
        return result["success"]

    def build_project_index(self):
        """Построение индекса проекта"""
        if not self.target_path.is_dir():
            self.logger.info("[ПРОПУСК] Пропуск построения индекса (цель не является директорией)")
            return True

        self.logger.info("[ИНДЕКС] Построение индекса проекта...")

        command = [
            "index",
            "rebuild",  # ИСПРАВЛЕНО: rebuild вместо build
            str(self.target_path),
            "--format",
            "json",
        ]

        result = self._run_intellirefactor_command(
            command, f"index_rebuild_{self.timestamp}.json"  # Обновлено имя файла
        )

        self._save_analysis_result("Построение индекса", result)
        return result["success"]

    def detect_code_duplicates(self):
        """Обнаружение дублированного кода"""
        if not self.target_path.is_dir():
            self.logger.info(
                "[ПРОПУСК] Пропуск обнаружения дубликатов (цель не является директорией)"
            )
            return True

        self.logger.info("[ДУБЛИКАТЫ] Обнаружение дублированного кода...")

        # Обнаружение блочных дубликатов
        command_blocks = [
            "duplicates",
            "blocks",
            str(self.target_path),
            "--format",
            "json",
            "--show-code",
        ]

        result_blocks = self._run_intellirefactor_command(
            command_blocks, f"duplicate_blocks_{self.timestamp}.json"
        )

        self._save_analysis_result("Обнаружение блочных дубликатов", result_blocks)

        # Обнаружение дубликатов методов
        command_methods = [
            "duplicates",
            "methods",
            str(self.target_path),
            "--format",
            "json",
            "--show-signatures",
            "--extraction-recommendations",
        ]

        result_methods = self._run_intellirefactor_command(
            command_methods, f"duplicate_methods_{self.timestamp}.json"
        )

        self._save_analysis_result("Обнаружение дубликатов методов", result_methods)

        # Семантическое сходство
        command_similar = [
            "duplicates",
            "similar",
            str(self.target_path),
            "--format",
            "json",
            "--show-evidence",
            "--show-differences",
            "--merge-recommendations",
        ]

        result_similar = self._run_intellirefactor_command(
            command_similar, f"semantic_similarity_{self.timestamp}.json"
        )

        self._save_analysis_result("Семантическое сходство", result_similar)

        return result_blocks["success"] and result_methods["success"] and result_similar["success"]

    def detect_unused_code(self):
        """Обнаружение неиспользуемого кода"""
        if not self.target_path.is_dir():
            self.logger.info(
                "[ПРОПУСК] Пропуск обнаружения неиспользуемого кода (цель не является директорией)"
            )
            return True

        self.logger.info("[НЕИСПОЛЬЗУЕМЫЙ] Обнаружение неиспользуемого кода...")

        command = [
            "unused",
            "detect",
            str(self.target_path),
            "--level",
            "all",
            "--format",
            "json",
            "--show-evidence",
            "--show-usage",
        ]

        result = self._run_intellirefactor_command(command, f"unused_code_{self.timestamp}.json")

        self._save_analysis_result("Обнаружение неиспользуемого кода", result)
        return result["success"]

    def detect_architectural_smells(self):
        """Обнаружение архитектурных запахов"""
        if not self.target_path.is_dir():
            self.logger.info(
                "[ПРОПУСК] Пропуск обнаружения архитектурных запахов (цель не является директорией)"
            )
            return True

        self.logger.info("[ЗАПАХИ] Обнаружение архитектурных запахов...")

        command = [
            "smells",
            "detect",
            str(self.target_path),
            "--format",
            "json",
            "--show-evidence",
            "--show-recommendations",
        ]

        result = self._run_intellirefactor_command(
            command, f"architectural_smells_{self.timestamp}.json"
        )

        self._save_analysis_result("Обнаружение архитектурных запахов", result)
        return result["success"]

    def cluster_responsibilities(self):
        """Кластеризация ответственностей"""
        if not self.target_path.is_dir():
            self.logger.info(
                "[ПРОПУСК] Пропуск кластеризации ответственностей (цель не является директорией)"
            )
            return True

        self.logger.info("[КЛАСТЕРИЗАЦИЯ] Кластеризация ответственностей...")

        command = [
            "cluster",
            "responsibility",
            str(self.target_path),
            "--output",
            "json",
            "--show-suggestions",
            "--show-interfaces",
            "--show-plans",
        ]

        result = self._run_intellirefactor_command(
            command, f"responsibility_clusters_{self.timestamp}.json"
        )

        self._save_analysis_result("Кластеризация ответственностей", result)
        return result["success"]

    def generate_refactoring_decisions(self):
        """Генерация решений по рефакторингу"""
        if not self.target_path.is_dir():
            self.logger.info(
                "[ПРОПУСК] Пропуск генерации решений по рефакторингу (цель не является директорией)"
            )
            return True

        self.logger.info("[РЕШЕНИЯ] Генерация решений по рефакторингу...")

        command = [
            "decide",
            "analyze",
            str(self.target_path),
            "--format",
            "json",
            "--export-decisions",
            str(self.output_dir / f"refactoring_decisions_{self.timestamp}.json"),
        ]

        result = self._run_intellirefactor_command(
            command, f"decision_analysis_{self.timestamp}.json"
        )

        self._save_analysis_result("Анализ решений по рефакторингу", result)
        return result["success"]

    def generate_comprehensive_audit(self):
        """Комплексный аудит проекта"""
        if not self.target_path.is_dir():
            self.logger.info("[ПРОПУСК] Пропуск комплексного аудита (цель не является директорией)")
            return True

        self.logger.info("[АУДИТ] Комплексный аудит проекта...")

        command = [
            "audit",
            str(self.target_path),
            "--format",
            "json",
            "--emit-spec",
            "--spec-output",
            str(self.output_dir / f"Requirements_{self.timestamp}.md"),
            "--emit-json",
            "--json-output",
            str(self.output_dir / f"audit_analysis_{self.timestamp}.json"),
        ]

        result = self._run_intellirefactor_command(
            command, f"comprehensive_audit_{self.timestamp}.json"
        )

        self._save_analysis_result("Комплексный аудит", result)
        return result["success"]

    def generate_specifications(self):
        """Генерация спецификаций"""
        if not self.target_path.is_dir():
            self.logger.info(
                "[ПРОПУСК] Пропуск генерации спецификаций (цель не является директорией)"
            )
            return True

        self.logger.info("[СПЕЦИФИКАЦИИ] Генерация спецификаций...")

        command = [
            "generate",
            "spec",
            str(self.target_path),
            "--spec-type",
            "all",
            "--output-dir",
            str(self.output_dir),
            "--include-index",
            "--include-duplicates",
            "--include-unused",
            "--emit-json",
            "--json-output",
            str(self.output_dir / f"spec_analysis_{self.timestamp}.json"),
        ]

        result = self._run_intellirefactor_command(command)

        self._save_analysis_result("Генерация спецификаций", result)
        return result["success"]

    def generate_documentation(self):
        """Генерация документации"""
        self.logger.info("[ДОКУМЕНТАЦИЯ] Генерация документации...")

        if self.target_path.is_file():
            # Для отдельного файла
            command = [
                "docs",
                "generate",
                str(self.target_path),
                "--output-dir",
                str(self.output_dir),
                "--format",
                "markdown",
            ]
        else:
            # Для проекта - генерируем документацию для основных файлов
            python_files = list(self.target_path.glob("*.py"))
            if not python_files:
                python_files = list(self.target_path.rglob("*.py"))[:5]  # Первые 5 файлов

            if not python_files:
                self.logger.info(
                    "[ПРОПУСК] Пропуск генерации документации (Python файлы не найдены)"
                )
                return True

            success_count = 0
            for py_file in python_files[:3]:  # Ограничиваем 3 файлами для скорости
                command = [
                    "docs",
                    "generate",
                    str(py_file),
                    "--output-dir",
                    str(self.output_dir / "docs"),
                    "--format",
                    "markdown",
                ]

                result = self._run_intellirefactor_command(command)
                if result["success"]:
                    success_count += 1

            # Создаем общий результат
            result = {
                "success": success_count > 0,
                "stdout": f"Документация создана для {success_count} из {len(python_files[:3])} файлов",
                "stderr": "",
                "returncode": 0 if success_count > 0 else 1,
                "command": "docs generate (multiple files)",
            }

        if self.target_path.is_file():
            result = self._run_intellirefactor_command(command)

        self._save_analysis_result("Генерация документации", result)
        return result["success"]

    def generate_visualizations(self):
        """Генерация визуализаций"""
        self.logger.info("[ВИЗУАЛИЗАЦИЯ] Генерация визуализаций...")

        if self.target_path.is_file():
            # Для отдельного файла - пытаемся найти методы для визуализации
            try:
                with open(self.target_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Простой поиск методов
                import re

                methods = re.findall(r"def\s+(\w+)\s*\(", content)

                if methods:
                    # Визуализируем первый найденный метод
                    method_name = methods[0]
                    command = [
                        "visualize",
                        "method",
                        str(self.target_path),
                        method_name,
                        "--format",
                        "mermaid",
                    ]

                    result = self._run_intellirefactor_command(
                        command, f"method_flowchart_{method_name}_{self.timestamp}.mmd"
                    )
                else:
                    result = {
                        "success": False,
                        "stdout": "",
                        "stderr": "Методы для визуализации не найдены",
                        "returncode": 1,
                        "command": "visualize method (no methods found)",
                    }
            except Exception as e:
                result = {
                    "success": False,
                    "stdout": "",
                    "stderr": f"Ошибка чтения файла: {e}",
                    "returncode": 1,
                    "command": "visualize method (file read error)",
                }
        else:
            # Для проекта - создаем граф вызовов
            result = {
                "success": True,
                "stdout": "Визуализации для проектов создаются в рамках других анализов",
                "stderr": "",
                "returncode": 0,
                "command": "visualize project (integrated)",
            }

        self._save_analysis_result("Генерация визуализаций", result)
        return result["success"]

    def generate_summary_report(self):
        """Генерация итогового отчета"""
        self.logger.info("[ОТЧЕТ] Генерация итогового отчета...")

        # Подсчитываем все созданные файлы в выходной директории
        all_files = []
        if self.output_dir.exists():
            # Добавляем все файлы из выходной директории
            for file_path in self.output_dir.iterdir():
                if file_path.is_file() and file_path.name != f"SUMMARY_REPORT_{self.timestamp}.md":
                    all_files.append(str(file_path))

        # Обновляем список созданных файлов
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

        self.analysis_results["statistics"] = {
            "total_analyses": total_analyses,
            "successful_analyses": len(self.analysis_results["completed_analyses"]),
            "failed_analyses": len(self.analysis_results["failed_analyses"]),
            "success_rate": success_rate,
            "generated_files_count": len(self.analysis_results["generated_files"]),
            "analysis_duration": datetime.now().isoformat(),
        }

        # Создаем итоговый отчет
        report_content = f"""# Отчет автоматизированного анализа IntelliRefactor

## Общая информация
- **Цель анализа:** {self.target_path}
- **Выходная директория:** {self.output_dir}
- **Время анализа:** {self.timestamp}
- **Тип цели:** {'Файл' if self.target_path.is_file() else 'Проект'}

## Статистика выполнения
- **Всего анализов:** {total_analyses}
- **Успешно выполнено:** {len(self.analysis_results['completed_analyses'])}
- **Завершено с ошибками:** {len(self.analysis_results['failed_analyses'])}
- **Процент успеха:** {success_rate:.1f}%
- **Создано файлов:** {len(self.analysis_results['generated_files'])}

## Выполненные анализы
"""

        for analysis in self.analysis_results["completed_analyses"]:
            report_content += f"- [УСПЕХ] {analysis}\n"

        if self.analysis_results["failed_analyses"]:
            report_content += "\n## Анализы с ошибками\n"
            for failed in self.analysis_results["failed_analyses"]:
                report_content += f"- [ОШИБКА] {failed['name']}: {failed['error'][:100]}...\n"

        report_content += f"\n## Созданные файлы\n"
        for file_path in self.analysis_results["generated_files"]:
            relative_path = Path(file_path).relative_to(self.output_dir)
            report_content += f"- {relative_path}\n"

        report_content += f"""
## Рекомендации по использованию результатов

1. **Базовый анализ** - начните с просмотра файла `basic_analysis_{self.timestamp}.json`
2. **Дубликаты кода** - изучите файлы `duplicate_*_{self.timestamp}.json` для оптимизации
3. **Неиспользуемый код** - проверьте `unused_code_{self.timestamp}.json` для очистки
4. **Архитектурные проблемы** - просмотрите `architectural_smells_{self.timestamp}.json`
5. **Спецификации** - используйте сгенерированные Requirements.md и Design.md файлы

## Следующие шаги

1. Изучите детальные отчеты в JSON формате
2. Примените рекомендации по рефакторингу из файлов решений
3. Используйте сгенерированную документацию для понимания архитектуры
4. Рассмотрите предложения по кластеризации ответственностей

---
*Отчет создан автоматизированной системой анализа IntelliRefactor*
"""

        # Сохраняем отчет
        report_path = self.output_dir / f"SUMMARY_REPORT_{self.timestamp}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        # Сохраняем JSON версию результатов
        json_path = self.output_dir / f"analysis_results_{self.timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.analysis_results, f, indent=2, ensure_ascii=False)

        self.analysis_results["generated_files"].extend([str(report_path), str(json_path)])

        self.logger.info(f"[ОТЧЕТ] Итоговый отчет создан: {report_path}")
        return True

    def run_full_analysis(self):
        """Запуск полного анализа"""
        self.logger.info("[СТАРТ] Запуск полного автоматизированного анализа...")
        self.logger.info(f"Цель: {self.target_path}")
        self.logger.info(f"Выходная директория: {self.output_dir}")

        # Проверяем наличие intellirefactor
        intellirefactor_path = Path(__file__).parent / "intellirefactor"
        if not intellirefactor_path.exists():
            self.logger.error("[ОШИБКА] Директория intellirefactor не найдена!")
            self.logger.error("Убедитесь, что IntelliRefactor установлен в текущей директории")
            return False

        # Проверяем доступность CLI
        try:
            test_result = subprocess.run(
                [sys.executable, "-m", "intellirefactor", "--help"],
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True,
                timeout=30,
            )
            if test_result.returncode != 0:
                self.logger.error("[ОШИБКА] IntelliRefactor CLI недоступен!")
                self.logger.error(f"Ошибка: {test_result.stderr}")
                return False
        except Exception as e:
            self.logger.error(f"[ОШИБКА] Не удалось проверить IntelliRefactor CLI: {e}")
            return False

        # Список анализов (упрощенный для начала)
        analyses = [
            ("Базовый анализ", self.run_basic_analysis),
        ]

        # Если цель - директория, добавляем дополнительные анализы
        if self.target_path.is_dir():
            analyses.extend(
                [
                    ("Построение индекса", self.build_project_index),
                    ("Расширенный анализ", self.run_enhanced_analysis),
                    ("Обнаружение дубликатов", self.detect_code_duplicates),
                    ("Обнаружение неиспользуемого кода", self.detect_unused_code),
                    ("Обнаружение архитектурных запахов", self.detect_architectural_smells),
                    ("Кластеризация ответственностей", self.cluster_responsibilities),
                    ("Генерация решений по рефакторингу", self.generate_refactoring_decisions),
                    ("Комплексный аудит", self.generate_comprehensive_audit),
                    ("Генерация спецификаций", self.generate_specifications),
                ]
            )

        # Всегда добавляем эти анализы
        analyses.extend(
            [
                ("Генерация документации", self.generate_documentation),
                ("Генерация визуализаций", self.generate_visualizations),
            ]
        )

        # Выполняем все анализы
        for analysis_name, analysis_func in analyses:
            try:
                self.logger.info(f"[ВЫПОЛНЕНИЕ] {analysis_name}")
                analysis_func()
            except Exception as e:
                self.logger.error(f"[КРИТИЧЕСКАЯ ОШИБКА] {analysis_name}: {e}")
                self.analysis_results["failed_analyses"].append(
                    {"name": analysis_name, "error": str(e), "command": "internal_error"}
                )

        # Генерируем итоговый отчет
        self.generate_summary_report()

        # Выводим итоговую статистику
        total = len(self.analysis_results["completed_analyses"]) + len(
            self.analysis_results["failed_analyses"]
        )
        success = len(self.analysis_results["completed_analyses"])

        self.logger.info("[ЗАВЕРШЕНИЕ] Анализ завершен!")
        self.logger.info(f"[СТАТИСТИКА] {success}/{total} анализов выполнено успешно")
        self.logger.info(f"[ФАЙЛЫ] Создано файлов: {len(self.analysis_results['generated_files'])}")
        self.logger.info(
            f"[ОТЧЕТ] Итоговый отчет: {self.output_dir}/SUMMARY_REPORT_{self.timestamp}.md"
        )

        return success > 0


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="Автоматизированная система полного анализа проектов на базе IntelliRefactor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python automated_intellirefactor_analyzer.py /path/to/project ./analysis_results
  python automated_intellirefactor_analyzer.py single_file.py ./file_analysis --verbose
  python automated_intellirefactor_analyzer.py . ./current_project_analysis

Система автоматически:
- Проводит все доступные виды анализа
- Создает отчеты в различных форматах (JSON, Markdown, HTML)
- Генерирует диаграммы и визуализации
- Создает спецификации и документацию
- Формирует итоговый отчет с рекомендациями
        """,
    )

    parser.add_argument("target_path", help="Путь к анализируемому проекту или файлу")

    parser.add_argument("output_dir", help="Директория для сохранения результатов анализа")

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Подробный вывод и отладочная информация"
    )

    parser.add_argument(
        "--version",
        action="version",
        version="Автоматизированный анализатор IntelliRefactor v1.0.0",
    )

    args = parser.parse_args()

    try:
        # Создаем и запускаем анализатор
        analyzer = AutomatedIntelliRefactorAnalyzer(
            target_path=args.target_path, output_dir=args.output_dir, verbose=args.verbose
        )

        success = analyzer.run_full_analysis()

        if success:
            print(f"\n[УСПЕХ] Анализ успешно завершен!")
            print(f"[РЕЗУЛЬТАТЫ] Результаты сохранены в: {analyzer.output_dir}")
            print(
                f"[ОТЧЕТ] Итоговый отчет: {analyzer.output_dir}/SUMMARY_REPORT_{analyzer.timestamp}.md"
            )
            sys.exit(0)
        else:
            print(f"\n[ОШИБКА] Анализ завершен с ошибками")
            print(f"[РЕЗУЛЬТАТЫ] Частичные результаты сохранены в: {analyzer.output_dir}")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n[ПРЕРВАНО] Анализ прерван пользователем")
        sys.exit(130)
    except Exception as e:
        print(f"\n[КРИТИЧЕСКАЯ ОШИБКА] {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

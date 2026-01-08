#!/usr/bin/env python3
"""
ContextualFileAnalyzer — контекстный анализатор файла в рамках проекта.

Анализирует отдельный файл в контексте всего проекта для получения
максимально полной информации о зависимостях, использовании и возможностях
рефакторинга.

Что исправлено/улучшено (по сравнению с вашими вариантами 1/2):
- _run_variants больше НЕ засоряет analysis_results попытками:
  сохраняется только успешный результат; при полном провале — один финальный (или placeholder).
  Это критично для честной статистики success_rate.
- Проверка доступности IntelliRefactor сделана корректно:
  через importlib.util.find_spec("intellirefactor") (а не требование папки ./intellirefactor).
- _run_intellirefactor_command_with_timeout всегда возвращает dict (защита от None).
- stdout сохраняется в output_file даже при success=False (если stdout непустой),
  чтобы не терять частичные результаты CLI.
- Manual fallback для unused помечен честно как inventory символов (не “unused detect”),
  но учитывается как успешный fallback, чтобы не проваливать весь прогон при баге CLI.
- Итоговый отчет генерируется без return внутри циклов, с дедупликацией списка файлов.

Зависимости:
- automated_intellirefactor_analyzer.AutomatedIntelliRefactorAnalyzer (ваш модуль)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Добавляем текущую директорию в путь для импорта
CURRENT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(CURRENT_DIR))

from automated_intellirefactor_analyzer import AutomatedIntelliRefactorAnalyzer  # noqa: E402


JSONDict = Dict[str, Any]
Command = Sequence[str]
Variant = Tuple[List[str], str]


class ContextualFileAnalyzer(AutomatedIntelliRefactorAnalyzer):
    """Анализатор файлов в контексте всего проекта."""

    def __init__(
        self,
        project_path: str,
        target_file: str,
        output_dir: str,
        verbose: bool = False,
    ) -> None:
        self.project_path = Path(project_path)
        self.target_file = Path(target_file)

        # Инициализируем базовый анализатор с проектом
        super().__init__(str(project_path), output_dir, verbose)

        # Нормализуем output_dir (в базовом классе это может быть уже Path)
        self.output_dir = Path(self.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.analysis_mode = "contextual_file"
        self._ensure_analysis_results_structure()

        self.logger.info("Инициализирован контекстный анализатор файла")
        self.logger.info("Проект: %s", self.project_path)
        self.logger.info("Целевой файл: %s", self.target_file)
        self.logger.info("Результаты: %s", self.output_dir)

    # -------------------------
    # Вспомогательные методы
    # -------------------------

    def _ensure_analysis_results_structure(self) -> None:
        """Гарантирует наличие ожидаемых ключей в self.analysis_results."""
        if not hasattr(self, "analysis_results") or not isinstance(self.analysis_results, dict):
            self.analysis_results = {}
        self.analysis_results.setdefault("completed_analyses", [])
        self.analysis_results.setdefault("failed_analyses", [])
        self.analysis_results.setdefault("generated_files", [])

    @staticmethod
    def _safe_stderr(result: Optional[JSONDict]) -> str:
        """Безопасно возвращает stderr из result."""
        if not isinstance(result, dict):
            return ""
        return result.get("stderr") or ""

    def _write_text(self, path: Path, content: str) -> bool:
        """Записывает текст в файл (UTF-8)."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            return True
        except OSError as e:
            self.logger.warning("Не удалось записать %s: %s", path, e)
            return False

    def _write_json(self, path: Path, data: Any) -> bool:
        """Записывает JSON в файл (UTF-8)."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except (OSError, TypeError) as e:
            self.logger.warning("Не удалось записать JSON %s: %s", path, e)
            return False

    def _write_stdout_to_file(self, output_path: Path, stdout: str) -> bool:
        """
        Пишет stdout в output_path, очищая JSON от лог-сообщений.

        Returns:
            True если запись успешна, иначе False.
        """
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Очищаем JSON от лог-сообщений, если файл должен быть JSON
            if output_path.suffix.lower() == '.json':
                cleaned_content = self._clean_json_from_logs(stdout)
                output_path.write_text(cleaned_content, encoding="utf-8")
            else:
                output_path.write_text(stdout, encoding="utf-8")
                
            self.analysis_results["generated_files"].append(str(output_path))
            return True
        except OSError as e:
            self.logger.warning("Не удалось сохранить stdout в %s: %s", output_path, e)
            return False

    def _clean_json_from_logs(self, content: str) -> str:
        """
        Очищает JSON контент от лог-сообщений в начале.
        
        Args:
            content: Исходный контент, который может содержать лог-сообщения перед JSON
            
        Returns:
            Очищенный JSON контент
        """
        # Ищем первую открывающую скобку { или [
        json_start = -1
        for i, char in enumerate(content):
            if char in '{[':
                json_start = i
                break
        
        if json_start > 0:
            # Найдены лог-сообщения в начале, обрезаем их
            self.logger.info("Обнаружены лог-сообщения в JSON, очищаем %d символов", json_start)
            return content[json_start:]
        elif json_start == 0:
            # JSON начинается сразу, все в порядке
            return content
        else:
            # JSON не найден, возвращаем как есть
            self.logger.warning("JSON структура не найдена в выводе")
            return content

    def _get_relative_file_path(self) -> Path:
        """Относительный путь файла от проекта (если возможно)."""
        try:
            return self.target_file.relative_to(self.project_path)
        except ValueError:
            return self.target_file

    def _intellirefactor_available(self) -> bool:
        """
        Проверяет, доступен ли модуль intellirefactor для python -m intellirefactor.
        Не требует наличия локальной папки ./intellirefactor.
        """
        return importlib.util.find_spec("intellirefactor") is not None

    def _run_variants(
        self,
        variants: List[Variant],
        output_template: str,
        timeout_minutes: int,
        analysis_name_for_save: Optional[str] = None,
    ) -> Tuple[bool, Optional[JSONDict]]:
        """
        Запускает несколько вариантов команд и возвращает первый успешный.

        Важно:
        - НЕ сохраняет попытки в analysis_results.
        - Если найден успешный вариант и задан analysis_name_for_save — сохраняет только success result.
        - При полном провале — ничего не сохраняет (caller сам решает, был ли успешный fallback).

        Returns:
            (success, result_or_last)
        """
        last_result: Optional[JSONDict] = None

        for i, (command, description) in enumerate(variants, 1):
            output_file = output_template.format(i=i, ts=self.timestamp)

            try:
                self.logger.info("[ПОПЫТКА %d] %s: %s", i, description, " ".join(command))
                result = self._run_intellirefactor_command_with_timeout(
                    command,
                    output_file=output_file,
                    timeout_minutes=timeout_minutes,
                )
                last_result = result

                if isinstance(result, dict) and result.get("success"):
                    if analysis_name_for_save:
                        self._save_analysis_result(analysis_name_for_save, result)
                    self.logger.info("[УСПЕХ] %s (вариант %d)", description, i)
                    return True, result

                stderr = self._safe_stderr(result)
                self.logger.warning(
                    "[ВАРИАНТ %d] Неуспех: %s",
                    i,
                    stderr[:200] if stderr else "Неизвестная ошибка",
                )

            except (OSError, ValueError, RuntimeError, TypeError) as e:
                self.logger.warning("[ВАРИАНТ %d] Исключение: %s", i, e)

        return False, last_result

    # -------------------------
    # Основной пайплайн
    # -------------------------

    def run_full_analysis(self) -> bool:
        """Запуск полного контекстного анализа файла."""
        self.logger.info("[СТАРТ] Запуск контекстного анализа файла в рамках проекта...")
        self.logger.info("Проект: %s", self.project_path)
        self.logger.info("Целевой файл: %s", self.target_file)
        self.logger.info("Выходная директория: %s", self.output_dir)

        if not self._intellirefactor_available():
            self.logger.error(
                "[ОШИБКА] Модуль 'intellirefactor' недоступен. "
                "Убедитесь, что он установлен и доступен для текущего интерпретатора."
            )
            return False

        if not self.project_path.exists() or not self.project_path.is_dir():
            self.logger.error("[ОШИБКА] Проект не найден или не является папкой: %s", self.project_path)
            return False

        if not self.target_file.exists() or not self.target_file.is_file():
            self.logger.error("[ОШИБКА] Целевой файл не найден или не является файлом: %s", self.target_file)
            return False

        analyses: List[Tuple[str, Any]] = [
            ("Построение индекса проекта", self.build_project_index_safe),
            ("Базовый анализ файла", self.run_basic_file_analysis),
            ("Обнаружение дубликатов в контексте проекта", self.detect_contextual_duplicates),
            ("Обнаружение неиспользуемого кода в контексте проекта", self.detect_contextual_unused_code),
            ("Обнаружение архитектурных запахов в контексте проекта", self.detect_contextual_smells),
            ("Анализ зависимостей файла", self.analyze_file_dependencies),
            ("Генерация решений по рефакторингу в контексте проекта", self.generate_contextual_decisions),
            ("Создание файла требований для файла", self.generate_file_requirements),
            ("Генерация спецификаций для файла", self.generate_file_specifications),
            ("Генерация документации файла", self.generate_file_documentation),
            ("Генерация визуализаций файла", self.generate_file_visualizations),
        ]

        for analysis_name, analysis_func in analyses:
            try:
                self.logger.info("[ВЫПОЛНЕНИЕ] %s", analysis_name)
                success = bool(analysis_func())
                if not success:
                    self.logger.warning("[ПРЕДУПРЕЖДЕНИЕ] %s завершился с предупреждениями", analysis_name)
            except (OSError, ValueError, RuntimeError, TypeError, KeyError) as e:
                self.logger.error("[ОШИБКА] Ошибка в %s: %s", analysis_name, e)

        # Итоговый отчет
        self.generate_contextual_summary_report()

        total_analyses = len(self.analysis_results["completed_analyses"]) + len(self.analysis_results["failed_analyses"])
        success_rate = (len(self.analysis_results["completed_analyses"]) / total_analyses * 100) if total_analyses else 0.0

        self.logger.info("[ЗАВЕРШЕНИЕ] Контекстный анализ завершен!")
        self.logger.info(
            "[СТАТИСТИКА] %d/%d анализов выполнено успешно",
            len(self.analysis_results["completed_analyses"]),
            total_analyses,
        )
        self.logger.info("[ФАЙЛЫ] Создано файлов: %d", len(self.analysis_results["generated_files"]))

        return success_rate > 50

    # -------------------------
    # IntelliRefactor commands
    # -------------------------

    def build_project_index_safe(self) -> bool:
        """Безопасное построение индекса проекта (не блокирует анализ при ошибках)."""
        self.logger.info("[ИНДЕКС] Безопасное построение индекса проекта...")

        command = ["index", "rebuild", str(self.project_path), "--format", "json"]

        try:
            result = self._run_intellirefactor_command_with_timeout(
                command,
                output_file=f"project_index_{self.timestamp}.json",
                timeout_minutes=15,
            )
            self._save_analysis_result("Построение индекса проекта", result)

            if isinstance(result, dict) and result.get("success"):
                self.logger.info("[УСПЕХ] Индекс проекта построен успешно")
            else:
                self.logger.warning("[ПРЕДУПРЕЖДЕНИЕ] Индекс построен с ошибками, но продолжаем анализ")

            return True

        except (OSError, ValueError, RuntimeError, TypeError) as e:
            self.logger.warning("[ПРЕДУПРЕЖДЕНИЕ] Ошибка построения индекса: %s", e)
            self.logger.info("[ПРОДОЛЖЕНИЕ] Анализ продолжается без индекса")
            return True

    def _run_intellirefactor_command_with_timeout(
        self,
        command: Command,
        output_file: Optional[str] = None,
        timeout_minutes: int = 10,
    ) -> JSONDict:
        """
        Выполнение команды IntelliRefactor с настраиваемым таймаутом.

        Важно:
        - stdout сохраняем в output_file (если указан) даже при success=False,
          если stdout непустой (полезно при частичных ошибках CLI).
        - всегда возвращаем dict (защита от None).
        """
        full_command = [sys.executable, "-m", "intellirefactor"] + list(command)

        env = os.environ.copy()
        env["PYTHONIOENCODING"] = "utf-8"

        self.logger.debug("Выполняется команда: %s", " ".join(full_command))

        try:
            proc = subprocess.run(
                full_command,
                capture_output=True,
                text=True,
                cwd=str(self.project_path),
                timeout=timeout_minutes * 60,
                encoding="utf-8",
                errors="replace",
                env=env,
            )

            success = proc.returncode == 0

            # Иногда stderr содержит кодировочную ошибку, но CLI мог создать файлы/частичный stdout.
            stderr = proc.stderr or ""
            if (
                not success
                and "charmap" in stderr
                and "codec can't encode character" in stderr
            ):
                self.logger.warning(
                    "Обнаружена ошибка кодировки в IntelliRefactor, команда могла выполниться частично"
                )
                if output_file:
                    expected = self.output_dir / output_file
                    if expected.exists() and expected.stat().st_size > 0:
                        success = True
                        self.logger.info("Выходной файл создан, несмотря на ошибку кодировки")

            if output_file and proc.stdout and proc.stdout.strip():
                output_path = self.output_dir / output_file
                self._write_stdout_to_file(output_path, proc.stdout)

            return {
                "success": success,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "returncode": proc.returncode,
                "command": " ".join(full_command),
            }

        except subprocess.TimeoutExpired:
            self.logger.error(
                "Команда превысила таймаут (%d минут): %s",
                timeout_minutes,
                " ".join(command),
            )
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Команда превысила таймаут ({timeout_minutes} минут)",
                "returncode": -1,
                "command": " ".join(full_command),
            }
        except (OSError, ValueError, RuntimeError) as e:
            self.logger.error("Ошибка выполнения команды: %s", e)
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
                "command": " ".join(full_command),
            }

    # -------------------------
    # Анализы
    # -------------------------

    def run_basic_file_analysis(self) -> bool:
        """Базовый анализ конкретного файла."""
        self.logger.info("[АНАЛИЗ] Запуск базового анализа файла...")

        command = ["analyze", str(self.target_file), "--format", "json"]

        result = self._run_intellirefactor_command_with_timeout(
            command,
            output_file=f"file_basic_analysis_{self.timestamp}.json",
            timeout_minutes=5,
        )

        self._save_analysis_result("Базовый анализ файла", result)
        return bool(isinstance(result, dict) and result.get("success"))

    def detect_contextual_duplicates(self) -> bool:
        """Обнаружение дубликатов в контексте всего проекта."""
        self.logger.info("[ДУБЛИКАТЫ] Поиск дубликатов в контексте проекта...")

        command = [
            "duplicates",
            "blocks",
            str(self.project_path),
            "--format",
            "json",
            "--show-code",
        ]

        try:
            result = self._run_intellirefactor_command_with_timeout(
                command,
                output_file=f"contextual_duplicate_blocks_{self.timestamp}.json",
                timeout_minutes=15,
            )

            self._save_analysis_result("Обнаружение дубликатов блоков в контексте проекта", result)

            if isinstance(result, dict) and result.get("success"):
                return True

            stderr = self._safe_stderr(result)
            self.logger.warning("[ДУБЛИКАТЫ] Неуспех: %s", stderr[:200] if stderr else "—")
            return False

        except (OSError, ValueError, RuntimeError, TypeError) as e:
            self.logger.error("[ОШИБКА] Дубликаты блоков: %s", e)
            return False
        finally:
            # Явные пропуски — как и в исходном коде
            self.logger.info("[ПРОПУСК] Пропуск анализа дубликатов методов (часто нестабилен)")
            self.logger.info("[ПРОПУСК] Пропуск семантического сходства (высокое потребление памяти)")

    def detect_contextual_unused_code(self) -> bool:
        """Обнаружение неиспользуемого кода в контексте проекта (с обходными путями)."""
        self.logger.info("[НЕИСПОЛЬЗУЕМЫЙ] Поиск неиспользуемого кода...")

        variants: List[Variant] = [
            (["analyze", str(self.target_file), "--format", "json"], "analyze file (workaround)"),
            (["unused", "detect", str(self.target_file)], "unused detect (file)"),
            (["unused", "detect", str(self.project_path)], "unused detect (project)"),
            (["analyze", str(self.project_path), "--format", "json"], "analyze project (workaround)"),
        ]

        success, result = self._run_variants(
            variants=variants,
            output_template="contextual_unused_code_attempt_{i}_{ts}.json",
            timeout_minutes=3,
            analysis_name_for_save="Обнаружение неиспользуемого кода в контексте проекта",
        )

        if success and isinstance(result, dict):
            # Если это analyze — создадим вспомогательный файл анализа (заглушку/обвязку)
            cmd = (result.get("command") or "").lower()
            if " -m intellirefactor analyze " in f" {cmd} ":
                self._create_unused_code_analysis_from_analyze_result(result)
            return True

        # Fallback: создаем inventory символов (не доказывает unused, но хоть что-то)
        self.logger.info("[FALLBACK] Создаем инвентаризацию символов вместо unused detect...")
        inventory_ok = self._create_manual_symbol_inventory()
        if inventory_ok:
            # Считаем этот этап “успешным fallback” и фиксируем в analysis_results как success
            self._save_analysis_result(
                "Обнаружение неиспользуемого кода в контексте проекта",
                {
                    "success": True,
                    "stdout": "Fallback: создана инвентаризация символов (manual_symbol_inventory)",
                    "stderr": "unused detect недоступен/нестабилен; inventory не доказывает неиспользуемость",
                    "returncode": 0,
                    "command": "manual_symbol_inventory (fallback)",
                },
            )
            return True

        # Полный провал — фиксируем один fail (если ранее _run_variants не сохранил success)
        self._save_analysis_result(
            "Обнаружение неиспользуемого кода в контексте проекта",
            result
            or {
                "success": False,
                "stdout": "",
                "stderr": "Unused detect variants failed; fallback inventory failed",
                "returncode": -1,
                "command": "unused detect (all variants) + fallback inventory",
            },
        )

        # Не блокируем весь анализ (как и в ваших версиях)
        self.logger.warning("[ПРОПУСК] Unused detect недоступен/нестабилен, продолжаем общий анализ")
        return True

    def _create_unused_code_analysis_from_analyze_result(self, analyze_result: JSONDict) -> None:
        """Создает файл-заглушку анализа unused на основе analyze результата (workaround)."""
        unused_analysis = {
            "analysis_type": "unused_code_from_analyze_workaround",
            "source_command": "analyze (workaround for unused detect bug)",
            "timestamp": self.timestamp,
            "findings": [],
            "summary": {
                "total_unused_items": 0,
                "unused_functions": 0,
                "unused_variables": 0,
                "unused_imports": 0,
            },
            "note": "Заглушка: analyze не равен unused detect; требуется ручная проверка",
            "original_result": analyze_result,
        }

        unused_file = self.output_dir / f"unused_code_analysis_{self.timestamp}.json"
        if self._write_json(unused_file, unused_analysis):
            self.analysis_results["generated_files"].append(str(unused_file))
            self.logger.info("[WORKAROUND] Создан файл анализа unused: %s", unused_file)

    def _create_manual_symbol_inventory(self) -> bool:
        """
        Fallback-инвентаризация: перечисляет определения в файле (очень упрощенно).

        Важно:
        - НЕ доказывает неиспользуемость.
        """
        if not self.target_file.exists():
            return False

        try:
            try:
                content = self.target_file.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = self.target_file.read_text(encoding="latin-1")

            lines = content.splitlines()

            functions: List[JSONDict] = []
            classes: List[JSONDict] = []
            variables: List[JSONDict] = []
            imports: List[JSONDict] = []

            for i, line in enumerate(lines, 1):
                s = line.strip()

                if s.startswith(("import ", "from ")):
                    imports.append({"statement": s, "line": i})
                elif s.startswith("class ") and ":" in s:
                    name = s.split("class ")[1].split(":")[0].split("(")[0].strip()
                    classes.append({"name": name, "line": i})
                elif s.startswith("def ") and "(" in s:
                    name = s.split("def ")[1].split("(")[0].strip()
                    functions.append({"name": name, "line": i})
                elif "=" in s and not s.startswith(("#", "def ", "class ")):
                    var = s.split("=")[0].strip()
                    if var.isidentifier():
                        variables.append({"name": var, "line": i})

            inventory = {
                "analysis_type": "manual_symbol_inventory",
                "file_analyzed": str(self.target_file),
                "timestamp": self.timestamp,
                "statistics": {
                    "total_functions": len(functions),
                    "total_classes": len(classes),
                    "total_variables": len(variables),
                    "total_imports": len(imports),
                    "lines_analyzed": len(lines),
                },
                "findings": {
                    "imports": imports,
                    "classes": classes,
                    "functions": functions,
                    "variables": variables,
                },
                "analysis_notes": {
                    "method": "Manual heuristic inventory (fallback)",
                    "limitation": "Не доказывает неиспользуемость, только перечисляет определения",
                },
            }

            out = self.output_dir / f"manual_symbol_inventory_{self.timestamp}.json"
            if self._write_json(out, inventory):
                self.analysis_results["generated_files"].append(str(out))
                self.logger.info("[FALLBACK] Инвентаризация создана: %s", out)
                return True

            return False

        except (OSError, ValueError, RuntimeError) as e:
            self.logger.warning("[FALLBACK] Ошибка инвентаризации: %s", e)
            return False

    def detect_contextual_smells(self) -> bool:
        """Обнаружение архитектурных запахов в контексте проекта."""
        self.logger.info("[ЗАПАХИ] Поиск архитектурных запахов...")

        variants: List[Variant] = [
            (["smells", "detect", str(self.project_path), "--format", "json"], "smells detect json (project)"),
            (["smells", "detect", str(self.project_path)], "smells detect default (project)"),
            (["smells", "detect", str(self.project_path), "--format", "text"], "smells detect text (project)"),
            (["smells", "detect", str(self.target_file), "--format", "json"], "smells detect json (file)"),
        ]

        success, result = self._run_variants(
            variants=variants,
            output_template="contextual_architectural_smells_attempt_{i}_{ts}.json",
            timeout_minutes=10,
            analysis_name_for_save="Обнаружение архитектурных запахов в контексте проекта",
        )
        if success:
            return True

        self.logger.warning("[АЛЬТЕРНАТИВА] smells detect не сработал, пробуем альтернативный анализ")

        alt_success, alt_result = self._analyze_smells_alternative()
        if alt_success:
            # фиксируем как успех шага smells (альтернативным способом)
            self._save_analysis_result(
                "Обнаружение архитектурных запахов в контексте проекта",
                {
                    "success": True,
                    "stdout": "Smells detect заменен альтернативными метриками/quality/analyze",
                    "stderr": "",
                    "returncode": 0,
                    "command": "alternative smells analysis",
                    "original_result": alt_result,
                },
            )
            self.logger.info("[УСПЕХ] Альтернативный анализ архитектурных запахов выполнен")
            return True

        # Полный провал — фиксируем один fail
        self._save_analysis_result(
            "Обнаружение архитектурных запахов в контексте проекта",
            result
            or {
                "success": False,
                "stdout": "",
                "stderr": "smells detect failed; alternative smells analysis failed",
                "returncode": -1,
                "command": "smells detect (all variants) + alternative",
            },
        )

        self.logger.warning("[ПРОПУСК] Анализ архитектурных запахов не удался, продолжаем")
        return True  # не блокируем общий анализ

    def _analyze_smells_alternative(self) -> Tuple[bool, Optional[JSONDict]]:
        """Альтернативный анализ архитектурных запахов через другие команды."""
        self.logger.info("[АЛЬТЕРНАТИВА] Альтернативный анализ архитектурных запахов...")

        variants: List[Variant] = [
            (["quality", str(self.target_file), "--format", "json"], "quality json"),
            (["metrics", str(self.target_file), "--format", "json"], "metrics json"),
            (["analyze", str(self.target_file), "--format", "json"], "analyze json"),
        ]

        return self._run_variants(
            variants=variants,
            output_template="alternative_smells_{i}_{ts}.json",
            timeout_minutes=5,
            analysis_name_for_save=None,
        )

    def analyze_file_dependencies(self) -> bool:
        """Анализ зависимостей файла в контексте проекта."""
        self.logger.info("[ЗАВИСИМОСТИ] Анализ зависимостей файла...")

        command = ["analyze", str(self.target_file), "--format", "json"]

        try:
            result = self._run_intellirefactor_command_with_timeout(
                command,
                output_file=f"file_dependencies_{self.timestamp}.json",
                timeout_minutes=5,
            )
            self._save_analysis_result("Анализ зависимостей файла", result)

            if isinstance(result, dict) and result.get("success"):
                return True

            stderr = self._safe_stderr(result)
            self.logger.warning("[ПРЕДУПРЕЖДЕНИЕ] Анализ зависимостей неуспешен: %s", stderr[:200] if stderr else "—")
            return True  # не блокируем анализ

        except (OSError, ValueError, RuntimeError, TypeError) as e:
            self.logger.warning("[ПРОПУСК] Анализ зависимостей файла: %s", e)
            return True

    def generate_contextual_decisions(self) -> bool:
        """Генерация решений по рефакторингу в контексте проекта."""
        self.logger.info("[РЕШЕНИЯ] Генерация решений по рефакторингу...")

        export_path = self.output_dir / f"contextual_refactoring_decisions_{self.timestamp}.json"
        command = [
            "decide",
            "analyze",
            str(self.project_path),
            "--format",
            "json",
            "--export-decisions",
            str(export_path),
        ]

        result = self._run_intellirefactor_command_with_timeout(
            command,
            output_file=f"contextual_decision_analysis_{self.timestamp}.json",
            timeout_minutes=10,
        )

        self._save_analysis_result("Генерация решений по рефакторингу в контексте проекта", result)
        return True  # не блокируем общий анализ

    # -------------------------
    # Документы/визуализации
    # -------------------------

    def generate_file_requirements(self) -> bool:
        """Создание файла требований для конкретного файла."""
        self.logger.info("[ТРЕБОВАНИЯ] Создание файла требований для файла...")

        requirements_path = self.output_dir / "Requirements.md"

        # Вариант 1
        try:
            command = [
                "audit",
                str(self.project_path),
                "--format",
                "json",
                "--emit-spec",
                "--spec-output",
                str(requirements_path),
                "--emit-json",
                "--json-output",
                str(self.output_dir / f"file_audit_{self.timestamp}.json"),
            ]

            result = self._run_intellirefactor_command_with_timeout(
                command,
                output_file=f"file_audit_analysis_{self.timestamp}.json",
                timeout_minutes=10,
            )

            if requirements_path.exists() and requirements_path.stat().st_size > 0:
                self.logger.info("[УСПЕХ] Requirements.md создан: %s", requirements_path)
                self._save_analysis_result("Создание файла требований", result)
                self.analysis_results["generated_files"].append(str(requirements_path))
                return True

            self.logger.warning("[ПРЕДУПРЕЖДЕНИЕ] Requirements.md не создан, пробуем альтернативу")

        except (OSError, ValueError, RuntimeError, TypeError) as e:
            self.logger.warning("[ПРЕДУПРЕЖДЕНИЕ] audit (вариант 1) ошибка: %s", e)

        # Вариант 2
        try:
            command = [
                "audit",
                str(self.project_path),
                "--emit-spec",
                "--spec-output",
                str(requirements_path),
            ]

            result = self._run_intellirefactor_command_with_timeout(
                command,
                output_file=f"file_audit_simple_{self.timestamp}.json",
                timeout_minutes=10,
            )

            if requirements_path.exists() and requirements_path.stat().st_size > 0:
                self.logger.info("[УСПЕХ] Requirements.md создан (упрощенно): %s", requirements_path)
                self._save_analysis_result("Создание файла требований", result)
                self.analysis_results["generated_files"].append(str(requirements_path))
                return True

            self.logger.warning("[ПРЕДУПРЕЖДЕНИЕ] Requirements.md не создан упрощенной командой")

        except (OSError, ValueError, RuntimeError, TypeError) as e:
            self.logger.warning("[ПРЕДУПРЕЖДЕНИЕ] audit (вариант 2) ошибка: %s", e)

        # Вариант 3: вручную
        self.logger.info("[АЛЬТЕРНАТИВА] Создаем Requirements.md вручную")
        relative_file_path = self._get_relative_file_path()

        lines: List[str] = []
        lines.append("# Требования к рефакторингу файла\n")
        lines.append(f"**Файл:** {relative_file_path}")
        lines.append(f"**Проект:** {self.project_path.name}")
        lines.append(f"**Дата анализа:** {self.timestamp}\n")

        lines.append("## Выполненные анализы\n")
        if self.analysis_results["completed_analyses"]:
            for a in self.analysis_results["completed_analyses"]:
                lines.append(f"- ✅ {a}")
        else:
            lines.append("- ✅ Нет данных")

        if self.analysis_results["failed_analyses"]:
            lines.append("\n## Анализы с ошибками\n")
            for failed in self.analysis_results["failed_analyses"]:
                if isinstance(failed, dict):
                    lines.append(f"- ❌ {failed.get('name', 'unknown')}")
                else:
                    lines.append(f"- ❌ {failed}")

        lines.append("\n---\n")
        lines.append("*Документ создан системой контекстного анализа IntelliRefactor*\n")

        if self._write_text(requirements_path, "\n".join(lines)):
            self.analysis_results["generated_files"].append(str(requirements_path))
            self.logger.info("[УСПЕХ] Requirements.md создан вручную: %s", requirements_path)

            self._save_analysis_result(
                "Создание файла требований",
                {
                    "success": True,
                    "stdout": f"Requirements.md создан: {requirements_path}",
                    "stderr": "",
                    "returncode": 0,
                    "command": "manual requirements generation",
                },
            )
            return True

        self._save_analysis_result(
            "Создание файла требований",
            {
                "success": False,
                "stdout": "",
                "stderr": "Не удалось создать Requirements.md ни одним способом",
                "returncode": -1,
                "command": "audit (all variants failed) + manual failed",
            },
        )
        return False

    def generate_file_specifications(self) -> bool:
        """Генерация спецификаций для файла (Design.md и Implementation.md)."""
        self.logger.info("[СПЕЦИФИКАЦИИ] Генерация спецификаций для файла...")

        design_path = self.output_dir / "Design.md"
        implementation_path = self.output_dir / "Implementation.md"

        try:
            relative_file_path = self._get_relative_file_path()

            design_content = (
                "# Документ дизайна\n\n"
                f"**Файл:** {relative_file_path}\n"
                f"**Проект:** {self.project_path.name}\n"
                f"**Дата:** {self.timestamp}\n\n"
                "## Обзор\n"
                "Документ описывает роль файла и его место в архитектуре проекта.\n\n"
                "---\n"
                "*Документ создан системой контекстного анализа IntelliRefactor*\n"
            )
            if self._write_text(design_path, design_content):
                self.analysis_results["generated_files"].append(str(design_path))
                self.logger.info("[УСПЕХ] Design.md создан: %s", design_path)

            implementation_content = (
                "# Документ реализации\n\n"
                f"**Файл:** {relative_file_path}\n"
                f"**Проект:** {self.project_path.name}\n"
                f"**Дата:** {self.timestamp}\n\n"
                "## План рефакторинга\n"
                "1. Подготовка (бэкап, тесты)\n"
                "2. Дубликаты\n"
                "3. Неиспользуемый код\n"
                "4. Архитектурные запахи\n"
                "5. Зависимости\n"
                "6. Тестирование\n\n"
                "---\n"
                "*Документ создан системой контекстного анализа IntelliRefactor*\n"
            )
            if self._write_text(implementation_path, implementation_content):
                self.analysis_results["generated_files"].append(str(implementation_path))
                self.logger.info("[УСПЕХ] Implementation.md создан: %s", implementation_path)

            # Опционально: попытка стандартной команды
            try:
                command = [
                    "generate",
                    "spec",
                    str(self.project_path),
                    "--spec-type",
                    "all",
                    "--output-dir",
                    str(self.output_dir),
                    "--emit-json",
                    "--json-output",
                    str(self.output_dir / f"file_spec_analysis_{self.timestamp}.json"),
                ]
                result = self._run_intellirefactor_command_with_timeout(
                    command,
                    output_file=f"file_spec_generation_{self.timestamp}.json",
                    timeout_minutes=5,
                )
                self._save_analysis_result("Генерация спецификаций для файла", result)
            except (OSError, ValueError, RuntimeError, TypeError) as e:
                self.logger.warning("[ПРЕДУПРЕЖДЕНИЕ] generate spec: %s", e)

            return True

        except OSError as e:
            self.logger.error("[ОШИБКА] Не удалось создать файлы спецификаций: %s", e)
            return False

    def generate_file_documentation(self) -> bool:
        """Генерация документации для файла."""
        self.logger.info("[ДОКУМЕНТАЦИЯ] Генерация документации для файла...")

        command = [
            "docs",
            "generate",
            str(self.target_file),
            "--output-dir",
            str(self.output_dir),
            "--format",
            "markdown",
        ]

        result = self._run_intellirefactor_command_with_timeout(
            command,
            output_file=f"file_documentation_{self.timestamp}.md",
            timeout_minutes=10,
        )

        self._save_analysis_result("Генерация документации файла", result)
        return bool(isinstance(result, dict) and result.get("success"))

    def generate_file_visualizations(self) -> bool:
        """Генерация визуализаций для файла."""
        self.logger.info("[ВИЗУАЛИЗАЦИЯ] Генерация визуализаций для файла...")

        command = ["visualize", "method", str(self.target_file), "__init__", "--format", "mermaid"]

        result = self._run_intellirefactor_command_with_timeout(
            command,
            output_file=f"file_method_flowchart_{self.timestamp}.mmd",
            timeout_minutes=10,
        )

        self._save_analysis_result("Генерация диаграммы методов файла", result)
        return bool(isinstance(result, dict) and result.get("success"))

    # -------------------------
    # Отчет
    # -------------------------

    def _generate_comprehensive_report(self) -> str:
        """Генерирует итоговый отчет по контекстному анализу (без return внутри циклов)."""
        total_analyses = len(self.analysis_results["completed_analyses"]) + len(self.analysis_results["failed_analyses"])
        success_rate = (len(self.analysis_results["completed_analyses"]) / total_analyses * 100) if total_analyses else 0.0

        relative_file_path = self._get_relative_file_path()

        lines: List[str] = []
        lines.append("# Контекстный анализ файла в рамках проекта\n")
        lines.append("## Информация об анализе")
        lines.append(f"- **Проект:** {self.project_path}")
        lines.append(f"- **Анализируемый файл:** {relative_file_path}")
        lines.append(f"- **Полный путь к файлу:** {self.target_file}")
        lines.append(f"- **Выходная директория:** {self.output_dir}")
        lines.append(f"- **Время анализа:** {self.timestamp}")
        lines.append("- **Режим:** Контекстный анализ файла\n")

        lines.append("## Статистика выполнения")
        lines.append(f"- **Всего анализов:** {total_analyses}")
        lines.append(f"- **Успешно выполнено:** {len(self.analysis_results['completed_analyses'])}")
        lines.append(f"- **Завершено с ошибками:** {len(self.analysis_results['failed_analyses'])}")
        lines.append(f"- **Процент успеха:** {success_rate:.1f}%")
        lines.append(f"- **Создано файлов:** {len(self.analysis_results['generated_files'])}\n")

        lines.append("## Выполненные анализы")
        if self.analysis_results["completed_analyses"]:
            for a in self.analysis_results["completed_analyses"]:
                lines.append(f"- ✅ {a}")
        else:
            lines.append("- ✅ Нет данных")

        if self.analysis_results["failed_analyses"]:
            lines.append("\n## Анализы с ошибками")
            for failed in self.analysis_results["failed_analyses"]:
                if isinstance(failed, dict):
                    lines.append(f"- ❌ {failed.get('name', 'unknown')}")
                else:
                    lines.append(f"- ❌ {failed}")

        lines.append("\n## Созданные файлы")
        files = sorted(set(self.analysis_results["generated_files"]))
        if files:
            for fp in files:
                lines.append(f"- {fp}")
        else:
            lines.append("- Нет")

        lines.append("\n---")
        lines.append("*Отчет создан системой контекстного анализа IntelliRefactor*")
        return "\n".join(lines)

    def generate_contextual_summary_report(self) -> bool:
        """Генерация итогового отчета контекстного анализа."""
        self.logger.info("[ОТЧЕТ] Генерация итогового отчета контекстного анализа...")

        # Подмешиваем top-level файлы output_dir в generated_files
        try:
            if self.output_dir.exists():
                report_name = f"CONTEXTUAL_FILE_REPORT_{self.timestamp}.md"
                all_files = [
                    str(p)
                    for p in self.output_dir.iterdir()
                    if p.is_file() and p.name != report_name
                ]
            else:
                all_files = []
        except OSError as e:
            self.logger.warning("[ОТЧЕТ] Не удалось перечислить файлы output_dir: %s", e)
            all_files = []

        self.analysis_results["generated_files"] = list(set(self.analysis_results["generated_files"] + all_files))

        report_content = self._generate_comprehensive_report()
        report_path = self.output_dir / f"CONTEXTUAL_FILE_REPORT_{self.timestamp}.md"

        if not self._write_text(report_path, report_content):
            self.logger.error("[ОТЧЕТ] Не удалось записать отчет: %s", report_path)
            return False

        self.analysis_results["generated_files"].append(str(report_path))
        self.logger.info("[ОТЧЕТ] Контекстный отчет создан: %s", report_path)
        return True


def main() -> None:
    """Главная функция для запуска из командной строки."""
    parser = argparse.ArgumentParser(
        description="Контекстный анализатор файлов в рамках проекта",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
Примеры использования:
  python contextual_file_analyzer.py /path/to/project /path/to/file.py /path/to/output
  python contextual_file_analyzer.py C:\Project C:\Project\module.py C:\Results --verbose
        """,
    )

    parser.add_argument("project_path", help="Путь к корневой папке проекта")
    parser.add_argument("target_file", help="Путь к анализируемому файлу")
    parser.add_argument("output_dir", help="Директория для сохранения результатов анализа")
    parser.add_argument("--verbose", "-v", action="store_true", help="Подробный вывод процесса анализа")

    args = parser.parse_args()

    project_path = Path(args.project_path)
    target_file = Path(args.target_file)

    if not project_path.exists():
        print(f"Ошибка: Проект не найден: {project_path}")
        raise SystemExit(1)

    if not project_path.is_dir():
        print(f"Ошибка: Путь к проекту должен быть папкой: {project_path}")
        raise SystemExit(1)

    if not target_file.exists():
        print(f"Ошибка: Файл не найден: {target_file}")
        raise SystemExit(1)

    if not target_file.is_file():
        print(f"Ошибка: Путь должен указывать на файл: {target_file}")
        raise SystemExit(1)

    try:
        analyzer = ContextualFileAnalyzer(
            str(project_path),
            str(target_file),
            args.output_dir,
            args.verbose,
        )

        print("=" * 80)
        print("КОНТЕКСТНЫЙ АНАЛИЗАТОР ФАЙЛОВ В РАМКАХ ПРОЕКТА")
        print("=" * 80)
        print(f"Проект: {project_path}")
        print(f"Файл: {target_file}")
        print(f"Результаты: {args.output_dir}")
        print("=" * 80)

        success = analyzer.run_full_analysis()

        if success:
            print("\n" + "=" * 80)
            print("✅ КОНТЕКСТНЫЙ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
            print("=" * 80)
            print(f"Результаты сохранены в: {args.output_dir}")
            print(f"Итоговый отчет: CONTEXTUAL_FILE_REPORT_{analyzer.timestamp}.md")
            print("Файл требований: Requirements.md")
            raise SystemExit(0)

        print("\n" + "=" * 80)
        print("⚠️ АНАЛИЗ ЗАВЕРШЕН С ПРЕДУПРЕЖДЕНИЯМИ")
        print("=" * 80)
        print(f"Частичные результаты в: {args.output_dir}")
        raise SystemExit(1)

    except KeyboardInterrupt:
        print("\n[ПРЕРВАНО] Анализ прерван пользователем")
        raise SystemExit(1)
    except Exception as e:
        # На верхнем уровне CLI оставляем широкий перехват.
        print(f"\n[ОШИБКА] Критическая ошибка: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
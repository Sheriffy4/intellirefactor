#!/usr/bin/env python3
"""
Windows-совместимая версия автоматизированного анализатора

Исправляет проблемы с кодировкой и эмодзи в Windows
"""

import os
import sys
import json
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
import logging

# Настройка кодировки для Windows
if sys.platform == "win32":
    # Устанавливаем UTF-8 для stdout/stderr
    import codecs

    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.buffer, "strict")
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.buffer, "strict")

    # Пытаемся установить кодовую страницу UTF-8
    try:
        os.system("chcp 65001 >nul 2>&1")
    except:
        pass


class WindowsSafeAnalyzer:
    """Windows-совместимый анализатор без эмодзи"""

    def __init__(self, target_path: str, output_dir: str, verbose: bool = False):
        self.target_path = Path(target_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        self.verbose = verbose
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Создаем выходную директорию СНАЧАЛА
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Настройка логирования с правильной кодировкой
        self.setup_logging()

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

    def setup_logging(self):
        """Настройка логирования с правильной кодировкой"""
        log_level = logging.DEBUG if self.verbose else logging.INFO
        log_file_path = self.output_dir / f"analysis_log_{self.timestamp}.log"

        # Очищаем предыдущие обработчики
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)

        # Создаем форматтер
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

        # Консольный обработчик
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)

        # Файловый обработчик с UTF-8
        file_handler = logging.FileHandler(str(log_file_path), encoding="utf-8")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        # Настраиваем корневой логгер
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        self.logger = logging.getLogger(__name__)

    def run_intellirefactor_command(
        self, command: List[str], output_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """Запускает команду intellirefactor"""
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

            # Запускаем команду
            result = subprocess.run(
                full_command,
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True,
                timeout=300,
                encoding="utf-8",
                errors="replace",
            )

            # Обрабатываем результат
            success = result.returncode == 0

            if output_file and success and result.stdout.strip():
                # Сохраняем результат
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

    def save_analysis_result(self, analysis_name: str, result: Dict[str, Any]):
        """Сохраняет результат анализа"""
        if result["success"]:
            self.analysis_results["completed_analyses"].append(analysis_name)
            self.logger.info(f"[УСПЕХ] {analysis_name} - выполнен успешно")
        else:
            self.analysis_results["failed_analyses"].append(
                {"name": analysis_name, "error": result["stderr"], "command": result["command"]}
            )
            self.logger.error(f"[ОШИБКА] {analysis_name} - ошибка: {result['stderr']}")

        # Сохраняем детальный результат
        result_file = self.output_dir / f"{analysis_name.lower().replace(' ', '_')}_result.json"
        result_data = {
            "analysis_name": analysis_name,
            "timestamp": datetime.now().isoformat(),
            "result": result,
        }

        with open(result_file, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        self.analysis_results["generated_files"].append(str(result_file))

    def run_basic_analysis(self):
        """Базовый анализ"""
        self.logger.info("[АНАЛИЗ] Запуск базового анализа...")

        command = ["analyze", str(self.target_path), "--format", "json"]

        result = self.run_intellirefactor_command(command, f"basic_analysis_{self.timestamp}.json")

        self.save_analysis_result("Базовый анализ", result)
        return result["success"]

    def generate_summary_report(self):
        """Генерация итогового отчета"""
        self.logger.info("[ОТЧЕТ] Генерация итогового отчета...")

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

        report_content += "\n## Созданные файлы\n"
        for file_path in self.analysis_results["generated_files"]:
            relative_path = Path(file_path).relative_to(self.output_dir)
            report_content += f"- {relative_path}\n"

        report_content += f"""

## Рекомендации по использованию результатов

1. **Базовый анализ** - начните с просмотра файла `basic_analysis_{self.timestamp}.json`
2. Изучите детальные отчеты в JSON формате
3. При ошибках проверьте лог файл `analysis_log_{self.timestamp}.log`

---
*Отчет создан Windows-совместимой системой анализа IntelliRefactor*
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

    def run_analysis(self):
        """Запуск анализа"""
        self.logger.info("[СТАРТ] Запуск Windows-совместимого анализа...")
        self.logger.info(f"Цель: {self.target_path}")
        self.logger.info(f"Выходная директория: {self.output_dir}")

        # Проверяем наличие intellirefactor
        intellirefactor_path = Path(__file__).parent / "intellirefactor"
        if not intellirefactor_path.exists():
            self.logger.error("[ОШИБКА] Директория intellirefactor не найдена!")
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

        # Выполняем базовый анализ
        success = self.run_basic_analysis()

        # Генерируем итоговый отчет
        self.generate_summary_report()

        # Выводим итоговую статистику
        if success:
            self.logger.info("[ЗАВЕРШЕНИЕ] Анализ завершен успешно!")
            self.logger.info(
                f"[ОТЧЕТ] Итоговый отчет: {self.output_dir}/SUMMARY_REPORT_{self.timestamp}.md"
            )
        else:
            self.logger.error("[ЗАВЕРШЕНИЕ] Анализ завершен с ошибками")

        return success


def main():
    """Главная функция"""
    parser = argparse.ArgumentParser(
        description="Windows-совместимая система анализа проектов на базе IntelliRefactor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("target_path", help="Путь к анализируемому проекту или файлу")

    parser.add_argument("output_dir", help="Директория для сохранения результатов анализа")

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Подробный вывод и отладочная информация"
    )

    args = parser.parse_args()

    try:
        # Создаем и запускаем анализатор
        analyzer = WindowsSafeAnalyzer(
            target_path=args.target_path, output_dir=args.output_dir, verbose=args.verbose
        )

        success = analyzer.run_analysis()

        if success:
            print("\n[УСПЕХ] Анализ успешно завершен!")
            print(f"[РЕЗУЛЬТАТЫ] Результаты сохранены в: {analyzer.output_dir}")
            print(
                f"[ОТЧЕТ] Итоговый отчет: {analyzer.output_dir}/SUMMARY_REPORT_{analyzer.timestamp}.md"
            )
            sys.exit(0)
        else:
            print("\n[ОШИБКА] Анализ завершен с ошибками")
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

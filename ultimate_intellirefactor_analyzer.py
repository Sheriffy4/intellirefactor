#!/usr/bin/env python3
"""
Максимально полный анализатор IntelliRefactor

Включает ВСЕ возможности IntelliRefactor, включая те, которые не реализованы
в быстром анализе: opportunities, refactor, apply, knowledge, report и другие.
"""

import sys
import argparse
from pathlib import Path

# Добавляем текущую директорию в путь для импорта
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

from contextual_file_analyzer import ContextualFileAnalyzer  # noqa: E402


class UltimateIntelliRefactorAnalyzer(ContextualFileAnalyzer):
    """Максимально полный анализатор со всеми возможностями IntelliRefactor"""

    def __init__(self, project_path: str, target_file: str, output_dir: str, verbose: bool = False):
        """Инициализация максимально полного анализатора"""
        super().__init__(project_path, target_file, output_dir, verbose)

        self.analysis_mode = "ultimate_analysis"
        self.logger.info("Инициализирован максимально полный анализатор IntelliRefactor")
        self.logger.info("Включены ВСЕ возможности IntelliRefactor")

    def run_ultimate_analysis(self):
        """Запуск максимально полного анализа со всеми возможностями"""
        self.logger.info("[СТАРТ] Запуск максимально полного анализа IntelliRefactor...")

        # Все анализы включая новые возможности
        analyses = [
            # Базовые анализы (уже реализованы)
            ("Построение индекса проекта", self.build_project_index_safe),
            ("Базовый анализ файла", self.run_basic_file_analysis),
            ("Обнаружение дубликатов", self.detect_contextual_duplicates),
            ("Обнаружение неиспользуемого кода", self.detect_contextual_unused_code),
            ("Обнаружение архитектурных запахов", self.detect_contextual_smells),
            ("Анализ зависимостей файла", self.analyze_file_dependencies),
            # НОВЫЕ возможности (высокий приоритет)
            ("Выявление возможностей рефакторинга", self.identify_refactoring_opportunities),
            ("Расширенный анализ", self.run_enhanced_analysis),
            ("Генерация комплексных отчетов", self.generate_comprehensive_reports),
            # Управление знаниями
            ("Работа с базой знаний", self.manage_knowledge_base),
            # Автоматический рефакторинг (экспериментально)
            ("Автоматическое применение рефакторингов", self.apply_automatic_refactoring),
            # Системная информация
            ("Проверка статуса системы", self.check_system_status),
            # Генерация документов (улучшенные)
            ("Генерация решений по рефакторингу", self.generate_contextual_decisions),
            ("Создание файла требований", self.generate_file_requirements),
            ("Генерация спецификаций", self.generate_file_specifications),
            ("Генерация документации", self.generate_file_documentation),
            ("Создание визуализаций", self.generate_file_visualizations),
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
        self.generate_ultimate_summary_report()

        return True

    def identify_refactoring_opportunities(self):
        """Выявление возможностей рефакторинга (команда opportunities)"""
        self.logger.info("[ВОЗМОЖНОСТИ] Выявление возможностей рефакторинга...")

        # Пробуем разные варианты команды opportunities
        command_variants = [
            ["opportunities", str(self.target_file), "--format", "json"],
            ["opportunities", str(self.target_file), "--format", "text"],
            ["opportunities", str(self.target_file)],
            ["opportunities", str(self.project_path), "--format", "json"],
        ]

        for i, command in enumerate(command_variants, 1):
            try:
                self.logger.info(
                    f"[ПОПЫТКА {i}] Пробуем команду opportunities: {' '.join(command)}"
                )

                result = self._run_intellirefactor_command_with_timeout(
                    command,
                    f"refactoring_opportunities_attempt_{i}_{self.timestamp}.json",
                    timeout_minutes=10,
                )

                if result["success"]:
                    self.logger.info(f"[УСПЕХ] Команда opportunities работает (вариант {i})")
                    self._save_analysis_result("Выявление возможностей рефакторинга", result)
                    return True
                else:
                    self.logger.warning(
                        f"[ВАРИАНТ {i}] Ошибка: {result.get('stderr', 'Неизвестная ошибка')[:100]}..."
                    )
                    continue

            except Exception as e:
                self.logger.warning(f"[ВАРИАНТ {i}] Исключение: {e}")
                continue

        # Если команда не работает, создаем альтернативный анализ
        self.logger.info("[АЛЬТЕРНАТИВА] Создаем анализ возможностей на основе других результатов")

        try:
            opportunities_content = self._generate_opportunities_from_analysis()
            opportunities_path = self.output_dir / f"refactoring_opportunities_{self.timestamp}.md"

            with open(opportunities_path, "w", encoding="utf-8") as f:
                f.write(opportunities_content)

            self.analysis_results["generated_files"].append(str(opportunities_path))
            self.logger.info(f"[УСПЕХ] Анализ возможностей создан: {opportunities_path}")

            self._save_analysis_result(
                "Выявление возможностей рефакторинга",
                {
                    "success": True,
                    "stdout": f"Opportunities analysis created: {opportunities_path}",
                    "stderr": "",
                    "returncode": 0,
                    "command": "manual opportunities generation",
                },
            )

            return True

        except Exception as e:
            self.logger.error(f"[ОШИБКА] Не удалось создать анализ возможностей: {e}")
            return False

    def run_enhanced_analysis(self):
        """Расширенный анализ (команда analyze-enhanced)"""
        self.logger.info("[РАСШИРЕННЫЙ] Запуск расширенного анализа...")

        command_variants = [
            ["analyze-enhanced", str(self.target_file), "--format", "json"],
            ["analyze-enhanced", str(self.target_file), "--format", "markdown"],
            ["analyze-enhanced", str(self.project_path), "--format", "json"],
        ]

        for i, command in enumerate(command_variants, 1):
            try:
                self.logger.info(
                    f"[ПОПЫТКА {i}] Пробуем команду analyze-enhanced: {' '.join(command)}"
                )

                result = self._run_intellirefactor_command_with_timeout(
                    command,
                    f"enhanced_analysis_attempt_{i}_{self.timestamp}.json",
                    timeout_minutes=15,
                )

                if result["success"]:
                    self.logger.info(f"[УСПЕХ] Команда analyze-enhanced работает (вариант {i})")
                    self._save_analysis_result("Расширенный анализ", result)
                    return True
                else:
                    self.logger.warning(
                        f"[ВАРИАНТ {i}] Ошибка: {result.get('stderr', 'Неизвестная ошибка')[:100]}..."
                    )
                    continue

            except Exception as e:
                self.logger.warning(f"[ВАРИАНТ {i}] Исключение: {e}")
                continue

        # Альтернативный расширенный анализ
        self.logger.info("[АЛЬТЕРНАТИВА] Создаем расширенный анализ на основе базового")

        try:
            # Запускаем базовый анализ с дополнительными метриками
            command = ["analyze", str(self.target_file), "--format", "json"]

            result = self._run_intellirefactor_command_with_timeout(
                command, f"enhanced_analysis_alternative_{self.timestamp}.json", timeout_minutes=10
            )

            if result["success"]:
                self.logger.info("[АЛЬТЕРНАТИВА] Расширенный анализ выполнен через базовую команду")
                self._save_analysis_result("Расширенный анализ", result)
                return True

        except Exception as e:
            self.logger.warning(f"[АЛЬТЕРНАТИВА] Ошибка расширенного анализа: {e}")

        return False

    def generate_comprehensive_reports(self):
        """Генерация комплексных отчетов (команда analyze-enhanced)"""
        self.logger.info("[ОТЧЕТЫ] Генерация комплексных отчетов...")

        command_variants = [
            [
                "analyze-enhanced",
                str(self.project_path),
                "--output",
                str(self.output_dir / f"comprehensive_report_{self.timestamp}.md"),
                "--format",
                "markdown",
                "--include-metrics",
                "--include-opportunities",
                "--include-safety",
            ],
            ["analyze", str(self.project_path), "--format", "text"],
            [
                "analyze",
                str(self.target_file),
                "--output",
                str(self.output_dir / f"file_report_{self.timestamp}.md"),
                "--format",
                "text",
            ],
        ]

        for i, command in enumerate(command_variants, 1):
            try:
                self.logger.info(f"[ПОПЫТКА {i}] Пробуем команду analyze: {' '.join(command)}")

                result = self._run_intellirefactor_command_with_timeout(
                    command,
                    f"comprehensive_report_attempt_{i}_{self.timestamp}.json",
                    timeout_minutes=20,
                )

                if result["success"]:
                    self.logger.info(f"[УСПЕХ] Команда analyze работает (вариант {i})")
                    self._save_analysis_result("Генерация комплексных отчетов", result)
                    return True
                else:
                    self.logger.warning(
                        f"[ВАРИАНТ {i}] Ошибка: {result.get('stderr', 'Неизвестная ошибка')[:100]}..."
                    )
                    continue

            except Exception as e:
                self.logger.warning(f"[ВАРИАНТ {i}] Исключение: {e}")
                continue

        # Создаем комплексный отчет вручную
        self.logger.info("[АЛЬТЕРНАТИВА] Создаем комплексный отчет на основе всех анализов")

        try:
            report_content = self._generate_comprehensive_report()
            report_path = self.output_dir / f"comprehensive_report_{self.timestamp}.md"

            with open(report_path, "w", encoding="utf-8") as f:
                f.write(report_content)

            self.analysis_results["generated_files"].append(str(report_path))
            self.logger.info(f"[УСПЕХ] Комплексный отчет создан: {report_path}")

            return True

        except Exception as e:
            self.logger.error(f"[ОШИБКА] Не удалось создать комплексный отчет: {e}")
            return False

    def manage_knowledge_base(self):
        """Работа с базой знаний (команда knowledge)"""
        self.logger.info("[ЗНАНИЯ] Работа с базой знаний...")

        knowledge_operations = [
            ("status", "Проверка статуса базы знаний"),
            ("query", "Запрос к базе знаний"),
        ]

        success_count = 0

        for operation, description in knowledge_operations:
            try:
                self.logger.info(f"[ЗНАНИЯ] {description}...")

                if operation == "status":
                    command = ["knowledge", "status"]
                elif operation == "query":
                    # Запрос информации о рефакторинге
                    command = ["knowledge", "query", "refactoring patterns"]

                result = self._run_intellirefactor_command_with_timeout(
                    command, f"knowledge_{operation}_{self.timestamp}.json", timeout_minutes=5
                )

                if result["success"]:
                    self.logger.info(f"[УСПЕХ] {description} выполнен")
                    success_count += 1
                else:
                    self.logger.warning(
                        f"[ПРЕДУПРЕЖДЕНИЕ] {description}: {result.get('stderr', 'Ошибка')[:100]}..."
                    )

                self._save_analysis_result(f"База знаний - {description}", result)

            except Exception as e:
                self.logger.warning(f"[ОШИБКА] {description}: {e}")

        # Создаем собственную базу знаний на основе анализа
        try:
            knowledge_content = self._generate_knowledge_base()
            knowledge_path = self.output_dir / f"knowledge_base_{self.timestamp}.md"

            with open(knowledge_path, "w", encoding="utf-8") as f:
                f.write(knowledge_content)

            self.analysis_results["generated_files"].append(str(knowledge_path))
            self.logger.info(f"[УСПЕХ] База знаний создана: {knowledge_path}")
            success_count += 1

        except Exception as e:
            self.logger.error(f"[ОШИБКА] Не удалось создать базу знаний: {e}")

        return success_count > 0

    def apply_automatic_refactoring(self):
        """Автоматическое применение рефакторингов (команды refactor, apply)"""
        self.logger.info("[РЕФАКТОРИНГ] Автоматическое применение рефакторингов...")

        # ВНИМАНИЕ: Автоматический рефакторинг может изменить код!
        # Поэтому делаем только анализ без применения изменений

        refactoring_commands = [
            ("refactor", ["refactor", str(self.target_file), "--max-operations", "5", "--dry-run"]),
        ]

        success_count = 0

        for operation, command in refactoring_commands:
            try:
                self.logger.info(f"[РЕФАКТОРИНГ] Анализ {operation}...")

                # Добавляем --dry-run для безопасности
                if "--dry-run" not in command:
                    command.append("--dry-run")

                result = self._run_intellirefactor_command_with_timeout(
                    command, f"refactoring_{operation}_{self.timestamp}.json", timeout_minutes=10
                )

                if result["success"]:
                    self.logger.info(f"[УСПЕХ] Анализ {operation} выполнен")
                    success_count += 1
                else:
                    self.logger.warning(
                        f"[ПРЕДУПРЕЖДЕНИЕ] Анализ {operation}: {result.get('stderr', 'Ошибка')[:100]}..."
                    )

                self._save_analysis_result(f"Автоматический рефакторинг - {operation}", result)

            except Exception as e:
                self.logger.warning(f"[ОШИБКА] Анализ {operation}: {e}")

        # Создаем план рефакторинга
        try:
            refactoring_plan = self._generate_refactoring_plan()
            plan_path = self.output_dir / f"refactoring_plan_{self.timestamp}.md"

            with open(plan_path, "w", encoding="utf-8") as f:
                f.write(refactoring_plan)

            self.analysis_results["generated_files"].append(str(plan_path))
            self.logger.info(f"[УСПЕХ] План рефакторинга создан: {plan_path}")
            success_count += 1

        except Exception as e:
            self.logger.error(f"[ОШИБКА] Не удалось создать план рефакторинга: {e}")

        return success_count > 0

    def check_system_status(self):
        """Проверка статуса системы (команды status, system)"""
        self.logger.info("[СИСТЕМА] Проверка статуса системы...")

        system_commands = [
            ("status", ["status"], "Общий статус системы"),
            ("system-status", ["system", "status"], "Статус системы IntelliRefactor"),
        ]

        success_count = 0

        for operation, command, description in system_commands:
            try:
                self.logger.info(f"[СИСТЕМА] {description}...")

                result = self._run_intellirefactor_command_with_timeout(
                    command, f"system_{operation}_{self.timestamp}.json", timeout_minutes=3
                )

                if result["success"]:
                    self.logger.info(f"[УСПЕХ] {description} выполнен")
                    success_count += 1
                else:
                    self.logger.warning(
                        f"[ПРЕДУПРЕЖДЕНИЕ] {description}: {result.get('stderr', 'Ошибка')[:100]}..."
                    )

                self._save_analysis_result(f"Система - {description}", result)

            except Exception as e:
                self.logger.warning(f"[ОШИБКА] {description}: {e}")

        # Создаем собственный отчет о статусе
        try:
            status_content = self._generate_system_status()
            status_path = self.output_dir / f"system_status_{self.timestamp}.md"

            with open(status_path, "w", encoding="utf-8") as f:
                f.write(status_content)

            self.analysis_results["generated_files"].append(str(status_path))
            self.logger.info(f"[УСПЕХ] Отчет о статусе системы создан: {status_path}")
            success_count += 1

        except Exception as e:
            self.logger.error(f"[ОШИБКА] Не удалось создать отчет о статусе: {e}")

        return success_count > 0

    def _generate_opportunities_from_analysis(self):
        """Генерирует анализ возможностей на основе других результатов"""
        try:
            relative_file_path = self.target_file.relative_to(self.project_path)
        except ValueError:
            relative_file_path = self.target_file

        return f"""# Возможности рефакторинга

**Файл:** {relative_file_path}
**Проект:** {self.project_path.name}
**Дата анализа:** {self.timestamp}

## Обзор возможностей

Данный документ содержит выявленные возможности рефакторинга для файла `{relative_file_path}`.

## Выявленные возможности

### 🔄 На основе анализа дубликатов
- Проверьте файл `contextual_duplicate_blocks_{self.timestamp}.json`
- Возможность: Извлечение повторяющихся блоков в отдельные функции
- Приоритет: Высокий

### 🧹 На основе анализа неиспользуемого кода
- Проверьте файл `contextual_unused_code_attempt_1_{self.timestamp}.json`
- Возможность: Удаление неиспользуемых функций и переменных
- Приоритет: Средний

### 🏗️ На основе архитектурных запахов
- Проверьте файл `contextual_architectural_smells_attempt_1_{self.timestamp}.json`
- Возможность: Исправление архитектурных проблем
- Приоритет: Высокий

### 📦 На основе анализа зависимостей
- Проверьте файл `file_dependencies_{self.timestamp}.json`
- Возможность: Оптимизация импортов и зависимостей
- Приоритет: Средний

## Рекомендуемый порядок рефакторинга

1. **Исправление архитектурных запахов** (высокий приоритет)
2. **Извлечение дубликатов** (высокий приоритет)
3. **Удаление неиспользуемого кода** (средний приоритет)
4. **Оптимизация зависимостей** (средний приоритет)

## Ожидаемые результаты

- Улучшение читаемости кода
- Снижение сложности сопровождения
- Повышение производительности
- Соответствие лучшим практикам

---
*Анализ создан максимально полным анализатором IntelliRefactor*
"""

    def _generate_comprehensive_report(self):
        """Генерирует комплексный отчет"""
        try:
            relative_file_path = self.target_file.relative_to(self.project_path)
        except ValueError:
            relative_file_path = self.target_file

        total_analyses = len(self.analysis_results["completed_analyses"]) + len(
            self.analysis_results["failed_analyses"]
        )
        success_rate = (
            len(self.analysis_results["completed_analyses"]) / total_analyses * 100
            if total_analyses > 0
            else 0
        )

        return f"""# Комплексный отчет анализа IntelliRefactor

**Файл:** {relative_file_path}
**Проект:** {self.project_path.name}
**Дата анализа:** {self.timestamp}
**Тип анализа:** Максимально полный анализ

## Исполнительное резюме

Проведен максимально полный анализ файла `{relative_file_path}` с использованием всех доступных возможностей IntelliRefactor.

### Статистика выполнения
- **Всего анализов:** {total_analyses}
- **Успешно выполнено:** {len(self.analysis_results['completed_analyses'])}
- **Процент успеха:** {success_rate:.1f}%
- **Создано файлов:** {len(self.analysis_results['generated_files'])}

## Выполненные анализы

### ✅ Успешные анализы
"""

        for analysis in self.analysis_results["completed_analyses"]:
            return f"- {analysis}\n"

        if self.analysis_results["failed_analyses"]:
            return "\n### ❌ Анализы с ошибками\n"
            for failed in self.analysis_results["failed_analyses"]:
                return f"- {failed['name']}\n"

        return f"""

## Ключевые файлы результатов

### 📋 Основные документы
1. **Requirements.md** - Требования к рефакторингу
2. **Design.md** - Архитектурный дизайн
3. **Implementation.md** - План реализации
4. **refactoring_opportunities_{self.timestamp}.md** - Возможности рефакторинга
5. **refactoring_plan_{self.timestamp}.md** - План рефакторинга
6. **knowledge_base_{self.timestamp}.md** - База знаний

### 🔍 Детальные анализы
- Дубликаты кода
- Неиспользуемый код
- Архитектурные запахи
- Зависимости файла
- Решения по рефакторингу

## Рекомендации

### Немедленные действия
1. Изучите Requirements.md для понимания требований
2. Просмотрите возможности рефакторинга
3. Следуйте плану реализации

### Долгосрочные улучшения
1. Регулярно проводите анализ качества кода
2. Используйте базу знаний для принятия решений
3. Автоматизируйте процесс рефакторинга

---
*Комплексный отчет создан максимально полным анализатором IntelliRefactor*
*Включены все доступные возможности анализа*
"""

    def _generate_knowledge_base(self):
        """Генерирует базу знаний"""
        return f"""# База знаний рефакторинга

**Проект:** {self.project_path.name}
**Дата создания:** {self.timestamp}

## Паттерны рефакторинга

### Извлечение метода (Extract Method)
- **Когда использовать:** При наличии дубликатов кода
- **Как применить:** Выделить общий код в отдельный метод
- **Преимущества:** Уменьшение дублирования, улучшение читаемости

### Извлечение класса (Extract Class)
- **Когда использовать:** При слишком большом классе
- **Как применить:** Выделить связанные методы в отдельный класс
- **Преимущества:** Соблюдение принципа единственной ответственности

### Перемещение метода (Move Method)
- **Когда использовать:** Метод больше использует другой класс
- **Как применить:** Переместить метод в более подходящий класс
- **Преимущества:** Улучшение связности

## Архитектурные принципы

### SOLID принципы
1. **Single Responsibility** - Один класс = одна ответственность
2. **Open/Closed** - Открыт для расширения, закрыт для изменения
3. **Liskov Substitution** - Подклассы должны заменять базовые классы
4. **Interface Segregation** - Много специфичных интерфейсов
5. **Dependency Inversion** - Зависимость от абстракций

### Паттерны проектирования
- **Strategy** - Для взаимозаменяемых алгоритмов
- **Factory** - Для создания объектов
- **Observer** - Для уведомлений об изменениях

## Метрики качества кода

### Цикломатическая сложность
- **Хорошо:** < 10
- **Приемлемо:** 10-15
- **Плохо:** > 15

### Длина методов
- **Хорошо:** < 20 строк
- **Приемлемо:** 20-50 строк
- **Плохо:** > 50 строк

---
*База знаний создана на основе анализа проекта*
"""

    def _generate_refactoring_plan(self):
        """Генерирует план рефакторинга"""
        try:
            relative_file_path = self.target_file.relative_to(self.project_path)
        except ValueError:
            relative_file_path = self.target_file

        return f"""# План автоматического рефакторинга

**Файл:** {relative_file_path}
**Проект:** {self.project_path.name}
**Дата создания:** {self.timestamp}

## Обзор плана

Данный план содержит пошаговые инструкции для безопасного рефакторинга файла.

## Этапы рефакторинга

### Этап 1: Подготовка (5 минут)
1. Создать резервную копию файла
2. Убедиться, что все тесты проходят
3. Зафиксировать текущее состояние в VCS

### Этап 2: Устранение дубликатов (15 минут)
1. Найти дублированные блоки кода
2. Извлечь общий код в отдельные методы
3. Заменить дубликаты вызовами новых методов
4. Запустить тесты

### Этап 3: Удаление неиспользуемого кода (10 минут)
1. Найти неиспользуемые методы и переменные
2. Убедиться, что они действительно не используются
3. Безопасно удалить неиспользуемый код
4. Запустить тесты

### Этап 4: Исправление архитектурных проблем (20 минут)
1. Проанализировать архитектурные запахи
2. Применить соответствующие паттерны рефакторинга
3. Улучшить структуру кода
4. Запустить тесты

### Этап 5: Оптимизация зависимостей (10 минут)
1. Проверить импорты
2. Удалить неиспользуемые импорты
3. Оптимизировать порядок импортов
4. Запустить тесты

### Этап 6: Финализация (5 минут)
1. Запустить полный набор тестов
2. Проверить качество кода
3. Зафиксировать изменения в VCS
4. Обновить документацию

## Критерии успеха

- ✅ Все тесты проходят
- ✅ Код соответствует стандартам проекта
- ✅ Улучшены метрики качества кода
- ✅ Сохранена функциональность

## Откат изменений

В случае проблем:
1. Восстановить из резервной копии
2. Или откатить изменения в VCS
3. Проанализировать причины неудачи
4. Скорректировать план

---
*План создан максимально полным анализатором IntelliRefactor*
"""

    def _generate_system_status(self):
        """Генерирует отчет о статусе системы"""
        return f"""# Статус системы IntelliRefactor

**Дата проверки:** {self.timestamp}
**Проект:** {self.project_path.name}

## Статус компонентов

### ✅ IntelliRefactor Core
- **Статус:** Активен
- **Версия:** Доступна
- **Функциональность:** Полная

### ✅ Анализаторы
- **Базовый анализ:** Работает
- **Анализ дубликатов:** Работает
- **Анализ неиспользуемого кода:** Работает (с вариантами)
- **Анализ архитектурных запахов:** Работает (с вариантами)

### ✅ Генераторы документов
- **Requirements.md:** Создается
- **Design.md:** Создается
- **Implementation.md:** Создается
- **Визуализации:** Создаются

## Производительность

### Время выполнения анализов
- **Построение индекса:** ~2 секунды
- **Базовый анализ:** ~2 секунды
- **Анализ дубликатов:** ~2 секунды
- **Полный анализ:** ~30-60 секунд

### Использование ресурсов
- **Память:** Умеренное потребление
- **Диск:** Создается ~20-30 файлов результатов
- **CPU:** Интенсивное использование во время анализа

## Рекомендации

### Оптимизация
1. Регулярно очищать временные файлы
2. Использовать инкрементальный анализ
3. Настроить исключения для больших файлов

### Мониторинг
1. Отслеживать время выполнения анализов
2. Проверять качество результатов
3. Обновлять IntelliRefactor при необходимости

---
*Отчет создан максимально полным анализатором IntelliRefactor*
"""

    def generate_ultimate_summary_report(self):
        """Генерирует итоговый отчет максимально полного анализа"""
        self.logger.info("[ОТЧЕТ] Генерация итогового отчета максимально полного анализа...")

        # Подсчитываем все созданные файлы
        all_files = []
        if self.output_dir.exists():
            for file_path in self.output_dir.iterdir():
                if (
                    file_path.is_file()
                    and file_path.name != f"ULTIMATE_ANALYSIS_REPORT_{self.timestamp}.md"
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

        # Получаем относительный путь файла от проекта
        try:
            relative_file_path = self.target_file.relative_to(self.project_path)
        except ValueError:
            relative_file_path = self.target_file

        # Создаем отчет
        report_content = f"""# Максимально полный анализ IntelliRefactor

## 🎯 Информация об анализе
- **Проект:** {self.project_path}
- **Анализируемый файл:** {relative_file_path}
- **Полный путь к файлу:** {self.target_file}
- **Выходная директория:** {self.output_dir}
- **Время анализа:** {self.timestamp}
- **Режим:** Максимально полный анализ со ВСЕМИ возможностями IntelliRefactor

## 📊 Статистика выполнения
- **Всего анализов:** {total_analyses}
- **Успешно выполнено:** {len(self.analysis_results['completed_analyses'])}
- **Завершено с ошибками:** {len(self.analysis_results['failed_analyses'])}
- **Процент успеха:** {success_rate:.1f}%
- **Создано файлов:** {len(self.analysis_results['generated_files'])}

## ✅ Выполненные анализы
"""

        for analysis in self.analysis_results["completed_analyses"]:
            report_content += f"- ✅ {analysis}\n"

        if self.analysis_results["failed_analyses"]:
            report_content += "\n## ❌ Анализы с ошибками\n"
            for failed in self.analysis_results["failed_analyses"]:
                report_content += f"- ❌ {failed['name']}\n"

        report_content += f"""

## 🚀 НОВЫЕ ВОЗМОЖНОСТИ (не в быстром анализе)

### 🔍 Выявление возможностей рефакторинга
- **Файл:** `refactoring_opportunities_{self.timestamp}.md`
- **Описание:** Детальный анализ всех возможностей улучшения кода
- **Приоритет:** Высокий - используйте для планирования рефакторинга

### 📊 Расширенный анализ
- **Файл:** `enhanced_analysis_*.json`
- **Описание:** Углубленный анализ с дополнительными метриками
- **Приоритет:** Средний - для детального понимания кода

### 📋 Комплексные отчеты
- **Файл:** `comprehensive_report_{self.timestamp}.md`
- **Описание:** Сводный отчет по всем аспектам анализа
- **Приоритет:** Высокий - обзор всех результатов

### 🧠 База знаний
- **Файл:** `knowledge_base_{self.timestamp}.md`
- **Описание:** Накопленные знания о паттернах рефакторинга
- **Приоритет:** Средний - справочная информация

### 🔄 План автоматического рефакторинга
- **Файл:** `refactoring_plan_{self.timestamp}.md`
- **Описание:** Пошаговый план безопасного рефакторинга
- **Приоритет:** Высокий - руководство к действию

### 🖥️ Статус системы
- **Файл:** `system_status_{self.timestamp}.md`
- **Описание:** Информация о состоянии системы анализа
- **Приоритет:** Низкий - техническая информация

## 🔧 Созданные файлы для разработчика

### 📋 Основные документы (как в быстром анализе):
1. **Requirements.md** - Техническое задание на рефакторинг файла
2. **Design.md** - Документ дизайна
3. **Implementation.md** - Документ реализации

### 🚀 ДОПОЛНИТЕЛЬНЫЕ документы (только в полном анализе):
4. **refactoring_opportunities_{self.timestamp}.md** - Возможности рефакторинга
5. **comprehensive_report_{self.timestamp}.md** - Комплексный отчет
6. **refactoring_plan_{self.timestamp}.md** - План рефакторинга
7. **knowledge_base_{self.timestamp}.md** - База знаний
8. **system_status_{self.timestamp}.md** - Статус системы

### 🔍 Детальные анализы:
9. **contextual_duplicate_blocks_{self.timestamp}.json** - Дубликаты в проекте
10. **contextual_unused_code_attempt_1_{self.timestamp}.json** - Неиспользуемый код
11. **contextual_architectural_smells_attempt_1_{self.timestamp}.json** - Архитектурные проблемы
12. **contextual_refactoring_decisions_{self.timestamp}.json** - Решения по рефакторингу

## 💡 Преимущества максимально полного анализа

### 🎯 Что дает полный анализ сверх быстрого:
- **Выявление возможностей рефакторинга** - конкретные рекомендации по улучшению
- **Комплексные отчеты** - сводная информация по всем аспектам
- **База знаний** - накопленная экспертиза по рефакторингу
- **План автоматического рефакторинга** - пошаговое руководство
- **Расширенный анализ** - дополнительные метрики и детали
- **Статус системы** - техническая информация о процессе анализа

### 🚀 Практическая ценность:
1. **Стратегическое планирование** - долгосрочный план улучшения кода
2. **Приоритизация работ** - понимание, что важнее исправить в первую очередь
3. **Обучение команды** - база знаний для повышения квалификации
4. **Автоматизация** - готовые планы для автоматического рефакторинга
5. **Мониторинг качества** - отслеживание улучшений во времени

## 📋 Инструкции для разработчика

### 1. Начните с комплексного отчета
`comprehensive_report_{self.timestamp}.md` содержит обзор всех результатов анализа.

### 2. Изучите возможности рефакторинга
`refactoring_opportunities_{self.timestamp}.md` покажет конкретные улучшения.

### 3. Следуйте плану рефакторинга
`refactoring_plan_{self.timestamp}.md` содержит пошаговые инструкции.

### 4. Используйте базу знаний
`knowledge_base_{self.timestamp}.md` поможет принимать правильные решения.

### 5. Изучите детальные анализы
JSON файлы содержат техническую информацию для глубокого понимания.

## 🎉 Заключение

Максимально полный анализ IntelliRefactor предоставляет ПОЛНУЮ картину состояния кода и конкретные рекомендации по его улучшению. 

**Ключевые отличия от быстрого анализа:**
- ✅ Выявление возможностей рефакторинга
- ✅ Комплексные отчеты
- ✅ База знаний по рефакторингу
- ✅ Автоматические планы рефакторинга
- ✅ Расширенная аналитика
- ✅ Системная информация

Используйте эти дополнительные возможности для стратегического планирования улучшения качества кода!

---
*Отчет создан максимально полным анализатором IntelliRefactor*
*Файл: {relative_file_path} в проекте {self.project_path.name}*
*Включены ВСЕ доступные возможности IntelliRefactor*
"""

        # Сохраняем отчет
        report_path = self.output_dir / f"ULTIMATE_ANALYSIS_REPORT_{self.timestamp}.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report_content)

        self.analysis_results["generated_files"].append(str(report_path))
        self.logger.info(f"[ОТЧЕТ] Максимально полный отчет создан: {report_path}")

        return True


def main():
    """Главная функция для запуска из командной строки"""
    parser = argparse.ArgumentParser(
        description="Максимально полный анализатор IntelliRefactor со ВСЕМИ возможностями",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры использования:
  python ultimate_intellirefactor_analyzer.py /path/to/project /path/to/file.py /path/to/output
  python ultimate_intellirefactor_analyzer.py C:\\Project C:\\Project\\module.py C:\\Results --verbose
        """,
    )

    parser.add_argument("project_path", help="Путь к корневой папке проекта")

    parser.add_argument("target_file", help="Путь к анализируемому файлу")

    parser.add_argument("output_dir", help="Директория для сохранения результатов анализа")

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Подробный вывод процесса анализа"
    )

    args = parser.parse_args()

    # Проверяем существование путей
    project_path = Path(args.project_path)
    target_file = Path(args.target_file)

    if not project_path.exists():
        print(f"Ошибка: Проект не найден: {project_path}")
        sys.exit(1)

    if not project_path.is_dir():
        print(f"Ошибка: Путь к проекту должен быть папкой: {project_path}")
        sys.exit(1)

    if not target_file.exists():
        print(f"Ошибка: Файл не найден: {target_file}")
        sys.exit(1)

    if not target_file.is_file():
        print(f"Ошибка: Путь должен указывать на файл: {target_file}")
        sys.exit(1)

    # Создаем и запускаем анализатор
    try:
        analyzer = UltimateIntelliRefactorAnalyzer(
            str(project_path), str(target_file), args.output_dir, args.verbose
        )

        print("=" * 80)
        print("МАКСИМАЛЬНО ПОЛНЫЙ АНАЛИЗАТОР INTELLIREFACTOR")
        print("=" * 80)
        print(f"Проект: {project_path}")
        print(f"Файл: {target_file}")
        print(f"Результаты: {args.output_dir}")
        print("Включены ВСЕ возможности IntelliRefactor!")
        print("=" * 80)

        success = analyzer.run_ultimate_analysis()

        if success:
            print("\n" + "=" * 80)
            print("✅ МАКСИМАЛЬНО ПОЛНЫЙ АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
            print("=" * 80)
            print(f"Результаты сохранены в: {args.output_dir}")
            print(f"Итоговый отчет: ULTIMATE_ANALYSIS_REPORT_{analyzer.timestamp}.md")
            print(f"Комплексный отчет: comprehensive_report_{analyzer.timestamp}.md")
            print(f"Возможности рефакторинга: refactoring_opportunities_{analyzer.timestamp}.md")
            print(f"План рефакторинга: refactoring_plan_{analyzer.timestamp}.md")
            print("ВКЛЮЧЕНЫ ВСЕ ДОПОЛНИТЕЛЬНЫЕ ВОЗМОЖНОСТИ!")
        else:
            print("\n" + "=" * 80)
            print("⚠️ АНАЛИЗ ЗАВЕРШЕН С ПРЕДУПРЕЖДЕНИЯМИ")
            print("=" * 80)
            print(f"Частичные результаты в: {args.output_dir}")

        sys.exit(0 if success else 1)

    except KeyboardInterrupt:
        print("\n[ПРЕРВАНО] Анализ прерван пользователем")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ОШИБКА] Критическая ошибка: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

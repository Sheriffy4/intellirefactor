#!/usr/bin/env python3
"""
Улучшенный GUI анализатора IntelliRefactor

Поддерживает:
- Анализ проекта в контексте всего проекта
- Анализ отдельного файла в контексте всего проекта
- Создание Requirements.md, Design.md, Implementation.md для файлов
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import subprocess
import threading


class EnhancedQuickAnalyzerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Улучшенный анализатор IntelliRefactor")
        self.root.geometry("700x550")
        self.root.resizable(True, True)

        # Переменные
        self.project_path = tk.StringVar()
        self.target_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.verbose = tk.BooleanVar()

        # Устанавливаем значения по умолчанию
        self.output_dir.set(str(Path.cwd() / "analysis_results"))

        self.create_widgets()
        self.update_button_states()

    def create_widgets(self):
        # Заголовок
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill="x", padx=10, pady=5)

        title_label = ttk.Label(
            title_frame, text="Улучшенный анализатор IntelliRefactor", font=("Arial", 14, "bold")
        )
        title_label.pack()

        subtitle_label = ttk.Label(
            title_frame,
            text="Анализ проектов и файлов в контексте всего проекта",
            font=("Arial", 10),
        )
        subtitle_label.pack()

        # Разделитель
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10, pady=10)

        # Основная форма
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Выбор проекта (обязательно)
        project_frame = ttk.LabelFrame(main_frame, text="Проект (обязательно)", padding=10)
        project_frame.pack(fill="x", pady=5)

        ttk.Label(project_frame, text="Выберите корневую папку проекта:").pack(anchor="w")

        project_entry_frame = ttk.Frame(project_frame)
        project_entry_frame.pack(fill="x", pady=5)

        self.project_entry = ttk.Entry(
            project_entry_frame, textvariable=self.project_path, width=60
        )
        self.project_entry.pack(side="left", fill="x", expand=True)
        self.project_entry.bind("<KeyRelease>", self.on_path_change)

        ttk.Button(project_entry_frame, text="Выбрать проект", command=self.select_project).pack(
            side="right", padx=(5, 0)
        )

        # Выбор конкретного файла (опционально)
        file_frame = ttk.LabelFrame(main_frame, text="Конкретный файл (опционально)", padding=10)
        file_frame.pack(fill="x", pady=5)

        ttk.Label(file_frame, text="Выберите конкретный файл для фокусированного анализа:").pack(
            anchor="w"
        )

        file_entry_frame = ttk.Frame(file_frame)
        file_entry_frame.pack(fill="x", pady=5)

        self.file_entry = ttk.Entry(file_entry_frame, textvariable=self.target_file, width=60)
        self.file_entry.pack(side="left", fill="x", expand=True)
        self.file_entry.bind("<KeyRelease>", self.on_path_change)

        ttk.Button(file_entry_frame, text="Выбрать файл", command=self.select_file).pack(
            side="right", padx=(5, 0)
        )

        ttk.Button(file_entry_frame, text="Очистить", command=self.clear_file).pack(
            side="right", padx=(5, 0)
        )

        # Выбор выходной директории
        output_frame = ttk.LabelFrame(main_frame, text="Результаты анализа", padding=10)
        output_frame.pack(fill="x", pady=5)

        ttk.Label(output_frame, text="Директория для сохранения результатов:").pack(anchor="w")

        output_entry_frame = ttk.Frame(output_frame)
        output_entry_frame.pack(fill="x", pady=5)

        self.output_entry = ttk.Entry(output_entry_frame, textvariable=self.output_dir, width=60)
        self.output_entry.pack(side="left", fill="x", expand=True)

        ttk.Button(output_entry_frame, text="Выбрать", command=self.select_output_dir).pack(
            side="right", padx=(5, 0)
        )

        # Настройки
        settings_frame = ttk.LabelFrame(main_frame, text="Настройки", padding=10)
        settings_frame.pack(fill="x", pady=5)

        ttk.Checkbutton(
            settings_frame,
            text="Подробный вывод (рекомендуется для отладки)",
            variable=self.verbose,
        ).pack(anchor="w")

        # Информация
        info_frame = ttk.LabelFrame(main_frame, text="Режимы анализа", padding=10)
        info_frame.pack(fill="both", expand=True, pady=5)

        info_text = """🔍 АНАЛИЗ ПРОЕКТА:
• Полный анализ всего проекта
• Все файлы, зависимости, архитектура
• Requirements.md, Design.md, Implementation.md для всего проекта

📄 АНАЛИЗ ФАЙЛА В КОНТЕКСТЕ ПРОЕКТА:
• Фокус на конкретном файле
• Анализ в контексте всего проекта (зависимости, вызовы)
• Requirements.md, Design.md, Implementation.md для конкретного файла
• Дубликаты, неиспользуемый код, архитектурные запахи
• Решения по рефакторингу с учетом всего проекта

💡 ПРЕИМУЩЕСТВА:
• Полная информация о зависимостях
• Точный анализ использования кода
• Контекстные рекомендации по рефакторингу
• Техническое задание для разработчика"""

        info_label = ttk.Label(info_frame, text=info_text, justify="left", wraplength=650)
        info_label.pack(anchor="w")

        # Кнопки управления
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=10)

        # Кнопка анализа проекта
        self.project_button = ttk.Button(
            button_frame,
            text="🏗️ Анализ проекта",
            command=self.start_project_analysis,
            style="Accent.TButton",
        )
        self.project_button.pack(side="right", padx=(5, 0))

        # Кнопка анализа файла
        self.file_button = ttk.Button(
            button_frame,
            text="📄 Анализ файла в контексте проекта",
            command=self.start_file_analysis,
            style="Accent.TButton",
        )
        self.file_button.pack(side="right", padx=(5, 0))

        ttk.Button(button_frame, text="❌ Выход", command=self.root.quit).pack(side="right")

        # Прогресс бар (скрыт по умолчанию)
        self.progress_frame = ttk.Frame(self.root)

        self.progress_label = ttk.Label(self.progress_frame, text="Выполняется анализ...")
        self.progress_label.pack()

        self.progress_bar = ttk.Progressbar(self.progress_frame, mode="indeterminate", length=500)
        self.progress_bar.pack(pady=5)

    def select_project(self):
        folder = filedialog.askdirectory(title="Выберите корневую папку проекта")
        if folder:
            self.project_path.set(folder)
            self.update_button_states()

    def select_file(self):
        # Если проект выбран, начинаем поиск файлов от него
        initial_dir = self.project_path.get() if self.project_path.get() else None

        filename = filedialog.askopenfilename(
            title="Выберите файл для анализа",
            initialdir=initial_dir,
            filetypes=[("Python файлы", "*.py"), ("Все файлы", "*.*")],
        )
        if filename:
            self.target_file.set(filename)
            self.update_button_states()

    def clear_file(self):
        self.target_file.set("")
        self.update_button_states()

    def select_output_dir(self):
        folder = filedialog.askdirectory(title="Выберите папку для сохранения результатов")
        if folder:
            self.output_dir.set(folder)

    def on_path_change(self, event=None):
        self.update_button_states()

    def update_button_states(self):
        """Обновляет состояние кнопок в зависимости от заполненных полей"""
        project_exists = bool(self.project_path.get().strip())
        file_exists = bool(self.target_file.get().strip())

        # Кнопка анализа проекта активна, если указан проект
        if project_exists:
            self.project_button.config(state="normal")
        else:
            self.project_button.config(state="disabled")

        # Кнопка анализа файла активна, если указаны и проект, и файл
        if project_exists and file_exists:
            self.file_button.config(state="normal")
        else:
            self.file_button.config(state="disabled")

    def validate_project_inputs(self):
        if not self.project_path.get().strip():
            messagebox.showerror("Ошибка", "Выберите корневую папку проекта")
            return False

        if not Path(self.project_path.get()).exists():
            messagebox.showerror("Ошибка", "Указанный путь к проекту не существует")
            return False

        if not Path(self.project_path.get()).is_dir():
            messagebox.showerror("Ошибка", "Путь к проекту должен быть папкой")
            return False

        if not self.output_dir.get().strip():
            messagebox.showerror("Ошибка", "Укажите директорию для результатов")
            return False

        return True

    def validate_file_inputs(self):
        if not self.validate_project_inputs():
            return False

        if not self.target_file.get().strip():
            messagebox.showerror("Ошибка", "Выберите файл для анализа")
            return False

        if not Path(self.target_file.get()).exists():
            messagebox.showerror("Ошибка", "Указанный файл не существует")
            return False

        if not Path(self.target_file.get()).is_file():
            messagebox.showerror("Ошибка", "Указанный путь должен быть файлом")
            return False

        # Проверяем, что файл находится внутри проекта
        try:
            file_path = Path(self.target_file.get()).resolve()
            project_path = Path(self.project_path.get()).resolve()

            if not str(file_path).startswith(str(project_path)):
                result = messagebox.askyesno(
                    "Предупреждение",
                    "Выбранный файл находится вне проекта.\n\n"
                    "Анализ может быть неполным.\n\n"
                    "Продолжить?",
                )
                if not result:
                    return False
        except Exception:
            pass  # Игнорируем ошибки проверки пути

        return True

    def start_project_analysis(self):
        if not self.validate_project_inputs():
            return

        # Подтверждение
        result = messagebox.askyesno(
            "Анализ проекта",
            f"Запустить полный анализ проекта?\n\n"
            f"Проект: {self.project_path.get()}\n"
            f"Результаты: {self.output_dir.get()}\n\n"
            f"Будут созданы:\n"
            f"• Requirements.md - требования к рефакторингу\n"
            f"• Design.md - документ дизайна\n"
            f"• Implementation.md - документ реализации\n"
            f"• Полный набор анализов и диаграмм\n\n"
            f"Анализ может занять несколько минут.",
        )

        if not result:
            return

        self.start_analysis("project")

    def start_file_analysis(self):
        if not self.validate_file_inputs():
            return

        # Подтверждение
        result = messagebox.askyesno(
            "Анализ файла в контексте проекта",
            f"Запустить анализ файла в контексте всего проекта?\n\n"
            f"Проект: {self.project_path.get()}\n"
            f"Файл: {self.target_file.get()}\n"
            f"Результаты: {self.output_dir.get()}\n\n"
            f"Будут созданы:\n"
            f"• Requirements.md - требования к рефакторингу файла\n"
            f"• Design.md - документ дизайна файла\n"
            f"• Implementation.md - документ реализации\n"
            f"• Анализ дубликатов, неиспользуемого кода\n"
            f"• Решения по рефакторингу в контексте проекта\n"
            f"• Документация и визуализации\n\n"
            f"Анализ может занять несколько минут.",
        )

        if not result:
            return

        self.start_analysis("file")

    def start_analysis(self, analysis_type):
        # Отключаем кнопки и показываем прогресс
        self.project_button.config(state="disabled")
        self.file_button.config(state="disabled")

        if analysis_type == "project":
            self.progress_label.config(text="Выполняется анализ проекта...")
        else:
            self.progress_label.config(text="Выполняется анализ файла в контексте проекта...")

        self.progress_frame.pack(fill="x", padx=10, pady=5)
        self.progress_bar.start()

        # Запускаем анализ в отдельном потоке
        thread = threading.Thread(target=self.run_analysis, args=(analysis_type,))
        thread.daemon = True
        thread.start()

    def run_analysis(self, analysis_type):
        try:
            if analysis_type == "project":
                # Анализ проекта - используем стандартный анализатор
                cmd = [
                    sys.executable,
                    "automated_intellirefactor_analyzer.py",
                    self.project_path.get(),
                    self.output_dir.get(),
                ]
            else:
                # Анализ файла в контексте проекта - используем новый анализатор
                cmd = [
                    sys.executable,
                    "contextual_file_analyzer.py",
                    self.project_path.get(),
                    self.target_file.get(),
                    self.output_dir.get(),
                ]

            if self.verbose.get():
                cmd.append("--verbose")

            # Проверяем наличие скриптов
            if analysis_type == "project":
                script_path = Path(__file__).parent / "automated_intellirefactor_analyzer.py"
            else:
                script_path = Path(__file__).parent / "contextual_file_analyzer.py"

            if not script_path.exists():
                self.root.after(0, self.analysis_error, f"Файл {script_path.name} не найден")
                return

            # Проверяем наличие intellirefactor
            intellirefactor_path = Path(__file__).parent / "intellirefactor"
            if not intellirefactor_path.exists():
                self.root.after(0, self.analysis_error, "Директория intellirefactor не найдена")
                return

            print(f"Запуск команды: {' '.join(cmd)}")
            print(f"Рабочая директория: {Path(__file__).parent}")

            # Запускаем процесс
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=Path(__file__).parent,
                timeout=1200,  # 20 минут максимум
                encoding="utf-8",
                errors="replace",
            )

            # Обновляем UI в главном потоке
            self.root.after(0, self.analysis_completed, result, analysis_type)

        except subprocess.TimeoutExpired:
            self.root.after(0, self.analysis_error, "Анализ превысил таймаут (20 минут)")
        except Exception as e:
            self.root.after(0, self.analysis_error, f"Ошибка запуска: {str(e)}")

    def analysis_completed(self, result, analysis_type):
        # Останавливаем прогресс бар
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
        self.project_button.config(state="normal")
        self.file_button.config(state="normal")
        self.update_button_states()

        analysis_name = "проекта" if analysis_type == "project" else "файла в контексте проекта"

        if result.returncode == 0:
            messagebox.showinfo(
                "Анализ завершен",
                f"Анализ {analysis_name} успешно завершен!\n\n"
                f"Результаты сохранены в:\n{self.output_dir.get()}\n\n"
                f"Созданы файлы:\n"
                f"• Requirements.md - требования к рефакторингу\n"
                f"• Design.md - документ дизайна\n"
                f"• Implementation.md - документ реализации\n"
                f"• SUMMARY_REPORT_*.md - итоговый отчет\n\n"
                f"Откройте итоговый отчет для просмотра результатов.",
            )

            # Предлагаем открыть папку с результатами
            if messagebox.askyesno("Открыть результаты", "Открыть папку с результатами?"):
                try:
                    if sys.platform == "win32":
                        os.startfile(self.output_dir.get())
                    elif sys.platform == "darwin":
                        subprocess.run(["open", self.output_dir.get()])
                    else:
                        subprocess.run(["xdg-open", self.output_dir.get()])
                except Exception:
                    pass
        else:
            # Показываем детальную информацию об ошибке
            error_details = f"Анализ {analysis_name}\n"
            error_details += f"Код ошибки: {result.returncode}\n\n"

            if result.stderr:
                error_details += f"Ошибки:\n{result.stderr}\n\n"

            if result.stdout:
                error_details += f"Вывод:\n{result.stdout[:1000]}"
                if len(result.stdout) > 1000:
                    error_details += "...\n(вывод обрезан)"

            # Создаем окно с детальной информацией
            error_window = tk.Toplevel(self.root)
            error_window.title(f"Детали ошибки анализа {analysis_name}")
            error_window.geometry("700x500")

            text_widget = tk.Text(error_window, wrap=tk.WORD)
            scrollbar = ttk.Scrollbar(error_window, orient="vertical", command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)

            text_widget.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            text_widget.insert("1.0", error_details)
            text_widget.config(state="disabled")

            messagebox.showerror(
                "Ошибка анализа",
                f"Анализ {analysis_name} завершился с ошибками.\n\n"
                f"Код ошибки: {result.returncode}\n"
                f"Частичные результаты могут быть доступны в:\n{self.output_dir.get()}\n\n"
                f"Откроется окно с подробной информацией об ошибке.",
            )

    def analysis_error(self, error_msg):
        # Останавливаем прогресс бар
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
        self.project_button.config(state="normal")
        self.file_button.config(state="normal")
        self.update_button_states()

        messagebox.showerror("Критическая ошибка", f"Произошла критическая ошибка:\n\n{error_msg}")

    def run(self):
        # Проверяем наличие основных скриптов
        required_files = ["automated_intellirefactor_analyzer.py", "contextual_file_analyzer.py"]

        missing_files = []
        for file in required_files:
            if not Path(file).exists():
                missing_files.append(file)

        if missing_files:
            messagebox.showerror(
                "Ошибка",
                "Не найдены файлы:\n" + "\n".join(missing_files) + "\n\n"
                "Убедитесь, что все файлы находятся в одной директории.",
            )
            return

        # Проверяем наличие intellirefactor
        if not Path("intellirefactor").exists():
            messagebox.showerror(
                "Ошибка",
                "Директория intellirefactor не найдена!\n\n"
                "Убедитесь, что IntelliRefactor установлен в текущей директории.",
            )
            return

        self.root.mainloop()


def main():
    """Главная функция"""
    try:
        app = EnhancedQuickAnalyzerGUI()
        app.run()
    except Exception as e:
        print(f"Ошибка запуска GUI: {e}")
        print("Используйте анализаторы из командной строки")
        sys.exit(1)


if __name__ == "__main__":
    main()

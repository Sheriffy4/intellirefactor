#!/usr/bin/env python3
"""
Быстрый запуск автоматизированного анализатора IntelliRefactor

Упрощенный интерфейс для быстрого анализа проектов
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import subprocess
import threading


class QuickAnalyzerGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Автоматизированный анализатор IntelliRefactor")
        self.root.geometry("600x400")
        self.root.resizable(True, True)

        # Переменные
        self.target_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.verbose = tk.BooleanVar()

        # Устанавливаем значения по умолчанию
        self.output_dir.set(str(Path.cwd() / "analysis_results"))

        self.create_widgets()

    def create_widgets(self):
        # Заголовок
        title_frame = ttk.Frame(self.root)
        title_frame.pack(fill="x", padx=10, pady=5)

        title_label = ttk.Label(
            title_frame,
            text="Автоматизированный анализатор IntelliRefactor",
            font=("Arial", 14, "bold"),
        )
        title_label.pack()

        subtitle_label = ttk.Label(
            title_frame,
            text="Полный анализ проектов с генерацией всех отчетов и диаграмм",
            font=("Arial", 10),
        )
        subtitle_label.pack()

        # Разделитель
        ttk.Separator(self.root, orient="horizontal").pack(fill="x", padx=10, pady=10)

        # Основная форма
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=5)

        # Выбор цели анализа
        target_frame = ttk.LabelFrame(main_frame, text="Цель анализа", padding=10)
        target_frame.pack(fill="x", pady=5)

        ttk.Label(target_frame, text="Выберите проект или файл для анализа:").pack(anchor="w")

        target_entry_frame = ttk.Frame(target_frame)
        target_entry_frame.pack(fill="x", pady=5)

        self.target_entry = ttk.Entry(target_entry_frame, textvariable=self.target_path, width=50)
        self.target_entry.pack(side="left", fill="x", expand=True)

        ttk.Button(target_entry_frame, text="Выбрать файл", command=self.select_file).pack(
            side="right", padx=(5, 0)
        )

        ttk.Button(target_entry_frame, text="Выбрать папку", command=self.select_folder).pack(
            side="right", padx=(5, 0)
        )

        # Выбор выходной директории
        output_frame = ttk.LabelFrame(main_frame, text="Результаты анализа", padding=10)
        output_frame.pack(fill="x", pady=5)

        ttk.Label(output_frame, text="Директория для сохранения результатов:").pack(anchor="w")

        output_entry_frame = ttk.Frame(output_frame)
        output_entry_frame.pack(fill="x", pady=5)

        self.output_entry = ttk.Entry(output_entry_frame, textvariable=self.output_dir, width=50)
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
        info_frame = ttk.LabelFrame(main_frame, text="Информация", padding=10)
        info_frame.pack(fill="both", expand=True, pady=5)

        info_text = """Автоматизированный анализатор выполнит следующие операции:

• Базовый и расширенный анализ кода
• Построение индекса проекта
• Обнаружение дублированного кода
• Поиск неиспользуемого кода
• Выявление архитектурных проблем
• Кластеризация ответственностей
• Генерация решений по рефакторингу
• Комплексный аудит проекта
• Создание спецификаций и документации
• Генерация диаграмм и визуализаций

Результаты будут сохранены в различных форматах (JSON, Markdown, HTML)
с итоговым отчетом и рекомендациями."""

        info_label = ttk.Label(info_frame, text=info_text, justify="left", wraplength=550)
        info_label.pack(anchor="w")

        # Кнопки управления
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", padx=10, pady=10)

        self.analyze_button = ttk.Button(
            button_frame,
            text="🚀 Запустить анализ",
            command=self.start_analysis,
            style="Accent.TButton",
        )
        self.analyze_button.pack(side="right", padx=(5, 0))

        ttk.Button(button_frame, text="❌ Выход", command=self.root.quit).pack(side="right")

        # Прогресс бар (скрыт по умолчанию)
        self.progress_frame = ttk.Frame(self.root)

        self.progress_label = ttk.Label(self.progress_frame, text="Выполняется анализ...")
        self.progress_label.pack()

        self.progress_bar = ttk.Progressbar(self.progress_frame, mode="indeterminate", length=400)
        self.progress_bar.pack(pady=5)

    def select_file(self):
        filename = filedialog.askopenfilename(
            title="Выберите файл для анализа",
            filetypes=[("Python файлы", "*.py"), ("Все файлы", "*.*")],
        )
        if filename:
            self.target_path.set(filename)

    def select_folder(self):
        folder = filedialog.askdirectory(title="Выберите папку проекта для анализа")
        if folder:
            self.target_path.set(folder)

    def select_output_dir(self):
        folder = filedialog.askdirectory(title="Выберите папку для сохранения результатов")
        if folder:
            self.output_dir.set(folder)

    def validate_inputs(self):
        if not self.target_path.get().strip():
            messagebox.showerror("Ошибка", "Выберите файл или папку для анализа")
            return False

        if not Path(self.target_path.get()).exists():
            messagebox.showerror("Ошибка", "Указанный путь не существует")
            return False

        if not self.output_dir.get().strip():
            messagebox.showerror("Ошибка", "Укажите директорию для результатов")
            return False

        return True

    def start_analysis(self):
        if not self.validate_inputs():
            return

        # Подтверждение
        result = messagebox.askyesno(
            "Подтверждение",
            f"Запустить полный анализ?\n\n"
            f"Цель: {self.target_path.get()}\n"
            f"Результаты: {self.output_dir.get()}\n\n"
            f"Анализ может занять несколько минут.",
        )

        if not result:
            return

        # Отключаем кнопку и показываем прогресс
        self.analyze_button.config(state="disabled")
        self.progress_frame.pack(fill="x", padx=10, pady=5)
        self.progress_bar.start()

        # Запускаем анализ в отдельном потоке
        thread = threading.Thread(target=self.run_analysis)
        thread.daemon = True
        thread.start()

    def run_analysis(self):
        try:
            # Формируем команду
            cmd = [
                sys.executable,
                "automated_intellirefactor_analyzer.py",
                self.target_path.get(),
                self.output_dir.get(),
            ]

            if self.verbose.get():
                cmd.append("--verbose")

            # Проверяем наличие основного скрипта
            script_path = Path(__file__).parent / "automated_intellirefactor_analyzer.py"
            if not script_path.exists():
                self.root.after(
                    0, self.analysis_error, "Файл automated_intellirefactor_analyzer.py не найден"
                )
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
                timeout=600,  # 10 минут максимум
                encoding="utf-8",
                errors="replace",
            )

            # Обновляем UI в главном потоке
            self.root.after(0, self.analysis_completed, result)

        except subprocess.TimeoutExpired:
            self.root.after(0, self.analysis_error, "Анализ превысил таймаут (10 минут)")
        except Exception as e:
            self.root.after(0, self.analysis_error, f"Ошибка запуска: {str(e)}")

    def analysis_completed(self, result):
        # Останавливаем прогресс бар
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
        self.analyze_button.config(state="normal")

        if result.returncode == 0:
            messagebox.showinfo(
                "Анализ завершен",
                f"Анализ успешно завершен!\n\n"
                f"Результаты сохранены в:\n{self.output_dir.get()}\n\n"
                f"Откройте файл SUMMARY_REPORT_*.md для просмотра итогового отчета.",
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
            error_details = f"Код ошибки: {result.returncode}\n\n"

            if result.stderr:
                error_details += f"Ошибки:\n{result.stderr}\n\n"

            if result.stdout:
                error_details += f"Вывод:\n{result.stdout[:500]}"
                if len(result.stdout) > 500:
                    error_details += "...\n(вывод обрезан)"

            # Создаем окно с детальной информацией
            error_window = tk.Toplevel(self.root)
            error_window.title("Детали ошибки анализа")
            error_window.geometry("600x400")

            text_widget = tk.Text(error_window, wrap=tk.WORD)
            scrollbar = ttk.Scrollbar(error_window, orient="vertical", command=text_widget.yview)
            text_widget.configure(yscrollcommand=scrollbar.set)

            text_widget.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")

            text_widget.insert("1.0", error_details)
            text_widget.config(state="disabled")

            messagebox.showerror(
                "Ошибка анализа",
                f"Анализ завершился с ошибками.\n\n"
                f"Код ошибки: {result.returncode}\n"
                f"Частичные результаты могут быть доступны в:\n{self.output_dir.get()}\n\n"
                f"Откроется окно с подробной информацией об ошибке.",
            )

    def analysis_error(self, error_msg):
        # Останавливаем прогресс бар
        self.progress_bar.stop()
        self.progress_frame.pack_forget()
        self.analyze_button.config(state="normal")

        messagebox.showerror("Критическая ошибка", f"Произошла критическая ошибка:\n\n{error_msg}")

    def run(self):
        # Проверяем наличие основного скрипта
        if not Path("automated_intellirefactor_analyzer.py").exists():
            messagebox.showerror(
                "Ошибка",
                "Файл automated_intellirefactor_analyzer.py не найден!\n\n"
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
        app = QuickAnalyzerGUI()
        app.run()
    except Exception as e:
        print(f"Ошибка запуска GUI: {e}")
        print("Используйте automated_intellirefactor_analyzer.py для запуска из командной строки")
        sys.exit(1)


if __name__ == "__main__":
    main()

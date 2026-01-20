#!/usr/bin/env python3
"""
Ultimate GUI Analyzer for IntelliRefactor

Исправления универсальности UI (в т.ч. для 4K/нестанд. масштабирования):
- Нижние кнопки ВСЕГДА видимы: вынесены в отдельную нижнюю панель (не «выталкиваются» контентом)
- Основной контент сделан прокручиваемым (если шрифты/масштаб большие — ничего не обрежется)
- Увеличены шрифт/паддинги кнопок
- Добавлена попытка включить DPI-aware режим на Windows (помогает при системном scaling)
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from pathlib import Path
import subprocess
import threading
from datetime import datetime


def enable_high_dpi_awareness():
    """Пытаемся включить DPI-aware режим на Windows до создания Tk()."""
    if sys.platform != "win32":
        return
    try:
        import ctypes  # noqa: F401
        # Windows 8.1+ (Per-monitor DPI aware)
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
            return
        except Exception:
            # Fall back to system DPI awareness
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                # If both fail, continue with default DPI handling
                pass
    except Exception:
        # If shcore is not available, continue with default DPI handling
        pass


class ScrollableFrame(ttk.Frame):
    """Простой прокручиваемый контейнер: внутри него можно pack/grid обычные виджеты."""

    def __init__(self, parent):
        super().__init__(parent)

        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.v_scroll = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.v_scroll.set)

        self.v_scroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Внутренняя рамка
        self.inner = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        # Обновляем scrollregion когда меняется размер inner
        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)

        # Колесо мыши
        self._bind_mousewheel()

    def _on_inner_configure(self, event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # Растягиваем inner по ширине canvas
        self.canvas.itemconfigure(self._window_id, width=event.width)

    def _bind_mousewheel(self):
        # Windows/macOS: <MouseWheel>, Linux: Button-4/5
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel_windows, add="+")
        self.canvas.bind_all("<Button-4>", self._on_mousewheel_linux, add="+")
        self.canvas.bind_all("<Button-5>", self._on_mousewheel_linux, add="+")

    def _on_mousewheel_windows(self, event):
        # Чтобы не скроллить весь UI где попало — проверяем, что курсор над canvas/inner
        x, y = self.canvas.winfo_pointerxy()
        w = self.canvas.winfo_containing(x, y)
        if w is None:
            return
        if not (w == self.canvas or str(w).startswith(str(self.inner))):
            return

        # event.delta: 120/-120 на Windows, на macOS может быть иначе
        delta = int(-1 * (event.delta / 120)) if event.delta else 0
        if delta:
            self.canvas.yview_scroll(delta, "units")

    def _on_mousewheel_linux(self, event):
        x, y = self.canvas.winfo_pointerxy()
        w = self.canvas.winfo_containing(x, y)
        if w is None:
            return
        if not (w == self.canvas or str(w).startswith(str(self.inner))):
            return

        if event.num == 4:
            self.canvas.yview_scroll(-1, "units")
        elif event.num == 5:
            self.canvas.yview_scroll(1, "units")


class UltimateGUIAnalyzer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Оптимизированный анализатор для рефакторинга")
        self.root.geometry("1400x900")
        self.root.minsize(1200, 800)

        # Попытка корректной Tk scaling на DPI системах
        # (обычно безопасно; если вдруг тема/система ведёт себя странно — можно закомментировать)
        try:
            # DPI (pixels per inch) / 72pt
            self.root.tk.call("tk", "scaling", self.root.winfo_fpixels("1i") / 72.0)
        except Exception:
            # If DPI scaling fails, continue with default scaling
            pass

        # Переменные
        self.project_path = tk.StringVar()
        self.target_file = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.verbose = tk.BooleanVar()

        self.output_dir.set(str(Path.cwd() / "analysis_results"))

        self._configure_styles()
        self._build_layout()
        self.update_button_states()

    def _configure_styles(self):
        style = ttk.Style(self.root)

        # Большие кнопки с нормальным padding — меньше шанс обрезки текста
        style.configure("Big.TButton", 
                       font=("Arial", 11, "bold"), 
                       padding=(20, 15),
                       width=25)
        style.configure("Accent.TButton", 
                       font=("Arial", 11, "bold"), 
                       padding=(20, 15),
                       width=25)

        style.configure("Hint.TLabel", font=("Arial", 9), foreground="gray")

    def _build_layout(self):
        # Корневой grid: контент растягивается, кнопки всегда внизу
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(2, weight=1)  # только середина растягивается

        # Header
        header = ttk.Frame(self.root)
        header.grid(row=0, column=0, sticky="ew", padx=10, pady=6)
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="Оптимизированный анализатор для рефакторинга",
            font=("Arial", 16, "bold"),
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            header,
            text="Собирает только нужную информацию для максимального плана рефакторинга",
            font=("Arial", 11, "italic"),
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        ttk.Separator(self.root, orient="horizontal").grid(
            row=1, column=0, sticky="ew", padx=10, pady=(0, 8)
        )

        # Прокручиваемый контент (все поля + описание)
        self.scrollable = ScrollableFrame(self.root)
        self.scrollable.grid(row=2, column=0, sticky="nsew", padx=10, pady=(0, 8))

        content = self.scrollable.inner

        # --- Проект ---
        project_frame = ttk.LabelFrame(content, text="Проект (обязательно)", padding=10)
        project_frame.pack(fill="x", pady=6)

        ttk.Label(project_frame, text="Выберите корневую папку проекта:").pack(anchor="w")

        project_entry_frame = ttk.Frame(project_frame)
        project_entry_frame.pack(fill="x", pady=6)

        self.project_entry = ttk.Entry(project_entry_frame, textvariable=self.project_path)
        self.project_entry.pack(side="left", fill="x", expand=True)
        self.project_entry.bind("<KeyRelease>", self.on_path_change)

        ttk.Button(
            project_entry_frame,
            text="Выбрать проект",
            command=self.select_project,
        ).pack(side="right", padx=(8, 0))

        # --- Файл ---
        file_frame = ttk.LabelFrame(content, text="Конкретный файл (опционально)", padding=10)
        file_frame.pack(fill="x", pady=6)

        ttk.Label(file_frame, text="Выберите конкретный файл для фокусированного анализа:").pack(
            anchor="w"
        )

        file_entry_frame = ttk.Frame(file_frame)
        file_entry_frame.pack(fill="x", pady=6)

        self.file_entry = ttk.Entry(file_entry_frame, textvariable=self.target_file)
        self.file_entry.pack(side="left", fill="x", expand=True)
        self.file_entry.bind("<KeyRelease>", self.on_path_change)

        ttk.Button(file_entry_frame, text="Выбрать файл", command=self.select_file).pack(
            side="right", padx=(8, 0)
        )
        ttk.Button(file_entry_frame, text="Очистить", command=self.clear_file).pack(
            side="right", padx=(8, 0)
        )

        # --- Output ---
        output_frame = ttk.LabelFrame(content, text="Результаты анализа", padding=10)
        output_frame.pack(fill="x", pady=6)

        ttk.Label(output_frame, text="Директория для сохранения результатов:").pack(anchor="w")

        output_entry_frame = ttk.Frame(output_frame)
        output_entry_frame.pack(fill="x", pady=6)

        self.output_entry = ttk.Entry(output_entry_frame, textvariable=self.output_dir)
        self.output_entry.pack(side="left", fill="x", expand=True)

        ttk.Button(output_entry_frame, text="Выбрать", command=self.select_output_dir).pack(
            side="right", padx=(8, 0)
        )

        # --- Settings ---
        settings_frame = ttk.LabelFrame(content, text="Настройки", padding=10)
        settings_frame.pack(fill="x", pady=6)

        ttk.Checkbutton(
            settings_frame,
            text="Подробный вывод (рекомендуется для отладки)",
            variable=self.verbose,
        ).pack(anchor="w")

        # --- Info (фиксируем высоту, чтобы не «выталкивало» кнопки) ---
        info_frame = ttk.LabelFrame(content, text="Режимы анализа", padding=10)
        info_frame.pack(fill="both", expand=True, pady=6)

        info_text = """🏗️ АНАЛИЗ ПРОЕКТА:
• Полный анализ всего проекта
• Requirements.md, Design.md, Implementation.md для всего проекта
• Базовые возможности IntelliRefactor

📄 АНАЛИЗ ФАЙЛА:
• Фокус на конкретном файле в контексте всего проекта
• Requirements.md, Design.md, Implementation.md для конкретного файла
• Дубликаты, неиспользуемый код, архитектурные запахи

🎯 ПЛАН РЕФАКТОРИНГА:
• ТОЛЬКО нужная информация для составления плана рефакторинга
• Реальные паттерны использования из production кода
• Структурированный анализ возможностей улучшения
• Экспертные рекомендации с приоритетами и рисками
• Готовый план действий с временными рамками

🏗️ АРХИТЕКТУРА:
• Анализ всего проекта для выявления архитектурных проблем
• Поиск God Objects и функциональных дубликатов
• Кластеризация модулей по функциональности
• Выявление мертвого кода и неиспользуемых модулей
• План архитектурной реорганизации проекта
• Диаграммы зависимостей и матрицы функциональности

🔧 ФУНКЦИОНАЛЬНАЯ ДЕКОМПОЗИЦИЯ:
• Новый подход: извлечение атомарных функциональных блоков
• Автоматическая категоризация по назначению (parsing, validation, etc.)
• Кластеризация похожей функциональности с оценкой похожести
• Планы безопасной консолидации (canonical + wrappers + migration)
• Пошаговые патчи с многоуровневой валидацией
• Режимы: analyze-only, plan-only, apply-safe, apply-assisted
• Детальные отчеты: JSON, Markdown, Mermaid диаграммы

💡 ПРЕИМУЩЕСТВА НОВЫХ АНАЛИЗОВ:
• Быстрее - 5-15 минут вместо 30+ минут
• Точнее - только релевантная информация
• Практичнее - готовые планы действий для специалиста
• Качественнее - экспертные рекомендации на основе реальных данных
• Безопаснее - пошаговые патчи с валидацией (функциональная декомпозиция)
• Современнее - использует передовые подходы анализа кода
"""

        # Текст со скроллом и фикс. высотой — не ломает раскладку на больших scaling
        info_text_frame = ttk.Frame(info_frame)
        info_text_frame.pack(fill="both", expand=True)

        self.info_text_widget = tk.Text(
            info_text_frame,
            height=16,            # ключевое: ограничиваем высоту
            wrap=tk.WORD,
            font=("Arial", 10),
        )
        info_scroll = ttk.Scrollbar(info_text_frame, orient="vertical", command=self.info_text_widget.yview)
        self.info_text_widget.configure(yscrollcommand=info_scroll.set)

        self.info_text_widget.pack(side="left", fill="both", expand=True)
        info_scroll.pack(side="right", fill="y")

        self.info_text_widget.insert("1.0", info_text)
        self.info_text_widget.config(state="disabled")

        # --- Нижняя панель кнопок (ВСЕГДА видна) ---
        self.buttons_panel = ttk.Frame(self.root)
        self.buttons_panel.grid(row=3, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.buttons_panel.columnconfigure(0, weight=1)

        buttons_grid = ttk.Frame(self.buttons_panel)
        buttons_grid.grid(row=0, column=0, sticky="ew")
        buttons_grid.columnconfigure(0, weight=1, uniform="btncol")
        buttons_grid.columnconfigure(1, weight=1, uniform="btncol")

        def make_button_card(parent, r, c, text, hint, command, style):
            card = ttk.Frame(parent)
            card.grid(row=r, column=c, sticky="ew", padx=12, pady=(0, 14))
            card.columnconfigure(0, weight=1)

            btn = ttk.Button(card, text=text, command=command, style=style, width=25)
            btn.grid(row=0, column=0, sticky="ew")

            ttk.Label(card, text=hint, style="Hint.TLabel", justify="center").grid(
                row=1, column=0, sticky="ew", pady=(6, 0)
            )
            return btn

        self.project_button = make_button_card(
            buttons_grid, 0, 0,
            "🏗️ Анализ проекта",
            "Быстрый анализ всего проекта\nRequirements, Design, Implementation",
            self.start_project_analysis,
            "Big.TButton",
        )
        self.file_button = make_button_card(
            buttons_grid, 0, 1,
            "📄 Анализ файла",
            "Файл в контексте проекта\nДубликаты, запахи кода",
            self.start_file_analysis,
            "Big.TButton",
        )
        self.ultimate_button = make_button_card(
            buttons_grid, 1, 0,
            "🎯 План рефакторинга",
            "Оптимизированный план\nТолько нужная информация",
            self.start_ultimate_analysis,
            "Accent.TButton",
        )
        self.decomposition_button = make_button_card(
            buttons_grid, 1, 1,
            "🔧 Функциональная декомпозиция",
            "Новый анализ: блоки, кластеры\nПланы безопасной консолидации",
            self.start_project_decomposition,
            "Big.TButton",
        )

        exit_row = ttk.Frame(buttons_grid)
        exit_row.grid(row=2, column=0, columnspan=2, sticky="ew")
        exit_row.columnconfigure(0, weight=1)
        ttk.Button(exit_row, text="❌ Выход", command=self.root.quit, width=18).grid(
            row=0, column=0, pady=(2, 0)
        )

        # --- Прогресс (показываем/прячем через grid_remove) ---
        self.progress_frame = ttk.Frame(self.root)
        self.progress_frame.grid(row=4, column=0, sticky="ew", padx=10, pady=(0, 10))
        self.progress_frame.columnconfigure(0, weight=1)

        self.progress_label = ttk.Label(self.progress_frame, text="Выполняется анализ...")
        self.progress_label.grid(row=0, column=0, sticky="w")

        self.progress_bar = ttk.Progressbar(self.progress_frame, mode="indeterminate", length=600)
        self.progress_bar.grid(row=1, column=0, sticky="w", pady=(6, 0))

        # Скрыт по умолчанию
        self.progress_frame.grid_remove()

    def select_project(self):
        folder = filedialog.askdirectory(title="Выберите корневую папку проекта")
        if folder:
            self.project_path.set(folder)
            self.update_button_states()

    def select_file(self):
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
        project_exists = bool(self.project_path.get().strip())
        file_exists = bool(self.target_file.get().strip())

        self.project_button.config(state="normal" if project_exists else "disabled")
        self.decomposition_button.config(state="normal" if project_exists else "disabled")

        enabled_file = project_exists and file_exists
        self.file_button.config(state="normal" if enabled_file else "disabled")
        self.ultimate_button.config(state="normal" if enabled_file else "disabled")

    def validate_project_inputs(self):
        if not self.project_path.get().strip():
            messagebox.showerror("Ошибка", "Выберите корневую папку проекта")
            return False

        p = Path(self.project_path.get())
        if not p.exists():
            messagebox.showerror("Ошибка", "Указанный путь к проекту не существует")
            return False
        if not p.is_dir():
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

        f = Path(self.target_file.get())
        if not f.exists():
            messagebox.showerror("Ошибка", "Указанный файл не существует")
            return False
        if not f.is_file():
            messagebox.showerror("Ошибка", "Указанный путь должен быть файлом")
            return False

        return True

    def start_project_analysis(self):
        if not self.validate_project_inputs():
            return

        result = messagebox.askyesno(
            "Анализ проекта",
            f"Запустить быстрый анализ проекта?\n\n"
            f"Проект: {self.project_path.get()}\n"
            f"Результаты: {self.output_dir.get()}\n\n"
            f"Будут созданы:\n"
            f"• Requirements.md\n"
            f"• Design.md\n"
            f"• Implementation.md\n"
            f"• Базовые анализы и диаграммы\n",
        )
        if result:
            self.start_analysis("project")

    def start_file_analysis(self):
        if not self.validate_file_inputs():
            return

        result = messagebox.askyesno(
            "Анализ файла",
            f"Запустить анализ файла в контексте всего проекта?\n\n"
            f"Проект: {self.project_path.get()}\n"
            f"Файл: {self.target_file.get()}\n"
            f"Результаты: {self.output_dir.get()}\n",
        )
        if result:
            self.start_analysis("file")

    def start_project_decomposition(self):
        if not self.validate_project_inputs():
            return

        result = messagebox.askyesno(
            "Функциональная декомпозиция",
            f"Запустить функциональную декомпозицию проекта?\n\n"
            f"Проект: {self.project_path.get()}\n"
            f"Результаты: {self.output_dir.get()}\n\n"
            f"Новый анализ включает:\n"
            f"• Извлечение функциональных блоков\n"
            f"• Категоризация и кластеризация похожего кода\n"
            f"• Планирование безопасной консолидации\n"
            f"• Пошаговые планы рефакторинга\n"
            f"• Детальные отчеты и диаграммы\n",
        )
        if result:
            self.start_analysis("functional_decomposition")

    def start_ultimate_analysis(self):
        if not self.validate_file_inputs():
            return

        result = messagebox.askyesno(
            "План рефакторинга",
            f"Создать оптимизированный план рефакторинга для файла?\n\n"
            f"Проект: {self.project_path.get()}\n"
            f"Файл: {self.target_file.get()}\n"
            f"Результаты: {self.output_dir.get()}\n",
        )
        if result:
            self.start_analysis("ultimate")

    def start_analysis(self, analysis_type: str):
        # Отключаем кнопки
        for b in (self.project_button, self.file_button, self.ultimate_button, self.decomposition_button):
            b.config(state="disabled")

        # Прогресс
        label = {
            "project": "Выполняется анализ проекта...",
            "file": "Выполняется анализ файла...",
            "ultimate": "Создается план рефакторинга...",
            "decomposition": "Анализируется архитектура...",
            "functional_decomposition": "Выполняется функциональная декомпозиция...",
        }.get(analysis_type, "Выполняется анализ...")
        self.progress_label.config(text=label)
        self.progress_frame.grid()
        self.progress_bar.start()

        thread = threading.Thread(target=self.run_analysis, args=(analysis_type,), daemon=True)
        thread.start()

    def run_analysis(self, analysis_type: str):
        try:
            script_names = {
                "project": "automated_intellirefactor_analyzer.py",
                "file": "contextual_file_analyzer.py",
                "ultimate": "optimized_refactoring_analyzer.py",
                "decomposition": "project_decomposition_analyzer.py",
                "functional_decomposition": "functional_decomposition_analyzer.py",
            }
            if analysis_type not in script_names:
                self.root.after(0, self.analysis_error, f"Неизвестный тип анализа: {analysis_type}")
                return

            base_dir = Path(__file__).parent
            script = script_names[analysis_type]
            script_path = base_dir / script

            if not script_path.exists():
                self.root.after(0, self.analysis_error, f"Файл {script} не найден")
                return

            if not (base_dir / "intellirefactor").exists():
                self.root.after(0, self.analysis_error, "Директория intellirefactor не найдена")
                return

            if analysis_type == "project":
                cmd = [sys.executable, script, self.project_path.get(), self.output_dir.get()]
            elif analysis_type in ["decomposition", "functional_decomposition"]:
                cmd = [sys.executable, script, self.project_path.get(), self.output_dir.get()]
            else:
                cmd = [
                    sys.executable,
                    script,
                    self.project_path.get(),
                    self.target_file.get(),
                    self.output_dir.get(),
                ]

            if self.verbose.get():
                cmd.append("--verbose")

            # УЛУЧШЕННОЕ ЛОГИРОВАНИЕ
            from datetime import datetime
            full_command = ' '.join(f'"{arg}"' if ' ' in arg else arg for arg in cmd)
            print(f"\n{'='*80}")
            print(f"ЗАПУСК АНАЛИЗА: {analysis_type.upper()}")
            print(f"{'='*80}")
            print(f"Команда: {full_command}")
            print(f"Рабочая директория: {base_dir}")
            print(f"Время запуска: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*80}\n")

            timeout = 1800 if analysis_type in ["ultimate", "decomposition", "functional_decomposition"] else 1200

            # Bandit B603: subprocess call - check for execution of untrusted input
            # This is safe because cmd comes from our controlled command construction
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=base_dir,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
                check=False,  # Explicitly disable check to handle errors manually
            )

            # ЛОГИРУЕМ РЕЗУЛЬТАТ
            print(f"\n{'='*80}")
            print(f"РЕЗУЛЬТАТ АНАЛИЗА: {analysis_type.upper()}")
            print(f"{'='*80}")
            print(f"Код возврата: {result.returncode}")
            print(f"Время завершения: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            if result.stdout:
                print("STDOUT (первые 1000 символов):")
                print(result.stdout[:1000])
                if len(result.stdout) > 1000:
                    print("... (вывод обрезан)")
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            print(f"{'='*80}\n")

            self.root.after(0, self.analysis_completed, result, analysis_type)

        except subprocess.TimeoutExpired:
            timeout_msg = "30 минут" if analysis_type in ["ultimate", "decomposition", "functional_decomposition"] else "20 минут"
            error_msg = f"Анализ превысил таймаут ({timeout_msg})"
            print(f"\nОШИБКА: {error_msg}")
            self.root.after(0, self.analysis_error, error_msg)
        except Exception as e:
            error_msg = f"Ошибка запуска: {str(e)}"
            print(f"\nОШИБКА: {error_msg}")
            import traceback
            traceback.print_exc()
            self.root.after(0, self.analysis_error, error_msg)

    def analysis_completed(self, result: subprocess.CompletedProcess, analysis_type: str):
        # Останавливаем прогресс бар
        self.progress_bar.stop()
        self.progress_frame.pack_forget()  # Исправлено: используем pack_forget вместо grid_remove
        
        # Включаем все кнопки
        self.project_button.config(state="normal")
        self.file_button.config(state="normal")
        self.ultimate_button.config(state="normal")
        self.decomposition_button.config(state="normal")
        self.update_button_states()

        analysis_names = {
            "project": "анализа проекта",
            "file": "анализа файла",
            "ultimate": "плана рефакторинга",
            "decomposition": "анализа архитектуры",
            "functional_decomposition": "функциональной декомпозиции",
        }
        analysis_name = analysis_names.get(analysis_type, analysis_type)

        # УЛУЧШЕННАЯ ОБРАБОТКА РЕЗУЛЬТАТОВ
        print(f"\n{'='*50}")
        print("ОБРАБОТКА РЕЗУЛЬТАТОВ GUI")
        print(f"{'='*50}")
        print(f"Тип анализа: {analysis_type}")
        print(f"Код возврата: {result.returncode}")
        print(f"Длина stdout: {len(result.stdout) if result.stdout else 0}")
        print(f"Длина stderr: {len(result.stderr) if result.stderr else 0}")
        print(f"{'='*50}")

        if result.returncode == 0:
            key_files = {
                "project": ["Requirements.md", "Design.md", "Implementation.md", "SUMMARY_REPORT_*.md"],
                "file": ["Requirements.md", "Design.md", "Implementation.md", "CONTEXTUAL_FILE_REPORT_*.md"],
                "ultimate": [
                    "OPTIMIZED_REFACTORING_PLAN_*.md",
                    "OPTIMIZED_REFACTORING_DATA_*.json",
                    "REFACTORING_SUMMARY_*.md",
                    "Requirements.md",
                    "Design.md",
                    "Implementation.md",
                ],
                "decomposition": [
                    "PROJECT_DECOMPOSITION_PLAN_*.md",
                    "PROJECT_DECOMPOSITION_DATA_*.json",
                    "DECOMPOSITION_SUMMARY_*.md",
                    "project_dependencies_*.mmd",
                    "functionality_matrix_*.md",
                ],
                "functional_decomposition": [
                    "FUNCTIONAL_DECOMPOSITION_SUMMARY_*.md",
                    "functional_map.json",
                    "clusters.json",
                    "consolidation_plan.md",
                    "catalog.md",
                    "summary.md",
                    "functional_graph.mmd",
                ],
            }

            msg = f"Анализ {analysis_name} успешно завершен!\n\n"
            msg += f"Результаты сохранены в:\n{self.output_dir.get()}\n\n"
            msg += "Созданы файлы:\n"
            for pattern in key_files.get(analysis_type, []):
                msg += f"• {pattern}\n"

            # Добавляем специфичные описания для каждого типа анализа
            if analysis_type == "ultimate":
                msg += "\n🎯 ОПТИМИЗИРОВАННЫЙ ПОДХОД:\n"
                msg += "• Фокус на рефакторинге - только нужная информация\n"
                msg += "• Реальные паттерны использования из кода\n"
                msg += "• Структурированный план действий\n"
                msg += "• Экспертные рекомендации с приоритетами\n"
                msg += "• Оценка рисков и временные рамки\n"
            
            elif analysis_type == "decomposition":
                msg += "\n🏗️ АРХИТЕКТУРНАЯ ДЕКОМПОЗИЦИЯ:\n"
                msg += "• Выявление God Objects и план их разделения\n"
                msg += "• Поиск функциональных дубликатов\n"
                msg += "• Кластеризация модулей по функциональности\n"
                msg += "• Обнаружение мертвого кода\n"
                msg += "• Диаграммы зависимостей и матрицы\n"
            
            elif analysis_type == "functional_decomposition":
                msg += "\n🔧 ФУНКЦИОНАЛЬНАЯ ДЕКОМПОЗИЦИЯ:\n"
                msg += "• Извлечение атомарных функциональных блоков\n"
                msg += "• Автоматическая категоризация по назначению\n"
                msg += "• Кластеризация похожей функциональности\n"
                msg += "• Планы безопасной консолидации (wrappers + migration)\n"
                msg += "• Пошаговые патчи с валидацией\n"
                msg += "• Детальные отчеты и визуализации\n"

            msg += "\nОткройте итоговый отчет для просмотра результатов."

            print(f"Показываем сообщение об успехе для {analysis_type}")
            messagebox.showinfo("Анализ завершен", msg)

            # Предлагаем открыть папку с результатами
            if messagebox.askyesno("Открыть результаты", "Открыть папку с результатами?"):
                try:
                    if sys.platform == "win32":
                        os.startfile(self.output_dir.get())
                    elif sys.platform == "darwin":
                        # Bandit B607: Starting process with partial executable path
                        # This is safe for system commands on macOS
                        subprocess.run(["open", self.output_dir.get()], check=False)
                    else:
                        # Bandit B607: Starting process with partial executable path
                        # This is safe for system commands on Linux
                        subprocess.run(["xdg-open", self.output_dir.get()], check=False)
                except Exception as e:
                    print(f"Не удалось открыть папку: {e}")
        else:
            # Обработка ошибок
            print(f"Анализ завершился с ошибкой. Код: {result.returncode}")
            error_details = f"Анализ {analysis_name}\nКод ошибки: {result.returncode}\n\n"
            
            if result.stderr:
                error_details += f"Ошибки:\n{result.stderr}\n\n"
            
            if result.stdout:
                error_details += "Вывод:\n" + (result.stdout[:2000] + ("\n...\n(вывод обрезан)" if len(result.stdout) > 2000 else ""))

            # Создаем окно с детальной информацией
            win = tk.Toplevel(self.root)
            win.title(f"Детали ошибки анализа {analysis_name}")
            win.geometry("800x550")

            txt = tk.Text(win, wrap=tk.WORD)
            scr = ttk.Scrollbar(win, orient="vertical", command=txt.yview)
            txt.configure(yscrollcommand=scr.set)

            txt.pack(side="left", fill="both", expand=True)
            scr.pack(side="right", fill="y")

            txt.insert("1.0", error_details)
            txt.config(state="disabled")

            messagebox.showerror(
                "Ошибка анализа",
                f"Анализ {analysis_name} завершился с ошибками.\n\n"
                f"Код ошибки: {result.returncode}\n"
                f"Частичные результаты могут быть в:\n{self.output_dir.get()}\n\n"
                f"Откроется окно с подробной информацией об ошибке.",
            )

    def analysis_error(self, error_msg: str):
        # Останавливаем прогресс бар
        self.progress_bar.stop()
        self.progress_frame.pack_forget()  # Исправлено: используем pack_forget вместо grid_remove

        # Включаем все кнопки
        for b in (self.project_button, self.file_button, self.ultimate_button, self.decomposition_button):
            b.config(state="normal")
        self.update_button_states()

        # Создаем окно с детальной информацией об ошибке
        error_window = tk.Toplevel(self.root)
        error_window.title("Детали ошибки")
        error_window.geometry("800x600")
        error_window.transient(self.root)
        error_window.grab_set()

        # Текстовое поле с возможностью копирования
        text_frame = ttk.Frame(error_window)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)

        error_text = tk.Text(text_frame, wrap=tk.WORD, font=("Consolas", 10))
        scrollbar = ttk.Scrollbar(text_frame, orient="vertical", command=error_text.yview)
        error_text.configure(yscrollcommand=scrollbar.set)

        error_text.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Вставляем текст ошибки
        full_error_text = f"""ОШИБКА АНАЛИЗА
{'='*50}

{error_msg}

ИНСТРУКЦИИ ПО ОТЛАДКЕ:
{'='*50}

1. Скопируйте этот текст (Ctrl+A, Ctrl+C)
2. Проверьте логи в папке результатов
3. Запустите анализ из командной строки для получения подробной информации

ВРЕМЯ ОШИБКИ: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        error_text.insert("1.0", full_error_text)
        error_text.config(state="disabled")

        # Кнопки
        button_frame = ttk.Frame(error_window)
        button_frame.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Button(
            button_frame, 
            text="Копировать в буфер", 
            command=lambda: self._copy_to_clipboard(full_error_text)
        ).pack(side="left", padx=(0, 10))

        ttk.Button(
            button_frame, 
            text="Закрыть", 
            command=error_window.destroy
        ).pack(side="right")

        # Показываем основное сообщение об ошибке
        messagebox.showerror("Ошибка анализа", "Произошла ошибка во время анализа.\n\nОткроется окно с подробной информацией для отладки.")

    def _copy_to_clipboard(self, text: str):
        """Копирует текст в буфер обмена"""
        try:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            messagebox.showinfo("Скопировано", "Текст ошибки скопирован в буфер обмена")
        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось скопировать в буфер: {e}")

    def run(self):
        base_dir = Path(__file__).parent
        required_files = [
            "automated_intellirefactor_analyzer.py",
            "contextual_file_analyzer.py",
            "optimized_refactoring_analyzer.py",
            "project_decomposition_analyzer.py",
        ]

        missing = [f for f in required_files if not (base_dir / f).exists()]
        if missing:
            messagebox.showerror(
                "Ошибка",
                "Не найдены файлы:\n" + "\n".join(missing) + "\n\n"
                "Убедитесь, что все файлы находятся в одной директории.",
            )
            return

        if not (base_dir / "intellirefactor").exists():
            messagebox.showerror(
                "Ошибка",
                "Директория intellirefactor не найдена!\n\n"
                "Убедитесь, что IntelliRefactor установлен рядом с этим GUI.",
            )
            return

        self.root.mainloop()


def main():
    enable_high_dpi_awareness()
    try:
        app = UltimateGUIAnalyzer()
        app.run()
    except Exception as e:
        print(f"Ошибка запуска GUI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
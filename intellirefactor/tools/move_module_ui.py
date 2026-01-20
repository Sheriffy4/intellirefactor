from __future__ import annotations

import os
import shutil
import threading
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Optional

import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import libcst as cst


# ---------- ЛОГИКА РЕФАКТОРИНГА ----------

PROJECT_MARKERS = ("pyproject.toml", "setup.py", "setup.cfg", ".git")


SKIP_DIRS = {
    ".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".tox",
    "venv", ".venv", "env", ".env",
    "build", "dist", ".eggs", "site-packages",
}


def find_project_root(start: Path) -> Path:
    p = start.resolve()
    if p.is_file():
        p = p.parent
    for parent in [p, *p.parents]:
        if any((parent / m).exists() for m in PROJECT_MARKERS):
            return parent
    return p


def pick_module_base(project_root: Path, any_project_file: Path) -> Path:
    """
    Поддержка src-layout:
    если есть <root>/src и выбранный файл лежит внутри него — считаем base=src.
    Иначе base=project_root.
    """
    src = project_root / "src"
    try:
        any_project_file.resolve().relative_to(src.resolve())
        if src.exists() and src.is_dir():
            return src
    except Exception:
        pass
    return project_root


def file_to_module(module_base: Path, file_path: Path) -> str:
    rel = file_path.resolve().relative_to(module_base.resolve())
    if rel.suffix != ".py":
        raise ValueError("Target is not a .py file")

    parts = list(rel.parts)
    if parts[-1] == "__init__.py":
        parts = parts[:-1]
    else:
        parts[-1] = parts[-1][:-3]  # remove .py
    return ".".join(parts)


def dotted_from_name_or_attr(node: Optional[cst.CSTNode]) -> str:
    if node is None:
        return ""
    if isinstance(node, cst.Name):
        return node.value
    if isinstance(node, cst.Attribute):
        parts = []
        cur = node
        while isinstance(cur, cst.Attribute):
            if not isinstance(cur.attr, cst.Name):
                return ""
            parts.append(cur.attr.value)
            cur = cur.value
        if isinstance(cur, cst.Name):
            parts.append(cur.value)
            parts.reverse()
            return ".".join(parts)
    return ""


def build_name_or_attr(dotted: str) -> cst.CSTNode:
    parts = dotted.split(".")
    expr: cst.CSTNode = cst.Name(parts[0])
    for p in parts[1:]:
        expr = cst.Attribute(value=expr, attr=cst.Name(p))
    return expr


def flatten_attr(expr: cst.BaseExpression) -> Optional[list[str]]:
    if isinstance(expr, cst.Name):
        return [expr.value]
    if not isinstance(expr, cst.Attribute):
        return None
    parts = []
    cur: cst.BaseExpression = expr
    while isinstance(cur, cst.Attribute):
        if not isinstance(cur.attr, cst.Name):
            return None
        parts.append(cur.attr.value)
        cur = cur.value
    if isinstance(cur, cst.Name):
        parts.append(cur.value)
        parts.reverse()
        return parts
    return None


def build_attr_from_parts(parts: list[str]) -> cst.BaseExpression:
    expr: cst.BaseExpression = cst.Name(parts[0])
    for p in parts[1:]:
        expr = cst.Attribute(value=expr, attr=cst.Name(p))
    return expr


@dataclass(frozen=True)
class MovePlan:
    project_root: Path
    module_base: Path
    old_file: Path
    new_file: Path
    old_mod: str          # e.g. utils.dispatcher
    new_mod: str          # e.g. manager.dispatcher
    old_parent: str       # e.g. utils
    new_parent: str       # e.g. manager
    basename: str         # e.g. dispatcher


def resolve_importfrom_abs_module(
    stmt: cst.ImportFrom,
    file_module: Optional[str],
    is_init: bool,
) -> Optional[str]:
    """
    Возвращает абсолютный module для "from <module> import ...".
    Если импорт относительный — пытаемся разрешить по file_module.
    """
    level = 0 if stmt.relative is None else len(stmt.relative)
    mod_part = dotted_from_name_or_attr(stmt.module)

    if level == 0:
        return mod_part

    # относительный импорт, нужен file_module
    if not file_module:
        return None

    file_parts = file_module.split(".") if file_module else []
    package_base = file_parts if is_init else file_parts[:-1]

    up = level - 1
    if up > len(package_base):
        base = []
    else:
        base = package_base[: len(package_base) - up]

    if mod_part:
        return ".".join(base + mod_part.split("."))
    return ".".join(base)


class RefactorTransformer(cst.CSTTransformer):
    """
    1) Меняет любые Attribute-цепочки с префиксом old_mod (например utils.dispatcher.*)
    2) Правит ImportFrom (включая относительные), плюс умеет делить:
       from utils import dispatcher, other  ->  from manager import dispatcher; from utils import other
    """

    def __init__(self, plan: MovePlan, file_module: Optional[str], file_is_init: bool):
        self.plan = plan
        self.file_module = file_module
        self.file_is_init = file_is_init
        self.changed = False

        self.old_parts = plan.old_mod.split(".")
        self.new_parts = plan.new_mod.split(".")

    def leave_Attribute(self, original_node: cst.Attribute, updated_node: cst.Attribute):
        parts = flatten_attr(updated_node)
        if not parts:
            return updated_node

        if parts[: len(self.old_parts)] == self.old_parts:
            new = self.new_parts + parts[len(self.old_parts):]
            self.changed = True
            return build_attr_from_parts(new)
        return updated_node

    def _rewrite_importfrom(self, stmt: cst.ImportFrom) -> list[cst.ImportFrom]:
        # star import — не трогаем
        if isinstance(stmt.names, cst.ImportStar):
            return [stmt]

        abs_module = resolve_importfrom_abs_module(stmt, self.file_module, self.file_is_init)
        if abs_module is None:
            # не смогли разрешить относительный — оставим как есть
            return [stmt]

        names = list(stmt.names)  # Sequence[ImportAlias]
        moved = []
        other = []
        for a in names:
            if isinstance(a, cst.ImportAlias) and isinstance(a.name, cst.Name) and a.name.value == self.plan.basename:
                moved.append(a)
            else:
                other.append(a)

        # A) from utils.dispatcher import X  -> from manager.dispatcher import X
        #    (и варианты глубже: from utils.dispatcher.sub import X)
        if abs_module == self.plan.old_mod or abs_module.startswith(self.plan.old_mod + "."):
            new_abs = self.plan.new_mod + abs_module[len(self.plan.old_mod):]
            new_stmt = stmt.with_changes(
                relative=None,
                module=build_name_or_attr(new_abs) if new_abs else None,
            )
            if new_stmt != stmt:
                self.changed = True
            return [new_stmt]

        # B) from utils import dispatcher -> from manager import dispatcher
        #    но если там ещё имена — делим на 2 импорта
        if abs_module == self.plan.old_parent and moved:
            out: list[cst.ImportFrom] = []
            # часть с moved уходит в new_parent
            out.append(
                stmt.with_changes(
                    relative=None,
                    module=build_name_or_attr(self.plan.new_parent) if self.plan.new_parent else None,
                    names=tuple(moved),
                )
            )
            # остаток остаётся в old_parent (делаем абсолютным)
            if other:
                out.append(
                    stmt.with_changes(
                        relative=None,
                        module=build_name_or_attr(self.plan.old_parent) if self.plan.old_parent else None,
                        names=tuple(other),
                    )
                )
            self.changed = True
            return out

        # C) относительные импорты, которые попали в B) через abs_module:
        # abs_module уже учтён выше (мы переводим их в absolute через relative=None)
        # если не попали — оставляем
        return [stmt]

    def leave_SimpleStatementLine(self, original_node: cst.SimpleStatementLine, updated_node: cst.SimpleStatementLine):
        # Обрабатываем только простой случай: 1 statement в строке
        if len(updated_node.body) != 1:
            return updated_node

        stmt = updated_node.body[0]
        if not isinstance(stmt, cst.ImportFrom):
            return updated_node

        rewritten = self._rewrite_importfrom(stmt)
        if len(rewritten) == 1:
            return updated_node.with_changes(body=(rewritten[0],))

        # нужно "развернуть" в 2 строки
        # ведущие комментарии/пустые строки — только первой строке
        first = cst.SimpleStatementLine(
            body=(rewritten[0],),
            leading_lines=updated_node.leading_lines,
            trailing_whitespace=cst.TrailingWhitespace(
                whitespace=updated_node.trailing_whitespace.whitespace,
                comment=None,
                newline=updated_node.trailing_whitespace.newline,
            ),
        )
        second = cst.SimpleStatementLine(
            body=(rewritten[1],),
            leading_lines=[],
            trailing_whitespace=updated_node.trailing_whitespace,
        )
        return cst.FlattenSentinel([first, second])


def safe_read_text(p: Path) -> str:
    data = p.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "cp1251"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            pass
    return data.decode("utf-8", errors="replace")


def safe_write_text(p: Path, text: str) -> None:
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)


def iter_py_files(project_root: Path):
    for root, dirs, files in os.walk(project_root):
        # фильтруем каталоги
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for fn in files:
            if fn.endswith(".py"):
                yield Path(root) / fn


def compute_file_module(module_base: Path, file_path: Path) -> tuple[Optional[str], bool]:
    """
    Возвращает (module_name, is_init).
    Если файл вне module_base — module_name=None.
    """
    file_path = file_path.resolve()
    module_base = module_base.resolve()
    try:
        rel = file_path.relative_to(module_base)
    except Exception:
        return None, False

    is_init = (rel.name == "__init__.py")
    mod = file_to_module(module_base, file_path)
    return mod, is_init


def build_plan(target_file: Path, dest_dir: Path) -> MovePlan:
    target_file = target_file.resolve()
    dest_dir = dest_dir.resolve()

    project_root = find_project_root(target_file)
    module_base = pick_module_base(project_root, target_file)

    if not target_file.exists() or not target_file.is_file():
        raise FileNotFoundError(f"Файл не найден: {target_file}")
    if target_file.suffix != ".py":
        raise ValueError("Можно переносить только .py файлы")
    if target_file.name == "__init__.py":
        raise ValueError("Перенос __init__.py намеренно запрещён (слишком много нюансов)")

    # запретим перенос "вне проекта" — иначе невозможно корректно сформировать новый import-путь
    try:
        dest_dir.resolve().relative_to(project_root.resolve())
    except Exception:
        raise ValueError("Целевая директория должна быть внутри проекта (внутри корня)")

    new_file = (dest_dir / target_file.name).resolve()
    if new_file.exists():
        raise FileExistsError(f"В целевой папке уже есть файл: {new_file}")

    old_mod = file_to_module(module_base, target_file)
    new_mod = file_to_module(module_base, new_file)

    if "." not in old_mod or "." not in new_mod:
        # допустимо, но тогда old_parent может быть пустым
        pass

    old_parts = old_mod.split(".")
    new_parts = new_mod.split(".")

    old_parent = ".".join(old_parts[:-1])
    new_parent = ".".join(new_parts[:-1])
    basename = old_parts[-1]

    return MovePlan(
        project_root=project_root,
        module_base=module_base,
        old_file=target_file,
        new_file=new_file,
        old_mod=old_mod,
        new_mod=new_mod,
        old_parent=old_parent,
        new_parent=new_parent,
        basename=basename,
    )


def backup_file(project_root: Path, file_path: Path, backup_root: Path) -> None:
    rel = file_path.resolve().relative_to(project_root.resolve())
    dst = backup_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file_path, dst)


def refactor_project(plan: MovePlan, make_backups: bool, log) -> int:
    changed_files = 0
    backup_root = None

    if make_backups:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_root = plan.project_root / ".move_refactor_backup" / ts
        backup_root.mkdir(parents=True, exist_ok=True)
        log(f"Backup: {backup_root}")

    for py in iter_py_files(plan.project_root):
        text = safe_read_text(py)
        try:
            mod = cst.parse_module(text)
        except Exception:
            log(f"[skip parse] {py}")
            continue

        file_module, is_init = compute_file_module(plan.module_base, py)
        tr = RefactorTransformer(plan, file_module=file_module, file_is_init=is_init)
        new_mod = mod.visit(tr)

        if tr.changed:
            if make_backups and backup_root is not None:
                backup_file(plan.project_root, py, backup_root)
            safe_write_text(py, new_mod.code)
            changed_files += 1
            log(f"[updated] {py}")

    return changed_files


def move_file(plan: MovePlan, create_init_if_missing: bool, log) -> None:
    plan.new_file.parent.mkdir(parents=True, exist_ok=True)

    if create_init_if_missing:
        init_path = plan.new_file.parent / "__init__.py"
        # не навязываем, но часто нужно для пакетных импортов
        if not init_path.exists():
            init_path.write_text("", encoding="utf-8")
            log(f"[created] {init_path}")

    shutil.move(str(plan.old_file), str(plan.new_file))
    log(f"[moved] {plan.old_file} -> {plan.new_file}")


def move_and_refactor(target_file: Path, dest_dir: Path, make_backups: bool, create_init_if_missing: bool, log) -> MovePlan:
    plan = build_plan(target_file, dest_dir)

    log(f"Project root : {plan.project_root}")
    log(f"Module base  : {plan.module_base}")
    log(f"Old module   : {plan.old_mod}")
    log(f"New module   : {plan.new_mod}")

    updated = refactor_project(plan, make_backups=make_backups, log=log)
    log(f"Files updated: {updated}")

    move_file(plan, create_init_if_missing=create_init_if_missing, log=log)
    return plan


# ---------- UI (Tkinter) ----------

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Python Module Mover (перенос модуля с правкой импортов)")
        self.geometry("900x520")

        self.queue: Queue[str] = Queue()

        self.target_var = tk.StringVar()
        self.dest_var = tk.StringVar()
        self.backup_var = tk.BooleanVar(value=True)
        self.init_var = tk.BooleanVar(value=True)

        frm = tk.Frame(self)
        frm.pack(fill="x", padx=10, pady=10)

        # target
        row1 = tk.Frame(frm)
        row1.pack(fill="x", pady=4)
        tk.Label(row1, text="Файл/модуль (.py) для переноса:", width=30, anchor="w").pack(side="left")
        tk.Entry(row1, textvariable=self.target_var).pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(row1, text="Выбрать…", command=self.pick_target).pack(side="left")

        # dest
        row2 = tk.Frame(frm)
        row2.pack(fill="x", pady=4)
        tk.Label(row2, text="Директория назначения:", width=30, anchor="w").pack(side="left")
        tk.Entry(row2, textvariable=self.dest_var).pack(side="left", fill="x", expand=True, padx=6)
        tk.Button(row2, text="Выбрать…", command=self.pick_dest).pack(side="left")

        # options
        row3 = tk.Frame(frm)
        row3.pack(fill="x", pady=6)
        tk.Checkbutton(row3, text="Делать backup изменяемых файлов", variable=self.backup_var).pack(side="left")
        tk.Checkbutton(row3, text="Создать __init__.py в папке назначения, если отсутствует", variable=self.init_var).pack(side="left", padx=16)

        # action
        row4 = tk.Frame(frm)
        row4.pack(fill="x", pady=6)
        self.btn_move = tk.Button(row4, text="Перенести", command=self.on_move, height=2)
        self.btn_move.pack(side="left")

        # log
        self.log = scrolledtext.ScrolledText(self, height=22)
        self.log.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        self.after(100, self.poll_log)

    def pick_target(self):
        path = filedialog.askopenfilename(
            title="Выберите .py файл",
            filetypes=[("Python files", "*.py"), ("All files", "*.*")]
        )
        if path:
            self.target_var.set(path)

    def pick_dest(self):
        path = filedialog.askdirectory(title="Выберите директорию назначения")
        if path:
            self.dest_var.set(path)

    def write_log(self, msg: str):
        self.queue.put(msg)

    def poll_log(self):
        try:
            while True:
                msg = self.queue.get_nowait()
                self.log.insert("end", msg + "\n")
                self.log.see("end")
        except Empty:
            pass
        self.after(100, self.poll_log)

    def on_move(self):
        target = self.target_var.get().strip()
        dest = self.dest_var.get().strip()
        if not target or not dest:
            messagebox.showerror("Ошибка", "Выберите файл и директорию назначения.")
            return

        self.btn_move.config(state="disabled")

        def worker():
            try:
                plan = move_and_refactor(
                    target_file=Path(target),
                    dest_dir=Path(dest),
                    make_backups=bool(self.backup_var.get()),
                    create_init_if_missing=bool(self.init_var.get()),
                    log=self.write_log,
                )
                self.write_log("DONE")
                self.write_log(f"{plan.old_mod}  ->  {plan.new_mod}")
            except Exception as e:
                self.write_log("ERROR:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)))
                messagebox.showerror("Ошибка", str(e))
            finally:
                self.btn_move.config(state="normal")

        threading.Thread(target=worker, daemon=True).start()


if __name__ == "__main__":
    App().mainloop()
"""
Rich terminal output utilities for IntelliRefactor CLI.

Provides beautiful, interactive terminal output with tables, trees, progress bars,
and other rich formatting features.
"""

from typing import Any, Dict, List, Optional, Union
from pathlib import Path
import json

try:
    from rich.console import Console
    from rich.table import Table
    from rich.tree import Tree
    from rich.progress import (
        Progress,
        TaskID,
        SpinnerColumn,
        TextColumn,
        BarColumn,
        TimeElapsedColumn,
    )
    from rich.panel import Panel
   
    from rich.syntax import Syntax
    from rich.columns import Columns
    from rich.align import Align
   
    from rich.prompt import Prompt, Confirm
    from rich.status import Status
   
    from rich.markdown import Markdown

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

    # Fallback console for when rich is not available
    class Console:
        def print(self, *args, **kwargs):
            print(*args)

        def rule(self, *args, **kwargs):
            print("=" * 60)


class RichOutputManager:
    """Manages rich terminal output with fallback to plain text."""

    def __init__(self, use_rich: bool = True):
        """Initialize the output manager."""
        self.use_rich = use_rich and RICH_AVAILABLE
        self.console = Console() if self.use_rich else Console()

    def print_header(self, title: str, subtitle: Optional[str] = None) -> None:
        """Print a formatted header."""
        if self.use_rich:
            if subtitle:
                header_text = f"[bold blue]{title}[/bold blue]\n[dim]{subtitle}[/dim]"
            else:
                header_text = f"[bold blue]{title}[/bold blue]"

            self.console.print(Panel(header_text, border_style="blue", padding=(1, 2)))
        else:
            self.console.print(f"\n=== {title} ===")
            if subtitle:
                self.console.print(f"{subtitle}")
            self.console.print()

    def print_section(self, title: str) -> None:
        """Print a section separator."""
        if self.use_rich:
            self.console.rule(f"[bold]{title}[/bold]", style="blue")
        else:
            self.console.print(f"\n--- {title} ---")

    def print_success(self, message: str) -> None:
        """Print a success message."""
        if self.use_rich:
            self.console.print(f"[green]✓[/green] {message}")
        else:
            self.console.print(f"✓ {message}")

    def print_warning(self, message: str) -> None:
        """Print a warning message."""
        if self.use_rich:
            self.console.print(f"[yellow]⚠[/yellow] {message}")
        else:
            self.console.print(f"⚠ {message}")

    def print_error(self, message: str) -> None:
        """Print an error message."""
        if self.use_rich:
            self.console.print(f"[red]✗[/red] {message}")
        else:
            self.console.print(f"✗ {message}")

    def print_info(self, message: str) -> None:
        """Print an info message."""
        if self.use_rich:
            self.console.print(f"[blue]ℹ[/blue] {message}")
        else:
            self.console.print(f"ℹ {message}")

    def create_table(self, title: str, columns: List[str]) -> "Table":
        """Create a rich table."""
        if self.use_rich:
            table = Table(title=title, show_header=True, header_style="bold blue")
            for column in columns:
                table.add_column(column)
            return table
        else:
            # Return a simple dict-based table for fallback
            return {"title": title, "columns": columns, "rows": []}

    def add_table_row(self, table: Union["Table", Dict], *values) -> None:
        """Add a row to the table."""
        if self.use_rich and hasattr(table, "add_row"):
            table.add_row(*[str(v) for v in values])
        elif isinstance(table, dict):
            table["rows"].append(values)

    def print_table(self, table: Union["Table", Dict]) -> None:
        """Print the table."""
        if self.use_rich and hasattr(table, "add_row"):
            self.console.print(table)
        elif isinstance(table, dict):
            # Fallback table printing
            self.console.print(f"\n{table['title']}")
            self.console.print("-" * len(table["title"]))

            # Print header
            header = " | ".join(table["columns"])
            self.console.print(header)
            self.console.print("-" * len(header))

            # Print rows
            for row in table["rows"]:
                row_str = " | ".join(str(v) for v in row)
                self.console.print(row_str)
            self.console.print()

    def create_tree(self, title: str) -> Union["Tree", Dict]:
        """Create a tree structure."""
        if self.use_rich:
            return Tree(title)
        else:
            return {"title": title, "children": []}

    def add_tree_node(
        self,
        tree: Union["Tree", Dict],
        label: str,
        parent: Optional[Union["Tree", Dict]] = None,
    ) -> Union["Tree", Dict]:
        """Add a node to the tree."""
        if self.use_rich and hasattr(tree, "add"):
            if parent is None:
                return tree.add(label)
            else:
                return parent.add(label)
        elif isinstance(tree, dict):
            node = {"label": label, "children": []}
            if parent is None:
                tree["children"].append(node)
            else:
                parent["children"].append(node)
            return node

    def print_tree(self, tree: Union["Tree", Dict]) -> None:
        """Print the tree."""
        if self.use_rich and hasattr(tree, "add"):
            self.console.print(tree)
        elif isinstance(tree, dict):
            self._print_tree_fallback(tree, 0)

    def _print_tree_fallback(self, node: Dict, indent: int) -> None:
        """Print tree in fallback mode."""
        prefix = "  " * indent
        if indent == 0:
            self.console.print(f"{node['title']}")
        else:
            self.console.print(f"{prefix}├── {node['label']}")

        for child in node.get("children", []):
            self._print_tree_fallback(child, indent + 1)

    def create_progress(self, description: str = "Processing...") -> Union["Progress", Dict]:
        """Create a progress bar."""
        if self.use_rich:
            return Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                console=self.console,
            )
        else:
            return {"description": description, "total": 0, "completed": 0}

    def add_progress_task(
        self, progress: Union["Progress", Dict], description: str, total: int
    ) -> Union[TaskID, str]:
        """Add a task to the progress bar."""
        if self.use_rich and hasattr(progress, "add_task"):
            return progress.add_task(description, total=total)
        elif isinstance(progress, dict):
            task_id = f"task_{len(progress.get('tasks', []))}"
            if "tasks" not in progress:
                progress["tasks"] = {}
            progress["tasks"][task_id] = {
                "description": description,
                "total": total,
                "completed": 0,
            }
            return task_id

    def update_progress(
        self,
        progress: Union["Progress", Dict],
        task_id: Union[TaskID, str],
        advance: int = 1,
    ) -> None:
        """Update progress."""
        if self.use_rich and hasattr(progress, "update"):
            progress.update(task_id, advance=advance)
        elif isinstance(progress, dict) and "tasks" in progress:
            if task_id in progress["tasks"]:
                progress["tasks"][task_id]["completed"] += advance
                task = progress["tasks"][task_id]
                percentage = (task["completed"] / task["total"]) * 100 if task["total"] > 0 else 0
                self.console.print(f"\r{task['description']}: {percentage:.1f}%", end="")

    def print_json(self, data: Any, title: Optional[str] = None) -> None:
        """Print JSON data with syntax highlighting."""
        if title:
            self.print_section(title)

        if self.use_rich:
            json_str = json.dumps(data, indent=2, default=str)
            syntax = Syntax(json_str, "json", theme="monokai", line_numbers=True)
            self.console.print(syntax)
        else:
            self.console.print(json.dumps(data, indent=2, default=str))

    def print_code(self, code: str, language: str = "python", title: Optional[str] = None) -> None:
        """Print code with syntax highlighting."""
        if title:
            self.print_section(title)

        if self.use_rich:
            syntax = Syntax(code, language, theme="monokai", line_numbers=True)
            self.console.print(syntax)
        else:
            self.console.print(code)

    def print_markdown(self, markdown_text: str) -> None:
        """Print markdown text."""
        if self.use_rich:
            md = Markdown(markdown_text)
            self.console.print(md)
        else:
            self.console.print(markdown_text)

    def prompt(self, message: str, default: Optional[str] = None) -> str:
        """Prompt user for input."""
        if self.use_rich:
            return Prompt.ask(message, default=default)
        else:
            prompt_text = f"{message}"
            if default:
                prompt_text += f" [{default}]"
            prompt_text += ": "

            response = input(prompt_text).strip()
            return response if response else (default or "")

    def confirm(self, message: str, default: bool = False) -> bool:
        """Ask user for confirmation."""
        if self.use_rich:
            return Confirm.ask(message, default=default)
        else:
            default_text = "Y/n" if default else "y/N"
            response = input(f"{message} [{default_text}]: ").strip().lower()

            if not response:
                return default
            return response in ["y", "yes", "true", "1"]

    def status(self, message: str) -> Union["Status", Dict]:
        """Create a status spinner."""
        if self.use_rich:
            return Status(message, console=self.console)
        else:
            self.console.print(f"{message}...")
            return {"message": message}

    def print_metrics_summary(self, metrics: Dict[str, Any]) -> None:
        """Print a formatted metrics summary."""
        self.print_section("Metrics Summary")

        if self.use_rich:
            # Create a grid of metric panels
            panels = []
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    if isinstance(value, float):
                        value_str = f"{value:.2f}"
                    else:
                        value_str = str(value)

                    panel = Panel(
                        Align.center(f"[bold green]{value_str}[/bold green]"),
                        title=key.replace("_", " ").title(),
                        border_style="green",
                    )
                    panels.append(panel)

            if panels:
                columns = Columns(panels, equal=True, expand=True)
                self.console.print(columns)
        else:
            for key, value in metrics.items():
                self.console.print(f"{key.replace('_', ' ').title()}: {value}")

    def print_file_tree(self, root_path: Path, max_depth: int = 3) -> None:
        """Print a file tree structure."""
        if not root_path.exists():
            self.print_error(f"Path does not exist: {root_path}")
            return

        tree = self.create_tree(f"📁 {root_path.name}")
        self._build_file_tree(tree, root_path, 0, max_depth)
        self.print_tree(tree)

    def _build_file_tree(
        self,
        parent_node: Union["Tree", Dict],
        path: Path,
        current_depth: int,
        max_depth: int,
    ) -> None:
        """Recursively build file tree."""
        if current_depth >= max_depth:
            return

        try:
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            for item in items:
                if item.name.startswith("."):
                    continue

                if item.is_dir():
                    icon = "📁"
                    node = self.add_tree_node(parent_node, f"{icon} {item.name}")
                    self._build_file_tree(node, item, current_depth + 1, max_depth)
                else:
                    icon = "📄"
                    if item.suffix == ".py":
                        icon = "🐍"
                    elif item.suffix in [".md", ".txt"]:
                        icon = "📝"
                    elif item.suffix in [".json", ".yaml", ".yml"]:
                        icon = "⚙️"

                    self.add_tree_node(parent_node, f"{icon} {item.name}")
        except PermissionError:
            self.add_tree_node(parent_node, "❌ Permission denied")


# Global instance
rich_output = RichOutputManager()


def set_rich_enabled(enabled: bool) -> None:
    """Enable or disable rich output globally."""
    global rich_output
    rich_output = RichOutputManager(use_rich=enabled)


def get_rich_output() -> RichOutputManager:
    """Get the global rich output manager."""
    return rich_output

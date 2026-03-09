from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, Iterable

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

_console = Console()


def print_banner(title: str, subtitle: str | None = None) -> None:
    text = title if subtitle is None else f"{title}\n[dim]{subtitle}[/dim]"
    _console.print(Panel(text, expand=False))


def print_kv(title: str, items: Dict[str, str]) -> None:
    table = Table(title=title, show_header=False, box=None)
    table.add_column("Key", style="bold")
    table.add_column("Value")
    for key, value in items.items():
        table.add_row(str(key), str(value))
    _console.print(table)


def print_stats_table(title: str, rows: Iterable[tuple[str, str]]) -> None:
    table = Table(title=title)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    for metric, value in rows:
        table.add_row(metric, value)
    _console.print(table)


@contextmanager
def progress_status(label: str):
    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
    )
    task_id = progress.add_task(label, start=True)
    with progress:
        yield
        progress.update(task_id, completed=1)

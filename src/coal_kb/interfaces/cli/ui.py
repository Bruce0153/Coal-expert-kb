"""集中提供 CLI 标题、键值和统计表格输出。"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


_CONSOLE = Console()


def print_banner(title: str, subtitle: str | None = None) -> None:
    """输出统一命令行标题。"""
    body = subtitle or ""
    _CONSOLE.print(Panel(body, title=title, expand=False))


def print_kv(title: str, values: Mapping[str, object]) -> None:
    """以两列表格输出配置键值。"""
    table = Table(title=title, show_header=False)
    table.add_column("Key")
    table.add_column("Value")
    for key, value in values.items():
        table.add_row(str(key), str(value))
    _CONSOLE.print(table)


def print_stats_table(title: str, rows: Iterable[tuple[str, object]]) -> None:
    """以两列表格输出运行统计。"""
    table = Table(title=title)
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in rows:
        table.add_row(str(key), str(value))
    _CONSOLE.print(table)

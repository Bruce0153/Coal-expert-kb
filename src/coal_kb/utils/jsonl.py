"""提供 JSONL 的稳定读取、迭代和写出函数。"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any


def iter_jsonl(path: str | Path) -> Iterator[tuple[int, Any]]:
    """逐行解析 JSONL，并返回从 1 开始的原始行号。"""
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if line.strip():
                yield row_number, json.loads(line)


def read_jsonl(path: str | Path) -> list[Any]:
    """读取 JSONL 中的全部非空记录。"""
    return [payload for _, payload in iter_jsonl(path)]


def write_jsonl(path: str | Path, rows: Iterable[Any], *, sort_keys: bool = True) -> None:
    """以 UTF-8 和结尾换行写出稳定 JSONL。"""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = [
        json.dumps(row, ensure_ascii=False, sort_keys=sort_keys)
        for row in rows
    ]
    destination.write_text(
        "\n".join(serialized) + ("\n" if serialized else ""),
        encoding="utf-8",
    )

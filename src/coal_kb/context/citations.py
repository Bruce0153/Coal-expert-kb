"""构建引用标签、证据条目和来源展示信息。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from coal_kb.context import config


def snippet(text: str, max_chars: int = config.DEFAULT_SNIPPET_CHARS) -> str:
    collapsed = " ".join((text or "").split())
    if len(collapsed) <= max_chars:
        return collapsed
    return collapsed[: max_chars - 3].rstrip() + "..."


def source_id(metadata: dict[str, Any]) -> str:
    source_file = str(metadata.get("source_file") or "unknown")
    title = str(metadata.get("title") or "").strip() or Path(source_file).stem
    return f"{title}|{source_file}"


def display_name(metadata: dict[str, Any]) -> str:
    source_file = str(metadata.get("source_file") or "unknown")
    title = str(metadata.get("title") or "").strip() or Path(source_file).name
    parts = [title]
    page = metadata.get("page")
    if page is not None:
        parts.append(f"page {page}")
    heading = str(metadata.get("heading_path") or "").strip()
    if heading:
        parts.append(heading)
    return " | ".join(parts)

"""解析回答中的证据标签并渲染引用列表。"""

from __future__ import annotations

import re


def extract_referenced_labels(text: str, available_labels: list[str]) -> list[str]:
    found = re.findall(r"\[(E\d+)\]", text)
    result = list(dict.fromkeys(found))
    return result or available_labels


def build_rendered_citations(
    citations: dict[str, dict],
    referenced_labels: list[str],
) -> list[str]:
    result: list[str] = []
    for label in referenced_labels:
        item = citations.get(label)
        if item is None:
            continue
        source = item.get("source_file", "unknown")
        page = item.get("page")
        if page is not None:
            result.append(f"[{label}] {source} (page {page})")
        else:
            result.append(f"[{label}] {source}")
    return result

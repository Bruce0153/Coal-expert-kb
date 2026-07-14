"""提供 Markdown 文本换行、连字符和重复页眉页脚清洗。"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def fix_hyphenation(text: str) -> str:
    return re.sub(r"([A-Za-z]{2,})-\n([A-Za-z]{2,})", r"\1\2", text)


def _should_merge(prev: str, cur: str) -> bool:
    if prev.endswith((".", "!", "?", ":", ";", "。", "！", "？", "：", "；")):
        return False
    if re.match(r"^(#|\-|\*|\+|\d+[\.)])\s+", cur):
        return False
    if cur and cur[0].isupper() and prev.endswith(")"):
        return False
    return True


def merge_wrapped_lines(text: str) -> str:
    lines = text.split("\n")
    output: List[str] = []
    for line in lines:
        current = line.strip()
        if not current:
            output.append("")
            continue
        if not output or not output[-1]:
            output.append(current)
            continue
        previous = output[-1]
        if _should_merge(previous, current):
            separator = "" if re.search(r"[\u4e00-\u9fff]$", previous) else " "
            output[-1] = f"{previous}{separator}{current}"
        else:
            output.append(current)
    return "\n".join(output)


def normalize_bullets(text: str) -> str:
    lines = text.split("\n")
    output: List[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^[•·‣◦]\s*", stripped):
            output.append("- " + re.sub(r"^[•·‣◦]\s*", "", stripped))
        elif re.match(r"^\(?\d+\)|^\d+\.\s+", stripped):
            output.append(re.sub(r"^\(?([0-9]+)\)\s*", r"\1. ", stripped))
        else:
            output.append(line)
    return "\n".join(output)


def collapse_repeated_headers(pages: Iterable[str], *, min_repeat: int = 2) -> List[str]:
    page_lines = [page.splitlines() for page in pages]
    header_counter: Counter[str] = Counter()
    footer_counter: Counter[str] = Counter()
    for lines in page_lines:
        if lines:
            header_counter[lines[0].strip()] += 1
            footer_counter[lines[-1].strip()] += 1
    noisy_headers = {text for text, count in header_counter.items() if text and count >= min_repeat}
    noisy_footers = {text for text, count in footer_counter.items() if text and count >= min_repeat}

    cleaned: List[str] = []
    for lines in page_lines:
        if lines and lines[0].strip() in noisy_headers:
            lines = lines[1:]
        if lines and lines[-1].strip() in noisy_footers:
            lines = lines[:-1]
        cleaned.append("\n".join(lines).strip())
    return cleaned

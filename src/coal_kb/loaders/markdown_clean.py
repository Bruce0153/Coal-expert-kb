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


def merge_wrapped_lines(text: str) -> str:
    lines = text.split("\n")
    out: List[str] = []
    for line in lines:
        cur = line.strip()
        if not cur:
            out.append("")
            continue
        if not out or not out[-1]:
            out.append(cur)
            continue
        prev = out[-1]
        if _should_merge(prev, cur):
            sep = "" if re.search(r"[\u4e00-\u9fff]$", prev) else " "
            out[-1] = f"{prev}{sep}{cur}"
        else:
            out.append(cur)
    return "\n".join(out)


def _should_merge(prev: str, cur: str) -> bool:
    if prev.endswith((".", "!", "?", ":", ";", "。", "！", "？", "：", "；")):
        return False
    if re.match(r"^(#|\-|\*|\+|\d+[\.)])\s+", cur):
        return False
    if cur and cur[0].isupper() and prev.endswith(")"):
        return False
    return True


def normalize_bullets(text: str) -> str:
    lines = text.split("\n")
    out: List[str] = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^[•·‣◦]\s*", stripped):
            out.append("- " + re.sub(r"^[•·‣◦]\s*", "", stripped))
        elif re.match(r"^\(?\d+\)|^\d+\.\s+", stripped):
            out.append(re.sub(r"^\(?([0-9]+)\)\s*", r"\1. ", stripped))
        else:
            out.append(line)
    return "\n".join(out)


def collapse_repeated_headers(pages: Iterable[str], *, min_repeat: int = 2) -> List[str]:
    page_lines = [p.splitlines() for p in pages]
    header_counter = Counter()
    footer_counter = Counter()
    for lines in page_lines:
        if lines:
            header_counter[lines[0].strip()] += 1
            footer_counter[lines[-1].strip()] += 1
    noisy_headers = {t for t, c in header_counter.items() if t and c >= min_repeat}
    noisy_footers = {t for t, c in footer_counter.items() if t and c >= min_repeat}

    cleaned: List[str] = []
    for lines in page_lines:
        if lines and lines[0].strip() in noisy_headers:
            lines = lines[1:]
        if lines and lines[-1].strip() in noisy_footers:
            lines = lines[:-1]
        cleaned.append("\n".join(lines).strip())
    return cleaned

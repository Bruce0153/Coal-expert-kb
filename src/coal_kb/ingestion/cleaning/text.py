"""提供文档文本清洗、断行修复和页眉页脚去除。"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List, Set

_WS_RE = re.compile(r"[ \t\r\f\v]+")
_HYPHEN_LINEBREAK_RE = re.compile(r"(\w)-\n(\w)")


def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [_WS_RE.sub(" ", ln).strip() for ln in text.split("\n")]
    cleaned = "\n".join([ln for ln in lines if ln != ""])
    return cleaned.strip()


def repair_hyphenation(text: str) -> str:
    return _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)


def basic_clean(text: str) -> str:
    return normalize_whitespace(repair_hyphenation(text))


def _norm_line(s: str) -> str:
    return _WS_RE.sub(" ", s).strip()


def find_common_header_footer_lines(
    page_texts: List[str],
    *,
    head_n: int = 3,
    tail_n: int = 3,
    min_ratio: float = 0.6,
    min_len: int = 6,
    max_len: int = 120,
) -> Set[str]:
    """
    Detect common header/footer lines by counting only first/last N lines of each page.
    """
    if not page_texts:
        return set()

    cnt = Counter()
    total_pages = len(page_texts)

    for text in page_texts:
        lines = [_norm_line(x) for x in text.split("\n") if _norm_line(x)]
        heads = lines[:head_n]
        tails = lines[-tail_n:] if tail_n > 0 else []
        for ln in heads + tails:
            if len(ln) < min_len or len(ln) > max_len:
                continue
            cnt[ln] += 1

    common = set()
    for ln, c in cnt.items():
        if c >= 2 and (c / total_pages) >= min_ratio:
            common.add(ln)
    return common


def remove_common_header_footer(
    text: str,
    common_lines: Set[str],
    *,
    head_n: int = 3,
    tail_n: int = 3,
) -> str:
    """
    Remove common lines only if they appear in header/footer positions.
    """
    if not common_lines:
        return text

    lines = [_norm_line(x) for x in text.split("\n") if _norm_line(x)]
    if not lines:
        return text

    def keep_line(i: int, ln: str) -> bool:
        if (i < head_n) and (ln in common_lines):
            return False
        if (i >= len(lines) - tail_n) and (ln in common_lines):
            return False
        return True

    kept = [ln for i, ln in enumerate(lines) if keep_line(i, ln)]
    return "\n".join(kept).strip()


# ---- line merging & bullet normalization ----

def _should_merge_lines(prev: str, cur: str) -> bool:
    if prev.endswith((".", "!", "?", ":", ";", "。", "！", "？", "：", "；")):
        return False
    if cur and cur[0].isupper() and prev.endswith(")"):
        return False
    return True


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
        if _should_merge_lines(prev, cur):
            sep = "" if re.search(r"[一-鿿]$", prev) else " "
            out[-1] = f"{prev}{sep}{cur}"
        else:
            out.append(cur)
    return "\n".join(out)


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
    header_counter: Counter = Counter()
    footer_counter: Counter = Counter()
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

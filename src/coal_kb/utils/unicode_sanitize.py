# src/coal_kb/utils/unicode_sanitize.py
from __future__ import annotations

import re
import unicodedata
from typing import Any, Tuple, Dict, List


_SURROGATE_MIN = 0xD800
_SURROGATE_MAX = 0xDFFF

# 常见“零宽”/方向控制字符，容易影响检索与显示
_ZERO_WIDTH_RE = re.compile(
    r"[\u200B\u200C\u200D\u200E\u200F\u202A-\u202E\u2060\uFEFF]"
)

# 统一空白（含不换行空格、窄空格等）
_WEIRD_SPACES_RE = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")


def _is_high_surrogate(cp: int) -> bool:
    return 0xD800 <= cp <= 0xDBFF


def _is_low_surrogate(cp: int) -> bool:
    return 0xDC00 <= cp <= 0xDFFF


def _combine_surrogates(high: int, low: int) -> int:
    return 0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00)


def repair_surrogates(text: str, *, replacement: str = "\uFFFD") -> str:
    """
    修复 UTF-16 surrogate pair（如果成对则组合成真实字符），
    孤立 surrogate 用 replacement 替换（默认 �），避免 UTF-8 写文件崩溃。
    """
    # 快路径：没有 surrogate
    if not any(_SURROGATE_MIN <= ord(ch) <= _SURROGATE_MAX for ch in text):
        return text

    out: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        cp = ord(text[i])
        if _is_high_surrogate(cp) and i + 1 < n:
            cp2 = ord(text[i + 1])
            if _is_low_surrogate(cp2):
                out.append(chr(_combine_surrogates(cp, cp2)))
                i += 2
                continue
            out.append(replacement)
            i += 1
            continue
        if _is_low_surrogate(cp):
            out.append(replacement)
            i += 1
            continue
        out.append(text[i])
        i += 1

    return "".join(out)


def normalize_for_retrieval(text: str) -> str:
    """
    让文本“更易检索”：
    1) 修复/替换 surrogate
    2) NFKC 归一化：把数学花体/粗体/黑板体等折叠成普通字母数字
    3) 移除零宽字符
    4) 统一奇怪空格为普通空格、并折叠多空白
    """
    t = repair_surrogates(text)

    # NFKC 会把很多兼容字符（尤其数学字母数字）折叠成普通 ASCII
    t = unicodedata.normalize("NFKC", t)

    # 去掉零宽/方向控制符
    t = _ZERO_WIDTH_RE.sub("", t)

    # 把奇怪空格换成普通空格
    t = _WEIRD_SPACES_RE.sub(" ", t)

    # 折叠多空白（保留换行结构的话可以改成只折叠空格）
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)

    t = "".join(ch if not (0xD800 <= ord(ch) <= 0xDFFF) else "\uFFFD" for ch in t)

    return t


def sanitize_obj(obj: Any) -> Tuple[Any, Dict[str, int]]:
    stats = {"strings_seen": 0, "strings_changed": 0}

    def _walk(x: Any) -> Any:
        if isinstance(x, str):
            stats["strings_seen"] += 1
            y = normalize_for_retrieval(x)
            if y != x:
                stats["strings_changed"] += 1
            return y
        if isinstance(x, list):
            return [_walk(v) for v in x]
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        return x

    return _walk(obj), stats

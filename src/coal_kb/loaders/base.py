from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List

from langchain_core.documents import Document


_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def detect_language(text: str) -> str:
    cjk = len(_CJK_RE.findall(text))
    latin = len(_LATIN_RE.findall(text))
    if cjk and latin:
        return "mixed"
    if cjk >= max(latin * 2, 1):
        return "zh"
    return "en"


def normalize_text(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass
class BaseLoader:
    def load(self, path: str) -> List[Document]:
        raise NotImplementedError

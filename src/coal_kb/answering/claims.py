"""将回答文本拆分为声明并关联证据标签。"""

from __future__ import annotations

import re
from typing import Any

from coal_kb.answering import config


def build_claim_items(text: str, referenced_labels: list[str]) -> list[dict[str, Any]]:
    if not text:
        return []
    sentences = re.split(r"(?<=[。.!?！？])\s*", text)
    claims: list[dict[str, Any]] = []
    for index, sentence in enumerate(sentences):
        sentence = sentence.strip()
        if not sentence or len(sentence) < config.MIN_CLAIM_CHARS:
            continue
        labels = list(dict.fromkeys(re.findall(r"\[(E\d+)\]", sentence)))
        claims.append({
            "claim_id": f"C{index + 1}",
            "text": sentence,
            "citations": labels,
            "support": "direct" if labels else "contextual",
        })
    if not claims:
        claims.append({
            "claim_id": "C1",
            "text": text[: config.FALLBACK_CLAIM_CHARS],
            "citations": referenced_labels,
            "support": "contextual",
        })
    return claims[: config.MAX_CLAIMS]

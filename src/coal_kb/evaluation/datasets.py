"""读取评估数据并生成手工标注模板。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from coal_kb.evaluation import config


@dataclass
class EvalItem:
    """保存问题和最小来源标注。"""

    question: str
    gold_sources: list[dict[str, Any]]


def load_eval_set(path: str) -> list[EvalItem]:
    """从 JSONL 加载评估问题，行为与旧实现一致。"""
    source = Path(path)
    items: list[EvalItem] = []
    for line in source.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        items.append(EvalItem(question=obj["question"], gold_sources=obj.get("gold_sources") or []))
    return items


def save_eval_template(path: str) -> None:
    """创建一行人工标注 JSONL 模板。"""
    destination = Path(path)
    sample = {
        "question": config.TEMPLATE_QUESTION,
        "gold_sources": [
            {"source_contains": config.TEMPLATE_SOURCE, "page": config.TEMPLATE_PAGE}
        ],
    }
    destination.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

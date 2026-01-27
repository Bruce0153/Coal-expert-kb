from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvalItem:
    question: str
    # Minimal ground truth: list of (source_file_contains, optional page)
    gold_sources: List[Dict[str, Any]]  # e.g. [{"source_contains":"paper1","page":12}]


def load_eval_set(path: str) -> List[EvalItem]:
    p = Path(path)
    items: List[EvalItem] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        items.append(EvalItem(question=obj["question"], gold_sources=obj.get("gold_sources") or []))
    return items


def save_eval_template(path: str) -> None:
    """
    Create a template JSONL for manual labeling.
    """
    p = Path(path)
    sample = {
        "question": "在蒸汽气化条件下，NH3与HCN的生成趋势如何？给出证据。",
        "gold_sources": [{"source_contains": "your_paper_name.pdf", "page": 5}],
    }
    p.write_text(json.dumps(sample, ensure_ascii=False) + "\n", encoding="utf-8")

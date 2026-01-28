from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


from coal_kb.metadata.normalize import Ontology
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.retriever import ExpertRetriever
from coal_kb.settings import load_config
from coal_kb.store.chroma_store import ChromaStore
from coal_kb.embeddings.factory import EmbeddingsConfig


@dataclass
class EvalItem:
    question: str
    gold_sources: List[Dict[str, Any]]


def load_eval_set(path: Path) -> List[EvalItem]:
    items: List[EvalItem] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        items.append(EvalItem(question=obj["question"], gold_sources=obj.get("gold_sources") or []))
    return items


def match_gold(gold: Dict[str, Any], meta: Dict[str, Any]) -> bool:
    contains = str(gold.get("source_contains", "")).lower()
    page = gold.get("page", None)
    src = str(meta.get("source_file", "")).lower()
    if contains and contains not in src:
        return False
    if page is not None:
        return meta.get("page") == page
    return True


def range_overlap(meta: Dict[str, Any], query_range: Optional[List[float]], *, key_point: str, key_min: str, key_max: str) -> bool:
    if query_range is None:
        return True
    qlo, qhi = float(query_range[0]), float(query_range[1])
    dmin = meta.get(key_min)
    dmax = meta.get(key_max)
    if dmin is not None and dmax is not None:
        return max(float(dmin), qlo) <= min(float(dmax), qhi)
    x = meta.get(key_point)
    if x is None:
        return False
    return qlo <= float(x) <= qhi


def filter_match(meta: Dict[str, Any], parsed: Dict[str, Any]) -> Dict[str, bool]:
    checks: Dict[str, bool] = {}
    stage = parsed.get("stage")
    if stage and stage != "unknown":
        checks["stage"] = str(meta.get("stage")) == stage

    gas = parsed.get("gas_agent") or []
    if gas:
        checks["gas_agent"] = any(meta.get(f"gas_{g}") for g in gas)

    targets = parsed.get("targets") or []
    if targets:
        checks["targets"] = any(meta.get(f"has_{t}") for t in targets)

    T_range = parsed.get("T_range_K")
    if T_range:
        checks["T_range_K"] = range_overlap(meta, T_range, key_point="T_K", key_min="T_min_K", key_max="T_max_K")

    P_range = parsed.get("P_range_MPa")
    if P_range:
        checks["P_range_MPa"] = range_overlap(meta, P_range, key_point="P_MPa", key_min="P_min_MPa", key_max="P_max_MPa")

    return checks


def recall_at_k(docs: List[Dict[str, Any]], gold_sources: List[Dict[str, Any]], k: int) -> bool:
    for d in docs[:k]:
        meta = d
        if any(match_gold(g, meta) for g in gold_sources):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval Recall@K and filter correctness.")
    parser.add_argument("--gold", default="data/eval/retrieval_gold.jsonl")
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config()
    onto = Ontology.load("configs/schema.yaml")
    parser_ = FilterParser(onto=onto)

    store = ChromaStore(
        persist_dir=cfg.paths.chroma_dir,
        collection_name=cfg.chroma.collection_name,
        embeddings_cfg=EmbeddingsConfig(**cfg.embeddings.model_dump()),
        embedding_model=cfg.embedding.model_name,
    )

    expert = ExpertRetriever(
        vector_retriever_factory=store.as_retriever,
        k=args.k,
        rerank_enabled=cfg.retrieval.rerank_enabled,
        rerank_model=cfg.retrieval.rerank_model,
        rerank_top_k=cfg.retrieval.rerank_top_k,
    )

    items = load_eval_set(Path(args.gold))

    recalls = {1: 0, 3: 0, 5: 0}
    filter_counts: Dict[str, int] = {}
    filter_hits: Dict[str, int] = {}

    for item in items:
        parsed = parser_.parse(item.question)
        docs = expert.retrieve(item.question, parsed)
        meta_list = [d.metadata or {} for d in docs]

        for k in (1, 3, 5):
            if recall_at_k(meta_list, item.gold_sources, k):
                recalls[k] += 1

        if meta_list:
            checks = filter_match(meta_list[0], parsed)
            for key, ok in checks.items():
                filter_counts[key] = filter_counts.get(key, 0) + 1
                if ok:
                    filter_hits[key] = filter_hits.get(key, 0) + 1

    total = max(len(items), 1)
    rows = [
        ["Recall@1", f"{recalls[1]}/{total}", f"{recalls[1] / total:.2f}"],
        ["Recall@3", f"{recalls[3]}/{total}", f"{recalls[3] / total:.2f}"],
        ["Recall@5", f"{recalls[5]}/{total}", f"{recalls[5] / total:.2f}"],
    ]

    for key in sorted(filter_counts.keys()):
        hit = filter_hits.get(key, 0)
        cnt = filter_counts[key]
        rows.append([f"Filter@1 {key}", f"{hit}/{cnt}", f"{hit / max(cnt, 1):.2f}"])

    headers = ["Metric", "Count", "Score"]
    col_widths = [max(len(str(row[i])) for row in ([headers] + rows)) for i in range(3)]

    def format_row(row: List[str]) -> str:
        return " | ".join(str(row[i]).ljust(col_widths[i]) for i in range(3))

    sep = "-|-".join("-" * w for w in col_widths)
    print(format_row(headers))
    print(sep)
    for row in rows:
        print(format_row(row))


if __name__ == "__main__":
    main()

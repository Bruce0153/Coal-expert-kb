from __future__ import annotations

"""Unified evaluation runner for retrieval/faithfulness/extraction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from coal_kb.eval.datasets import load_eval_set
from coal_kb.metadata.normalize import Ontology
from coal_kb.query.planner import QueryPlanner
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.retriever import ExpertRetriever


@dataclass
class EvalRunner:
    planner: QueryPlanner
    retriever: ExpertRetriever

    def run(self, *, task: str, gold_path: str, k: int = 5) -> Dict[str, float]:
        if task != "retrieval":
            return {"status": 0.0}

        items = load_eval_set(Path(gold_path))
        hit = 0
        parent_hit = 0
        for item in items:
            plan = self.planner.build_plan(item.query)
            trace: Dict[str, object] = {}
            docs = self.retriever.execute(plan, trace=trace)
            meta_list = [d.metadata or {} for d in docs[:k]]
            if any(any((g.get("source_file") == m.get("source_file")) for g in item.expected_sources) for m in meta_list):
                hit += 1
            if trace.get("stage1_parent_hits", 0):
                parent_hit += 1
        total = max(len(items), 1)
        return {
            "recall@k": hit / total,
            "parents_recall": parent_hit / total,
            "n": float(total),
        }


def make_default_planner(cfg) -> QueryPlanner:
    onto = Ontology.load("configs/schema.yaml")
    parser = FilterParser(onto=onto)
    return QueryPlanner(cfg=cfg, parser=parser)

"""验证研究实验复用统一评估并写出实验清单。"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.documents import Document

from coal_kb.research import ExperimentSpec, ResearchExperiment, ResearchRoute


def test_research_experiment_writes_route_manifest(tmp_path: Path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "case-1",
                "query": "steam gasification",
                "query_type": "fact",
                "expected_answer": "",
                "expected_evidence": [{"source_file": "paper.pdf", "page": 2}],
                "expected_filters": {},
                "answerable": True,
                "metadata": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "output"

    def retrieve(case, k):
        return [
            Document(
                page_content=case.query,
                metadata={"chunk_id": "c1", "source_file": "paper.pdf", "page": 2},
            )
        ][:k], {"research_route": {"route": "graph"}}

    manifest = ResearchExperiment(
        spec=ExperimentSpec(
            name="graph-smoke",
            route=ResearchRoute.GRAPH,
            dataset_path=dataset,
            output_dir=output,
            k_values=(1,),
            metadata={"purpose": "test"},
        ),
        retrieve_fn=retrieve,
    ).process()

    saved = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    assert manifest["route"] == "graph"
    assert saved["experiment_id"] == manifest["experiment_id"]
    assert saved["metadata"]["purpose"] == "test"
    assert (output / "manifest.json").exists()
    assert (output / "metrics.json").exists()

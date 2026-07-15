"""验证统一 Evaluation Pipeline 的数据、指标和产物。"""

from __future__ import annotations

import json

from langchain_core.documents import Document

from coal_kb.evaluation import AnswerObservation, ClaimObservation, EvaluationPipeline, EvidenceReference
from coal_kb.evaluation.datasets import load_evaluation_cases


def test_evaluation_pipeline_writes_versioned_artifacts(tmp_path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "id": "case_1",
                "query": "steam gasification NH3",
                "query_type": "fact",
                "expected_evidence": [{"source_file": "paper.pdf", "page": 2, "chunk_id": "c1"}],
                "answerable": True,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    def retrieve(case, k):
        assert case.case_id == "case_1"
        assert k == 3
        return [Document(page_content="NH3 evidence", metadata={"source_file": "paper.pdf", "page": 2, "chunk_id": "c1"})]

    def answer(case, documents):
        assert documents
        return AnswerObservation(
            answer_text="Supported claim [E1]",
            citations=(EvidenceReference(source_file="paper.pdf", page=2, chunk_id="c1"),),
            claims=(ClaimObservation(text="Supported claim", citations=("E1",), supported=True),),
            abstained=False,
        )

    output = tmp_path / "output"
    metrics = EvaluationPipeline(
        dataset_path=dataset,
        output_dir=output,
        retrieve_fn=retrieve,
        answer_fn=answer,
        k_values=(1, 3),
        run_metadata={"backend": "fake"},
    ).process()

    assert metrics["retrieval"]["recall_at_1"] == 1.0
    assert metrics["answer"]["citation_precision"] == 1.0
    assert (output / "metrics.json").exists()
    assert (output / "case_results.jsonl").exists()
    assert (output / "failures.jsonl").read_text(encoding="utf-8") == ""
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["backend"] == "fake"
    assert manifest["case_count"] == 1


def test_dataset_accepts_previous_question_and_gold_source_fields(tmp_path) -> None:
    dataset = tmp_path / "cases.jsonl"
    dataset.write_text(
        '{"question":"q","gold_sources":[{"source_contains":"paper.pdf","page":1}]}\n',
        encoding="utf-8",
    )
    cases = load_evaluation_cases(dataset)
    assert cases[0].query == "q"
    assert cases[0].expected_evidence[0].source_file == "paper.pdf"

"""运行统一 RAG 评估 Pipeline 并生成版本化报告。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from coal_kb.application.ask import build_runtime
from coal_kb.evaluation import (
    AnswerObservation,
    ClaimObservation,
    EvaluationCase,
    EvaluationPipeline,
    EvidenceReference,
    config,
)
from coal_kb.infra.config import AppConfig, load_config


@dataclass
class Evaluate:
    """组装当前运行时并执行统一评估。"""

    cfg: AppConfig
    dataset_path: Path
    output_dir: Path
    k_values: tuple[int, ...]
    enable_answers: bool

    def _retrieve(self, case: EvaluationCase, k: int):
        runtime = self._runtime
        plan = runtime.planner.build_plan(case.query, self.cfg, enable_llm=False, llm_config=None)
        trace = {}
        documents = runtime.complex_question_service.process(plan, trace=trace)[:k]
        return documents, trace

    def _answer(self, case: EvaluationCase, documents) -> AnswerObservation:
        runtime = self._runtime
        plan = runtime.planner.build_plan(case.query, self.cfg, enable_llm=False, llm_config=None)
        context = runtime.context_builder.build(plan, documents)
        result = runtime.answerer.answer(plan, context)
        citations = tuple(
            EvidenceReference(
                source_file=item.get("source_file"),
                document_id=item.get("document_id"),
                page=item.get("page"),
                section=item.get("heading_path") or item.get("section"),
                chunk_id=item.get("chunk_id"),
                text_span=item.get("snippet"),
            )
            for item in result.citations.values()
        )
        claims = tuple(
            ClaimObservation(
                text=str(item.get("text") or ""),
                citations=tuple(item.get("citations") or ()),
                supported=str(item.get("support") or "").lower() in {"direct", "supported", "partial"},
            )
            for item in result.claim_items
        )
        return AnswerObservation(
            answer_text=result.answer_text,
            citations=citations,
            claims=claims,
            abstained=result.evidence_sufficiency == "insufficient",
        )

    def process(self) -> dict:
        self._runtime = build_runtime(
            self.cfg,
            backend=self.cfg.backend,
            k=max(self.k_values),
            enable_llm=self.enable_answers,
        )
        pipeline = EvaluationPipeline(
            dataset_path=self.dataset_path,
            output_dir=self.output_dir,
            retrieve_fn=self._retrieve,
            answer_fn=self._answer if self.enable_answers else None,
            k_values=self.k_values,
            run_metadata={
                "backend": self.cfg.backend,
                "embedding_model": self.cfg.embeddings.model,
                "embedding_version": self.cfg.model_versions.embedding_version,
                "answers_enabled": self.enable_answers,
            },
        )
        return pipeline.process()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the unified RAG evaluation pipeline.")
    parser.add_argument("--dataset", default=config.DEFAULT_DATASET_PATH)
    parser.add_argument("--output-dir", default=config.DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k", default="1,3,5,10", help="Comma-separated retrieval cutoffs.")
    parser.add_argument("--answers", action="store_true", help="Also evaluate answer citations and claims.")
    args = parser.parse_args()
    k_values = tuple(sorted({int(value) for value in args.k.split(",") if value.strip()}))
    if not k_values:
        raise ValueError("At least one positive k value is required")
    Evaluate(
        cfg=load_config(),
        dataset_path=Path(args.dataset),
        output_dir=Path(args.output_dir),
        k_values=k_values,
        enable_answers=args.answers,
    ).process()


if __name__ == "__main__":
    main()

# 运行命令：python scripts/evaluate.py --dataset data/eval/evaluation_sample.jsonl

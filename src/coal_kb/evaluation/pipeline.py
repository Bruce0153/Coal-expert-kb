"""执行复杂问答评估案例、计算指标并生成版本化报告。"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from langchain_core.documents import Document
from tqdm import tqdm

from coal_kb.evaluation import config
from coal_kb.evaluation.attribution import classify_failure
from coal_kb.evaluation.datasets import load_evaluation_cases
from coal_kb.evaluation.metrics import (
    aggregate_metrics,
    answer_metrics,
    complex_question_metrics,
    retrieval_metrics,
)
from coal_kb.evaluation.models import (
    AnswerObservation,
    CaseEvaluationResult,
    EvaluationCase,
    RetrievedEvidence,
)
from coal_kb.evaluation.reporting import EvaluationReportWriter

RetrieveOutput = list[Document] | tuple[list[Document], dict[str, Any]]
RetrieveFunction = Callable[[EvaluationCase, int], RetrieveOutput]
AnswerFunction = Callable[[EvaluationCase, list[Document]], AnswerObservation]


@dataclass
class EvaluationPipeline:
    """持有运行依赖并以 process() 执行完整评估。"""

    dataset_path: Path
    output_dir: Path
    retrieve_fn: RetrieveFunction
    answer_fn: AnswerFunction | None = None
    k_values: tuple[int, ...] = config.DEFAULT_K_VALUES
    run_metadata: dict[str, Any] | None = None

    def process(self) -> dict[str, Any]:
        cases = load_evaluation_cases(self.dataset_path)
        max_k = max(self.k_values)
        results: list[CaseEvaluationResult] = []
        for case in tqdm(cases, total=len(cases), desc=self.__class__.__name__):
            started_at = time.monotonic()
            output = self.retrieve_fn(case, max_k)
            if isinstance(output, tuple):
                documents, trace = output
            else:
                documents, trace = output, {}
            retrieved = tuple(
                RetrievedEvidence.from_document(document, rank=rank)
                for rank, document in enumerate(documents[:max_k], start=1)
            )
            answer = self.answer_fn(case, documents) if self.answer_fn else None
            retrieval_values = retrieval_metrics(case, retrieved, self.k_values)
            answer_values = answer_metrics(case, answer)
            complex_values = complex_question_metrics(case, trace, retrieved)
            failure = classify_failure(case, retrieval_values, answer_values, complex_values)
            results.append(
                CaseEvaluationResult(
                    case_id=case.case_id,
                    query=case.query,
                    query_type=case.query_type.value,
                    retrieval_metrics=retrieval_values,
                    answer_metrics=answer_values,
                    complex_metrics=complex_values,
                    failure_category=failure,
                    latency_ms=round((time.monotonic() - started_at) * 1000, 3),
                    retrieved=retrieved,
                    trace=trace,
                )
            )
        metrics = {
            "retrieval": aggregate_metrics([result.retrieval_metrics for result in results]),
            "complex": aggregate_metrics([result.complex_metrics for result in results]),
            "answer": aggregate_metrics([result.answer_metrics for result in results]),
            "by_query_type": self._by_query_type(results),
        }
        EvaluationReportWriter(self.output_dir).process(
            metrics=metrics,
            results=results,
            manifest={
                "dataset_path": str(self.dataset_path),
                "case_count": len(cases),
                "k_values": list(self.k_values),
                **(self.run_metadata or {}),
            },
        )
        return metrics

    @staticmethod
    def _by_query_type(results: list[CaseEvaluationResult]) -> dict[str, dict[str, float]]:
        grouped: dict[str, list[CaseEvaluationResult]] = {}
        for result in results:
            grouped.setdefault(result.query_type, []).append(result)
        output: dict[str, dict[str, float]] = {}
        for query_type, items in sorted(grouped.items()):
            output[query_type] = {
                "case_count": float(len(items)),
                **aggregate_metrics([item.complex_metrics for item in items]),
            }
        return output

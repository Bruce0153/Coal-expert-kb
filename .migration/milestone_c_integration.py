"""接入 Milestone C 运行时、回答提示、评估指标、配置和文档。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path.cwd()


def _write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _replace(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    if old not in text:
        raise ValueError(f"未找到替换目标: {path}: {old[:100]}")
    target.write_text(text.replace(old, new), encoding="utf-8")


def process() -> None:
    app_path = ROOT / "configs/app.yaml"
    app_text = app_path.read_text(encoding="utf-8")
    if "\ncomplex_qa:\n" not in app_text:
        block = '''\ncomplex_qa:\n  enabled: true\n  max_subqueries: 4\n  max_multi_hop_steps: 3\n  comparison_k_per_side: 4\n  cross_document_min_sources: 2\n  cross_document_max_per_source: 2\n  aggregation_record_limit: 500\n  aggregation_evidence_limit: 12\n  table_records_path: "data/interim/table_records.jsonl"\n  table_top_k: 5\n  base_context_tokens: 2400\n  base_evidence_chunks: 10\n'''
        app_text = app_text.replace("\ntokenizer:\n", block + "\ntokenizer:\n")
    app_path.write_text(app_text, encoding="utf-8")

    ask_path = ROOT / "src/coal_kb/application/ask.py"
    ask_text = ask_path.read_text(encoding="utf-8")
    ask_text = ask_text.replace(
        "    from coal_kb.context import ContextBuilder\n",
        "    from coal_kb.complex_qa import ComplexQuestionService\n    from coal_kb.context import ContextBuilder\n",
    )
    ask_text = ask_text.replace(
        "    context_builder: ContextBuilder\n    answerer: Answerer\n",
        "    context_builder: ContextBuilder\n    complex_question_service: ComplexQuestionService\n    answerer: Answerer\n",
    )
    ask_text = ask_text.replace(
        "    from coal_kb.answering import Answerer\n    from coal_kb.context import ContextBuilder\n",
        "    from coal_kb.answering import Answerer\n    from coal_kb.complex_qa import ComplexQuestionService\n    from coal_kb.context import ContextBuilder\n",
    )
    ask_text = ask_text.replace(
        "        context_builder=ContextBuilder(token_counter=make_tokenizer(cfg.tokenizer).count_tokens),\n        answerer=Answerer(enable_llm=enable_llm and llm_config is not None, llm_config=llm_config),\n",
        "        context_builder=ContextBuilder(token_counter=make_tokenizer(cfg.tokenizer).count_tokens),\n        complex_question_service=ComplexQuestionService(\n            retriever=retriever,\n            sqlite_path=cfg.paths.sqlite_path,\n            table_records_path=cfg.complex_qa.table_records_path,\n            comparison_k_per_side=cfg.complex_qa.comparison_k_per_side,\n            max_multi_hop_steps=cfg.complex_qa.max_multi_hop_steps,\n            aggregation_record_limit=cfg.complex_qa.aggregation_record_limit,\n            aggregation_evidence_limit=cfg.complex_qa.aggregation_evidence_limit,\n            table_top_k=cfg.complex_qa.table_top_k,\n            cross_document_min_sources=cfg.complex_qa.cross_document_min_sources,\n            cross_document_max_per_source=cfg.complex_qa.cross_document_max_per_source,\n        ),\n        answerer=Answerer(enable_llm=enable_llm and llm_config is not None, llm_config=llm_config),\n",
    )
    ask_text = ask_text.replace(
        "    docs = runtime.retriever.execute(plan, trace=trace)\n",
        "    docs = runtime.complex_question_service.process(plan, trace=trace)\n",
    )
    ask_path.write_text(ask_text, encoding="utf-8")

    _write(
        "src/coal_kb/answering/prompts.py",
        '''"""集中维护事实检索和复杂科学问答的回答 Prompt。"""

from __future__ import annotations

_ROUTE_INSTRUCTIONS = {
    "comparison": "按共同条件、差异条件、相同点和主要差异组织答案；两侧证据不足时不要强行比较。",
    "multi_hop": "按证据链顺序解释中间过程，每一步都要引用对应证据，并明确链路缺口。",
    "aggregation": "直接采用证据目录中的程序计算结果，报告样本量和筛选范围，不要由模型重新心算。",
    "table": "优先给出表格标题、行列或单元格值及单位，并明确对应页码。",
    "cross_document": "区分主要共识、不同结论、条件差异和证据覆盖范围，不要把同一文档的多个片段视为多篇文献。",
    "unanswerable": "明确说明知识库无法证实，不得猜测。",
    "fact": "先给出直接结论，再解释适用条件和证据局限。",
}


def build_answer_prompt(user_question: str, context_markdown: str, *, query_type: str = "fact") -> str:
    """根据问题路线生成受证据约束的中文回答 Prompt。"""
    route_instruction = _ROUTE_INSTRUCTIONS.get(query_type, _ROUTE_INSTRUCTIONS["fact"])
    return f"""你是一个面向煤热解、气化和燃烧领域的科研问答助手。

请严格基于下面提供的证据片段回答用户问题，要求：
1. 只能依据给出的证据回答，不要编造文献中没有的信息。
2. 回答中必须保留引用标记，例如 [E1] [E2]，并把引用放在对应结论句末。
3. 如果证据之间存在阶段、工况、煤种、反应器或单位差异，要明确区分。
4. 如果证据不足以支持强结论，要明确说明证据边界。
5. 输出用中文和 Markdown，不要输出空泛的上下文提示语。
6. 不要捏造不存在的引用编号。
7. 当前问题路线为 `{query_type}`：{route_instruction}

用户问题：
{user_question}

证据片段：
{context_markdown}

请先给出总括结论，再按当前路线要求组织证据、适用条件和局限。
"""
''',
    )

    answer_path = ROOT / "src/coal_kb/answering/service.py"
    answer_text = answer_path.read_text(encoding="utf-8")
    answer_text = answer_text.replace(
        "        prompt = build_answer_prompt(user_question, context_package.markdown)\n",
        "        prompt = build_answer_prompt(\n            user_question,\n            context_package.markdown,\n            query_type=plan.complex.query_type,\n        )\n",
    )
    answer_path.write_text(answer_text, encoding="utf-8")

    _write(
        "src/coal_kb/evaluation/models.py",
        '''"""定义复杂科学问答评估数据、运行观察和结果模型。"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class QueryType(StrEnum):
    """评估问题类型。"""

    FACT = "fact"
    CONDITION = "condition"
    COMPARISON = "comparison"
    MULTI_HOP = "multi_hop"
    AGGREGATION = "aggregation"
    TABLE = "table"
    CROSS_DOCUMENT = "cross_document"
    UNANSWERABLE = "unanswerable"


@dataclass(frozen=True)
class EvidenceReference:
    """保存可追溯到原始文档的证据标注。"""

    source_file: str | None = None
    document_id: str | None = None
    page: int | None = None
    section: str | None = None
    chunk_id: str | None = None
    text_span: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EvidenceReference:
        return cls(
            source_file=value.get("source_file") or value.get("source_contains"),
            document_id=value.get("document_id"),
            page=value.get("page"),
            section=value.get("section"),
            chunk_id=value.get("chunk_id"),
            text_span=value.get("text_span"),
        )


@dataclass(frozen=True)
class EvaluationCase:
    """表示一条版本化复杂科学问答评估样本。"""

    case_id: str
    query: str
    query_type: QueryType = QueryType.FACT
    expected_answer: str | None = None
    expected_evidence: tuple[EvidenceReference, ...] = ()
    expected_filters: dict[str, Any] = field(default_factory=dict)
    expected_subqueries: tuple[str, ...] = ()
    expected_operation: str | None = None
    expected_min_sources: int = 1
    expected_table_ids: tuple[str, ...] = ()
    answerable: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, row_number: int) -> EvaluationCase:
        evidence_values = value.get("expected_evidence") or value.get("gold_sources") or []
        case_id = str(value.get("id") or value.get("case_id") or f"case_{row_number:05d}")
        query = str(value.get("query") or value.get("question") or "").strip()
        if not query:
            raise ValueError(f"Evaluation case {case_id} has an empty query")
        raw_type = str(value.get("query_type") or QueryType.FACT.value)
        if raw_type == "global":
            raw_type = QueryType.CROSS_DOCUMENT.value
        return cls(
            case_id=case_id,
            query=query,
            query_type=QueryType(raw_type),
            expected_answer=value.get("expected_answer"),
            expected_evidence=tuple(EvidenceReference.from_dict(item) for item in evidence_values),
            expected_filters=dict(value.get("expected_filters") or {}),
            expected_subqueries=tuple(str(item) for item in value.get("expected_subqueries") or ()),
            expected_operation=value.get("expected_operation"),
            expected_min_sources=max(1, int(value.get("expected_min_sources", 1))),
            expected_table_ids=tuple(str(item) for item in value.get("expected_table_ids") or ()),
            answerable=bool(value.get("answerable", True)),
            metadata=dict(value.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["id"] = payload.pop("case_id")
        payload["query_type"] = self.query_type.value
        payload["expected_evidence"] = list(payload["expected_evidence"])
        payload["expected_subqueries"] = list(payload["expected_subqueries"])
        payload["expected_table_ids"] = list(payload["expected_table_ids"])
        return payload


@dataclass(frozen=True)
class RetrievedEvidence:
    """保存一个排序后的检索结果。"""

    rank: int
    source_file: str | None
    document_id: str | None
    page: int | None
    section: str | None
    chunk_id: str | None
    text: str
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_document(cls, document: Any, *, rank: int) -> RetrievedEvidence:
        metadata = dict(document.metadata or {})
        return cls(
            rank=rank,
            source_file=metadata.get("source_file"),
            document_id=metadata.get("document_id"),
            page=metadata.get("page"),
            section=metadata.get("section") or metadata.get("heading_path"),
            chunk_id=metadata.get("chunk_id"),
            text=document.page_content or "",
            score=metadata.get("score") or metadata.get("retrieval_score"),
            metadata=metadata,
        )


@dataclass(frozen=True)
class ClaimObservation:
    """保存回答中的一个 Claim 及其证据状态。"""

    text: str
    citations: tuple[str, ...] = ()
    supported: bool = False


@dataclass(frozen=True)
class AnswerObservation:
    """保存回答评估所需的结构化输出。"""

    answer_text: str
    citations: tuple[EvidenceReference, ...] = ()
    claims: tuple[ClaimObservation, ...] = ()
    abstained: bool = False


@dataclass(frozen=True)
class EvaluationObservation:
    """保存一次案例执行产生的检索与回答观察。"""

    retrieved: tuple[RetrievedEvidence, ...]
    answer: AnswerObservation | None = None
    trace: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


@dataclass(frozen=True)
class CaseEvaluationResult:
    """保存一条案例的全部指标和失败归因。"""

    case_id: str
    query: str
    query_type: str
    retrieval_metrics: dict[str, float]
    answer_metrics: dict[str, float]
    complex_metrics: dict[str, float]
    failure_category: str
    latency_ms: float
    retrieved: tuple[RetrievedEvidence, ...]
    trace: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
''',
    )

    _write(
        "src/coal_kb/evaluation/metrics.py",
        '''"""计算检索、复杂路线、引用、Claim 和拒答指标。"""

from __future__ import annotations

import math
import re

from coal_kb.evaluation.models import AnswerObservation, EvidenceReference, EvaluationCase, RetrievedEvidence


def evidence_matches(expected: EvidenceReference, actual: RetrievedEvidence | EvidenceReference) -> bool:
    """按最具体的可用字段判断证据是否匹配。"""
    if expected.chunk_id:
        return expected.chunk_id == actual.chunk_id
    if expected.document_id and expected.document_id != actual.document_id:
        return False
    if expected.source_file:
        source = (actual.source_file or "").lower()
        if expected.source_file.lower() not in source:
            return False
    if expected.page is not None and expected.page != actual.page:
        return False
    if expected.section:
        section = (actual.section or "").lower()
        if expected.section.lower() not in section:
            return False
    return bool(expected.document_id or expected.source_file or expected.page is not None)


def retrieval_metrics(case: EvaluationCase, retrieved: tuple[RetrievedEvidence, ...], k_values: tuple[int, ...]) -> dict[str, float]:
    """计算分层检索指标。"""
    expected = case.expected_evidence
    metrics: dict[str, float] = {}
    first_relevant_rank: int | None = None
    relevance = [1 if any(evidence_matches(gold, item) for gold in expected) else 0 for item in retrieved]
    for index, relevant in enumerate(relevance, start=1):
        if relevant and first_relevant_rank is None:
            first_relevant_rank = index
    metrics["mrr"] = 1.0 / first_relevant_rank if first_relevant_rank else 0.0
    for k in k_values:
        prefix = retrieved[:k]
        matched_gold = {
            gold_index
            for gold_index, gold in enumerate(expected)
            if any(evidence_matches(gold, item) for item in prefix)
        }
        relevant_count = sum(1 for item in prefix if any(evidence_matches(gold, item) for gold in expected))
        metrics[f"hit_at_{k}"] = 1.0 if matched_gold else 0.0
        metrics[f"recall_at_{k}"] = len(matched_gold) / len(expected) if expected else 1.0
        metrics[f"precision_at_{k}"] = relevant_count / len(prefix) if prefix else 0.0
        gains = relevance[:k]
        dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
        ideal_hits = min(len(expected), k)
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
        metrics[f"ndcg_at_{k}"] = dcg / ideal_dcg if ideal_dcg else 1.0
    expected_sources = {item.source_file.lower() for item in expected if item.source_file}
    actual_sources = {(item.source_file or "").lower() for item in retrieved if item.source_file}
    metrics["source_recall"] = (
        sum(1 for source in expected_sources if any(source in actual for actual in actual_sources)) / len(expected_sources)
        if expected_sources
        else 1.0
    )
    expected_pages = {((item.source_file or "").lower(), item.page) for item in expected if item.page is not None}
    actual_pages = {((item.source_file or "").lower(), item.page) for item in retrieved}
    metrics["page_recall"] = len(expected_pages & actual_pages) / len(expected_pages) if expected_pages else 1.0
    return metrics


def complex_question_metrics(case: EvaluationCase, trace: dict, retrieved: tuple[RetrievedEvidence, ...]) -> dict[str, float]:
    """计算路由、分解、证据链、表格和跨文档指标。"""
    route = trace.get("complex_route") or trace.get("plan") or {}
    execution = trace.get("complex_execution") or {}
    actual_type = str(route.get("query_type") or execution.get("query_type") or "fact")
    expected_type = "fact" if case.query_type == case.query_type.CONDITION else case.query_type.value
    metrics = {"route_accuracy": float(actual_type == expected_type)}

    actual_subqueries = [str(item.get("query") or "") for item in route.get("subqueries") or []]
    if case.expected_subqueries:
        matched = sum(
            1
            for expected in case.expected_subqueries
            if any(_text_overlap(expected, actual) >= 0.5 for actual in actual_subqueries)
        )
        metrics["subquery_recall"] = matched / len(case.expected_subqueries)

    if case.expected_operation:
        actual_operation = ((route.get("aggregation") or {}).get("operation") or execution.get("operation"))
        metrics["aggregation_operation_accuracy"] = float(actual_operation == case.expected_operation)

    sources = {item.source_file for item in retrieved if item.source_file and item.source_file != "computed_aggregation"}
    metrics["source_diversity"] = float(len(sources))
    if case.query_type == case.query_type.CROSS_DOCUMENT:
        metrics["cross_document_coverage"] = min(1.0, len(sources) / case.expected_min_sources)
    if case.query_type == case.query_type.COMPARISON:
        sides = {str(item.metadata.get("complex_role")) for item in retrieved if item.metadata.get("complex_role")}
        metrics["comparison_side_coverage"] = min(1.0, len(sides) / 2)
    if case.query_type == case.query_type.MULTI_HOP:
        steps = execution.get("steps") or []
        metrics["chain_completeness"] = float(bool(steps) and all(int(step.get("hits", 0)) > 0 for step in steps))
    if case.query_type == case.query_type.TABLE:
        table_hits = [item for item in retrieved if item.metadata.get("table_id")]
        metrics["table_evidence_rate"] = len(table_hits) / len(retrieved) if retrieved else 0.0
        if case.expected_table_ids:
            actual_ids = {str(item.metadata.get("table_id")) for item in table_hits}
            metrics["table_id_recall"] = sum(1 for value in case.expected_table_ids if value in actual_ids) / len(case.expected_table_ids)
    return metrics


def answer_metrics(case: EvaluationCase, answer: AnswerObservation | None) -> dict[str, float]:
    """计算引用、Claim 支撑和拒答指标。"""
    if answer is None:
        return {}
    expected = case.expected_evidence
    matched_citations = sum(1 for citation in answer.citations if any(evidence_matches(gold, citation) for gold in expected))
    citation_precision = matched_citations / len(answer.citations) if answer.citations else (1.0 if not expected else 0.0)
    matched_gold = sum(1 for gold in expected if any(evidence_matches(gold, citation) for citation in answer.citations))
    citation_recall = matched_gold / len(expected) if expected else 1.0
    unsupported = sum(1 for claim in answer.claims if not claim.supported)
    unsupported_rate = unsupported / len(answer.claims) if answer.claims else 0.0
    abstention_correct = float(answer.abstained == (not case.answerable))
    return {
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "claim_support_rate": 1.0 - unsupported_rate,
        "unsupported_claim_rate": unsupported_rate,
        "abstention_correct": abstention_correct,
    }


def aggregate_metrics(results: list[dict[str, float]]) -> dict[str, float]:
    """对案例级指标做宏平均。"""
    keys = sorted({key for result in results for key in result})
    return {
        key: sum(result[key] for result in results if key in result) / sum(1 for result in results if key in result)
        for key in keys
    }


def _text_overlap(left: str, right: str) -> float:
    left_terms = set(re.findall(r"[a-z0-9_.+-]+|[\u4e00-\u9fff]", left.lower()))
    right_terms = set(re.findall(r"[a-z0-9_.+-]+|[\u4e00-\u9fff]", right.lower()))
    return len(left_terms & right_terms) / max(1, len(left_terms))
''',
    )

    _write(
        "src/coal_kb/evaluation/attribution.py",
        '''"""根据案例指标生成稳定的复杂问答失败归因。"""

from __future__ import annotations

from coal_kb.evaluation import config
from coal_kb.evaluation.models import EvaluationCase


def classify_failure(
    case: EvaluationCase,
    retrieval: dict[str, float],
    answer: dict[str, float],
    complex_values: dict[str, float] | None = None,
) -> str:
    """按照最靠前的失败环节进行归因。"""
    complex_values = complex_values or {}
    if complex_values.get("route_accuracy", 1.0) < 1.0:
        return "ROUTE"
    if complex_values.get("subquery_recall", 1.0) < 1.0:
        return "DECOMPOSITION"
    if complex_values.get("aggregation_operation_accuracy", 1.0) < 1.0:
        return "AGGREGATION"
    if complex_values.get("table_id_recall", 1.0) < 1.0:
        return "TABLE"
    if complex_values.get("cross_document_coverage", 1.0) < 1.0:
        return "CROSS_DOCUMENT"
    if complex_values.get("chain_completeness", 1.0) < 1.0:
        return "MULTI_HOP"
    if retrieval.get("hit_at_10", retrieval.get("hit_at_5", 0.0)) == 0.0 and case.expected_evidence:
        return "RECALL"
    if retrieval.get("recall_at_10", retrieval.get("recall_at_5", 1.0)) < 1.0:
        return "EVIDENCE_COVERAGE"
    if answer:
        if answer.get("citation_precision", 1.0) < 1.0:
            return "CITATION"
        if answer.get("unsupported_claim_rate", 0.0) > 0.0:
            return "GENERATION"
        if answer.get("abstention_correct", 1.0) < 1.0:
            return "ABSTENTION"
    if case.expected_filters and retrieval.get("hit_at_1", 0.0) == 0.0:
        return "QUERY_PLAN"
    return config.FAILURE_NONE
''',
    )

    _write(
        "src/coal_kb/evaluation/pipeline.py",
        '''"""执行复杂问答评估案例、计算指标并生成版本化报告。"""

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
from coal_kb.evaluation.metrics import aggregate_metrics, answer_metrics, complex_question_metrics, retrieval_metrics
from coal_kb.evaluation.models import AnswerObservation, CaseEvaluationResult, EvaluationCase, RetrievedEvidence
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
''',
    )

    report_path = ROOT / "src/coal_kb/evaluation/reporting.py"
    report_text = report_path.read_text(encoding="utf-8")
    report_text = report_text.replace('for group_name in ("retrieval", "answer"):', 'for group_name in ("retrieval", "complex", "answer"):')
    report_text = report_text.replace(
        '            for key, value in sorted(values.items()):\n                lines.append(f"- `{key}`: {value:.4f}")\n',
        '            for key, value in sorted(values.items()):\n                if isinstance(value, (int, float)):\n                    lines.append(f"- `{key}`: {value:.4f}")\n',
    )
    report_path.write_text(report_text, encoding="utf-8")

    evaluate_path = ROOT / "scripts/evaluate.py"
    evaluate_text = evaluate_path.read_text(encoding="utf-8")
    evaluate_text = evaluate_text.replace(
        "        plan = runtime.planner.build_plan(case.query, self.cfg, enable_llm=False, llm_config=None)\n        return runtime.retriever.execute(plan, trace={})[:k]\n",
        "        plan = runtime.planner.build_plan(case.query, self.cfg, enable_llm=False, llm_config=None)\n        trace = {}\n        documents = runtime.complex_question_service.process(plan, trace=trace)[:k]\n        return documents, trace\n",
    )
    evaluate_path.write_text(evaluate_text, encoding="utf-8")

    _write(
        "data/eval/complex_science_sample.jsonl",
        '''{"id":"comparison_0001","query":"比较蒸汽气化与CO2气化对H2/CO比的影响","query_type":"comparison","expected_evidence":[{"source_file":"steam_gasification_sample.pdf"},{"source_file":"co2_gasification_sample.pdf"}],"expected_subqueries":["蒸汽气化 实验条件 结果","CO2气化 实验条件 结果"],"expected_min_sources":2,"answerable":true,"metadata":{"sample_only":true}}
{"id":"comparison_0002","query":"高温与低温热解对焦油组成有何不同","query_type":"comparison","expected_evidence":[{"source_file":"pyrolysis_temperature_sample.pdf"}],"expected_subqueries":["高温 实验条件 结果","低温热解 实验条件 结果"],"answerable":true,"metadata":{"sample_only":true}}
{"id":"multi_hop_0001","query":"高温蒸汽气化为什么会通过水煤气反应提高氢气产率","query_type":"multi_hop","expected_evidence":[{"source_file":"mechanism_sample.pdf"}],"expected_subqueries":["关键反应 中间过程","中间过程 对目标结果的影响"],"answerable":true,"metadata":{"sample_only":true}}
{"id":"multi_hop_0002","query":"催化剂如何通过焦油裂解路径降低焦油产率","query_type":"multi_hop","expected_evidence":[{"source_file":"catalyst_tar_sample.pdf"}],"expected_subqueries":["关键反应 中间过程","机制链"],"answerable":true,"metadata":{"sample_only":true}}
{"id":"aggregation_0001","query":"结构化实验记录中的平均气化温度是多少","query_type":"aggregation","expected_operation":"average","expected_evidence":[],"answerable":true,"metadata":{"sample_only":true}}
{"id":"aggregation_0002","query":"列出H2产率最高的前5条实验记录","query_type":"aggregation","expected_operation":"top_k","expected_evidence":[],"answerable":true,"metadata":{"sample_only":true}}
{"id":"table_0001","query":"表格中蒸汽流量对应的H2产率是多少","query_type":"table","expected_evidence":[{"source_file":"table_sample.pdf","section":"table"}],"expected_table_ids":["table_001"],"answerable":true,"metadata":{"sample_only":true}}
{"id":"table_0002","query":"Table 2中最高温度一行的NH3数值是多少","query_type":"table","expected_evidence":[{"source_file":"table_sample.pdf","section":"table"}],"expected_table_ids":["table_002"],"answerable":true,"metadata":{"sample_only":true}}
{"id":"cross_document_0001","query":"多篇文献对压力影响NH3生成的主要共识和冲突是什么","query_type":"cross_document","expected_evidence":[{"source_file":"pressure_study_a.pdf"},{"source_file":"pressure_study_b.pdf"}],"expected_subqueries":["支持性证据","相反结果","实验条件差异"],"expected_min_sources":2,"answerable":true,"metadata":{"sample_only":true}}
{"id":"cross_document_0002","query":"不同研究对催化气化降低焦油的总体结论是否一致","query_type":"cross_document","expected_evidence":[{"source_file":"catalyst_a.pdf"},{"source_file":"catalyst_b.pdf"}],"expected_min_sources":2,"answerable":true,"metadata":{"sample_only":true}}
{"id":"fact_0001","query":"煤气化的主要气化剂有哪些","query_type":"fact","expected_evidence":[{"source_file":"gasification_review_sample.pdf"}],"answerable":true,"metadata":{"sample_only":true}}
{"id":"condition_0001","query":"只考虑1200K蒸汽气化条件下的NH3生成","query_type":"condition","expected_evidence":[{"source_file":"condition_sample.pdf"}],"expected_filters":{"stage":"gasification","T_range_K":[1140,1260]},"answerable":true,"metadata":{"sample_only":true}}
{"id":"unanswerable_0001","query":"未公开的私人实验具体操作压力是多少","query_type":"unanswerable","expected_evidence":[],"answerable":false,"metadata":{"sample_only":true}}
''',
    )

    _write(
        "docs/architecture/complex_science_qa.md",
        '''# Milestone C：复杂科学问答

## 统一运行链

```text
用户问题
  → QueryPlanner
  → ComplexQuestionSpec
  → ComplexQuestionService
  → Comparison / Multi-hop / Aggregation / Table / Cross-document
  → Document 证据
  → ContextBuilder
  → Answerer
```

复杂路线不会建立平行的 Context 或 Answering 实现。所有执行器都返回标准 `Document`，因此引用、Token 预算、Reranker 和 UI 继续使用现有正式链路。

## C0 复杂问答评估集

`EvaluationCase` 支持：

- `expected_subqueries`：预期子问题；
- `expected_operation`：预期聚合操作；
- `expected_min_sources`：跨文档最少来源数；
- `expected_table_ids`：预期表格；
- 比较、多跳、聚合、表格、跨文档和不可回答类型。

样例文件为 `data/eval/complex_science_sample.jsonl`。样例仅用于格式和离线测试，正式实验必须替换为人工核验的真实文献标注。

## C1 比较问题

每个比较对象独立生成子查询和检索预算。证据通过 `comparison_entity` 和 `complex_role` 标记，避免只有一侧证据时形成伪比较。

## C2 多跳问题

最多执行配置中的受限步骤。每一跳保存查询、依赖、桥接术语和命中数量；禁止无限检索循环。

## C3 统计聚合

聚合只读取 SQLite 结构化实验记录，由 Python 执行 `count`、`sum`、`average`、`median`、`min`、`max`、`group_by` 和 `top_k`。LLM 只解释程序结果，不负责心算。

## C4 表格问题

标准表格资产使用 JSONL：

```json
{"table_id":"table_001","source_file":"paper.pdf","page":7,"caption":"...","headers":["T_K","H2"],"rows":[{"T_K":1200,"H2":42.1}],"nearby_text":"..."}
```

表格路线返回带 `table_id`、页码和命中行的证据。没有表格资产时才回退到文档检索。

## C5 跨文档综合

分别检索支持、冲突和条件差异证据，并限制每个来源的最大证据数。Trace 记录真实来源数量及是否满足最少来源要求。

## C6 统一路由

路由类型：

```text
fact
comparison
multi_hop
aggregation
table
cross_document
unanswerable
```

第一版使用可解释规则；每个计划保存置信度、路由原因、子问题和结构化操作，便于回放和评估。

## 配置

`configs/app.yaml` 的 `complex_qa` 控制子问题数量、多跳步数、聚合记录数、表格路径、跨文档来源数和上下文预算。
''',
    )

    readme_path = ROOT / "README.md"
    readme_text = readme_path.read_text(encoding="utf-8")
    if "complex_science_qa.md" not in readme_text:
        readme_text = readme_text.replace(
            "- `docs/architecture/evaluation_operations.md`\n",
            "- `docs/architecture/evaluation_operations.md`\n- `docs/architecture/complex_science_qa.md`\n",
        )
    readme_path.write_text(readme_text, encoding="utf-8")


if __name__ == "__main__":
    process()

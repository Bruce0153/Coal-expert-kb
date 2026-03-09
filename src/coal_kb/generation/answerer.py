from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from coal_kb.context.types import ContextPackage
from coal_kb.query.plan import QueryPlan


@dataclass
class AnswerResult:
    answer_text: str
    citations: Dict[str, dict]
    used_chunks: List[str]
    debug: Dict[str, Any]


class Answerer:
    def answer(self, plan: QueryPlan, context_package: ContextPackage) -> AnswerResult:
        ev_count = len(context_package.used_chunks)
        if ev_count < plan.answer.min_evidence:
            return AnswerResult(
                answer_text="无法可靠回答：证据不足。请补充更明确的工况/目标污染物证据。",
                citations={k: v.model_dump() for k, v in context_package.citations.items()},
                used_chunks=context_package.used_chunks,
                debug={"reason": "insufficient_evidence", "evidence": ev_count},
            )

        if plan.answer.output_format == "json":
            text = '{"summary":"见 evidence", "citations": %s}' % list(context_package.citations.keys())
        else:
            refs = " ".join(f"[{k}]" for k in context_package.citations.keys())
            text = f"基于检索证据，结论如下：\n\n- 关键结论请结合证据片段核验 {refs}\n"
        return AnswerResult(
            answer_text=text,
            citations={k: v.model_dump() for k, v in context_package.citations.items()},
            used_chunks=context_package.used_chunks,
            debug={"context_debug": context_package.debug},
        )

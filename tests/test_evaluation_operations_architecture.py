"""验证评估、可观测性、安全与运维分层。"""

from pathlib import Path

import pytest
from langchain_core.documents import Document

from coal_kb.evaluation import EvalItem, RetrievalEvaluator, simple_faithfulness_check
from coal_kb.infra.observability.trace import build_retrieval_trace_summary
from coal_kb.infra.security.uploads import build_upload_path, safe_upload_name
from coal_kb.operations import health_status

ROOT = Path(__file__).resolve().parents[1] / "src" / "coal_kb"


def test_evaluation_public_api_is_canonical() -> None:
    assert EvalItem.__module__ == "coal_kb.evaluation.datasets"
    assert RetrievalEvaluator.__module__ == "coal_kb.evaluation.retrieval"
    assert simple_faithfulness_check.__module__ == "coal_kb.evaluation.faithfulness"
    assert not (ROOT / "eval").exists()


def test_evaluation_formulas_remain_unchanged() -> None:
    docs = [Document(page_content="evidence", metadata={"source_file": "paper.pdf", "page": 3})]
    assert simple_faithfulness_check("[1] [2] [3]", docs) == pytest.approx(0.5)
    evaluator = RetrievalEvaluator(lambda question: docs)
    result = evaluator.evaluate(
        [EvalItem(question="q", gold_sources=[{"source_contains": "paper", "page": 3}])]
    )
    assert result == {"recall": 1.0, "total": 1.0, "hit": 1.0}


def test_upload_security_preserves_existing_behavior(tmp_path: Path) -> None:
    assert safe_upload_name("../../paper.pdf") == "paper.pdf"
    first = build_upload_path(tmp_path, "../../paper.pdf")
    assert first == tmp_path / "paper.pdf"
    first.write_text("old", encoding="utf-8")
    second = build_upload_path(tmp_path, "paper.pdf")
    assert second.parent == tmp_path
    assert second.name.startswith("paper_")
    assert second.suffix == ".pdf"


def test_health_and_trace_protocols_remain_stable() -> None:
    assert health_status() == {"status": "ok"}
    assert build_retrieval_trace_summary(
        retrieval_query="query",
        history_used=False,
        history_reason="standalone_query",
        trace={"vector_candidates": 4, "postfiltered_count": 2},
    ) == {
        "retrieval_query": "query",
        "history_used": False,
        "history_reason": "standalone_query",
        "vector_candidates": 4,
        "postfiltered_count": 2,
        "source_distribution": None,
        "heading_distribution": None,
    }


def test_architecture_dependencies_are_one_way() -> None:
    evaluation_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "evaluation").glob("*.py")
    )
    security_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "infra" / "security").glob("*.py")
    )
    operations_text = "\n".join(
        path.read_text(encoding="utf-8") for path in (ROOT / "operations").glob("*.py")
    )
    assert "fastapi" not in evaluation_text
    assert "coal_kb.application" not in security_text
    assert "coal_kb.interfaces" not in operations_text

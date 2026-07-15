"""验证评估、可观测性、安全与运维分层。"""

from pathlib import Path

from coal_kb.evaluation import EvaluationCase, EvaluationPipeline, EvidenceReference, QueryType
from coal_kb.infra.observability.trace import build_retrieval_trace_summary
from coal_kb.infra.security.uploads import build_upload_path, safe_upload_name
from coal_kb.operations import health_status

ROOT = Path(__file__).resolve().parents[1] / "src" / "coal_kb"


def test_evaluation_public_api_is_canonical() -> None:
    assert EvaluationCase.__module__ == "coal_kb.evaluation.models"
    assert EvidenceReference.__module__ == "coal_kb.evaluation.models"
    assert QueryType.FACT.value == "fact"
    assert EvaluationPipeline.__module__ == "coal_kb.evaluation.pipeline"
    assert not (ROOT / "evaluation" / "faithfulness.py").exists()
    assert not (ROOT / "evaluation" / "retrieval.py").exists()


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
    )["postfiltered_count"] == 2


def test_evaluation_layer_does_not_depend_on_interfaces() -> None:
    text = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "evaluation").glob("*.py"))
    assert "fastapi" not in text
    assert "coal_kb.interfaces" not in text
    assert "coal_kb.application" not in text

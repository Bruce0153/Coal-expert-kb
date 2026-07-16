"""验证多模态路线的模态识别、重排和元数据隔离。"""

from langchain_core.documents import Document

from coal_kb.research import MultimodalRoute, RouteResult


def test_multimodal_route_promotes_requested_table_evidence() -> None:
    text = Document(
        page_content="Narrative explanation",
        metadata={"chunk_id": "text", "source_file": "paper.pdf"},
    )
    table = Document(
        page_content="Table 2: temperature and NH3",
        metadata={"chunk_id": "table", "source_file": "paper.pdf", "table_id": "t2"},
    )

    result = MultimodalRoute().process(
        "表格中温度对应的 NH3 是多少",
        lambda: RouteResult(documents=[text, table], trace={"base": True}),
    )

    assert [item.metadata["chunk_id"] for item in result.documents] == ["table", "text"]
    assert result.documents[0].metadata["research_modality"] == "table"
    assert "research_modality" not in table.metadata
    assert result.trace["multimodal"]["requested_modalities"] == ["table"]
    assert result.trace["multimodal"]["available_modalities"] == {"text": 1, "table": 1}


def test_multimodal_route_detects_figure_caption() -> None:
    figure = Document(
        page_content="Figure 3. Product gas composition curve",
        metadata={"chunk_id": "fig3", "section": "Results"},
    )
    assert MultimodalRoute.infer_modality(figure) == "image"

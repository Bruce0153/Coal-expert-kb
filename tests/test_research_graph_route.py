"""验证 Graph route 只重排基础证据并输出可解释关系。"""

from langchain_core.documents import Document

from coal_kb.research import GraphRoute, RouteResult


def test_graph_route_promotes_evidence_connected_to_seed() -> None:
    documents = [
        Document(
            page_content="steam gasification increases hydrogen yield",
            metadata={"chunk_id": "seed", "source_file": "a.pdf", "parent_id": "p1"},
        ),
        Document(
            page_content="unrelated coal storage note",
            metadata={"chunk_id": "unrelated", "source_file": "b.pdf", "parent_id": "p2"},
        ),
        Document(
            page_content="steam gasification hydrogen mechanism",
            metadata={"chunk_id": "linked", "source_file": "a.pdf", "parent_id": "p1"},
        ),
    ]

    result = GraphRoute(seed_count=1).process(
        lambda: RouteResult(documents=documents, trace={"base": True})
    )

    assert [item.metadata["chunk_id"] for item in result.documents] == [
        "seed",
        "linked",
        "unrelated",
    ]
    graph = result.trace["graph"]
    assert graph["node_count"] == 3
    assert graph["edge_count"] >= 1
    assert graph["seed_chunk_ids"] == ["seed"]
    assert any("same_parent" in edge["reasons"] for edge in graph["edges"])


def test_graph_route_deduplicates_without_fetching_new_documents() -> None:
    duplicate = Document(page_content="same", metadata={"chunk_id": "c1"})
    result = GraphRoute().process(
        lambda: RouteResult(documents=[duplicate, duplicate], trace={})
    )
    assert len(result.documents) == 1
    assert result.trace["graph"]["node_count"] == 1

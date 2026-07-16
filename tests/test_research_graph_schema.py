"""验证 Graph Schema、抽取来源和升级后的 Graph route。"""

from __future__ import annotations

from langchain_core.documents import Document

from coal_kb.research import GraphRoute, RouteResult
from coal_kb.research.graph_extraction import KnowledgeGraphExtractor
from coal_kb.research.graph_schema import GRAPH_SCHEMA_VERSION, GraphNodeType, GraphRelationType


def _documents() -> list[Document]:
    return [
        Document(
            page_content="Steam gasification increases Hydrogen yield and promotes char conversion.",
            metadata={
                "chunk_id": "seed",
                "source_file": "a.pdf",
                "parent_id": "p1",
                "heading_path": "Gasification > Mechanism",
                "entities": ["Steam gasification", "Hydrogen"],
            },
        ),
        Document(
            page_content="Hydrogen production is affected by steam concentration.",
            metadata={
                "chunk_id": "linked",
                "source_file": "b.pdf",
                "parent_id": "p2",
                "heading_path": "Results",
                "entities": ["Hydrogen"],
            },
        ),
        Document(
            page_content="Coal storage note without a mechanistic relation.",
            metadata={"chunk_id": "other", "source_file": "c.pdf", "parent_id": "p3"},
        ),
    ]


def test_graph_extractor_builds_versioned_typed_graph_with_provenance() -> None:
    graph = KnowledgeGraphExtractor().process(_documents())
    payload = graph.to_dict()

    assert payload["schema_version"] == GRAPH_SCHEMA_VERSION
    assert len(graph.nodes_of_type(GraphNodeType.EVIDENCE)) == 3
    assert graph.nodes_of_type(GraphNodeType.ENTITY)
    assert graph.nodes_of_type(GraphNodeType.CLAIM)
    mention = graph.relations_of_type(GraphRelationType.MENTIONS)[0]
    assert mention.provenance["chunk_id"]
    assert mention.provenance["extractor"]
    assert payload["statistics"]["relation_types"]["supports"] >= 1


def test_graph_route_promotes_document_connected_by_shared_entity() -> None:
    result = GraphRoute(seed_count=1, use_shared_terms=False).process(
        lambda: RouteResult(documents=_documents(), trace={"base": True})
    )

    assert [document.metadata["chunk_id"] for document in result.documents] == [
        "seed",
        "linked",
        "other",
    ]
    trace = result.trace["graph"]
    assert trace["schema_version"] == GRAPH_SCHEMA_VERSION
    assert trace["typed_node_count"] > trace["node_count"]
    assert trace["statistics"]["node_types"]["entity"] >= 2
    assert any(relation["type"] == "mentions" for relation in trace["relations"])

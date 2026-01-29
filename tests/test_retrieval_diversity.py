from __future__ import annotations

import tempfile
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from coal_kb.metadata.normalize import Ontology, flatten_for_filtering
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.retriever import ExpertRetriever


class HashEmbeddings:
    def _vec(self, text: str) -> List[float]:
        s = sum(ord(c) for c in text[:200])
        return [(s % (i + 17)) / 100.0 for i in range(8)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


def _strip_lists(doc: Document) -> Document:
    doc.metadata.pop("gas_agent", None)
    doc.metadata.pop("targets", None)
    return doc


def test_diversity_max_per_source():
    onto = Ontology.load("configs/schema.yaml")
    parser = FilterParser(onto=onto)

    with tempfile.TemporaryDirectory() as tmp:
        vs = Chroma(
            collection_name="test",
            embedding_function=HashEmbeddings(),
            persist_directory=tmp,
        )

        docs = []
        for idx in range(3):
            docs.append(
                Document(
                    page_content=f"Steam gasification example {idx}",
                    metadata=flatten_for_filtering(
                        {
                            "source_file": "a.pdf",
                            "page": idx,
                            "stage": "gasification",
                            "T_K": 1200.0,
                            "P_MPa": 2.0,
                            "gas_agent": ["steam"],
                            "targets": ["NH3"],
                            "chunk_id": f"a{idx}",
                            "section": "results",
                        },
                        onto,
                    ),
                )
            )

        docs.append(
            Document(
                page_content="Steam gasification from another source.",
                metadata=flatten_for_filtering(
                    {
                        "source_file": "b.pdf",
                        "page": 1,
                        "stage": "gasification",
                        "T_K": 1200.0,
                        "P_MPa": 2.0,
                        "gas_agent": ["steam"],
                        "targets": ["NH3"],
                        "chunk_id": "b1",
                        "section": "results",
                    },
                    onto,
                ),
            )
        )

        vs.add_documents([_strip_lists(d) for d in docs])

        def factory(k: int, where=None):
            return vs.as_retriever(search_kwargs={"k": k, "filter": where} if where else {"k": k})

        expert = ExpertRetriever(vector_retriever_factory=factory, k=3, k_candidates=10, max_per_source=1)
        query = "steam gasification NH3 1200K"
        parsed = parser.parse(query)
        results = expert.retrieve(query, parsed)

        sources = [d.metadata.get("source_file") for d in results]
        assert len(set(sources)) == len(sources)

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document

from coal_kb.metadata.normalize import Ontology, flatten_for_filtering
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.retriever import ExpertRetriever


class HashEmbeddings:
    """
    Lightweight deterministic embeddings for tests (no model download).
    """

    def _vec(self, text: str) -> List[float]:
        s = sum(ord(c) for c in text[:200])
        # fixed-length vector
        return [(s % (i + 17)) / 100.0 for i in range(8)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


def test_filter_and_postfilter():
    onto = Ontology.load("configs/schema.yaml")
    parser = FilterParser(onto=onto)

    with tempfile.TemporaryDirectory() as tmp:
        vs = Chroma(
            collection_name="test",
            embedding_function=HashEmbeddings(),
            persist_directory=tmp,
        )

        d1 = Document(
            page_content="Steam gasification at 1200 K and 2 MPa produces NH3 and HCN.",
            metadata=flatten_for_filtering(
                {
                    "source_file": "a.pdf",
                    "page": 1,
                    "stage": "gasification",
                    "T_K": 1200.0,
                    "P_MPa": 2.0,
                    "gas_agent": ["steam"],
                    "targets": ["NH3", "HCN"],
                    "chunk_id": "c1",
                    "section": "results",
                },
                onto,
            ),
        )
        d2 = Document(
            page_content="Pyrolysis at 800 C produces tar and phenols.",
            metadata=flatten_for_filtering(
                {
                    "source_file": "b.pdf",
                    "page": 2,
                    "stage": "pyrolysis",
                    "T_K": 1073.15,
                    "P_MPa": 0.1,
                    "gas_agent": None,
                    "targets": ["phenols"],
                    "chunk_id": "c2",
                    "section": "results",
                },
                onto,
            ),
        )

        vs.add_documents([d1, d2])

        def factory(k: int, where=None):
            return vs.as_retriever(search_kwargs={"k": k, "filter": where} if where else {"k": k})

        expert = ExpertRetriever(vector_retriever_factory=factory, k=3, k_candidates=10)

        f = parser.parse("steam gasification 1200K 2MPa NH3")
        docs = expert.retrieve("steam gasification 1200K 2MPa NH3", f)

        assert len(docs) >= 1
        assert "a.pdf" in (docs[0].metadata.get("source_file") or "")
        assert docs[0].metadata.get("stage") == "gasification"

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


def test_reference_like_filtered():
    onto = Ontology.load("configs/schema.yaml")
    parser = FilterParser(onto=onto)

    with tempfile.TemporaryDirectory() as tmp:
        vs = Chroma(
            collection_name="test",
            embedding_function=HashEmbeddings(),
            persist_directory=tmp,
        )

        ref_text = (
            "References\n"
            "1. Smith J. Fuel 2019, 123, 45-56. doi:10.1016/j.fuel.2019.01.001"
        )
        ref_doc = Document(
            page_content=ref_text,
            metadata=flatten_for_filtering(
                {
                    "source_file": "refs.pdf",
                    "page": 10,
                    "stage": "gasification",
                    "chunk_id": "ref1",
                    "section": "references",
                },
                onto,
            ),
        )
        good_doc = Document(
            page_content="Steam gasification produces NH3.",
            metadata=flatten_for_filtering(
                {
                    "source_file": "good.pdf",
                    "page": 1,
                    "stage": "gasification",
                    "chunk_id": "good1",
                    "section": "results",
                    "gas_agent": ["steam"],
                    "targets": ["NH3"],
                },
                onto,
            ),
        )

        vs.add_documents([_strip_lists(ref_doc), _strip_lists(good_doc)])

        def factory(k: int, where=None):
            return vs.as_retriever(search_kwargs={"k": k, "filter": where} if where else {"k": k})

        expert = ExpertRetriever(
            vector_retriever_factory=factory,
            k=2,
            k_candidates=5,
            drop_sections=["references"],
            drop_reference_like=True,
        )
        query = "steam gasification NH3"
        parsed = parser.parse(query)
        results = expert.retrieve(query, parsed)

        assert all(d.metadata.get("source_file") != "refs.pdf" for d in results)

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..embeddings.factory import make_embeddings, EmbeddingsConfig

@dataclass
class ChromaStore:
    """
    Chroma vector store wrapper.

    - If embeddings_cfg is provided: use DashScope/OpenAI-compatible embeddings via make_embeddings()
    - Else: fallback to local HuggingFace embeddings (bge-m3 by default)
    """
    persist_dir: str
    collection_name: str

    # Preferred: DashScope/OpenAI-compatible embeddings config (from app.yaml `embeddings:`)
    embeddings_cfg: Optional[EmbeddingsConfig] = None

    # Fallback: local HF embeddings model (only used if embeddings_cfg is None)
    embedding_model: str = "BAAI/bge-m3"

    def __post_init__(self) -> None:
        self._embeddings = self._build_embeddings()
        self._vs = Chroma(
            collection_name=self.collection_name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_dir,
        )

    def _build_embeddings(self):
        if self.embeddings_cfg is not None:
            # DashScope (OpenAI-compatible) embeddings
            return make_embeddings(self.embeddings_cfg)

        # Fallback to local HF embeddings (keep your old behavior)
        from langchain_huggingface import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name=self.embedding_model)

    @property
    def vectorstore(self) -> Chroma:
        return self._vs

    def add_documents(self, docs: List[Document], *, ids: Optional[List[str]] = None) -> None:
        if not docs:
            return
        if ids is not None:
            self._vs.add_documents(docs, ids=ids)
        else:
            self._vs.add_documents(docs)

    def delete_where(self, where: Dict[str, Any]) -> None:
        if not where:
            return
        self._vs.delete(where=where)

    def as_retriever(self, *, k: int = 5, where: Optional[Dict[str, Any]] = None):
        """
        Return a retriever with optional metadata filter.
        """
        search_kwargs: Dict[str, Any] = {"k": k}
        if where:
            search_kwargs["filter"] = where  # Chroma uses `filter`
        return self._vs.as_retriever(search_kwargs=search_kwargs)

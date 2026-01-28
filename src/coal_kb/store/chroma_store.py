from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document

from ..embeddings.factory import make_embeddings, EmbeddingsConfig

logger = logging.getLogger(__name__)

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
        filtered_docs: List[Document] = []
        filtered_ids: Optional[List[str]] = [] if ids is not None else None
        for idx, doc in enumerate(docs):
            text = (doc.page_content or "").strip()
            if not text:
                continue
            filtered_docs.append(doc)
            if filtered_ids is not None:
                filtered_ids.append(ids[idx])
        if len(filtered_docs) != len(docs):
            logger.info("Filtered empty texts | removed=%d", len(docs) - len(filtered_docs))
        if not filtered_docs:
            return
        batch_size = len(filtered_docs)
        logger.info(
            "Chroma add_documents | docs=%d ids=%s batch_size=%d",
            len(filtered_docs),
            ids is not None,
            batch_size,
        )
        try:
            if filtered_ids is not None:
                self._vs.add_documents(filtered_docs, ids=filtered_ids)
            else:
                self._vs.add_documents(filtered_docs)
        except Exception as e:
            logger.error("Chroma add_documents failed | error=%s", e)
            raise

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

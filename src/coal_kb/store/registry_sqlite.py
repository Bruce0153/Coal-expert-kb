from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import DateTime, Integer, String, Text, UniqueConstraint, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class Base(DeclarativeBase):
    pass


class DocumentModel(Base):
    __tablename__ = "documents"

    document_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_file: Mapped[str] = mapped_column(Text, index=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    mtime: Mapped[int] = mapped_column(Integer)
    size: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="active")
    tenant_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ChunkModel(Base):
    __tablename__ = "chunks"

    chunk_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str] = mapped_column(String(64), index=True)
    page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    section: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    chunk_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    text: Mapped[str] = mapped_column(Text)
    metadata_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding_model: Mapped[str] = mapped_column(String(128))
    embedding_dim: Mapped[int] = mapped_column(Integer)
    embedding_version: Mapped[str] = mapped_column(String(64), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class ModelModel(Base):
    __tablename__ = "models"
    __table_args__ = (
        UniqueConstraint("embedding_model", "embedding_dim", "embedding_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    embedding_model: Mapped[str] = mapped_column(String(128))
    embedding_dim: Mapped[int] = mapped_column(Integer)
    embedding_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class QueryLogModel(Base):
    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(Text)
    filters_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    top_chunk_ids_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(nullable=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    embedding_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


@dataclass
class RegistrySQLite:
    sqlite_path: str

    def __post_init__(self) -> None:
        self._engine = create_engine(f"sqlite:///{self.sqlite_path}", future=True)
        Base.metadata.create_all(self._engine)

    def upsert_document(
        self,
        *,
        document_id: str,
        source_file: str,
        sha256: str,
        mtime: int,
        size: int,
        status: str = "active",
        tenant_id: Optional[str] = None,
    ) -> None:
        now = datetime.utcnow()
        with Session(self._engine) as sess:
            obj = sess.get(DocumentModel, document_id)
            if obj is None:
                obj = DocumentModel(
                    document_id=document_id,
                    source_file=source_file,
                    sha256=sha256,
                    mtime=mtime,
                    size=size,
                    status=status,
                    tenant_id=tenant_id,
                    created_at=now,
                    updated_at=now,
                )
                sess.add(obj)
            else:
                obj.source_file = source_file
                obj.sha256 = sha256
                obj.mtime = mtime
                obj.size = size
                obj.status = status
                obj.tenant_id = tenant_id
                obj.updated_at = now
            sess.commit()

    def upsert_chunks_bulk(
        self,
        chunks: Iterable[Dict[str, Any]],
    ) -> None:
        now = datetime.utcnow()
        with Session(self._engine) as sess:
            for chunk in chunks:
                chunk_id = str(chunk["chunk_id"])
                obj = sess.get(ChunkModel, chunk_id)
                payload = {
                    "document_id": str(chunk["document_id"]),
                    "page": chunk.get("page"),
                    "section": chunk.get("section"),
                    "chunk_index": chunk.get("chunk_index"),
                    "text": chunk.get("text", ""),
                    "metadata_json": chunk.get("metadata_json"),
                    "embedding_model": str(chunk["embedding_model"]),
                    "embedding_dim": int(chunk["embedding_dim"]),
                    "embedding_version": str(chunk["embedding_version"]),
                    "created_at": now,
                }
                if obj is None:
                    obj = ChunkModel(chunk_id=chunk_id, **payload)
                    sess.add(obj)
                else:
                    for key, value in payload.items():
                        setattr(obj, key, value)
            sess.commit()

    def delete_chunks_by_document_id(self, document_id: str) -> None:
        with Session(self._engine) as sess:
            stmt = delete(ChunkModel).where(ChunkModel.document_id == document_id)
            sess.execute(stmt)
            sess.commit()

    def log_model(
        self,
        *,
        embedding_model: str,
        embedding_dim: int,
        embedding_version: str,
    ) -> None:
        with Session(self._engine) as sess:
            stmt = select(ModelModel).where(
                ModelModel.embedding_model == embedding_model,
                ModelModel.embedding_dim == embedding_dim,
                ModelModel.embedding_version == embedding_version,
            )
            if sess.scalars(stmt).first() is None:
                sess.add(
                    ModelModel(
                        embedding_model=embedding_model,
                        embedding_dim=embedding_dim,
                        embedding_version=embedding_version,
                        created_at=datetime.utcnow(),
                    )
                )
                sess.commit()

    def log_query(
        self,
        *,
        query: str,
        filters: Optional[Dict[str, Any]],
        top_chunk_ids: Optional[list[str]],
        latency_ms: Optional[float],
        tenant_id: Optional[str],
        embedding_version: Optional[str],
    ) -> None:
        with Session(self._engine) as sess:
            obj = QueryLogModel(
                query=query,
                filters_json=json.dumps(filters, ensure_ascii=False) if filters else None,
                top_chunk_ids_json=json.dumps(top_chunk_ids, ensure_ascii=False) if top_chunk_ids else None,
                latency_ms=latency_ms,
                tenant_id=tenant_id,
                embedding_version=embedding_version,
                created_at=datetime.utcnow(),
            )
            sess.add(obj)
            sess.commit()

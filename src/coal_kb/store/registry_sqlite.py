from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import (
    DateTime,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    select,
    text as sql_text,
)
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
    doc_type: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    language: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    parser: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
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
    constraints_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    top_chunk_ids_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    top_source_files_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(nullable=True)
    backend: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    rerank_enabled: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    mode: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    relax_steps_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    diversity_k: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    tenant_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    embedding_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class IngestionRunModel(Base):
    __tablename__ = "ingestion_runs"

    run_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    finished_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    embedding_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    embedding_model: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    embedding_dim: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    schema_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    chunking_signature: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    stats_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)


class RunMetricsModel(Base):
    __tablename__ = "run_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), index=True)
    index_name: Mapped[str] = mapped_column(Text)
    embedding_version: Mapped[str] = mapped_column(String(64))
    schema_hash: Mapped[str] = mapped_column(String(64))
    doc_count: Mapped[int] = mapped_column(Integer)
    chunks: Mapped[int] = mapped_column(Integer)
    precision_at_k: Mapped[Optional[float]] = mapped_column(nullable=True)
    recall_at_k: Mapped[Optional[float]] = mapped_column(nullable=True)
    mrr: Mapped[Optional[float]] = mapped_column(nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


@dataclass
class RegistrySQLite:
    sqlite_path: str

    def __post_init__(self) -> None:
        self._engine = create_engine(f"sqlite:///{self.sqlite_path}", future=True)
        Base.metadata.create_all(self._engine)
        self._ensure_migrations()

    def _has_column(self, table: str, col: str) -> bool:
        with self._engine.connect() as conn:
            rows = conn.execute(sql_text(f"PRAGMA table_info({table})")).fetchall()
            return any(r[1] == col for r in rows)

    def _ensure_column(self, table: str, col: str, ddl: str) -> None:
        if self._has_column(table, col):
            return
        with self._engine.connect() as conn:
            conn.execute(sql_text(f"ALTER TABLE {table} ADD COLUMN {ddl}"))
            conn.commit()

    def _ensure_migrations(self) -> None:
        self._ensure_column("query_logs", "filters_json", "filters_json TEXT")
        self._ensure_column("query_logs", "constraints_json", "constraints_json TEXT")
        self._ensure_column("query_logs", "top_source_files_json", "top_source_files_json TEXT")
        self._ensure_column("query_logs", "rerank_enabled", "rerank_enabled INTEGER")
        self._ensure_column("query_logs", "backend", "backend VARCHAR(32)")
        self._ensure_column("query_logs", "mode", "mode VARCHAR(32)")
        self._ensure_column("query_logs", "relax_steps_json", "relax_steps_json TEXT")
        self._ensure_column("query_logs", "diversity_k", "diversity_k INTEGER")
        self._ensure_column("documents", "title", "title TEXT")
        self._ensure_column("documents", "doc_type", "doc_type VARCHAR(32)")
        self._ensure_column("documents", "language", "language VARCHAR(16)")
        self._ensure_column("documents", "parser", "parser VARCHAR(64)")
        self._ensure_column("ingestion_runs", "embedding_model", "embedding_model VARCHAR(128)")
        self._ensure_column("ingestion_runs", "embedding_dim", "embedding_dim INTEGER")

    def upsert_document(
        self,
        *,
        document_id: str,
        source_file: str,
        sha256: str,
        mtime: int,
        size: int,
        doc_type: Optional[str] = None,
        language: Optional[str] = None,
        parser: Optional[str] = None,
        title: Optional[str] = None,
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
                    doc_type=doc_type,
                    language=language,
                    parser=parser,
                    title=title,
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
                obj.doc_type = doc_type
                obj.language = language
                obj.parser = parser
                obj.title = title
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

    def delete_by_document_id(self, document_id: str) -> None:
        with Session(self._engine) as sess:
            sess.execute(delete(DocumentModel).where(DocumentModel.document_id == document_id))
            sess.execute(delete(ChunkModel).where(ChunkModel.document_id == document_id))
            sess.commit()

    def start_run(
        self,
        *,
        run_id: str,
        embedding_version: str,
        embedding_model: str,
        embedding_dim: int,
        schema_hash: str,
        chunking_signature: str,
    ) -> None:
        with Session(self._engine) as sess:
            obj = IngestionRunModel(
                run_id=run_id,
                started_at=datetime.utcnow(),
                embedding_version=embedding_version,
                embedding_model=embedding_model,
                embedding_dim=embedding_dim,
                schema_hash=schema_hash,
                chunking_signature=chunking_signature,
                status="running",
            )
            sess.add(obj)
            sess.commit()

    def finish_run(self, *, run_id: str, stats: Dict[str, Any], status: str) -> None:
        with Session(self._engine) as sess:
            obj = sess.get(IngestionRunModel, run_id)
            if obj is None:
                return
            obj.finished_at = datetime.utcnow()
            obj.stats_json = json.dumps(stats, ensure_ascii=False)
            obj.status = status
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
        constraints: Optional[Dict[str, Any]],
        top_chunk_ids: Optional[list[str]],
        top_source_files: Optional[list[str]],
        latency_ms: Optional[float],
        backend: Optional[str],
        tenant_id: Optional[str],
        embedding_version: Optional[str],
        rerank_enabled: Optional[bool],
        mode: Optional[str] = None,
        relax_steps: Optional[list[str]] = None,
        diversity_k: Optional[int] = None,
    ) -> None:
        with Session(self._engine) as sess:
            obj = QueryLogModel(
                query=query,
                filters_json=json.dumps(filters, ensure_ascii=False) if filters else None,
                constraints_json=json.dumps(constraints, ensure_ascii=False) if constraints else None,
                top_chunk_ids_json=json.dumps(top_chunk_ids, ensure_ascii=False) if top_chunk_ids else None,
                top_source_files_json=json.dumps(top_source_files, ensure_ascii=False)
                if top_source_files
                else None,
                latency_ms=latency_ms,
                backend=backend,
                rerank_enabled=1 if rerank_enabled else 0 if rerank_enabled is not None else None,
                mode=mode,
                relax_steps_json=json.dumps(relax_steps, ensure_ascii=False) if relax_steps else None,
                diversity_k=diversity_k,
                tenant_id=tenant_id,
                embedding_version=embedding_version,
                created_at=datetime.utcnow(),
            )
            sess.add(obj)
            sess.commit()

    def log_run_metrics(
        self,
        *,
        run_id: str,
        index_name: str,
        embedding_version: str,
        schema_hash: str,
        doc_count: int,
        chunks: int,
        precision_at_k: Optional[float],
        recall_at_k: Optional[float],
        mrr: Optional[float],
    ) -> None:
        with Session(self._engine) as sess:
            obj = RunMetricsModel(
                run_id=run_id,
                index_name=index_name,
                embedding_version=embedding_version,
                schema_hash=schema_hash,
                doc_count=doc_count,
                chunks=chunks,
                precision_at_k=precision_at_k,
                recall_at_k=recall_at_k,
                mrr=mrr,
                created_at=datetime.utcnow(),
            )
            sess.add(obj)
            sess.commit()

    def get_document(self, document_id: str) -> Optional[DocumentModel]:
        with Session(self._engine) as sess:
            return sess.get(DocumentModel, document_id)

    def get_chunk(self, chunk_id: str) -> Optional[ChunkModel]:
        with Session(self._engine) as sess:
            return sess.get(ChunkModel, chunk_id)

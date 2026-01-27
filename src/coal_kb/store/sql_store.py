from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text, create_engine, select, text as sql_text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class RecordModel(Base):
    __tablename__ = "records"

    record_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    source_file: Mapped[str] = mapped_column(Text, index=True)

    stage: Mapped[str] = mapped_column(String(32), default="unknown", index=True)
    coal_name: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    reactor_type: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    T_K: Mapped[Optional[float]] = mapped_column(nullable=True)
    P_MPa: Mapped[Optional[float]] = mapped_column(nullable=True)

    gas_agent_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    ratios_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    pollutants_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # new: signature & conflict marker
    signature: Mapped[Optional[str]] = mapped_column(String(96), nullable=True, index=True)
    is_conflict: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    evidence: Mapped[List["EvidenceModel"]] = relationship(
        "EvidenceModel",
        back_populates="record",
        cascade="all, delete-orphan",
    )


class EvidenceModel(Base):
    __tablename__ = "evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    record_id: Mapped[str] = mapped_column(ForeignKey("records.record_id"), index=True)

    source_file: Mapped[str] = mapped_column(Text, index=True)
    page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    chunk_id: Mapped[str] = mapped_column(String(64), index=True)
    quote: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    record: Mapped[RecordModel] = relationship("RecordModel", back_populates="evidence")


class ConflictModel(Base):
    __tablename__ = "conflicts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    signature: Mapped[str] = mapped_column(String(96), index=True)

    existing_record_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    new_record_id: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    existing_pollutants_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    new_pollutants_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    note: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


@dataclass
class SQLiteStore:
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
        # records.signature, records.is_conflict for older dbs
        self._ensure_column("records", "signature", "signature VARCHAR(96)")
        self._ensure_column("records", "is_conflict", "is_conflict INTEGER DEFAULT 0")

        # conflicts table exists?
        # Base.metadata.create_all already creates if missing; keep here for clarity
        Base.metadata.create_all(self._engine)

    def find_any_by_signature(self, signature: str) -> Optional[RecordModel]:
        with Session(self._engine) as sess:
            stmt = select(RecordModel).where(RecordModel.signature == signature).order_by(RecordModel.updated_at.desc()).limit(1)
            return sess.scalars(stmt).first()

    def log_conflict(
        self,
        *,
        signature: str,
        existing_record_id: Optional[str],
        new_record_id: Optional[str],
        existing_pollutants: Optional[Dict[str, Any]],
        new_pollutants: Optional[Dict[str, Any]],
        note: str,
    ) -> None:
        with Session(self._engine) as sess:
            obj = ConflictModel(
                signature=signature,
                existing_record_id=existing_record_id,
                new_record_id=new_record_id,
                existing_pollutants_json=json.dumps(existing_pollutants, ensure_ascii=False) if existing_pollutants else None,
                new_pollutants_json=json.dumps(new_pollutants, ensure_ascii=False) if new_pollutants else None,
                note=note,
                created_at=datetime.utcnow(),
            )
            sess.add(obj)
            sess.commit()

    def upsert_record(
        self,
        *,
        record_id: str,
        source_file: str,
        stage: str = "unknown",
        coal_name: Optional[str] = None,
        reactor_type: Optional[str] = None,
        T_K: Optional[float] = None,
        P_MPa: Optional[float] = None,
        gas_agent: Optional[List[str]] = None,
        ratios: Optional[Dict[str, float]] = None,
        pollutants: Optional[Dict[str, Any]] = None,
        signature: Optional[str] = None,
        is_conflict: int = 0,
    ) -> None:
        now = datetime.utcnow()
        with Session(self._engine) as sess:
            obj = sess.get(RecordModel, record_id)
            if obj is None:
                obj = RecordModel(
                    record_id=record_id,
                    source_file=source_file,
                    stage=stage,
                    coal_name=coal_name,
                    reactor_type=reactor_type,
                    T_K=T_K,
                    P_MPa=P_MPa,
                    gas_agent_json=json.dumps(gas_agent, ensure_ascii=False) if gas_agent else None,
                    ratios_json=json.dumps(ratios, ensure_ascii=False) if ratios else None,
                    pollutants_json=json.dumps(pollutants, ensure_ascii=False) if pollutants else None,
                    signature=signature,
                    is_conflict=is_conflict,
                    created_at=now,
                    updated_at=now,
                )
                sess.add(obj)
            else:
                obj.source_file = source_file
                obj.stage = stage
                obj.coal_name = coal_name
                obj.reactor_type = reactor_type
                obj.T_K = T_K
                obj.P_MPa = P_MPa
                obj.gas_agent_json = json.dumps(gas_agent, ensure_ascii=False) if gas_agent else None
                obj.ratios_json = json.dumps(ratios, ensure_ascii=False) if ratios else None
                obj.pollutants_json = json.dumps(pollutants, ensure_ascii=False) if pollutants else None
                obj.signature = signature
                obj.is_conflict = is_conflict
                obj.updated_at = now

            sess.commit()

    def add_evidence(
        self,
        *,
        record_id: str,
        source_file: str,
        page: Optional[int],
        chunk_id: str,
        quote: Optional[str] = None,
    ) -> None:
        with Session(self._engine) as sess:
            ev = EvidenceModel(
                record_id=record_id,
                source_file=source_file,
                page=page,
                chunk_id=chunk_id,
                quote=quote,
            )
            sess.add(ev)
            sess.commit()

    def list_records(self, limit: int = 50) -> List[RecordModel]:
        with Session(self._engine) as sess:
            stmt = select(RecordModel).order_by(RecordModel.updated_at.desc()).limit(limit)
            return list(sess.scalars(stmt))

    def get_record(self, record_id: str) -> Optional[RecordModel]:
        with Session(self._engine) as sess:
            return sess.get(RecordModel, record_id)

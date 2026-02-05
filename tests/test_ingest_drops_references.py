from __future__ import annotations

from pathlib import Path

import pytest
from langchain_core.documents import Document

from coal_kb.pipelines.ingest_pipeline import IngestPipeline
from coal_kb.settings import load_config


class DummyEmbeddings:
    def embed_query(self, _text: str):
        return [0.1, 0.2, 0.3]

    def embed_documents(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def test_ingest_drops_references(tmp_path: Path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    pdf_path = raw_dir / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4")

    ref_text = (
        "References\n"
        "1. Smith J. Fuel 2019, 123, 45-56. doi:10.1016/j.fuel.2019.01.001\n"
        "2. Zhang L. Energy & Fuels 2020, 34, 100-110. doi:10.1021/ef2c00001\n"
    )

    def fake_pdf_load(self, _path: str):
        return [Document(page_content=ref_text, metadata={"source_file": str(pdf_path), "page": 0, "format": "text"})]

    monkeypatch.setattr("coal_kb.pipelines.ingest_pipeline.PDFLoader.load", fake_pdf_load)
    monkeypatch.setattr("coal_kb.pipelines.ingest_pipeline.make_embeddings", lambda cfg: DummyEmbeddings())
    monkeypatch.setattr("coal_kb.store.chroma_store.make_embeddings", lambda cfg: DummyEmbeddings())

    captured = {"docs": []}

    def fake_add_documents(self, docs, ids=None):
        captured["docs"].extend(docs)

    monkeypatch.setattr("coal_kb.store.chroma_store.ChromaStore.add_documents", fake_add_documents)

    cfg = load_config()
    cfg.paths.raw_pdfs_dir = str(raw_dir)
    cfg.paths.interim_dir = str(tmp_path / "interim")
    Path(cfg.paths.interim_dir).mkdir(parents=True, exist_ok=True)
    cfg.paths.chroma_dir = str(tmp_path / "chroma")
    cfg.paths.manifest_path = str(tmp_path / "manifest.json")
    cfg.registry.sqlite_path = str(tmp_path / "kb.db")
    cfg.backend = "chroma"
    cfg.chunking.embedding_backend = "lexical"

    pipe = IngestPipeline(cfg=cfg)
    stats = pipe.run(rebuild=True, force=True)

    assert stats["dropped_chunks"] >= 1

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from langchain_core.documents import Document
from tqdm import tqdm
from ..llm.factory import LLMConfig
from coal_kb.embeddings.factory import EmbeddingsConfig
from ..chunking.splitter import split_page_docs
from ..metadata.extract import MetadataExtractor
from ..metadata.normalize import Ontology, flatten_for_filtering
from ..parsing.pdf_loader import load_pdf_pages
from ..parsing.table_extractor import TableExtractor
from ..settings import AppConfig
from ..store.chroma_store import ChromaStore
from ..store.manifest import Manifest, ManifestEntry
from ..utils.file_hash import sha256_file
from ..utils.hash import stable_chunk_id

logger = logging.getLogger(__name__)


def _cache_path_for_pdf(interim_dir: Path, pdf_path: str) -> Path:
    # stable cache file name
    p = Path(pdf_path)
    try:
        stamp = str(p.stat().st_mtime_ns)
    except FileNotFoundError:
        stamp = "missing"
    safe = stable_chunk_id(pdf_path, stamp)
    return interim_dir / f"pages_{safe}.jsonl"


def _save_pages_cache(path: Path, docs: List[Document]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for d in docs:
            payload = {"text": d.page_content, "metadata": d.metadata}
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _load_pages_cache(path: Path) -> Optional[List[Document]]:
    if not path.exists():
        return None
    docs: List[Document] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            docs.append(Document(page_content=obj["text"], metadata=obj.get("metadata") or {}))
    return docs


@dataclass
class IngestPipeline:
    cfg: AppConfig
    enable_table_extraction: bool = False
    table_flavor: str = "lattice"

    # metadata extraction
    enable_llm_metadata: bool = False
    llm_provider: str = "none"  # "openai"

    def run(self, *, rebuild: bool = False, force: bool = False) -> dict:
        raw_dir = Path(self.cfg.paths.raw_pdfs_dir)
        interim_dir = Path(self.cfg.paths.interim_dir)
        manifest_path = Path(self.cfg.paths.manifest_path)

        onto = Ontology.load("configs/schema.yaml")

        llm_provider = self.llm_provider
        if self.enable_llm_metadata and llm_provider == "none":
            llm_provider = self.cfg.llm.provider
        provider = llm_provider if llm_provider != "none" else self.cfg.llm.provider
        llm_cfg = LLMConfig(**{**self.cfg.llm.model_dump(), "provider": provider})

        extractor = MetadataExtractor(
            onto=onto,
            enable_llm=self.enable_llm_metadata,
            llm_provider=llm_provider,
            llm_config=llm_cfg,
        )

        if rebuild:
            shutil.rmtree(self.cfg.paths.chroma_dir, ignore_errors=True)
            if manifest_path.exists():
                manifest_path.unlink()

        # Vectorstore
        store = ChromaStore(
            persist_dir=self.cfg.paths.chroma_dir,
            collection_name=self.cfg.chroma.collection_name,
            embeddings_cfg=EmbeddingsConfig(**self.cfg.embeddings.model_dump()),
            embedding_model=self.cfg.embedding.model_name,  # fallback：本地 bge-m3
        )

        manifest = Manifest.load(manifest_path)
        embed_sig = stable_chunk_id(json.dumps(self.cfg.embeddings.model_dump(), sort_keys=True, ensure_ascii=False))
        chunk_sig = stable_chunk_id(json.dumps(self.cfg.chunking.model_dump(), sort_keys=True, ensure_ascii=False))
        schema_sig = stable_chunk_id(Path("configs/schema.yaml").read_text(encoding="utf-8"))
        mismatches = manifest.signature_mismatch(
            embeddings=embed_sig,
            chunking=chunk_sig,
            schema=schema_sig,
        )
        if mismatches and not (rebuild or force):
            msg = (
                "Manifest signature mismatch detected. "
                "Rebuild is recommended to avoid stale embeddings/chunks. "
                "Use --rebuild to clear the KB or --force to continue."
            )
            logger.warning("%s Mismatches=%s", msg, mismatches)
            raise RuntimeError(msg)

        manifest.embeddings_signature = embed_sig
        manifest.chunking_signature = chunk_sig
        manifest.schema_signature = schema_sig

        current_files = {str(p.resolve()): p for p in sorted(raw_dir.rglob("*.pdf"))}
        removed = set(manifest.files.keys()) - set(current_files.keys())
        for path in removed:
            store.delete_where({"source_file": path})
            manifest.files.pop(path, None)

        # Load PDFs (page-level)
        page_docs: List[Document] = []
        changed_files: Dict[str, ManifestEntry] = {}
        for path_str, pdf_path in current_files.items():
            stat = pdf_path.stat()
            entry = manifest.files.get(path_str)
            if entry and entry.mtime_ns == stat.st_mtime_ns and entry.size == stat.st_size:
                continue

            file_hash = sha256_file(pdf_path)
            if entry and entry.sha256 == file_hash:
                manifest.files[path_str] = ManifestEntry(
                    path=path_str,
                    mtime_ns=stat.st_mtime_ns,
                    size=stat.st_size,
                    sha256=file_hash,
                    chunk_count=entry.chunk_count,
                )
                continue

            changed_files[path_str] = ManifestEntry(
                path=path_str,
                mtime_ns=stat.st_mtime_ns,
                size=stat.st_size,
                sha256=file_hash,
            )

            cache_path = _cache_path_for_pdf(interim_dir, str(pdf_path))
            cached = _load_pages_cache(cache_path)
            if cached is not None:
                page_docs.extend(cached)
                continue

            docs = load_pdf_pages(pdf_path)
            if docs:
                _save_pages_cache(cache_path, docs)
                page_docs.extend(docs)

        # Optional: table extraction (as extra docs)
        if self.enable_table_extraction:
            tex = TableExtractor(flavor=self.table_flavor)
            # naive: run over files again
            for path_str, pdf_path in current_files.items():
                if path_str not in changed_files:
                    continue
                page_docs.extend(tex.extract(pdf_path))

        if not page_docs:
            manifest.save(manifest_path)
            return {"docs": 0, "chunks": 0, "indexed": 0, "skipped": len(current_files)}

        # Chunk
        chunks = split_page_docs(
            page_docs,
            chunk_size=self.cfg.chunking.chunk_size,
            chunk_overlap=self.cfg.chunking.chunk_overlap,
        )

        indexed_docs: Dict[str, Document] = {}
        for i, ch in enumerate(tqdm(chunks, desc="Enrich metadata")):
            # generate chunk_id deterministically
            src = (ch.metadata or {}).get("source_file", "unknown")
            page = str((ch.metadata or {}).get("page", ""))
            section = str((ch.metadata or {}).get("section", "unknown"))

            chunk_id = stable_chunk_id(src, page, section, ch.page_content[:200])
            meta = extractor.extract(ch)

            # flatten meta for robust filtering
            meta = flatten_for_filtering(meta, onto)

            # attach chunk_id + canonical source/page for citations
            meta["chunk_id"] = chunk_id
            meta.setdefault("source_file", src)

            # store the chunk text
            indexed_docs[chunk_id] = Document(page_content=ch.page_content, metadata=meta)

        for path_str in changed_files.keys():
            store.delete_where({"source_file": path_str})

        docs_to_add = list(indexed_docs.values())
        store.add_documents(docs_to_add, ids=list(indexed_docs.keys()))

        for path_str, entry in changed_files.items():
            entry.chunk_count = sum(1 for d in docs_to_add if d.metadata.get("source_file") == path_str)
            manifest.files[path_str] = entry

        manifest.save(manifest_path)
        logger.info(
            "Ingest complete: page_docs=%d chunks=%d indexed=%d",
            len(page_docs),
            len(chunks),
            len(docs_to_add),
        )

        return {
            "docs": len(page_docs),
            "chunks": len(chunks),
            "indexed": len(docs_to_add),
            "skipped": len(current_files) - len(changed_files),
        }

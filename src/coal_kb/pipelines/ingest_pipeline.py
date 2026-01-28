from __future__ import annotations

import json
import logging
import shutil
import time
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

_KEEP_META_KEYS = {
    "source_file",
    "page",
    "page_label",
    "section",
    "chunk_id",
    "stage",
    "gas_agent",
    "targets",
    "T_K",
    "T_min_K",
    "T_max_K",
    "P_MPa",
    "P_min_MPa",
    "P_max_MPa",
    "coal_name",
    "ratios",
}


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


def _trim_metadata(meta: Dict[str, object]) -> Dict[str, object]:
    trimmed: Dict[str, object] = {}
    for k, v in meta.items():
        if k in _KEEP_META_KEYS or k.startswith("gas_") or k.startswith("has_"):
            trimmed[k] = v
    return trimmed


def _stringify_value(value: object) -> object:
    if isinstance(value, list):
        return ",".join(str(v) for v in value) if value else None
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _stringify_metadata(meta: Dict[str, object]) -> Dict[str, object]:
    return {k: _stringify_value(v) for k, v in meta.items()}


def _merge_list_field(meta: Dict[str, object], key: str, values: Optional[List[str]]) -> None:
    if not values:
        return
    existing = meta.get(key)
    if existing is None:
        meta[key] = values
        return
    if isinstance(existing, list):
        meta[key] = sorted(set(existing) | set(values))


@dataclass
class IngestPipeline:
    cfg: AppConfig
    enable_table_extraction: bool = False
    table_flavor: str = "lattice"

    # metadata extraction
    enable_llm_metadata: bool = False
    llm_provider: str = "none"  # "openai"

    def run(self, *, rebuild: bool = False, force: bool = False) -> dict:
        overall_start = time.monotonic()
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
            logger.info("Rebuild requested; cleared vectorstore and manifest.")

        # Vectorstore
        store = ChromaStore(
            persist_dir=self.cfg.paths.chroma_dir,
            collection_name=self.cfg.chroma.collection_name,
            embeddings_cfg=EmbeddingsConfig(**self.cfg.embeddings.model_dump()),
            embedding_model=self.cfg.embedding.model_name,  # fallback：本地 bge-m3
        )

        stage_start = time.monotonic()
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
        logger.info(
            "Stage: manifest_load_scan | pdfs=%d | elapsed=%.2fs",
            len(current_files),
            time.monotonic() - stage_start,
        )
        removed = set(manifest.files.keys()) - set(current_files.keys())
        if removed:
            logger.info("Stage: removed_files | count=%d", len(removed))
        for path in removed:
            logger.info("Removed file; deleting entries: %s", path)
            store.delete_where({"source_file": path})
            manifest.files.pop(path, None)

        # Load PDFs (page-level)
        page_docs: List[Document] = []
        changed_files: Dict[str, ManifestEntry] = {}
        skipped_unchanged = 0
        skipped_same_hash = 0
        cache_hits = 0
        parsed_pages = 0
        stage_start = time.monotonic()
        for path_str, pdf_path in current_files.items():
            stat = pdf_path.stat()
            entry = manifest.files.get(path_str)
            if entry and entry.mtime_ns == stat.st_mtime_ns and entry.size == stat.st_size:
                skipped_unchanged += 1
                logger.debug("File unchanged; skipped: %s", path_str)
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
                skipped_same_hash += 1
                logger.info("File metadata updated only (hash unchanged): %s", path_str)
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
                cache_hits += 1
                parsed_pages += len(cached)
                continue

            try:
                docs = load_pdf_pages(pdf_path)
            except Exception as e:
                logger.warning("Failed to parse PDF: %s error=%s", pdf_path, e)
                continue
            if docs:
                _save_pages_cache(cache_path, docs)
                page_docs.extend(docs)
                parsed_pages += len(docs)
            else:
                logger.warning("Parsed 0 pages; skipping PDF: %s", pdf_path)

        logger.info(
            "Stage: pdf_parse | changed=%d skipped=%d metadata_only=%d cache_hits=%d pages=%d | elapsed=%.2fs",
            len(changed_files),
            skipped_unchanged,
            skipped_same_hash,
            cache_hits,
            parsed_pages,
            time.monotonic() - stage_start,
        )

        # Optional: table extraction (as extra docs)
        if self.enable_table_extraction:
            stage_start = time.monotonic()
            tex = TableExtractor(flavor=self.table_flavor)
            # naive: run over files again
            table_docs = 0
            for path_str, pdf_path in current_files.items():
                if path_str not in changed_files:
                    continue
                extracted = tex.extract(pdf_path)
                table_docs += len(extracted)
                page_docs.extend(extracted)
            logger.info(
                "Stage: table_extraction | tables=%d | elapsed=%.2fs",
                table_docs,
                time.monotonic() - stage_start,
            )

        if not page_docs:
            manifest.save(manifest_path)
            return {
                "pdfs_scanned": len(current_files),
                "pdfs_changed": len(changed_files),
                "pdfs_removed": len(removed),
                "pages_parsed": parsed_pages,
                "docs": 0,
                "chunks": 0,
                "indexed": 0,
                "skipped": skipped_unchanged + skipped_same_hash,
                "elapsed_s": round(time.monotonic() - overall_start, 2),
            }

        # Page-level metadata extraction
        stage_start = time.monotonic()
        pages_with_stage = 0
        pages_with_gas = 0
        pages_with_T = 0
        pages_with_P = 0
        for d in tqdm(page_docs, desc="Extract page metadata"):
            base_meta = dict(d.metadata or {})
            meta = extractor.extract(d)
            for key in ("source_file", "page", "page_label", "section"):
                if base_meta.get(key) is not None:
                    meta[key] = base_meta[key]
            d.metadata = meta
            if meta.get("stage") and meta.get("stage") != "unknown":
                pages_with_stage += 1
            if meta.get("gas_agent"):
                pages_with_gas += 1
            if meta.get("T_K") is not None or meta.get("T_range_K") is not None:
                pages_with_T += 1
            if meta.get("P_MPa") is not None or meta.get("P_range_MPa") is not None:
                pages_with_P += 1
        logger.info(
            "Stage: page_metadata | pages=%d stage=%d gas=%d T=%d P=%d | elapsed=%.2fs",
            len(page_docs),
            pages_with_stage,
            pages_with_gas,
            pages_with_T,
            pages_with_P,
            time.monotonic() - stage_start,
        )

        # Chunk
        stage_start = time.monotonic()
        chunks = split_page_docs(
            page_docs,
            chunk_size=self.cfg.chunking.chunk_size,
            chunk_overlap=self.cfg.chunking.chunk_overlap,
        )
        if chunks:
            avg_len = sum(len(c.page_content or "") for c in chunks) / len(chunks)
        else:
            avg_len = 0.0
        logger.info(
            "Stage: chunking | chunks=%d avg_len=%.1f | elapsed=%.2fs",
            len(chunks),
            avg_len,
            time.monotonic() - stage_start,
        )

        indexed_docs: Dict[str, Document] = {}
        stage_start = time.monotonic()
        chunks_with_gas_flags = 0
        chunks_with_target_flags = 0
        chunks_with_any_flags = 0
        for i, ch in enumerate(tqdm(chunks, desc="Enrich metadata")):
            # generate chunk_id deterministically
            src = (ch.metadata or {}).get("source_file", "unknown")
            page = str((ch.metadata or {}).get("page", ""))
            section = str((ch.metadata or {}).get("section", "unknown"))

            chunk_id = stable_chunk_id(src, page, section, ch.page_content[:200])
            meta: Dict[str, object] = dict(ch.metadata or {})

            chunk_meta = extractor.extract(Document(page_content=ch.page_content, metadata={}))
            _merge_list_field(meta, "targets", chunk_meta.get("targets"))

            chunk_gas = chunk_meta.get("gas_agent")
            if chunk_gas:
                _merge_list_field(meta, "gas_agent", chunk_gas)

            if meta.get("stage") in (None, "unknown") and chunk_meta.get("stage") not in (None, "unknown"):
                meta["stage"] = chunk_meta.get("stage")

            for key in (
                "T_K",
                "T_range_K",
                "T_min_K",
                "T_max_K",
                "P_MPa",
                "P_range_MPa",
                "P_min_MPa",
                "P_max_MPa",
                "coal_name",
                "ratios",
            ):
                if meta.get(key) is None and chunk_meta.get(key) is not None:
                    meta[key] = chunk_meta.get(key)

            # flatten meta for robust filtering
            meta = flatten_for_filtering(meta, onto)

            # attach chunk_id + canonical source/page for citations
            meta["chunk_id"] = chunk_id
            meta.setdefault("source_file", src)

            meta = _trim_metadata(meta)
            meta = _stringify_metadata(meta)

            has_gas_flags = any(k.startswith("gas_") and meta.get(k) is True for k in meta)
            has_target_flags = any(k.startswith("has_") and meta.get(k) is True for k in meta)
            if has_gas_flags:
                chunks_with_gas_flags += 1
            if has_target_flags:
                chunks_with_target_flags += 1
            if has_gas_flags or has_target_flags:
                chunks_with_any_flags += 1

            # store the chunk text
            indexed_docs[chunk_id] = Document(page_content=ch.page_content, metadata=meta)

        logger.info(
            "Stage: chunk_enrichment | chunks=%d gas_flags=%d target_flags=%d any_flags=%d | elapsed=%.2fs",
            len(indexed_docs),
            chunks_with_gas_flags,
            chunks_with_target_flags,
            chunks_with_any_flags,
            time.monotonic() - stage_start,
        )

        for path_str in changed_files.keys():
            store.delete_where({"source_file": path_str})
            logger.info("File changed; deleted entries: %s", path_str)

        docs_to_add = list(indexed_docs.values())
        stage_start = time.monotonic()
        batch_size = 128
        ids = list(indexed_docs.keys())
        for start in range(0, len(docs_to_add), batch_size):
            batch_docs = docs_to_add[start : start + batch_size]
            batch_ids = ids[start : start + batch_size]
            try:
                store.add_documents(batch_docs, ids=batch_ids)
            except Exception as e:
                batch_sources = {d.metadata.get("source_file") for d in batch_docs}
                pages = [d.metadata.get("page") for d in batch_docs if isinstance(d.metadata.get("page"), int)]
                page_range = f"{min(pages)}-{max(pages)}" if pages else "unknown"
                logger.error(
                    "Embedding/upsert failed | batch=%d-%d sources=%s pages=%s error=%s",
                    start,
                    start + len(batch_docs) - 1,
                    sorted(s for s in batch_sources if s),
                    page_range,
                    e,
                )
                if force:
                    logger.warning("Force enabled; continuing after batch failure.")
                    continue
                raise
        logger.info(
            "Stage: vectorstore_write | docs_to_add=%d batch_size=%d | elapsed=%.2fs",
            len(docs_to_add),
            batch_size,
            time.monotonic() - stage_start,
        )

        for path_str, entry in changed_files.items():
            entry.chunk_count = sum(1 for d in docs_to_add if d.metadata.get("source_file") == path_str)
            manifest.files[path_str] = entry

        stage_start = time.monotonic()
        manifest.save(manifest_path)
        logger.info("Stage: manifest_save | elapsed=%.2fs", time.monotonic() - stage_start)
        logger.info(
            "Ingest complete: page_docs=%d chunks=%d indexed=%d elapsed=%.2fs",
            len(page_docs),
            len(chunks),
            len(docs_to_add),
            time.monotonic() - overall_start,
        )

        return {
            "pdfs_scanned": len(current_files),
            "pdfs_changed": len(changed_files),
            "pdfs_removed": len(removed),
            "pages_parsed": parsed_pages,
            "docs": len(page_docs),
            "chunks": len(chunks),
            "indexed": len(docs_to_add),
            "skipped": skipped_unchanged + skipped_same_hash,
            "elapsed_s": round(time.monotonic() - overall_start, 2),
        }

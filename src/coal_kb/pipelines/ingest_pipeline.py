from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.documents import Document
from tqdm import tqdm
from ..llm.factory import LLMConfig
from coal_kb.embeddings.factory import EmbeddingsConfig, make_embeddings
from ..chunking.splitter import split_page_docs
from ..metadata.extract import MetadataExtractor
from ..metadata.normalize import Ontology, flatten_for_filtering
from ..parsing.pdf_loader import load_pdf_pages
from ..parsing.table_extractor import TableExtractor
from ..settings import AppConfig
from ..store.chroma_store import ChromaStore
from ..store.elastic_store import ElasticStore
from ..store.manifest import Manifest, ManifestEntry
from ..store.registry_sqlite import RegistrySQLite
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

    def run(
        self,
        *,
        rebuild: bool = False,
        force: bool = False,
        elastic_index_override: Optional[str] = None,
    ) -> dict:
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

        backend = self.cfg.backend
        if backend not in {"chroma", "elastic", "both"}:
            raise ValueError(f"Unsupported backend: {backend}")

        registry = RegistrySQLite(self.cfg.registry.sqlite_path)

        store: Optional[ChromaStore] = None
        if backend in {"chroma", "both"}:
            store = ChromaStore(
                persist_dir=self.cfg.paths.chroma_dir,
                collection_name=self.cfg.chroma.collection_name,
                embeddings_cfg=EmbeddingsConfig(**self.cfg.embeddings.model_dump()),
                embedding_model=self.cfg.embedding.model_name,  # fallback：本地 bge-m3
            )

        elastic_store: Optional[ElasticStore] = None
        elastic_index: Optional[str] = None
        embeddings = None
        if backend in {"elastic", "both"}:
            elastic_store = ElasticStore(
                host=self.cfg.elastic.host,
                verify_certs=self.cfg.elastic.verify_certs,
            )
            embeddings = make_embeddings(EmbeddingsConfig(**self.cfg.embeddings.model_dump()))

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

        embedding_version = self.cfg.model_versions.embedding_version
        embedding_dim = self.cfg.embeddings.dimensions or 0
        if embeddings is not None and not embedding_dim:
            embedding_dim = len(embeddings.embed_query("dimension probe"))
        registry.log_model(
            embedding_model=self.cfg.embeddings.model,
            embedding_dim=embedding_dim,
            embedding_version=embedding_version,
        )
        if elastic_store and embeddings is not None:
            dims = embedding_dim or len(embeddings.embed_query("dimension probe"))
            if elastic_index_override:
                elastic_index = elastic_index_override
                elastic_store.create_index(elastic_index, dims)
            else:
                current = elastic_store.resolve_current_index(self.cfg.elastic.alias_current)
                if rebuild or not current:
                    schema_hash = schema_sig[:8]
                    index_name = elastic_store.build_index_name(
                        index_prefix=self.cfg.elastic.index_prefix,
                        embedding_version=embedding_version,
                        schema_hash=schema_hash,
                    )
                    elastic_store.create_index(index_name, dims)
                    elastic_store.switch_alias(
                        alias_current=self.cfg.elastic.alias_current,
                        alias_prev=self.cfg.elastic.alias_prev,
                        new_index=index_name,
                    )
                    elastic_index = index_name
                else:
                    elastic_index = self.cfg.elastic.alias_current

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
            entry = manifest.files.get(path)
            document_id = entry.sha256 if entry else stable_chunk_id(path)
            if store:
                store.delete_where({"source_file": path})
            if elastic_store and elastic_index:
                elastic_store.delete_by_document_id(elastic_index, document_id)
            registry.delete_chunks_by_document_id(document_id)
            if entry:
                registry.upsert_document(
                    document_id=document_id,
                    source_file=path,
                    sha256=entry.sha256,
                    mtime=entry.mtime_ns,
                    size=entry.size,
                    status="removed",
                )
            manifest.files.pop(path, None)

        # Load PDFs (page-level)
        page_docs: List[Document] = []
        changed_files: Dict[str, ManifestEntry] = {}
        changed_old_doc_ids: Dict[str, str] = {}
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
                registry.upsert_document(
                    document_id=file_hash,
                    source_file=path_str,
                    sha256=file_hash,
                    mtime=stat.st_mtime_ns,
                    size=stat.st_size,
                    status="active",
                )
                continue

            changed_files[path_str] = ManifestEntry(
                path=path_str,
                mtime_ns=stat.st_mtime_ns,
                size=stat.st_size,
                sha256=file_hash,
            )
            if entry:
                changed_old_doc_ids[path_str] = entry.sha256

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

        document_id_by_source = {
            path_str: entry.sha256 for path_str, entry in changed_files.items()
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
        registry_chunks: List[Dict[str, Any]] = []
        es_docs: List[Dict[str, Any]] = []
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
            document_id = document_id_by_source.get(str(src), stable_chunk_id(str(src)))
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
            meta_for_registry = _stringify_metadata(meta)

            has_gas_flags = any(k.startswith("gas_") and meta.get(k) is True for k in meta)
            has_target_flags = any(k.startswith("has_") and meta.get(k) is True for k in meta)
            if has_gas_flags:
                chunks_with_gas_flags += 1
            if has_target_flags:
                chunks_with_target_flags += 1
            if has_gas_flags or has_target_flags:
                chunks_with_any_flags += 1

            # store the chunk text
            if store:
                indexed_docs[chunk_id] = Document(page_content=ch.page_content, metadata=meta_for_registry)

            registry_chunks.append(
                {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "page": ch.metadata.get("page"),
                    "section": ch.metadata.get("section"),
                    "chunk_index": i,
                    "text": ch.page_content,
                    "metadata_json": json.dumps(meta_for_registry, ensure_ascii=False),
                    "embedding_model": self.cfg.embeddings.model,
                    "embedding_dim": int(embedding_dim),
                    "embedding_version": embedding_version,
                }
            )

            if elastic_store and elastic_index:
                es_doc = {
                    "chunk_id": chunk_id,
                    "document_id": document_id,
                    "source_file": meta.get("source_file"),
                    "page": ch.metadata.get("page"),
                    "page_label": ch.metadata.get("page_label"),
                    "section": ch.metadata.get("section"),
                    "chunk_index": i,
                    "text": ch.page_content,
                    "stage": meta.get("stage"),
                    "gas_agent": meta.get("gas_agent"),
                    "targets": meta.get("targets"),
                    "T_K": meta.get("T_K"),
                    "T_min_K": meta.get("T_min_K"),
                    "T_max_K": meta.get("T_max_K"),
                    "P_MPa": meta.get("P_MPa"),
                    "P_min_MPa": meta.get("P_min_MPa"),
                    "P_max_MPa": meta.get("P_max_MPa"),
                    "coal_name": meta.get("coal_name"),
                    "metadata_json": json.dumps(meta_for_registry, ensure_ascii=False),
                }
                for key, value in meta.items():
                    if key.startswith("gas_") or key.startswith("has_"):
                        es_doc[key] = value
                es_docs.append(es_doc)

        logger.info(
            "Stage: chunk_enrichment | chunks=%d gas_flags=%d target_flags=%d any_flags=%d | elapsed=%.2fs",
            len(chunks),
            chunks_with_gas_flags,
            chunks_with_target_flags,
            chunks_with_any_flags,
            time.monotonic() - stage_start,
        )

        for path_str, entry in changed_files.items():
            registry.upsert_document(
                document_id=entry.sha256,
                source_file=path_str,
                sha256=entry.sha256,
                mtime=entry.mtime_ns,
                size=entry.size,
                status="active",
            )
        registry.upsert_chunks_bulk(registry_chunks)

        for path_str in changed_files.keys():
            if store:
                store.delete_where({"source_file": path_str})
            old_doc_id = changed_old_doc_ids.get(path_str)
            if old_doc_id:
                if elastic_store and elastic_index:
                    elastic_store.delete_by_document_id(elastic_index, old_doc_id)
                registry.delete_chunks_by_document_id(old_doc_id)
            logger.info("File changed; deleted entries: %s", path_str)

        docs_to_add = list(indexed_docs.values())
        if store:
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

        if elastic_store and elastic_index and embeddings is not None:
            stage_start = time.monotonic()
            batch_size = 64
            for start in range(0, len(es_docs), batch_size):
                batch_docs = es_docs[start : start + batch_size]
                texts = [d["text"] for d in batch_docs]
                try:
                    vectors = embeddings.embed_documents(texts)
                except Exception as e:
                    logger.error(
                        "Embedding failed | batch=%d-%d error=%s",
                        start,
                        start + len(batch_docs) - 1,
                        e,
                    )
                    if force:
                        logger.warning("Force enabled; continuing after embedding failure.")
                        continue
                    raise
                for doc, vec in zip(batch_docs, vectors):
                    doc["embedding"] = vec
                elastic_store.bulk_upsert_chunks(elastic_index, batch_docs)
            logger.info(
                "Stage: elastic_write | docs_to_add=%d batch_size=%d | elapsed=%.2fs",
                len(es_docs),
                batch_size,
                time.monotonic() - stage_start,
            )

        for path_str, entry in changed_files.items():
            if store:
                entry.chunk_count = sum(
                    1 for d in docs_to_add if d.metadata.get("source_file") == path_str
                )
            else:
                entry.chunk_count = sum(
                    1 for d in es_docs if d.get("source_file") == path_str
                )
            manifest.files[path_str] = entry

        stage_start = time.monotonic()
        manifest.save(manifest_path)
        logger.info("Stage: manifest_save | elapsed=%.2fs", time.monotonic() - stage_start)
        indexed_count = len(docs_to_add) if store else len(es_docs)
        logger.info(
            "Ingest complete: page_docs=%d chunks=%d indexed=%d elapsed=%.2fs",
            len(page_docs),
            len(chunks),
            indexed_count,
            time.monotonic() - overall_start,
        )

        return {
            "pdfs_scanned": len(current_files),
            "pdfs_changed": len(changed_files),
            "pdfs_removed": len(removed),
            "pages_parsed": parsed_pages,
            "docs": len(page_docs),
            "chunks": len(chunks),
            "indexed": indexed_count,
            "skipped": skipped_unchanged + skipped_same_hash,
            "elapsed_s": round(time.monotonic() - overall_start, 2),
        }

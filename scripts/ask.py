#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coal KB Ask (enterprise-ish)
- Hybrid retrieve (Chroma / Elastic / both)
- On-demand LLM metadata enrichment for only top candidates (lazy, save cost)
- Persistent enrichment cache (sqlite) => resumable + no duplicate spend
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import signal
import sqlite3
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from langchain_core.documents import Document

from coal_kb.cli_ui import print_banner, print_kv, print_stats_table
from coal_kb.embeddings.factory import EmbeddingsConfig
from coal_kb.logging import setup_logging
from coal_kb.llm.factory import LLMConfig
from coal_kb.metadata.normalize import Ontology
from coal_kb.qa.rag_answer import RAGAnswerer
from coal_kb.retrieval.bm25 import rrf_fuse
from coal_kb.retrieval.elastic_retriever import make_elastic_retriever_factory
from coal_kb.retrieval.filter_parser import FilterParser
from coal_kb.retrieval.query_rewrite import rewrite_query
from coal_kb.retrieval.retriever import ExpertRetriever
from coal_kb.settings import load_config
from coal_kb.store.chroma_store import ChromaStore
from coal_kb.store.elastic_store import ElasticStore
from coal_kb.store.registry_sqlite import RegistrySQLite

logger = logging.getLogger(__name__)


# -----------------------------
# Helpers: safe retriever invoke
# -----------------------------
def _invoke_retriever(ret: Any, query: str) -> List[Document]:
    """
    LangChain retrievers changed APIs across versions:
    - new: retriever.invoke(query)
    - old: retriever.get_relevant_documents(query)
    """
    if hasattr(ret, "invoke"):
        return ret.invoke(query)  # type: ignore
    if hasattr(ret, "get_relevant_documents"):
        return ret.get_relevant_documents(query)  # type: ignore
    raise TypeError(f"Retriever {type(ret)} has neither invoke() nor get_relevant_documents().")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime())


def _sha1_text(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()


def _json_dumps(x: Any) -> str:
    return json.dumps(x, ensure_ascii=False, sort_keys=True)


# -----------------------------
# Enrichment cache (sqlite)
# -----------------------------
@dataclass
class EnrichCache:
    db_path: Path

    def __post_init__(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS enrich_cache (
              chunk_id TEXT NOT NULL,
              content_sha1 TEXT NOT NULL,
              model_key TEXT NOT NULL,
              schema_sig TEXT NOT NULL,
              enriched_json TEXT NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY(chunk_id, content_sha1, model_key, schema_sig)
            );
            """
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.commit()

    def get(
        self, *, chunk_id: str, content_sha1: str, model_key: str, schema_sig: str
    ) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute(
            """
            SELECT enriched_json FROM enrich_cache
            WHERE chunk_id=? AND content_sha1=? AND model_key=? AND schema_sig=?
            """,
            (chunk_id, content_sha1, model_key, schema_sig),
        )
        row = cur.fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except Exception:
            return None

    def put(
        self,
        *,
        chunk_id: str,
        content_sha1: str,
        model_key: str,
        schema_sig: str,
        enriched: Dict[str, Any],
    ) -> None:
        self._conn.execute(
            """
            INSERT OR REPLACE INTO enrich_cache
            (chunk_id, content_sha1, model_key, schema_sig, enriched_json, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (chunk_id, content_sha1, model_key, schema_sig, _json_dumps(enriched), _now_iso()),
        )
        self._conn.commit()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass


# -----------------------------
# Metadata heuristics (free)
# -----------------------------
_GAS_AGENTS = {"steam", "h2o", "co2", "o2", "air", "ar", "n2", "h2", "ch4"}
_TARGETS = {"nh3", "hcn", "h2s", "so2", "no", "nox", "co", "co2", "h2", "ch4", "tar", "phenols", "phenol"}

_STAGE_WORDS = {
    "gasification": ["gasification", "气化", "地下气化", "ucg"],
    "pyrolysis": ["pyrolysis", "热解"],
    "combustion": ["combustion", "燃烧"],
    "oxidation": ["oxidation", "氧化"],
}


def _guess_stage(text: str) -> Optional[str]:
    tl = text.lower()
    for stage, kws in _STAGE_WORDS.items():
        if any(k.lower() in tl for k in kws):
            return stage
    return None


def _extract_temperature_K(text: str) -> Optional[float]:
    """
    Try: 1200 K, 800 C, 800°C
    """
    tl = text.lower()
    m = re.search(r"(\d{2,4}(?:\.\d+)?)\s*k\b", tl)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m = re.search(r"(\d{2,4}(?:\.\d+)?)\s*°?\s*c\b", tl)
    if m:
        try:
            c = float(m.group(1))
            return c + 273.15
        except Exception:
            pass
    return None


def _extract_pressure_MPa(text: str) -> Optional[float]:
    """
    Try: 2 MPa, 0.1 MPa, 10 bar, 1 atm
    """
    tl = text.lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*mpa\b", tl)
    if m:
        try:
            return float(m.group(1))
        except Exception:
            pass
    m = re.search(r"(\d+(?:\.\d+)?)\s*bar\b", tl)
    if m:
        try:
            return float(m.group(1)) * 0.1
        except Exception:
            pass
    m = re.search(r"(\d+(?:\.\d+)?)\s*atm\b", tl)
    if m:
        try:
            return float(m.group(1)) * 0.101325
        except Exception:
            pass
    return None


def _extract_list_by_vocab(text: str, vocab: set[str]) -> List[str]:
    tl = text.lower()
    found: List[str] = []
    for w in vocab:
        # word boundary-ish for ascii tokens
        if re.search(rf"\b{re.escape(w.lower())}\b", tl):
            found.append(w.upper() if w.isalpha() and len(w) <= 4 else w)
    # normalize some common
    norm = []
    for x in found:
        if x.lower() == "h2o":
            norm.append("steam")
        elif x.lower() == "nox":
            norm.append("NOx")
        else:
            norm.append(x)
    # unique keep order
    out = []
    seen = set()
    for x in norm:
        k = x.lower()
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _apply_flags(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert gas_agent/targets into boolean flags used in your filtering/scoring:
      - gas_agent: ["steam", "co2"] -> gas_steam=True, gas_co2=True
      - targets: ["NH3"] -> has_NH3=True
    """
    out = dict(meta)
    gas_agent = out.get("gas_agent")
    if isinstance(gas_agent, list):
        for g in gas_agent:
            if not isinstance(g, str):
                continue
            key = "gas_" + g.strip().lower()
            out[key] = True

    targets = out.get("targets")
    if isinstance(targets, list):
        for t in targets:
            if not isinstance(t, str):
                continue
            # keep original casing in has_*
            key = "has_" + t.strip()
            out[key] = True
    return out


def _heuristic_enrich(doc: Document) -> Dict[str, Any]:
    text = doc.page_content or ""
    meta = dict(doc.metadata or {})

    # stage
    if not meta.get("stage"):
        st = _guess_stage(text)
        if st:
            meta["stage"] = st

    # T_K / P_MPa
    if meta.get("T_K") is None:
        tk = _extract_temperature_K(text)
        if tk is not None:
            meta["T_K"] = float(tk)
    if meta.get("P_MPa") is None:
        pm = _extract_pressure_MPa(text)
        if pm is not None:
            meta["P_MPa"] = float(pm)

    # gas_agent / targets
    if meta.get("gas_agent") in (None, "", []):
        ga = _extract_list_by_vocab(text, _GAS_AGENTS)
        if ga:
            # normalize
            meta["gas_agent"] = [g.lower() for g in ga]
    if meta.get("targets") in (None, "", []):
        tg = _extract_list_by_vocab(text, _TARGETS)
        if tg:
            # keep chemical casing
            meta["targets"] = [t if t != "steam" else "steam" for t in tg]

    return _apply_flags(meta)


# -----------------------------
# LLM enrichment (paid) - on demand
# -----------------------------
def _make_lightweight_llm(provider: str, cfg_llm: LLMConfig):
    """
    Try to reuse your project's LLM factory. If not available, fail gracefully.
    """
    try:
        # Many repos expose make_chat_model in coal_kb.llm.factory
        from coal_kb.llm.factory import make_chat_model  # type: ignore

        return make_chat_model(cfg_llm)
    except Exception:
        return None


def _llm_enrich_one(
    *,
    llm: Any,
    onto: Ontology,
    query: str,
    doc: Document,
    max_chars: int = 2500,
) -> Optional[Dict[str, Any]]:
    """
    Ask LLM to extract metadata as JSON.
    Keep prompt small to save tokens.
    """
    text = (doc.page_content or "").strip()
    if len(text) > max_chars:
        text = text[:max_chars]

    # Keep schema minimal and aligned to your filters
    schema_hint = {
        "stage": ["gasification", "pyrolysis", "combustion", "oxidation", None],
        "T_K": "number or null",
        "P_MPa": "number or null",
        "gas_agent": "list[str] or null (steam/co2/o2/air/n2/ar/h2/ch4)",
        "targets": "list[str] or null (NH3/HCN/NOx/H2S/SO2/CO/CO2/H2/CH4/tar/phenols ...)",
        "section": "one of (abstract/introduction/methods/results/discussion/conclusion/references/unknown) or null",
    }

    prompt = (
        "You are an information extraction system for coal thermochemical papers.\n"
        "Extract ONLY the requested fields as strict JSON. No extra keys, no markdown.\n"
        f"User query: {query}\n"
        f"JSON schema: {json.dumps(schema_hint, ensure_ascii=False)}\n"
        "Text:\n"
        f"{text}\n"
        "Return JSON now."
    )

    try:
        # LangChain chat model typically supports invoke(str)
        resp = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
        content = getattr(resp, "content", resp)
        if not isinstance(content, str):
            content = str(content)

        # best-effort JSON parse
        m = re.search(r"\{.*\}", content, flags=re.S)
        if not m:
            return None
        obj = json.loads(m.group(0))
        if not isinstance(obj, dict):
            return None
        # sanitize types
        out: Dict[str, Any] = {}
        for k in ("stage", "T_K", "P_MPa", "gas_agent", "targets", "section"):
            if k in obj:
                out[k] = obj[k]
        # normalize lists
        if isinstance(out.get("gas_agent"), list):
            out["gas_agent"] = [str(x).lower() for x in out["gas_agent"] if isinstance(x, (str, int, float))]
        if isinstance(out.get("targets"), list):
            out["targets"] = [str(x) for x in out["targets"] if isinstance(x, (str, int, float))]
        # numbers
        for nk in ("T_K", "P_MPa"):
            if out.get(nk) is not None:
                try:
                    out[nk] = float(out[nk])
                except Exception:
                    out[nk] = None
        return _apply_flags(out)
    except Exception as e:
        logger.warning("LLM enrich failed: %s", e)
        return None


def _should_enrich(parsed_filter: Any, trace: Dict[str, Any]) -> bool:
    """
    Trigger enrich only when we likely lost candidates due to missing metadata.
    Heuristic: postfiltered becomes 0 while vector/fused had >0.
    """
    try:
        counts = trace.get("counts") or {}
        return (counts.get("fused", 0) > 0) and (counts.get("postfiltered", 0) == 0)
    except Exception:
        return False


def _merge_meta(doc: Document, enriched: Dict[str, Any]) -> None:
    meta = dict(doc.metadata or {})
    # do not clobber existing keys unless missing
    for k, v in enriched.items():
        if meta.get(k) in (None, "", [], {}):
            meta[k] = v
        # flags are safe to set
        if k.startswith("gas_") or k.startswith("has_"):
            meta[k] = v
    doc.metadata = meta


# -----------------------------
# Printing
# -----------------------------
def _print_docs(docs: List[Document]) -> None:
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        src = meta.get("source_file", "unknown")
        page = meta.get("page", "?")
        cid = meta.get("chunk_id", meta.get("id", ""))
        print(f"\n[{i}] {src} (page {page}) #{cid}")
        snippet = (d.page_content or "").strip().replace("\n", " ")
        print(snippet[:600] + ("..." if len(snippet) > 600 else ""))


def _print_trace(trace: Dict[str, Any], docs: List[Document]) -> None:
    print("\nRetrieval trace:")
    if trace.get("where") is not None:
        print(f"  where: {trace.get('where')}")
    counts = trace.get("counts") or {}
    if counts:
        print(
            f"  counts: vector={counts.get('vector', 0)} fused={counts.get('fused', 0)} postfiltered={counts.get('postfiltered', 0)}"
        )
    tv = trace.get("top_vector_candidates") or []
    if tv:
        print("  top vector candidates:")
        for x in tv[:3]:
            print(f"    - {x}")
    te = trace.get("top_evidence_citations") or []
    if te:
        print("  top evidence citations:")
        for x in te[:5]:
            print(f"    - {x}")
    else:
        print("  top evidence citations: (none)")
    if docs:
        print(f"  returned: {len(docs)}")


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the expert KB with on-demand enrichment.")
    parser.add_argument("--k", type=int, default=None, help="Final top-k docs to return.")
    parser.add_argument("--llm", action="store_true", help="Enable LLM answer generation.")
    parser.add_argument("--backend", default=None, choices=["chroma", "elastic", "both"], help="Override backend.")
    parser.add_argument("--mode", default=None, choices=["strict", "balanced", "broad"], help="Constraint mode.")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder rerank (if configured).")
    parser.add_argument("--enrich", action="store_true", help="Enable on-demand LLM metadata enrich (paid).")
    parser.add_argument("--enrich-top-n", type=int, default=20, help="How many top candidates to enrich when triggered.")
    parser.add_argument("--enrich-cache", default="storage/enrich_cache.db", help="SQLite path for enrich cache.")
    parser.add_argument("--print-trace", action="store_true", help="Print retrieval trace.")
    args = parser.parse_args()

    setup_logging()
    cfg = load_config()

    backend = args.backend or getattr(cfg, "backend", "chroma")
    print_banner("Coal KB Ask", subtitle=f"backend={backend}")

    onto = Ontology.load("configs/schema.yaml")
    filter_parser = FilterParser(onto=onto)

    # query rewrite (optional)
    qr_cfg = getattr(cfg, "query_rewrite", None)

    # retrieval config
    retr_cfg = getattr(cfg, "retrieval", None)
    k = int(args.k or (getattr(retr_cfg, "k", 5) if retr_cfg else 5))
    candidates = int(getattr(retr_cfg, "candidates", 50) if retr_cfg else 50)
    rrf_k = int(getattr(retr_cfg, "rrf_k", 60) if retr_cfg else 60)
    max_per_source = int(getattr(retr_cfg, "max_per_source", 2) if retr_cfg else 2)

    # drop sections / reference-like
    drop_sections = list(getattr(retr_cfg, "drop_sections", []) if retr_cfg else [])
    drop_reference_like = bool(getattr(retr_cfg, "drop_reference_like", True) if retr_cfg else True)

    # LLM config (for answer + enrich)
    llm_provider = getattr(cfg.llm, "provider", "none")
    llm_cfg = LLMConfig(**cfg.llm.model_dump())

    # embeddings config for stores
    emb_cfg = EmbeddingsConfig(**cfg.embeddings.model_dump())

    # stores
    chroma_store: Optional[ChromaStore] = None
    if backend in ("chroma", "both"):
        chroma_store = ChromaStore(
            persist_dir=cfg.paths.chroma_dir,
            collection_name=cfg.chroma.collection_name,
            embeddings_cfg=emb_cfg,
            embedding_model=cfg.embedding.model_name,
        )

    elastic_store: Optional[ElasticStore] = None
    elastic_factory = None
    if backend in ("elastic", "both"):
        timeout_s = getattr(cfg.elastic, "timeout_s", 60)
        elastic_store = ElasticStore(
            host=cfg.elastic.host,
            verify_certs=cfg.elastic.verify_certs,
            timeout_s=timeout_s,
        )
        elastic_factory = make_elastic_retriever_factory(
            store=elastic_store,
            index_alias=cfg.elastic.index_alias,
            k_rrf=rrf_k,
        )

    # registry (optional usage)
    registry = RegistrySQLite(cfg.registry.sqlite_path)

    # enrichment cache
    schema_sig = _sha1_text(Path("configs/schema.yaml").read_text(encoding="utf-8", errors="ignore"))[:12]
    model_key = f"{llm_provider}:{getattr(cfg.llm, 'model', 'unknown')}"
    enrich_cache = EnrichCache(db_path=Path(args.enrich_cache))

    # handle Ctrl+C gracefully
    interrupted = {"flag": False}

    def _sigint(_signum, _frame):
        interrupted["flag"] = True
        print("\n[CTRL+C] interrupted. Cached enrich results are saved; re-run query to continue.")
        # do not exit immediately; allow cleanup
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, _sigint)

    # interactive loop
    while True:
        try:
            query = input("\n你的问题> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            break

        # rewrite query (free/cheap depending on your config)
        rewritten = query
        if qr_cfg and getattr(qr_cfg, "enabled", False):
            try:
                rewritten = rewrite_query(cfg, query)
            except Exception as e:
                logger.warning("Query rewrite failed: %s", e)

        parsed = filter_parser.parse(rewritten)
        print("\n解析到的过滤条件:")
        print(_json_dumps(parsed.to_dict() if hasattr(parsed, "to_dict") else parsed.__dict__))

        # build vector retriever factory
        def chroma_factory(k_: int, where: Optional[dict] = None):
            assert chroma_store is not None
            vs = chroma_store._vs  # internal
            kw = {"k": k_}
            if where:
                kw["filter"] = where
            return vs.as_retriever(search_kwargs=kw)

        def vector_factory(k_: int, where: Optional[dict] = None):
            if backend == "chroma":
                return chroma_factory(k_, where)
            if backend == "elastic":
                assert elastic_factory is not None
                return elastic_factory(k_, where)
            # both: we still use chroma as vector, elastic for bm25 fuse
            return chroma_factory(k_, where)

        expert = ExpertRetriever(
            vector_retriever_factory=vector_factory,
            k=k,
            k_candidates=candidates,
            drop_sections=drop_sections,
            drop_reference_like=drop_reference_like,
            max_per_source=max_per_source,
        )

        # run retrieve
        trace: Dict[str, Any] = {}
        t0 = time.time()
        docs = expert.retrieve(rewritten, parsed, trace=trace)
        latency_ms = (time.time() - t0) * 1000

        # if both: fuse with elastic BM25 (optional, your ExpertRetriever may already do it;
        # we keep compatibility: do explicit BM25+RRF when backend=both and elastic available)
        if backend == "both" and elastic_store is not None:
            try:
                # bm25 candidates
                bm25_docs = elastic_store.bm25_search(
                    index_alias=cfg.elastic.index_alias,
                    query=rewritten,
                    k=candidates,
                    where=getattr(parsed, "compat_where", None),
                )
                # fuse vector docs (pre) with bm25 docs
                fused = rrf_fuse(docs, bm25_docs, k_rrf=rrf_k)
                docs = fused[:k]
            except Exception as e:
                logger.warning("BM25+RRF fuse skipped (error=%s)", e)

        # on-demand enrich trigger
        did_enrich = False
        if args.enrich and _should_enrich(parsed, trace):
            # fetch raw candidates without the post-filter effects:
            try:
                where = getattr(parsed, "compat_where", None)
                raw_ret = vector_factory(args.enrich_top_n, where=where)
                cand_docs = _invoke_retriever(raw_ret, rewritten)
            except Exception as e:
                logger.warning("Candidate fetch for enrich failed: %s", e)
                cand_docs = []

            # build lightweight llm only if needed
            llm = _make_lightweight_llm(llm_provider, llm_cfg)

            for d in cand_docs[: args.enrich_top_n]:
                if interrupted["flag"]:
                    break
                meta = d.metadata or {}
                chunk_id = str(meta.get("chunk_id") or meta.get("id") or "")
                if not chunk_id:
                    # fallback stable id
                    chunk_id = _sha1_text((meta.get("source_file", "") + str(meta.get("page", "")) + (d.page_content or ""))[:500])

                content_sha1 = _sha1_text(d.page_content or "")

                cached = enrich_cache.get(
                    chunk_id=chunk_id,
                    content_sha1=content_sha1,
                    model_key=model_key,
                    schema_sig=schema_sig,
                )
                if cached:
                    _merge_meta(d, cached)
                    did_enrich = True
                    continue

                # always do free heuristic first
                heur = _heuristic_enrich(d)
                _merge_meta(d, heur)

                # if still too empty -> optionally call LLM
                need_llm = (
                    (d.metadata or {}).get("stage") in (None, "", [])
                    or (d.metadata or {}).get("gas_agent") in (None, "", [])
                    or (d.metadata or {}).get("targets") in (None, "", [])
                )
                enriched_final = dict(d.metadata or {})
                if need_llm and llm is not None:
                    llm_out = _llm_enrich_one(llm=llm, onto=onto, query=rewritten, doc=d)
                    if llm_out:
                        _merge_meta(d, llm_out)
                        enriched_final = dict(d.metadata or {})

                # cache (store the enriched subset fields to reduce size)
                keep_keys = ["stage", "T_K", "P_MPa", "gas_agent", "targets", "section"]
                # plus flags
                for k2 in list(enriched_final.keys()):
                    if k2.startswith("gas_") or k2.startswith("has_"):
                        keep_keys.append(k2)
                compact = {k2: enriched_final.get(k2) for k2 in keep_keys if k2 in enriched_final}
                enrich_cache.put(
                    chunk_id=chunk_id,
                    content_sha1=content_sha1,
                    model_key=model_key,
                    schema_sig=schema_sig,
                    enriched=compact,
                )
                did_enrich = True

            # after enriching candidates, rerun retrieve (still no extra cost for same chunks)
            t1 = time.time()
            docs = expert.retrieve(rewritten, parsed, trace=trace)
            latency_ms = (time.time() - t1) * 1000

        # output
        if args.print_trace:
            _print_trace(trace, docs)

        print_stats_table(
            {"docs": len(docs), "latency_ms": round(latency_ms, 2), "enriched": bool(did_enrich)}
        )

        if not docs:
            print("\n未检索到相关证据。建议：放宽条件（温度/压力范围）或减少过滤条件（气化剂/阶段）。")
            continue

        _print_docs(docs)

        if args.llm:
            try:
                answerer = RAGAnswerer(cfg=cfg, onto=onto)
                answer = answerer.answer(query=query, docs=docs)
                print("\n====================\nLLM Answer\n====================")
                print(answer)
            except Exception as e:
                logger.warning("LLM answer generation failed: %s", e)

    enrich_cache.close()


if __name__ == "__main__":
    main()

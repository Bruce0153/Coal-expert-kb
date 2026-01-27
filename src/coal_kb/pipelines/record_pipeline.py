from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from ..schema.models import ExperimentRecord
from ..schema.pollutants import normalize_pollutants_dict
from ..schema.validators import (
    safe_model_validate,
    validate_pollutants,
    validate_pressure_mpa,
    validate_temperature_k,
)
from ..settings import AppConfig
from ..store.sql_store import SQLiteStore
from ..utils.hash import stable_chunk_id

logger = logging.getLogger(__name__)

# sentences (EN/CN)
_SENT_SPLIT = re.compile(r"(?<=[\.\!\?。！？])\s+")

# simple numeric pattern for finding evidence lines
_NUM = r"\d+(?:\.\d+)?"


def build_signature(
    *,
    stage: str,
    coal_name: Optional[str],
    reactor_type: Optional[str],
    T_K: Optional[float],
    P_MPa: Optional[float],
    gas_agent: Optional[List[str]],
    ratios: Dict[str, float],
) -> str:
    """
    Signature for "same operating condition".
    Keep it stable and insensitive to ordering.
    """
    gas = ",".join(sorted([str(x).lower() for x in (gas_agent or [])]))
    rat = ",".join(f"{k}={ratios[k]:.4g}" for k in sorted(ratios.keys()))
    return stable_chunk_id(
        stage or "unknown",
        (coal_name or "").strip().lower(),
        (reactor_type or "").strip().lower(),
        f"{T_K:.3f}" if isinstance(T_K, (int, float)) else "",
        f"{P_MPa:.3f}" if isinstance(P_MPa, (int, float)) else "",
        gas,
        rat,
    )


def extract_evidence_quote(text: str, pollutants: Dict[str, Any]) -> Optional[str]:
    """
    Find sentences containing pollutant keys + a number (best-effort).
    This is crucial for later human review & LoRA supervision.
    """
    if not text.strip() or not pollutants:
        return None

    sents = _SENT_SPLIT.split(text.replace("\n", " "))
    keys = [k for k in pollutants.keys() if isinstance(k, str)]
    if not keys:
        return None

    hits: List[str] = []
    for s in sents:
        s_low = s.lower()
        if not re.search(_NUM, s):
            continue
        if any(k.lower() in s_low for k in keys):
            hits.append(s.strip())

    if not hits:
        return None

    # keep short & informative
    quote = " ".join(hits[:3])
    return quote[:800]


@dataclass
class RecordExtractor:
    enable_llm: bool = False
    llm_provider: str = "none"

    def extract_records(self, *, text: str) -> List[Dict[str, Any]]:
        if not self.enable_llm or self.llm_provider == "none":
            return []
        if self.llm_provider == "openai":
            return self._extract_with_openai(text=text)
        return []

    def _extract_with_openai(self, *, text: str) -> List[Dict[str, Any]]:
        try:
            from langchain_openai import ChatOpenAI  # type: ignore
            from langchain_core.messages import SystemMessage, HumanMessage
        except Exception as e:
            logger.warning("langchain-openai not installed; record extraction disabled. %s", e)
            return []

        prompt_path = "configs/prompts/record_extract.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            sys_prompt = f.read().strip()

        model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        msgs = [SystemMessage(content=sys_prompt), HumanMessage(content=f"片段:\n{text}\n\n请输出严格 JSON 数组。")]
        rsp = model.invoke(msgs)
        content = getattr(rsp, "content", None) or ""

        try:
            arr = json.loads(content)
            if isinstance(arr, dict):
                arr = [arr]
            if not isinstance(arr, list):
                return []
            return [x for x in arr if isinstance(x, dict)]
        except Exception:
            logger.debug("LLM returned non-JSON array; ignored.")
            return []


@dataclass
class RecordPipeline:
    cfg: AppConfig
    enable_llm_records: bool = False
    llm_provider: str = "none"
    allowed_sections: tuple[str, ...] = ("methods", "conditions", "results", "discussion", "table")

    def run(self, docs: List[Document]) -> dict:
        store = SQLiteStore(self.cfg.paths.sqlite_path)
        extractor = RecordExtractor(enable_llm=self.enable_llm_records, llm_provider=self.llm_provider)

        n_attempt = 0
        n_saved = 0
        n_conflict = 0

        for d in docs:
            meta = d.metadata or {}
            section = str(meta.get("section", "unknown")).lower()
            if section not in self.allowed_sections:
                continue

            n_attempt += 1
            records = extractor.extract_records(text=d.page_content)
            if not records:
                continue

            for rec in records:
                src = str(meta.get("source_file", rec.get("source_file", "unknown")))
                page = meta.get("page", None)
                chunk_id = str(meta.get("chunk_id") or stable_chunk_id(src, str(page), d.page_content[:120]))

                record_id = stable_chunk_id(src, str(page), chunk_id, json.dumps(rec, ensure_ascii=False)[:240])

                rec_payload = dict(rec)
                rec_payload["record_id"] = record_id
                rec_payload["source_file"] = src
                rec_payload.setdefault("stage", meta.get("stage", "unknown"))

                model = safe_model_validate(ExperimentRecord, rec_payload)

                validate_temperature_k(model.T_K)
                validate_pressure_mpa(model.P_MPa)

                # ---- normalize pollutants for training ----
                normalized_pollutants = normalize_pollutants_dict(model.pollutants)
                validate_pollutants(normalized_pollutants)

                # ---- evidence quote ----
                quote = extract_evidence_quote(d.page_content, normalized_pollutants)

                # ---- conflict detection by signature ----
                signature = build_signature(
                    stage=model.stage,
                    coal_name=model.coal_name,
                    reactor_type=model.reactor_type,
                    T_K=model.T_K,
                    P_MPa=model.P_MPa,
                    gas_agent=model.gas_agent,
                    ratios=model.ratios,
                )

                existing = store.find_any_by_signature(signature)
                is_conflict = 0
                if existing and existing.pollutants_json:
                    try:
                        old_pol = json.loads(existing.pollutants_json)
                    except Exception:
                        old_pol = None
                    # basic conflict rule: different numeric values for same key (norm if available)
                    if isinstance(old_pol, dict):
                        if self._pollutants_conflict(old_pol, normalized_pollutants):
                            is_conflict = 1
                            n_conflict += 1
                            store.log_conflict(
                                signature=signature,
                                existing_record_id=existing.record_id,
                                new_record_id=record_id,
                                existing_pollutants=old_pol,
                                new_pollutants=normalized_pollutants,
                                note="Same signature but pollutant values differ (check units/basis/source).",
                            )

                store.upsert_record(
                    record_id=model.record_id,
                    source_file=model.source_file,
                    stage=model.stage,
                    coal_name=model.coal_name,
                    reactor_type=model.reactor_type,
                    T_K=model.T_K,
                    P_MPa=model.P_MPa,
                    gas_agent=model.gas_agent,
                    ratios=model.ratios,
                    pollutants=normalized_pollutants,
                    signature=signature,
                    is_conflict=is_conflict,
                )
                store.add_evidence(
                    record_id=model.record_id,
                    source_file=src,
                    page=page if isinstance(page, int) else None,
                    chunk_id=chunk_id,
                    quote=quote,
                )
                n_saved += 1

        logger.info("RecordPipeline: attempted=%d saved=%d conflicts=%d", n_attempt, n_saved, n_conflict)
        return {"attempted": n_attempt, "saved": n_saved, "conflicts": n_conflict}

    def _pollutants_conflict(self, old_pol: Dict[str, Any], new_pol: Dict[str, Any]) -> bool:
        """
        Very conservative conflict check:
        - compare value_norm if present else value
        - if both present and differ > 20% relative => conflict
        """
        for k, vnew in new_pol.items():
            if k not in old_pol:
                continue
            vold = old_pol.get(k)
            if not isinstance(vold, dict) or not isinstance(vnew, dict):
                continue

            a = vold.get("value_norm", vold.get("value"))
            b = vnew.get("value_norm", vnew.get("value"))
            if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
                continue

            if a == 0:
                if abs(b) > 1e-9:
                    return True
            else:
                rel = abs(float(a) - float(b)) / abs(float(a))
                if rel > 0.2:
                    return True
        return False

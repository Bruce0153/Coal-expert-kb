from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from langchain_core.messages import HumanMessage, SystemMessage

from ..llm.factory import LLMConfig, make_chat_llm
from ..schema.units import atm_to_mpa, bar_to_mpa, c_to_k
from .evidence import FieldEvidence, clip_span
from .normalize import Ontology, detect_targets, normalize_gas_agents, normalize_stage

logger = logging.getLogger(__name__)

# ---------- Regex helpers ----------
_NUM = r"(\d+(?:\.\d+)?)"

# Temperature
_RE_T_K_SINGLE = re.compile(rf"(?:T\s*=?\s*)?{_NUM}\s*K\b", re.I)
_RE_T_C_SINGLE = re.compile(rf"(?:T\s*=?\s*)?{_NUM}\s*°?\s*C\b", re.I)

# Range: 1000-1400 K / 1000 to 1400 K / between 1000 and 1400 K
_RE_T_K_RANGE = re.compile(rf"{_NUM}\s*(?:[-~～]|to|and)\s*{_NUM}\s*K\b", re.I)
_RE_T_C_RANGE = re.compile(rf"{_NUM}\s*(?:[-~～]|to|and)\s*{_NUM}\s*°?\s*C\b", re.I)

# Pressure
_RE_P_MPA_SINGLE = re.compile(rf"(?:P\s*=?\s*)?{_NUM}\s*MPa\b", re.I)
_RE_P_MPA_RANGE = re.compile(rf"{_NUM}\s*(?:[-~～]|to|and)\s*{_NUM}\s*MPa\b", re.I)

_RE_P_BAR_SINGLE = re.compile(rf"(?:P\s*=?\s*)?{_NUM}\s*bar\b", re.I)
_RE_P_ATM_SINGLE = re.compile(rf"(?:P\s*=?\s*)?{_NUM}\s*atm\b", re.I)

# Coal name (very conservative)
_RE_COAL_CN = re.compile(r"(煤种|煤)\s*[:：]\s*([^\n，。;；]+)")
_RE_COAL_EN = re.compile(r"\bcoal\b\s*[:\-]?\s*([A-Za-z0-9\-\s]{2,80})", re.I)

# Ratios
_RE_SC = re.compile(rf"\bS\s*/\s*C\b\s*=?\s*{_NUM}", re.I)
_RE_STEAM_C = re.compile(rf"(steam[-\s]?to[-\s]?carbon)\s*=?\s*{_NUM}", re.I)

_RE_ER = re.compile(rf"\bER\b\s*=?\s*{_NUM}", re.I)
_RE_EQ_RATIO = re.compile(rf"(equivalence\s*ratio)\s*=?\s*{_NUM}", re.I)

_RE_OC = re.compile(rf"\bO\s*/\s*C\b\s*=?\s*{_NUM}", re.I)
_RE_O2C = re.compile(rf"\bO2\s*/\s*C\b\s*=?\s*{_NUM}", re.I)

_RE_H2O_O2 = re.compile(rf"(?:H2O|steam)\s*/\s*O2\s*=?\s*{_NUM}", re.I)


def _first_match(rx: re.Pattern[str], text: str) -> Optional[re.Match[str]]:
    return rx.search(text)


def _parse_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def _extract_range(
    text: str,
    rx_range: re.Pattern[str],
    rx_single: re.Pattern[str],
    *,
    unit: str,
    to_canonical: Any,
    conf_range: float,
    conf_single: float,
    field_name: str,
) -> Tuple[Optional[List[float]], Optional[float], Optional[FieldEvidence]]:
    """
    Return: (range[min,max], single_value, evidence)
    - If range found => range + representative single (midpoint) + evidence
    - Else if single found => single + range(None) + evidence
    """
    m = _first_match(rx_range, text)
    if m:
        a = _parse_float(m.group(1))
        b = _parse_float(m.group(2))
        if a is not None and b is not None:
            lo, hi = (a, b) if a <= b else (b, a)
            clo = to_canonical(lo)
            chi = to_canonical(hi)
            rep = (clo + chi) / 2.0
            s, e = m.span()
            _, _, snippet = clip_span(text, s, e, window=60)
            ev = FieldEvidence(
                field=field_name,
                value=[clo, chi],
                confidence=conf_range,
                start=s,
                end=e,
                evidence_text=snippet,
            )
            return [clo, chi], rep, ev

    m = _first_match(rx_single, text)
    if m:
        v = _parse_float(m.group(1))
        if v is not None:
            cv = to_canonical(v)
            s, e = m.span()
            _, _, snippet = clip_span(text, s, e, window=60)
            ev = FieldEvidence(
                field=field_name,
                value=cv,
                confidence=conf_single,
                start=s,
                end=e,
                evidence_text=snippet,
            )
            return None, cv, ev

    return None, None, None


def _extract_pressure_range_single(text: str) -> Tuple[Optional[List[float]], Optional[float], List[FieldEvidence]]:
    evidences: List[FieldEvidence] = []

    # MPa range/single
    r, v, ev = _extract_range(
        text,
        _RE_P_MPA_RANGE,
        _RE_P_MPA_SINGLE,
        unit="MPa",
        to_canonical=lambda x: x,
        conf_range=0.90,
        conf_single=0.90,
        field_name="P_range_MPa",
    )
    if ev:
        evidences.append(ev)

    if r is not None:
        return r, v, evidences
    if v is not None:
        return None, v, evidences

    # bar
    m = _first_match(_RE_P_BAR_SINGLE, text)
    if m:
        bar = _parse_float(m.group(1))
        if bar is not None:
            v2 = bar_to_mpa(bar)
            s, e = m.span()
            _, _, snippet = clip_span(text, s, e, window=60)
            evidences.append(
                FieldEvidence("P_MPa", v2, 0.88, s, e, snippet)
            )
            return None, v2, evidences

    # atm
    m = _first_match(_RE_P_ATM_SINGLE, text)
    if m:
        atm = _parse_float(m.group(1))
        if atm is not None:
            v2 = atm_to_mpa(atm)
            s, e = m.span()
            _, _, snippet = clip_span(text, s, e, window=60)
            evidences.append(
                FieldEvidence("P_MPa", v2, 0.85, s, e, snippet)
            )
            return None, v2, evidences

    return None, None, evidences


def _extract_temperature_range_single(text: str) -> Tuple[Optional[List[float]], Optional[float], List[FieldEvidence]]:
    evidences: List[FieldEvidence] = []

    # K range/single
    r, v, ev = _extract_range(
        text,
        _RE_T_K_RANGE,
        _RE_T_K_SINGLE,
        unit="K",
        to_canonical=lambda x: x,
        conf_range=0.90,
        conf_single=0.90,
        field_name="T_range_K",
    )
    if ev:
        evidences.append(ev)
    if r is not None or v is not None:
        return r, v, evidences

    # C range/single => K
    r, v, ev = _extract_range(
        text,
        _RE_T_C_RANGE,
        _RE_T_C_SINGLE,
        unit="C",
        to_canonical=lambda x: c_to_k(x),
        conf_range=0.88,
        conf_single=0.88,
        field_name="T_range_K",
    )
    if ev:
        evidences.append(ev)
    return r, v, evidences


def _extract_coal_name(text: str) -> Tuple[Optional[str], Optional[FieldEvidence]]:
    m = _RE_COAL_CN.search(text)
    if m:
        name = m.group(2).strip()
        s, e = m.span()
        _, _, snippet = clip_span(text, s, e, window=60)
        return name, FieldEvidence("coal_name", name, 0.85, s, e, snippet)

    m = _RE_COAL_EN.search(text)
    if m:
        name = m.group(1).strip()
        # avoid long capture
        name = name[:60].strip()
        s, e = m.span()
        _, _, snippet = clip_span(text, s, e, window=60)
        return (name if name else None), FieldEvidence("coal_name", name, 0.70, s, e, snippet)

    return None, None


def _extract_ratios(text: str) -> Tuple[Optional[Dict[str, float]], List[FieldEvidence]]:
    ratios: Dict[str, float] = {}
    evs: List[FieldEvidence] = []

    def add_ratio(key: str, rx: re.Pattern[str], group_idx: int = 1, conf: float = 0.85):
        m = rx.search(text)
        if not m:
            return
        v = _parse_float(m.group(group_idx))
        if v is None:
            return
        ratios[key] = float(v)
        s, e = m.span()
        _, _, snippet = clip_span(text, s, e, window=60)
        evs.append(FieldEvidence(f"ratios.{key}", float(v), conf, s, e, snippet))

    add_ratio("S/C", _RE_SC, 1, 0.88)
    add_ratio("S/C", _RE_STEAM_C, 2, 0.85)  # group(2) holds value
    add_ratio("ER", _RE_ER, 1, 0.88)
    add_ratio("ER", _RE_EQ_RATIO, 2, 0.85)
    add_ratio("O/C", _RE_OC, 1, 0.85)
    add_ratio("O2/C", _RE_O2C, 1, 0.83)
    add_ratio("H2O/O2", _RE_H2O_O2, 1, 0.80)

    return (ratios or None), evs


@dataclass
class MetadataExtractor:
    onto: Ontology
    enable_llm: bool = False
    llm_provider: str = "none"  # "openai"/"dashscope"/"openai_compatible"
    llm_config: Optional[LLMConfig] = None

    def extract(self, doc: Document) -> Dict[str, Any]:
        text = doc.page_content or ""
        meta: Dict[str, Any] = dict(doc.metadata or {})

        # ---- stage/gas/targets (ontology-based) ----
        stage = normalize_stage(text, self.onto)
        gas_agent = normalize_gas_agents(text, self.onto)
        targets = detect_targets(text, self.onto)

        meta["stage"] = stage
        meta["gas_agent"] = gas_agent
        meta["targets"] = targets

        conf: Dict[str, float] = {}
        evid: Dict[str, dict] = {}

        conf["stage"] = 0.70 if stage != "unknown" else 0.30
        if gas_agent:
            conf["gas_agent"] = 0.70
        if targets:
            conf["targets"] = 0.75

        # ---- temperature ----
        T_range, T_single, evs_T = _extract_temperature_range_single(text)
        if T_range is not None:
            meta["T_range_K"] = T_range
            meta["T_min_K"], meta["T_max_K"] = T_range[0], T_range[1]
            meta["T_K"] = T_single
            conf["T_range_K"] = 0.90
            conf["T_K"] = 0.85
            for ev in evs_T:
                evid[ev.field] = ev.to_dict()
        elif T_single is not None:
            meta["T_K"] = T_single
            conf["T_K"] = 0.90
            for ev in evs_T:
                # could be field "T_range_K" evidence; still keep
                evid[ev.field] = ev.to_dict()

        # ---- pressure ----
        P_range, P_single, evs_P = _extract_pressure_range_single(text)
        if P_range is not None:
            meta["P_range_MPa"] = P_range
            meta["P_min_MPa"], meta["P_max_MPa"] = P_range[0], P_range[1]
            meta["P_MPa"] = P_single
            conf["P_range_MPa"] = 0.90
            conf["P_MPa"] = 0.85
            for ev in evs_P:
                evid[ev.field] = ev.to_dict()
        elif P_single is not None:
            meta["P_MPa"] = P_single
            conf["P_MPa"] = max(conf.get("P_MPa", 0.0), 0.88)
            for ev in evs_P:
                evid[ev.field] = ev.to_dict()

        # ---- coal name ----
        coal, ev_coal = _extract_coal_name(text)
        if coal:
            meta["coal_name"] = coal
            conf["coal_name"] = 0.80
        if ev_coal:
            evid["coal_name"] = ev_coal.to_dict()

        # ---- ratios ----
        ratios, evs_R = _extract_ratios(text)
        if ratios:
            meta["ratios"] = ratios
            conf["ratios"] = 0.82
            for ev in evs_R:
                evid[ev.field] = ev.to_dict()

        # attach evidence/conf (JSON-serializable)
        meta["meta_confidence"] = conf
        meta["meta_evidence"] = evid

        # Optional LLM augmentation (merge conservatively)
        if self.enable_llm and self.llm_provider != "none":
            meta = self._augment_with_openai(text=text, meta=meta)

        return meta

    def _augment_with_openai(self, *, text: str, meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Requires:
          pip install langchain-openai
        and OPENAI_API_KEY in env.
        Merge policy:
          - if field missing from rule-extraction, accept LLM value.
          - keep rule evidence/conf as primary.
        """
        prompt_path = "configs/prompts/metadata_extract.txt"
        with open(prompt_path, "r", encoding="utf-8") as f:
            sys_prompt = f.read().strip()

        if self.llm_config is None:
            logger.warning("LLM config missing; skip LLM augmentation.")
            return meta

        model = make_chat_llm(self.llm_config)
        msgs = [
            SystemMessage(content=sys_prompt),
            HumanMessage(
                content=(
                    f"片段:\n{text}\n\n"
                    f"当前已抽取(仅供参考):\n{json.dumps(meta, ensure_ascii=False)}\n\n"
                    "请输出严格 JSON。"
                )
            ),
        ]
        rsp = model.invoke(msgs)
        content = getattr(rsp, "content", None) or ""
        try:
            new_meta = json.loads(content)
        except Exception:
            return meta

        # Merge only missing keys
        for k, v in (new_meta or {}).items():
            if meta.get(k) is None and v is not None:
                meta[k] = v
                # add low-confidence record
                meta.setdefault("meta_confidence", {})
                meta["meta_confidence"][k] = 0.70

        return meta

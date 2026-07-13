"""提供摄入阶段的metadata实现。"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.documents import Document

from coal_kb.schema.units import atm_to_mpa, bar_to_mpa, c_to_k

from .normalize import Ontology, detect_targets, normalize_gas_agents, normalize_stage

logger = logging.getLogger(__name__)

_NUM = r"(\d+(?:\.\d+)?)"

_RE_T_K_SINGLE = re.compile(rf"(?:T\s*=?\s*)?{_NUM}\s*K\b", re.I)
_RE_T_C_SINGLE = re.compile(rf"(?:T\s*=?\s*)?{_NUM}\s*掳?\s*C\b", re.I)
_RE_T_K_RANGE = re.compile(rf"{_NUM}\s*(?:-|~|to|and)\s*{_NUM}\s*K\b", re.I)
_RE_T_C_RANGE = re.compile(rf"{_NUM}\s*(?:-|~|to|and)\s*{_NUM}\s*掳?\s*C\b", re.I)

_RE_P_MPA_SINGLE = re.compile(rf"(?:P\s*=?\s*)?{_NUM}\s*MPa\b", re.I)
_RE_P_MPA_RANGE = re.compile(rf"{_NUM}\s*(?:-|~|to|and)\s*{_NUM}\s*MPa\b", re.I)
_RE_P_BAR_SINGLE = re.compile(rf"(?:P\s*=?\s*)?{_NUM}\s*bar\b", re.I)
_RE_P_ATM_SINGLE = re.compile(rf"(?:P\s*=?\s*)?{_NUM}\s*atm\b", re.I)

_RE_COAL_CN = re.compile(r"(煤种|煤)\s*[:：]?\s*([^\n，。,;；]{1,80})")
_RE_COAL_EN = re.compile(r"\bcoal\b\s*[:\-]?\s*([A-Za-z0-9\-\s]{2,80})", re.I)

_RE_SC = re.compile(rf"\bS\s*/\s*C\b\s*=?\s*{_NUM}", re.I)
_RE_STEAM_C = re.compile(rf"(steam[-\s]?to[-\s]?carbon)\s*=?\s*{_NUM}", re.I)
_RE_ER = re.compile(rf"\bER\b\s*=?\s*{_NUM}", re.I)
_RE_EQ_RATIO = re.compile(rf"(equivalence\s*ratio)\s*=?\s*{_NUM}", re.I)
_RE_OC = re.compile(rf"\bO\s*/\s*C\b\s*=?\s*{_NUM}", re.I)
_RE_O2C = re.compile(rf"\bO2\s*/\s*C\b\s*=?\s*{_NUM}", re.I)
_RE_H2O_O2 = re.compile(rf"(?:H2O|steam)\s*/\s*O2\s*=?\s*{_NUM}", re.I)


def _first_match(rx: re.Pattern[str], text: str) -> Optional[re.Match[str]]:
    return rx.search(text)


def _parse_float(value: str) -> Optional[float]:
    try:
        return float(value)
    except Exception:
        return None


def _extract_range(
    text: str,
    rx_range: re.Pattern[str],
    rx_single: re.Pattern[str],
    *,
    to_canonical: Any,
) -> Tuple[Optional[List[float]], Optional[float]]:
    match = _first_match(rx_range, text)
    if match:
        first = _parse_float(match.group(1))
        second = _parse_float(match.group(2))
        if first is not None and second is not None:
            low, high = sorted((to_canonical(first), to_canonical(second)))
            return [low, high], (low + high) / 2.0

    match = _first_match(rx_single, text)
    if match:
        value = _parse_float(match.group(1))
        if value is not None:
            canonical = to_canonical(value)
            return None, canonical

    return None, None


def _extract_pressure_range_single(text: str) -> Tuple[Optional[List[float]], Optional[float]]:
    pressure_range, pressure_value = _extract_range(
        text,
        _RE_P_MPA_RANGE,
        _RE_P_MPA_SINGLE,
        to_canonical=lambda value: value,
    )
    if pressure_range is not None or pressure_value is not None:
        return pressure_range, pressure_value

    match = _first_match(_RE_P_BAR_SINGLE, text)
    if match:
        value = _parse_float(match.group(1))
        if value is not None:
            return None, bar_to_mpa(value)

    match = _first_match(_RE_P_ATM_SINGLE, text)
    if match:
        value = _parse_float(match.group(1))
        if value is not None:
            return None, atm_to_mpa(value)

    return None, None


def _extract_temperature_range_single(text: str) -> Tuple[Optional[List[float]], Optional[float]]:
    temp_range, temp_value = _extract_range(
        text,
        _RE_T_K_RANGE,
        _RE_T_K_SINGLE,
        to_canonical=lambda value: value,
    )
    if temp_range is not None or temp_value is not None:
        return temp_range, temp_value

    return _extract_range(
        text,
        _RE_T_C_RANGE,
        _RE_T_C_SINGLE,
        to_canonical=c_to_k,
    )


def _extract_coal_name(text: str) -> Optional[str]:
    match = _RE_COAL_CN.search(text)
    if match:
        return match.group(2).strip()

    match = _RE_COAL_EN.search(text)
    if match:
        return match.group(1).strip()[:60].strip() or None

    return None


def _extract_ratios(text: str) -> Optional[Dict[str, float]]:
    ratios: Dict[str, float] = {}

    def add_ratio(key: str, rx: re.Pattern[str], group_index: int = 1) -> None:
        match = rx.search(text)
        if not match:
            return
        value = _parse_float(match.group(group_index))
        if value is not None:
            ratios[key] = float(value)

    add_ratio("S/C", _RE_SC, 1)
    add_ratio("S/C", _RE_STEAM_C, 2)
    add_ratio("ER", _RE_ER, 1)
    add_ratio("ER", _RE_EQ_RATIO, 2)
    add_ratio("O/C", _RE_OC, 1)
    add_ratio("O2/C", _RE_O2C, 1)
    add_ratio("H2O/O2", _RE_H2O_O2, 1)
    return ratios or None


@dataclass
class MetadataExtractor:
    onto: Ontology

    def extract(self, doc: Document) -> Dict[str, Any]:
        text = doc.page_content or ""
        meta: Dict[str, Any] = dict(doc.metadata or {})

        meta["stage"] = normalize_stage(text, self.onto)
        meta["gas_agent"] = normalize_gas_agents(text, self.onto)
        meta["targets"] = detect_targets(text, self.onto)

        temperature_range, temperature_value = _extract_temperature_range_single(text)
        if temperature_range is not None:
            meta["T_range_K"] = temperature_range
            meta["T_min_K"], meta["T_max_K"] = temperature_range
            meta["T_K"] = temperature_value
        elif temperature_value is not None:
            meta["T_K"] = temperature_value

        pressure_range, pressure_value = _extract_pressure_range_single(text)
        if pressure_range is not None:
            meta["P_range_MPa"] = pressure_range
            meta["P_min_MPa"], meta["P_max_MPa"] = pressure_range
            meta["P_MPa"] = pressure_value
        elif pressure_value is not None:
            meta["P_MPa"] = pressure_value

        coal_name = _extract_coal_name(text)
        if coal_name:
            meta["coal_name"] = coal_name

        ratios = _extract_ratios(text)
        if ratios:
            meta["ratios"] = ratios

        return meta

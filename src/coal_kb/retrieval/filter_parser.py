from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from ..metadata.normalize import Ontology, normalize_gas_agents

_RE_NUM = r"(\d+(?:\.\d+)?)"


def _range_from_single(v: float, rel: float = 0.05) -> Tuple[float, float]:
    # +/-5% window by default
    return (v * (1 - rel), v * (1 + rel))


@dataclass
class FilterParser:
    onto: Ontology

    def parse(self, query: str) -> Dict[str, Any]:
        q = query.strip()

        # stage & gas
        stage, stage_candidates = self._detect_stage(q)
        gas = normalize_gas_agents(q, self.onto)

        # targets: keyword detect (same as ontology)
        from ..metadata.normalize import detect_targets
        targets = detect_targets(q, self.onto)

        # temperature: accept "1200K" or "900C"
        T_range = self._parse_temperature_range(q)
        P_range = self._parse_pressure_range(q)

        coal_name = self._parse_coal_name(q)

        return {
            "stage": stage,
            "stage_candidates": stage_candidates,
            "coal_name": coal_name,
            "T_range_K": T_range,
            "P_range_MPa": P_range,
            "gas_agent": gas,
            "targets": targets,
        }

    def _detect_stage(self, q: str) -> Tuple[str, List[str]]:
        t = q.lower()
        candidates: List[str] = []
        for canonical, aliases in self.onto.stage_aliases.items():
            if any(alias.lower() in t for alias in aliases):
                candidates.append(canonical)

        if not candidates:
            return "unknown", []

        priority = ["combustion", "ignition", "oxidation", "gasification", "pyrolysis", "coupled"]
        for stage in priority:
            if stage in candidates:
                return stage, candidates
        return candidates[0], candidates

    def _detect_stage(self, q: str) -> Tuple[str, List[str]]:
        t = q.lower()
        candidates: List[str] = []
        for canonical, aliases in self.onto.stage_aliases.items():
            if any(alias.lower() in t for alias in aliases):
                candidates.append(canonical)

        if not candidates:
            return "unknown", []

        priority = ["combustion", "ignition", "oxidation", "gasification", "pyrolysis", "coupled"]
        for stage in priority:
            if stage in candidates:
                return stage, candidates
        return candidates[0], candidates

    def _detect_stage(self, q: str) -> Tuple[str, List[str]]:
        t = q.lower()
        candidates: List[str] = []
        for canonical, aliases in self.onto.stage_aliases.items():
            if any(alias.lower() in t for alias in aliases):
                candidates.append(canonical)

        if not candidates:
            return "unknown", []

        priority = ["combustion", "ignition", "oxidation", "gasification", "pyrolysis", "coupled"]
        for stage in priority:
            if stage in candidates:
                return stage, candidates
        return candidates[0], candidates

    def _parse_temperature_range(self, q: str) -> Optional[List[float]]:
        # explicit range like "1100-1300 K"
        m = re.search(rf"{_RE_NUM}\s*[-~～]\s*{_RE_NUM}\s*K", q, re.I)
        if m:
            return [float(m.group(1)), float(m.group(2))]

        m = re.search(rf"{_RE_NUM}\s*K", q, re.I)
        if m:
            v = float(m.group(1))
            lo, hi = _range_from_single(v, 0.05)
            return [lo, hi]

        # Celsius
        m = re.search(rf"{_RE_NUM}\s*[-~～]\s*{_RE_NUM}\s*°?\s*C", q, re.I)
        if m:
            lo = float(m.group(1)) + 273.15
            hi = float(m.group(2)) + 273.15
            return [lo, hi]

        m = re.search(rf"{_RE_NUM}\s*°?\s*C", q, re.I)
        if m:
            v = float(m.group(1)) + 273.15
            lo, hi = _range_from_single(v, 0.05)
            return [lo, hi]

        return None

    def _parse_pressure_range(self, q: str) -> Optional[List[float]]:
        m = re.search(rf"{_RE_NUM}\s*[-~～]\s*{_RE_NUM}\s*MPa", q, re.I)
        if m:
            return [float(m.group(1)), float(m.group(2))]

        m = re.search(rf"{_RE_NUM}\s*MPa", q, re.I)
        if m:
            v = float(m.group(1))
            lo, hi = _range_from_single(v, 0.1)
            return [lo, hi]

        return None

    def _parse_coal_name(self, q: str) -> Optional[str]:
        # very lightweight
        m = re.search(r"(煤种|煤)\s*[:：]\s*([^\n，。;；]+)", q)
        if m:
            return m.group(2).strip()
        return None

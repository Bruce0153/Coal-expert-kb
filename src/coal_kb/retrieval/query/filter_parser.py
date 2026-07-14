"""解析查询中的工况、阶段和污染物约束。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from coal_kb.ingestion.metadata.normalize import Ontology, normalize_gas_agents
from coal_kb.retrieval.constraints import Constraint, ConstraintSet

_RE_NUM = r"(\d+(?:\.\d+)?)"


def _range_from_single(v: float, rel: float = 0.05) -> Tuple[float, float]:
    return (v * (1 - rel), v * (1 + rel))


@dataclass
class FilterParser:
    onto: Ontology

    def parse(self, query: str) -> ConstraintSet:
        q = query.strip()
        must_keywords = ["必须", "only", "strict", "严格", "仅"]

        stage, stage_candidates = self._detect_stage(q)
        gas = normalize_gas_agents(q, self.onto)

        from coal_kb.ingestion.metadata.normalize import detect_targets
        targets = detect_targets(q, self.onto)

        T_range = self._parse_temperature_range(q)
        P_range = self._parse_pressure_range(q)
        coal_name = self._parse_coal_name(q)

        compat = {
            "stage": stage,
            "stage_candidates": stage_candidates,
            "coal_name": coal_name,
            "T_range_K": T_range,
            "P_range_MPa": P_range,
            "gas_agent": gas,
            "targets": targets,
        }

        constraints: List[Constraint] = []

        if stage and stage != "unknown":
            priority = "hard" if any(k in q for k in must_keywords) else "soft"
            constraints.append(
                Constraint(
                    name="stage",
                    ctype="enum",
                    value=stage,
                    confidence=0.6,
                    source="rule",
                    priority=priority,
                )
            )

        if gas:
            constraints.append(
                Constraint(
                    name="gas_agent",
                    ctype="set",
                    value=gas,
                    confidence=0.6,
                    source="rule",
                    priority="soft",
                )
            )

        if targets:
            constraints.append(
                Constraint(
                    name="targets",
                    ctype="set",
                    value=targets,
                    confidence=0.6,
                    source="rule",
                    priority="soft",
                )
            )

        if T_range:
            priority = "hard" if any(k in q for k in must_keywords) else "soft"
            constraints.append(
                Constraint(
                    name="T_range_K",
                    ctype="range",
                    value=T_range,
                    confidence=0.9,
                    source="rule",
                    priority=priority,
                )
            )

        if P_range:
            priority = "hard" if any(k in q for k in must_keywords) else "soft"
            constraints.append(
                Constraint(
                    name="P_range_MPa",
                    ctype="range",
                    value=P_range,
                    confidence=0.9,
                    source="rule",
                    priority=priority,
                )
            )

        if coal_name:
            constraints.append(
                Constraint(
                    name="coal_name",
                    ctype="text",
                    value=coal_name,
                    confidence=0.5,
                    source="rule",
                    priority="soft",
                )
            )

        return ConstraintSet(constraints=constraints, compat_where=compat)

    def _detect_stage(self, q: str) -> Tuple[str, List[str]]:
        """
        修复点：
        - NOx / 氮氧化物 只属于 targets，不自动映射成 oxidation stage
        - 只有显式 stage 词才设 stage
        """
        stage_patterns = {
            "combustion": [r"燃烧", r"\bcombustion\b", r"\bburning\b"],
            "ignition": [r"点火", r"着火", r"\bignition\b"],
            "oxidation": [
                r"(^|[\s,，。;；()（）])氧化($|[\s,，。;；()（）])",
                r"氧化阶段",
                r"\boxidation\b",
                r"\boxidative\b",
            ],
            "gasification": [r"气化", r"\bgasification\b"],
            "pyrolysis": [r"热解", r"裂解", r"\bpyrolysis\b"],
            "coupled": [r"耦合", r"\bcoupled\b"],
        }

        candidates: List[str] = []
        for canonical, patterns in stage_patterns.items():
            if any(re.search(p, q, flags=re.I) for p in patterns):
                candidates.append(canonical)

        if not candidates:
            return "unknown", []

        priority = ["combustion", "ignition", "oxidation", "gasification", "pyrolysis", "coupled"]
        for stage in priority:
            if stage in candidates:
                return stage, candidates
        return candidates[0], candidates

    def _parse_temperature_range(self, q: str) -> Optional[List[float]]:
        m = re.search(rf"{_RE_NUM}\s*[-~～]\s*{_RE_NUM}\s*K", q, re.I)
        if m:
            return [float(m.group(1)), float(m.group(2))]

        m = re.search(rf"{_RE_NUM}\s*K", q, re.I)
        if m:
            v = float(m.group(1))
            lo, hi = _range_from_single(v, 0.05)
            return [lo, hi]

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
        m = re.search(r"(煤种|煤)\s*[:：]\s*([^\n，。;；]+)", q)
        if m:
            return m.group(2).strip()
        return None
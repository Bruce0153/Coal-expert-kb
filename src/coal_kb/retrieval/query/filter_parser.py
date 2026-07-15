"""解析查询中的工况、阶段和污染物约束。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from coal_kb.ingestion.metadata.normalize import (
    Ontology,
    detect_targets,
    normalize_gas_agents,
)
from coal_kb.retrieval.constraints import Constraint, ConstraintSet

_RE_NUM = r"(\d+(?:\.\d+)?)"
_MUST_KEYWORDS = ("必须", "only", "strict", "严格", "仅")
_STAGE_PATTERNS = {
    "combustion": (r"燃烧", r"\bcombustion\b", r"\bburning\b"),
    "ignition": (r"点火", r"着火", r"\bignition\b"),
    "oxidation": (
        r"(^|[\s,，。;；()（）])氧化($|[\s,，。;；()（）])",
        r"氧化阶段",
        r"\boxidation\b",
        r"\boxidative\b",
    ),
    "gasification": (r"气化", r"\bgasification\b"),
    "pyrolysis": (r"热解", r"裂解", r"\bpyrolysis\b"),
    "coupled": (r"耦合", r"\bcoupled\b"),
}
_STAGE_PRIORITY = (
    "combustion",
    "ignition",
    "oxidation",
    "gasification",
    "pyrolysis",
    "coupled",
)


def _range_from_single(value: float, relative_margin: float) -> list[float]:
    return [
        value * (1 - relative_margin),
        value * (1 + relative_margin),
    ]


@dataclass
class FilterParser:
    onto: Ontology

    def parse(self, query: str) -> ConstraintSet:
        """将查询解析为唯一正式约束集合。"""
        normalized = query.strip()
        strict = any(keyword in normalized for keyword in _MUST_KEYWORDS)
        constraints: list[Constraint] = []

        stage = self._detect_stage(normalized)
        if stage:
            constraints.append(
                Constraint(
                    name="stage",
                    ctype="enum",
                    value=stage,
                    confidence=0.6,
                    source="rule",
                    priority="hard" if strict else "soft",
                )
            )

        gas_agents = normalize_gas_agents(normalized, self.onto)
        if gas_agents:
            constraints.append(
                Constraint(
                    name="gas_agent",
                    ctype="set",
                    value=gas_agents,
                    confidence=0.6,
                    source="rule",
                )
            )

        targets = detect_targets(normalized, self.onto)
        if targets:
            constraints.append(
                Constraint(
                    name="targets",
                    ctype="set",
                    value=targets,
                    confidence=0.6,
                    source="rule",
                )
            )

        temperature_range = self._parse_temperature_range(normalized)
        if temperature_range:
            constraints.append(
                Constraint(
                    name="T_range_K",
                    ctype="range",
                    value=temperature_range,
                    confidence=0.9,
                    source="rule",
                    priority="hard" if strict else "soft",
                )
            )

        pressure_range = self._parse_pressure_range(normalized)
        if pressure_range:
            constraints.append(
                Constraint(
                    name="P_range_MPa",
                    ctype="range",
                    value=pressure_range,
                    confidence=0.9,
                    source="rule",
                    priority="hard" if strict else "soft",
                )
            )

        coal_name = self._parse_coal_name(normalized)
        if coal_name:
            constraints.append(
                Constraint(
                    name="coal_name",
                    ctype="text",
                    value=coal_name,
                    confidence=0.5,
                    source="rule",
                )
            )

        return ConstraintSet(constraints=constraints)

    @staticmethod
    def _detect_stage(query: str) -> str | None:
        """只将显式阶段词映射为阶段约束。"""
        matches = {
            stage
            for stage, patterns in _STAGE_PATTERNS.items()
            if any(re.search(pattern, query, flags=re.I) for pattern in patterns)
        }
        return next((stage for stage in _STAGE_PRIORITY if stage in matches), None)

    @staticmethod
    def _parse_temperature_range(query: str) -> list[float] | None:
        match = re.search(rf"{_RE_NUM}\s*[-~～]\s*{_RE_NUM}\s*K", query, re.I)
        if match:
            return [float(match.group(1)), float(match.group(2))]

        match = re.search(rf"{_RE_NUM}\s*K", query, re.I)
        if match:
            return _range_from_single(float(match.group(1)), 0.05)

        match = re.search(rf"{_RE_NUM}\s*[-~～]\s*{_RE_NUM}\s*°?\s*C", query, re.I)
        if match:
            return [float(match.group(1)) + 273.15, float(match.group(2)) + 273.15]

        match = re.search(rf"{_RE_NUM}\s*°?\s*C", query, re.I)
        if match:
            return _range_from_single(float(match.group(1)) + 273.15, 0.05)
        return None

    @staticmethod
    def _parse_pressure_range(query: str) -> list[float] | None:
        match = re.search(rf"{_RE_NUM}\s*[-~～]\s*{_RE_NUM}\s*MPa", query, re.I)
        if match:
            return [float(match.group(1)), float(match.group(2))]

        match = re.search(rf"{_RE_NUM}\s*MPa", query, re.I)
        if match:
            return _range_from_single(float(match.group(1)), 0.1)
        return None

    @staticmethod
    def _parse_coal_name(query: str) -> str | None:
        match = re.search(r"(?:煤种|煤)\s*[:：]\s*([^\n，。;；]+)", query)
        return match.group(1).strip() if match else None

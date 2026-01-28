from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import yaml
import json

@dataclass(frozen=True)
class Ontology:
    stage_aliases: Dict[str, List[str]]
    gas_aliases: Dict[str, List[str]]
    pollutant_aliases: Dict[str, List[str]]

    @staticmethod
    def load(schema_path: str = "configs/schema.yaml") -> "Ontology":
        p = Path(schema_path)
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        onto = raw.get("ontology", {}) or {}

        stage = onto.get("stage", {}) or {}
        gas = onto.get("gas_agent", {}) or {}
        pol = onto.get("pollutants", {}) or {}

        # normalize keys to lower for stage/gas, keep pollutant canonical as-is (NH3, etc.)
        stage_aliases = {k.lower(): [a for a in v] for k, v in stage.items()}
        gas_aliases = {k.lower(): [a for a in v] for k, v in gas.items()}
        pollutant_aliases = {k: [a for a in v] for k, v in pol.items()}
        return Ontology(stage_aliases, gas_aliases, pollutant_aliases)


def _contains_any(text_low: str, aliases: Iterable[str]) -> bool:
    return any(a.lower() in text_low for a in aliases)


def normalize_stage(text: str, onto: Ontology) -> str:
    t = text.lower()
    for canonical, aliases in onto.stage_aliases.items():
        if _contains_any(t, aliases):
            return canonical  # "pyrolysis"/"gasification"/"coupled"
    return "unknown"


def normalize_gas_agents(text: str, onto: Ontology) -> Optional[List[str]]:
    t = text.lower()
    hits: List[str] = []
    for canonical, aliases in onto.gas_aliases.items():
        if _contains_any(t, aliases):
            hits.append(canonical)
    return sorted(set(hits)) if hits else None


def detect_targets(text: str, onto: Ontology) -> Optional[List[str]]:
    t = text.lower()
    hits: List[str] = []
    for canonical, aliases in onto.pollutant_aliases.items():
        # canonical may contain uppercase; compare with aliases
        if _contains_any(t, aliases) or canonical.lower() in t:
            hits.append(canonical)
    return sorted(set(hits)) if hits else None


def normalize_coal_name(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = raw.strip()
    return s if s else None


def flatten_for_filtering(meta: Dict[str, Any], onto: Ontology) -> Dict[str, Any]:
    out: Dict[str, Any] = dict(meta)

    # Ensure section exists
    out.setdefault("section", "unknown")

    # Normalize some canonical strings
    if "stage" in out and isinstance(out["stage"], str):
        out["stage"] = out["stage"].lower()

    coal_name = out.get("coal_name")
    out["coal_name"] = normalize_coal_name(coal_name)

    # Derive min/max from ranges (if present)
    tr = out.get("T_range_K")
    if isinstance(tr, list) and len(tr) == 2:
        out.setdefault("T_min_K", tr[0])
        out.setdefault("T_max_K", tr[1])

    pr = out.get("P_range_MPa")
    if isinstance(pr, list) and len(pr) == 2:
        out.setdefault("P_min_MPa", pr[0])
        out.setdefault("P_max_MPa", pr[1])

    # Flatten multi-value fields into boolean flags
    gas = out.get("gas_agent")
    if isinstance(gas, list):
        for g in gas:
            out[f"gas_{str(g).lower()}"] = True

    targets = out.get("targets")
    if isinstance(targets, list):
        for tg in targets:
            out[f"has_{tg}"] = True

    allowed = (str, int, float, bool, type(None))

    def sanitize(v: Any) -> Any:
        if isinstance(v, allowed):
            return v
        try:
            return json.dumps(v, ensure_ascii=False)
        except TypeError:
            return str(v)

    for k, v in list(out.items()):
        out[k] = sanitize(v)

    return out

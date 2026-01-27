from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from .units import mol_percent_to_ppmv

# Molecular weights (g/mol)
MW = {
    "NH3": 17.031,
    "HCN": 27.025,
    "H2S": 34.081,
    "SO2": 64.066,
    "COS": 60.075,
    "benzene": 78.114,
    "phenol": 94.113,
    # NOx is ambiguous (NO/NO2 equivalence varies); do not convert mg/Nm3 -> ppmv for NOx by default
}

# mg/Nm3 -> ppmv at 0°C, 1 atm:
# ppmv = mg/Nm3 * (22.414 / MW)
# derived from ideal gas at STP (Nm3)
STP_MOLAR_VOLUME_L = 22.414


def mgNm3_to_ppmv(mg_per_Nm3: float, mw: float) -> float:
    return mg_per_Nm3 * (STP_MOLAR_VOLUME_L / mw)


def normalize_one_pollutant(name: str, item: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize pollutant measurement to a canonical representation when possible.
    Keeps the original fields and adds:
      - value_norm
      - unit_norm
    """
    out = dict(item or {})
    if "basis" not in out:
        out["basis"] = "unknown"

    v = out.get("value", None)
    u = str(out.get("unit", "") or "").strip().lower()

    # Only normalize scalar numeric values
    if not isinstance(v, (int, float)):
        return out

    # common unit aliases
    if u == "ppm":
        u = "ppmv"

    if u in ("mol%", "mol %", "mole%", "mole %"):
        out["value_norm"] = float(mol_percent_to_ppmv(float(v)))
        out["unit_norm"] = "ppmv"
        return out

    if u in ("ppmv",):
        out["value_norm"] = float(v)
        out["unit_norm"] = "ppmv"
        return out

    if u in ("mg/nm3", "mg/nm^3", "mg per nm3"):
        mw = MW.get(name)
        if mw is None:
            # can't safely convert
            out["value_norm"] = float(v)
            out["unit_norm"] = "mg/Nm3"
            return out
        out["value_norm"] = float(mgNm3_to_ppmv(float(v), mw))
        out["unit_norm"] = "ppmv"
        return out

    # fallback: keep as-is
    out["value_norm"] = float(v)
    out["unit_norm"] = out.get("unit")
    return out


def normalize_pollutants_dict(pollutants: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in (pollutants or {}).items():
        if isinstance(v, dict):
            out[k] = normalize_one_pollutant(k, v)
        else:
            out[k] = v
    return out

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ValidationError


def validate_temperature_k(T_K: Optional[float]) -> None:
    if T_K is None:
        return
    # loose bounds for gasification/pyrolysis literature
    if not (200.0 <= T_K <= 3000.0):
        raise ValueError(f"Unreasonable temperature (K): {T_K}")


def validate_pressure_mpa(P_MPa: Optional[float]) -> None:
    if P_MPa is None:
        return
    if not (0.0 < P_MPa <= 50.0):
        raise ValueError(f"Unreasonable pressure (MPa): {P_MPa}")


def validate_pollutants(pollutants: Dict[str, Any]) -> None:
    """
    Minimal validation:
    - each pollutant item should have value/unit/basis (basis may be 'unknown').
    """
    for k, v in (pollutants or {}).items():
        if not isinstance(v, dict):
            raise ValueError(f"pollutant '{k}' must be an object with value/unit/basis")
        if "value" not in v or "unit" not in v:
            raise ValueError(f"pollutant '{k}' must have 'value' and 'unit'")
        # value could be number or (min,max) range; keep flexible
        if "basis" not in v:
            v["basis"] = "unknown"


def safe_model_validate(model: type[BaseModel], data: dict) -> BaseModel:
    """
    Validate with clearer error message for pipelines.
    """
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ValueError(str(e)) from e

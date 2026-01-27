from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Stage = Literal["pyrolysis", "gasification", "coupled", "unknown"]


class ChunkMetadata(BaseModel):
    source_file: str
    page: Optional[int] = None
    section: str = "unknown"

    # Expert filters (may be partial/null initially)
    stage: Stage = "unknown"
    coal_name: Optional[str] = None
    reactor_type: Optional[str] = None

    T_K: Optional[float] = None
    P_MPa: Optional[float] = None

    gas_agent: Optional[List[str]] = None  # ["steam","o2",...]
    targets: Optional[List[str]] = None  # ["NH3","HCN",...]
    ratios: Optional[Dict[str, float]] = None  # {"S/C":1.0, "ER":0.3}


class Chunk(BaseModel):
    chunk_id: str
    text: str
    metadata: ChunkMetadata


class Evidence(BaseModel):
    source_file: str
    page: Optional[int] = None
    chunk_id: str
    quote: Optional[str] = None


class ExperimentRecord(BaseModel):
    record_id: str
    source_file: str

    stage: Stage = "unknown"
    coal_name: Optional[str] = None
    reactor_type: Optional[str] = None

    T_K: Optional[float] = None
    P_MPa: Optional[float] = None
    gas_agent: Optional[List[str]] = None

    ratios: Dict[str, float] = Field(default_factory=dict)

    # pollutants: e.g. {"NH3":{"value":120,"unit":"ppmv","basis":"dry"}}
    pollutants: Dict[str, Any] = Field(default_factory=dict)

    evidence: List[Evidence] = Field(default_factory=list)

    created_at: datetime = Field(default_factory=datetime.utcnow)

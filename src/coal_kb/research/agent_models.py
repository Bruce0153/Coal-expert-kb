"""定义受控 Agent 的计划、工具调用、预算和执行协议。"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

AGENT_PLAN_VERSION = "controlled-agent-plan.v1"


class AgentAction(str, Enum):
    """受控 Agent 唯一允许的动作。"""

    RETRIEVE = "retrieve"
    GRAPH = "graph"
    MULTIMODAL = "multimodal"


@dataclass(frozen=True)
class AgentPlanStep:
    """保存一个有序、可验证的计划步骤。"""

    index: int
    action: AgentAction
    reason: str
    inputs: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action.value,
            "reason": self.reason,
            "inputs": self.inputs,
        }


@dataclass(frozen=True)
class AgentPlan:
    """保存版本化计划和截断原因。"""

    query: str
    steps: tuple[AgentPlanStep, ...]
    truncated: bool = False
    version: str = AGENT_PLAN_VERSION
    planner: str = "controlled-rule-planner-v1"

    def validate(self) -> None:
        if self.version != AGENT_PLAN_VERSION:
            raise ValueError(f"Unsupported agent plan version: {self.version}")
        if not self.steps or self.steps[0].action is not AgentAction.RETRIEVE:
            raise ValueError("Controlled agent plan must start with retrieve")
        expected = list(range(1, len(self.steps) + 1))
        if [step.index for step in self.steps] != expected:
            raise ValueError("Agent plan step indexes must be contiguous")
        actions = [step.action for step in self.steps]
        if len(actions) != len(set(actions)):
            raise ValueError("Controlled agent plan cannot repeat actions")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "version": self.version,
            "planner": self.planner,
            "query": self.query,
            "truncated": self.truncated,
            "steps": [step.to_dict() for step in self.steps],
        }


@dataclass(frozen=True)
class AgentExecutionBudget:
    """限制工具调用次数和总执行时间。"""

    max_calls: int = 3
    max_duration_ms: float = 5000.0

    def validate(self) -> None:
        if self.max_calls < 1:
            raise ValueError("Agent max_calls must be positive")
        if self.max_duration_ms <= 0:
            raise ValueError("Agent max_duration_ms must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_calls": self.max_calls,
            "max_duration_ms": self.max_duration_ms,
        }


@dataclass(frozen=True)
class ToolExecutionRecord:
    """保存一次工具执行的可回放结果。"""

    index: int
    action: AgentAction
    reason: str
    status: str
    input_count: int
    output_count: int
    latency_ms: float
    inputs: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "action": self.action.value,
            "reason": self.reason,
            "status": self.status,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "latency_ms": round(self.latency_ms, 3),
            "inputs": self.inputs,
            "error_type": self.error_type,
            "error": self.error,
        }

    def to_legacy_step(self) -> dict[str, object]:
        return {
            "index": self.index,
            "action": self.action.value,
            "reason": self.reason,
            "input_count": self.input_count,
            "output_count": self.output_count,
            "latency_ms": round(self.latency_ms, 3),
            "status": self.status,
        }

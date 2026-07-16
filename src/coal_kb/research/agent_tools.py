"""注册并在预算内执行受控 Agent 工具。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from coal_kb.research.agent_models import (
    AgentAction,
    AgentExecutionBudget,
    AgentPlanStep,
    ToolExecutionRecord,
)
from coal_kb.research.graph import GraphRoute
from coal_kb.research.models import RouteResult
from coal_kb.research.multimodal import MultimodalRoute

BaseRoute = Callable[[], RouteResult]
ToolHandler = Callable[["AgentToolContext", dict[str, Any]], RouteResult]


@dataclass
class AgentToolContext:
    """保存工具可访问的最小运行上下文。"""

    query: str
    base_route: BaseRoute
    graph_route: GraphRoute
    multimodal_route: MultimodalRoute
    current: RouteResult | None = None


@dataclass(frozen=True)
class ToolSpec:
    """定义工具名、输入字段和处理函数。"""

    action: AgentAction
    handler: ToolHandler
    allowed_input_fields: frozenset[str] = frozenset()
    description: str = ""

    def validate_inputs(self, inputs: dict[str, Any]) -> None:
        unknown = sorted(set(inputs) - set(self.allowed_input_fields))
        if unknown:
            raise ValueError(
                f"Tool {self.action.value} received unsupported inputs: {', '.join(unknown)}"
            )


@dataclass
class ToolRegistry:
    """只允许显式注册的固定动作。"""

    _tools: dict[AgentAction, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        if spec.action in self._tools:
            raise ValueError(f"Agent tool is already registered: {spec.action.value}")
        self._tools[spec.action] = spec

    def resolve(self, action: AgentAction | str) -> ToolSpec:
        try:
            normalized = action if isinstance(action, AgentAction) else AgentAction(action)
        except ValueError as exc:
            raise ValueError(f"Agent tool is not registered: {action}") from exc
        if normalized not in self._tools:
            raise ValueError(f"Agent tool is not registered: {normalized.value}")
        return self._tools[normalized]

    def describe(self) -> list[dict[str, Any]]:
        return [
            {
                "action": action.value,
                "allowed_input_fields": sorted(spec.allowed_input_fields),
                "description": spec.description,
            }
            for action, spec in sorted(self._tools.items(), key=lambda item: item[0].value)
        ]


@dataclass
class BudgetedToolExecutor:
    """验证工具和输入，并执行调用次数与耗时预算。"""

    registry: ToolRegistry
    budget: AgentExecutionBudget
    records: list[ToolExecutionRecord] = field(default_factory=list)
    _started_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self.budget.validate()

    def execute(self, step: AgentPlanStep, context: AgentToolContext) -> RouteResult:
        self._check_budget()
        spec = self.registry.resolve(step.action)
        spec.validate_inputs(step.inputs)
        input_count = len(context.current.documents) if context.current is not None else 0
        started = time.monotonic()
        try:
            result = spec.handler(context, step.inputs)
        except Exception as exc:
            self.records.append(
                ToolExecutionRecord(
                    index=step.index,
                    action=step.action,
                    reason=step.reason,
                    status="failed",
                    input_count=input_count,
                    output_count=0,
                    latency_ms=(time.monotonic() - started) * 1000,
                    inputs=step.inputs,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
            )
            raise
        context.current = result
        self.records.append(
            ToolExecutionRecord(
                index=step.index,
                action=step.action,
                reason=step.reason,
                status="completed",
                input_count=input_count,
                output_count=len(result.documents),
                latency_ms=(time.monotonic() - started) * 1000,
                inputs=step.inputs,
            )
        )
        self._check_duration()
        return result

    def _check_budget(self) -> None:
        if len(self.records) >= self.budget.max_calls:
            raise RuntimeError("Agent tool call budget exceeded")
        self._check_duration()

    def _check_duration(self) -> None:
        elapsed_ms = (time.monotonic() - self._started_at) * 1000
        if elapsed_ms > self.budget.max_duration_ms:
            raise TimeoutError("Agent execution time budget exceeded")

    @property
    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._started_at) * 1000


def build_controlled_tool_registry() -> ToolRegistry:
    """创建唯一的 retrieve、graph、multimodal 工具注册表。"""

    registry = ToolRegistry()

    def retrieve(context: AgentToolContext, _: dict[str, Any]) -> RouteResult:
        if context.current is not None:
            raise ValueError("Retrieve can only run as the first controlled agent action")
        return context.base_route()

    def graph(context: AgentToolContext, _: dict[str, Any]) -> RouteResult:
        if context.current is None:
            raise ValueError("Graph requires retrieved evidence")
        return context.graph_route.process(lambda result=context.current: result)

    def multimodal(context: AgentToolContext, _: dict[str, Any]) -> RouteResult:
        if context.current is None:
            raise ValueError("Multimodal requires retrieved evidence")
        return context.multimodal_route.process(
            context.query,
            lambda result=context.current: result,
        )

    registry.register(
        ToolSpec(
            AgentAction.RETRIEVE,
            retrieve,
            description="Run the standard evidence route once.",
        )
    )
    registry.register(
        ToolSpec(
            AgentAction.GRAPH,
            graph,
            description="Apply typed graph reasoning to current evidence.",
        )
    )
    registry.register(
        ToolSpec(
            AgentAction.MULTIMODAL,
            multimodal,
            description="Apply multimodal and configured asset retrieval to current evidence.",
        )
    )
    return registry

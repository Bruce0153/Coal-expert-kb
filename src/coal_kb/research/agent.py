"""使用独立 Planner、注册表和预算执行器运行受控 Agent。"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from coal_kb.research.agent_models import AgentAction, AgentExecutionBudget
from coal_kb.research.agent_planner import ControlledAgentPlanner
from coal_kb.research.agent_tools import (
    AgentToolContext,
    BudgetedToolExecutor,
    ToolRegistry,
    build_controlled_tool_registry,
)
from coal_kb.research.graph import GraphRoute
from coal_kb.research.models import RouteResult
from coal_kb.research.multimodal import MultimodalRoute

BaseRoute = Callable[[], RouteResult]


@dataclass
class ControlledAgentRoute:
    """规划、验证并在固定工具和预算内执行研究路线。"""

    max_steps: int = 3
    max_duration_ms: float = 5000.0
    allowed_actions: tuple[AgentAction, ...] = (
        AgentAction.RETRIEVE,
        AgentAction.GRAPH,
        AgentAction.MULTIMODAL,
    )
    planner: ControlledAgentPlanner = field(default_factory=ControlledAgentPlanner)
    registry_factory: Callable[[], ToolRegistry] = build_controlled_tool_registry

    def process(
        self,
        query: str,
        *,
        base_route: BaseRoute,
        graph_route: GraphRoute,
        multimodal_route: MultimodalRoute,
    ) -> RouteResult:
        if self.max_steps < 1:
            raise ValueError("Controlled agent max_steps must be positive")
        if AgentAction.RETRIEVE not in self.allowed_actions:
            raise ValueError("Controlled agent must allow the retrieve action")
        started = time.monotonic()
        context = AgentToolContext(
            query=query,
            base_route=base_route,
            graph_route=graph_route,
            multimodal_route=multimodal_route,
        )
        registry = self.registry_factory()
        executor = BudgetedToolExecutor(
            registry=registry,
            budget=AgentExecutionBudget(
                max_calls=self.max_steps,
                max_duration_ms=self.max_duration_ms,
            ),
        )
        retrieve_step = self.planner.process(
            query,
            retrieved=RouteResult(documents=[], trace={}),
            multimodal_route=multimodal_route,
            max_steps=1,
        ).steps[0]
        retrieved = executor.execute(retrieve_step, context)
        plan = self.planner.process(
            query,
            retrieved=retrieved,
            multimodal_route=multimodal_route,
            max_steps=self.max_steps,
        )
        self.planner.validate_allowed(plan, self.allowed_actions)
        for step in plan.steps[1:]:
            executor.execute(step, context)
        if context.current is None:
            raise RuntimeError("Controlled agent completed without evidence")
        stop_reason = "max_steps_reached" if plan.truncated else "plan_completed"
        return RouteResult(
            documents=context.current.documents,
            trace={
                "route": "agent",
                "agent": {
                    "policy": "controlled-v2",
                    "allowed_actions": [action.value for action in self.allowed_actions],
                    "max_steps": self.max_steps,
                    "steps": [record.to_legacy_step() for record in executor.records],
                    "stop_reason": stop_reason,
                    "duration_ms": round((time.monotonic() - started) * 1000, 3),
                    "plan": plan.to_dict(),
                    "tool_registry": registry.describe(),
                    "budget": executor.budget.to_dict(),
                    "executions": [record.to_dict() for record in executor.records],
                },
                "result_trace": context.current.trace,
            },
        )


__all__ = ["AgentAction", "ControlledAgentRoute"]

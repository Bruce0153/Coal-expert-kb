"""执行白名单动作、固定步数和完整 Trace 的受控 Agent 路线。"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from coal_kb.research.graph import GraphRoute
from coal_kb.research.models import RouteResult
from coal_kb.research.multimodal import MultimodalRoute

BaseRoute = Callable[[], RouteResult]


class AgentAction(str, Enum):
    """受控 Agent 唯一允许执行的动作。"""

    RETRIEVE = "retrieve"
    GRAPH = "graph"
    MULTIMODAL = "multimodal"


@dataclass
class ControlledAgentRoute:
    """根据可解释规则执行最多三步，不接受任意工具名。"""

    max_steps: int = 3
    allowed_actions: tuple[AgentAction, ...] = (
        AgentAction.RETRIEVE,
        AgentAction.GRAPH,
        AgentAction.MULTIMODAL,
    )

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
        base = base_route()
        current = base
        steps = [
            self._step(
                index=1,
                action=AgentAction.RETRIEVE,
                input_count=0,
                output_count=len(current.documents),
                reason="required_initial_retrieval",
            )
        ]
        planned = self._plan_actions(query, current, multimodal_route)
        for action, reason in planned:
            if len(steps) >= self.max_steps:
                break
            if action not in self.allowed_actions:
                raise ValueError(f"Agent action is not allowed: {action.value}")
            input_count = len(current.documents)
            action_started = time.monotonic()
            if action is AgentAction.GRAPH:
                current = graph_route.process(lambda result=current: result)
            elif action is AgentAction.MULTIMODAL:
                current = multimodal_route.process(
                    query,
                    lambda result=current: result,
                )
            else:
                raise ValueError(f"Unexpected repeated action: {action.value}")
            steps.append(
                self._step(
                    index=len(steps) + 1,
                    action=action,
                    input_count=input_count,
                    output_count=len(current.documents),
                    reason=reason,
                    latency_ms=(time.monotonic() - action_started) * 1000,
                )
            )

        stop_reason = "plan_completed"
        if len(steps) >= self.max_steps and len(planned) + 1 > self.max_steps:
            stop_reason = "max_steps_reached"
        return RouteResult(
            documents=current.documents,
            trace={
                "route": "agent",
                "agent": {
                    "policy": "controlled-v1",
                    "allowed_actions": [action.value for action in self.allowed_actions],
                    "max_steps": self.max_steps,
                    "steps": steps,
                    "stop_reason": stop_reason,
                    "duration_ms": round((time.monotonic() - started) * 1000, 3),
                },
                "result_trace": current.trace,
            },
        )

    def _plan_actions(
        self,
        query: str,
        current: RouteResult,
        multimodal_route: MultimodalRoute,
    ) -> list[tuple[AgentAction, str]]:
        lowered = query.lower()
        actions: list[tuple[AgentAction, str]] = []
        if re.search(r"比较|关系|机制|为什么|导致|路径|compare|relationship|mechanism|why|cause|pathway", lowered):
            actions.append((AgentAction.GRAPH, "relationship_or_mechanism_query"))
        requested = multimodal_route.requested_modalities(query)
        available = {
            multimodal_route.infer_modality(document)
            for document in current.documents
        }
        if requested != {"text"} or available - {"text"}:
            actions.append((AgentAction.MULTIMODAL, "multimodal_evidence_available_or_requested"))
        return actions

    @staticmethod
    def _step(
        *,
        index: int,
        action: AgentAction,
        input_count: int,
        output_count: int,
        reason: str,
        latency_ms: float = 0.0,
    ) -> dict[str, object]:
        return {
            "index": index,
            "action": action.value,
            "reason": reason,
            "input_count": input_count,
            "output_count": output_count,
            "latency_ms": round(latency_ms, 3),
            "status": "completed",
        }

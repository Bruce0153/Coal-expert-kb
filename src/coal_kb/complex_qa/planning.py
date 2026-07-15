"""把路由结果转换为可执行子问题和结构化操作。"""

from __future__ import annotations

import re

from coal_kb.complex_qa.router import route_question
from coal_kb.core.models.query import (
    AggregationSpec,
    ComplexQuestionSpec,
    QueryPlan,
    SubQuerySpec,
)

_FIELD_PATTERNS = (
    (r"温度|temperature", "T_K"),
    (r"压力|pressure", "P_MPa"),
    (r"nh3|氨", "pollutants.NH3"),
    (r"hcn|氰化氢", "pollutants.HCN"),
    (r"h2|氢气|氢产率", "pollutants.H2"),
    (r"co2|二氧化碳", "pollutants.CO2"),
    (r"co(?!2)|一氧化碳", "pollutants.CO"),
    (r"ch4|甲烷", "pollutants.CH4"),
    (r"焦油|tar", "pollutants.tar"),
)
_GROUP_PATTERNS = (
    (r"按煤种|不同煤种", "coal_name"),
    (r"按阶段|不同阶段", "stage"),
    (r"按反应器|不同反应器", "reactor_type"),
    (r"按气化剂|不同气化剂", "gas_agent"),
)


def clone_plan_for_subquery(plan: QueryPlan, subquery: SubQuerySpec) -> QueryPlan:
    """克隆计划并将复杂子问题转换为普通事实检索。"""
    cloned = plan.model_copy(deep=True)
    cloned.query.raw = subquery.query
    cloned.query.normalized = subquery.query
    cloned.query.rewritten = None
    cloned.complex = ComplexQuestionSpec(
        query_type="fact",
        confidence=1.0,
        reason="复杂路线内部子检索",
    )
    return cloned


def _comparison_entities(query: str) -> list[str]:
    patterns = (
        r"比较\s*(.+?)\s*(?:与|和|及|vs\.?|versus)\s*(.+?)(?:对|在|的|之间|有何|$)",
        r"(.+?)\s*(?:与|和|vs\.?)\s*(.+?)\s*(?:有何不同|的区别|的差异)",
    )
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.I)
        if match:
            values = [" ".join(value.strip(" ，。；;：:").split()) for value in match.groups()[:2]]
            return [value for value in values if value]
    return []


def _aggregation_spec(query: str) -> AggregationSpec:
    lowered = query.lower()
    operation = "count"
    if re.search(r"平均|均值|average|mean", lowered):
        operation = "average"
    elif re.search(r"中位数|median", lowered):
        operation = "median"
    elif re.search(r"前\s*\d+|top\s*\d+|排名", lowered):
        operation = "top_k"
    elif re.search(r"最高|最大|max", lowered):
        operation = "max"
    elif re.search(r"最低|最小|min", lowered):
        operation = "min"
    elif re.search(r"按.+统计|分组|group", lowered):
        operation = "group_by"

    field = None
    for pattern, canonical in _FIELD_PATTERNS:
        if re.search(pattern, lowered, flags=re.I):
            field = canonical
            break

    group_by = None
    for pattern, canonical in _GROUP_PATTERNS:
        if re.search(pattern, lowered, flags=re.I):
            group_by = canonical
            break
    if group_by and operation == "count":
        operation = "group_by"

    top_match = re.search(r"(?:前|top)\s*(\d+)", lowered, flags=re.I)
    top_k = max(1, min(100, int(top_match.group(1)))) if top_match else 10
    return AggregationSpec(operation=operation, field=field, group_by=group_by, top_k=top_k)


def build_complex_spec(
    query: str,
    *,
    max_subqueries: int,
    max_multi_hop_steps: int,
) -> ComplexQuestionSpec:
    """根据问题生成统一复杂问答计划。"""
    query_type, confidence, reason = route_question(query)
    subqueries: list[SubQuerySpec] = []
    entities: list[str] = []
    aggregation = None

    if query_type == "comparison":
        entities = _comparison_entities(query)
        if len(entities) == 2:
            for index, entity in enumerate(entities, start=1):
                subqueries.append(
                    SubQuerySpec(
                        subquery_id=f"comparison_{index}",
                        query=f"{entity} 实验条件 结果 反应机理",
                        purpose=f"检索比较对象 {entity} 的独立证据",
                    )
                )
        else:
            subqueries = [
                SubQuerySpec(
                    subquery_id="comparison_1",
                    query=f"{query} 对象一 证据",
                    purpose="检索第一侧证据",
                ),
                SubQuerySpec(
                    subquery_id="comparison_2",
                    query=f"{query} 对象二 证据",
                    purpose="检索第二侧证据",
                ),
            ]
    elif query_type == "multi_hop":
        templates: tuple[tuple[str, str, str, list[str]], ...] = (
            ("hop_1", f"{query} 关键反应 中间过程", "识别关键中间过程", []),
            ("hop_2", f"{query} 中间过程 对目标结果的影响", "连接中间过程与目标结果", ["hop_1"]),
            ("hop_3", f"{query} 实验条件 机制链 证据", "核验完整机制链及适用条件", ["hop_2"]),
        )
        for subquery_id, subquery, purpose, dependencies in templates[:max_multi_hop_steps]:
            subqueries.append(
                SubQuerySpec(
                    subquery_id=subquery_id,
                    query=subquery,
                    purpose=purpose,
                    depends_on=dependencies,
                )
            )
    elif query_type == "aggregation":
        aggregation = _aggregation_spec(query)
    elif query_type == "table":
        subqueries.append(
            SubQuerySpec(
                subquery_id="table_1",
                query=query,
                purpose="定位相关表格、行和单元格",
            )
        )
    elif query_type == "cross_document":
        templates = (
            ("cross_support", f"{query} 支持性证据", "检索支持主要结论的文档"),
            ("cross_conflict", f"{query} 相反结果 冲突证据", "检索相反或冲突结论"),
            ("cross_conditions", f"{query} 实验条件差异", "解释文献差异的条件来源"),
        )
        subqueries = [
            SubQuerySpec(subquery_id=item[0], query=item[1], purpose=item[2])
            for item in templates
        ]

    return ComplexQuestionSpec(
        query_type=query_type,
        confidence=confidence,
        reason=reason,
        subqueries=subqueries[:max_subqueries],
        comparison_entities=entities,
        aggregation=aggregation,
        require_table=query_type == "table",
        require_cross_document=query_type == "cross_document",
    )

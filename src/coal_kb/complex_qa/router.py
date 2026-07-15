"""使用可解释规则识别复杂科学问答路线。"""

from __future__ import annotations

import re

from coal_kb.core.models.query import QuestionType


def route_question(query: str) -> tuple[QuestionType, float, str]:
    """返回问题类型、置信度和可解释原因。"""
    normalized = " ".join(query.strip().lower().split())
    if not normalized:
        return "unanswerable", 1.0, "问题为空"

    if re.search(r"未公开|无公开记录|无法从文献确认|不存在的实验|undocumented|not published", normalized):
        return "unanswerable", 0.95, "问题明确要求无法由知识库证实的信息"

    if re.search(r"表中|表格|第\s*\d+\s*行|哪一列|单元格|table\s*\d+", normalized):
        return "table", 0.95, "问题明确引用表格、行、列或单元格"

    if re.search(r"平均|均值|中位数|最高|最低|最大|最小|前\s*\d+|top\s*\d+|排名|频率|出现最多|共有多少|多少篇|多少条|统计", normalized):
        return "aggregation", 0.92, "问题要求确定性统计、排序或聚合"

    if re.search(r"多篇文献|不同研究|文献共识|总体结论|综合来看|研究结论是否一致|冲突结论|跨文档|综述", normalized):
        return "cross_document", 0.91, "问题要求跨文档共识、差异或冲突综合"

    if re.search(r"比较|对比|区别|差异|相比| versus |vs\.?|与.+有何不同|和.+有何不同", f" {normalized} "):
        return "comparison", 0.9, "问题包含明确比较关系"

    if re.search(r"为什么|如何通过|怎样导致|共同导致|反应路径|机制链|从而|进而|因果链|multi[- ]hop", normalized):
        return "multi_hop", 0.84, "问题需要连接中间过程或因果证据"

    return "fact", 0.8, "未命中复杂路线规则，使用事实检索"

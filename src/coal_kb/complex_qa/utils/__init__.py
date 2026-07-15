"""导出复杂科学问答的文档处理公共函数。"""

from coal_kb.complex_qa.utils.documents import (
    clone_plan_for_subquery,
    deduplicate_documents,
    tag_documents,
)

__all__ = ["clone_plan_for_subquery", "deduplicate_documents", "tag_documents"]

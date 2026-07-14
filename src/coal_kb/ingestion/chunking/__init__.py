"""导出摄入阶段的文档切分策略。"""

from .splitter import split_docs_markdown_hierarchical_semantic, split_page_docs

__all__ = ["split_docs_markdown_hierarchical_semantic", "split_page_docs"]

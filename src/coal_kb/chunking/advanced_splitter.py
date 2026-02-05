"""Legacy shim.

This module is kept for backward compatibility. New ingestion defaults to
markdown_hierarchical_semantic strategy. Legacy behavior lives in
`coal_kb.chunking.legacy.advanced_splitter`.
"""

from .legacy.advanced_splitter import split_page_docs_section_aware

__all__ = ["split_page_docs_section_aware"]

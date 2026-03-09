from __future__ import annotations

from langchain_core.documents import Document

from coal_kb.chunking.markdown_hierarchical_semantic import (
    ChunkingParams,
    parse_markdown_sections,
)
from coal_kb.chunking.splitter import split_docs_markdown_hierarchical_semantic
from coal_kb.chunking.tokenizer import count_tokens


def test_markdown_heading_parse_tree() -> None:
    md = "# A\nintro\n## B\nbody\n### C\nchild"
    sections = parse_markdown_sections(md, heading_max_depth=4)
    paths = [s.heading_path for s in sections]
    assert "A" in paths
    assert "A > B" in paths
    assert "A > B > C" in paths


def test_table_block_is_atomic() -> None:
    md = "# Sec\n|c1|c2|\n|---|---|\n|1|2|\n\nNext sentence."
    docs = [Document(page_content=md, metadata={"source_file": "x.pdf", "format": "markdown"})]
    chunks = split_docs_markdown_hierarchical_semantic(
        docs,
        {
            "max_parent_tokens": 50,
            "max_child_tokens": 20,
            "overlap_tokens": 0,
            "similarity_threshold": 0.9,
            "heading_max_depth": 4,
            "embedding_backend": "lexical",
        },
    )
    child_texts = [c.page_content for c in chunks if not c.metadata.get("is_parent")]
    assert any("|c1|c2|" in t and "|1|2|" in t for t in child_texts)


def test_token_limit_enforced() -> None:
    md = "# T\n" + "句子。" * 200
    docs = [Document(page_content=md, metadata={"source_file": "x.pdf", "format": "markdown"})]
    chunks = split_docs_markdown_hierarchical_semantic(
        docs,
        {
            "max_parent_tokens": 80,
            "max_child_tokens": 30,
            "overlap_tokens": 0,
            "similarity_threshold": 0.0,
            "heading_max_depth": 4,
            "embedding_backend": "lexical",
        },
    )
    children = [c for c in chunks if not c.metadata.get("is_parent")]
    assert children
    assert all(count_tokens(c.page_content) <= 30 for c in children)


def test_semantic_boundary_split_on_low_similarity() -> None:
    md = "# S\ncoal gasification kinetics reaction conversion.\nbanana apple fruit smoothie recipe."
    docs = [Document(page_content=md, metadata={"source_file": "x", "format": "markdown"})]
    chunks = split_docs_markdown_hierarchical_semantic(
        docs,
        {
            "max_parent_tokens": 200,
            "max_child_tokens": 200,
            "overlap_tokens": 0,
            "similarity_threshold": 0.95,
            "heading_max_depth": 4,
            "embedding_backend": "lexical",
        },
    )
    children = [c.page_content for c in chunks if not c.metadata.get("is_parent")]
    assert len(children) >= 2

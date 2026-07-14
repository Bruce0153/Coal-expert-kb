"""兼容旧 BM25 与 RRF 导入路径。"""

from coal_kb.recall import bm25_rank, rrf_fuse, tokenize

__all__ = ["bm25_rank", "rrf_fuse", "tokenize"]

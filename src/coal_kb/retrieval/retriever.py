"""兼容旧 ExpertRetriever 导入及测试注入路径。"""

from coal_kb.retrieval import service as _service

make_embeddings = _service.make_embeddings


class ExpertRetriever(_service.ExpertRetriever):
    """保留旧模块 monkeypatch `make_embeddings` 的兼容行为。"""

    def __post_init__(self) -> None:
        _service.make_embeddings = make_embeddings
        super().__post_init__()


__all__ = ["ExpertRetriever", "make_embeddings"]

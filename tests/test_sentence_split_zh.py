from __future__ import annotations

from coal_kb.chunking.sentence_split import split_sentences


def test_split_sentences_merges_short_zh() -> None:
    text = "热解生成酚类。机理复杂！需考虑缩聚。"
    sentences = split_sentences(text, min_len=6)
    assert len(sentences) >= 2
    assert any("热解" in s for s in sentences)

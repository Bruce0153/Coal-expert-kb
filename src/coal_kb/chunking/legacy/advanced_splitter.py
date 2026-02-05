from __future__ import annotations

from typing import Dict, List

from langchain_core.documents import Document

from ..sectioner import infer_section_with_debug
from ..sentence_split import split_sentences
from ...loaders import detect_language


def _chunk_sentences(sentences: List[str], *, chunk_size: int, overlap: int) -> List[str]:
    if not sentences:
        return []
    chunks: List[str] = []
    buffer: List[str] = []
    buffer_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if buffer_len + sent_len + (1 if buffer else 0) > chunk_size and buffer:
            chunk_text = " ".join(buffer).strip()
            chunks.append(chunk_text)

            if overlap > 0:
                tail: List[str] = []
                tail_len = 0
                for s in reversed(buffer):
                    if tail_len + len(s) + (1 if tail else 0) > overlap:
                        break
                    tail.insert(0, s)
                    tail_len += len(s) + (1 if tail else 0)
                buffer = tail
                buffer_len = sum(len(s) for s in buffer) + max(len(buffer) - 1, 0)
            else:
                buffer = []
                buffer_len = 0

        buffer.append(sent)
        buffer_len += sent_len + (1 if buffer_len else 0)

    if buffer:
        chunks.append(" ".join(buffer).strip())
    return chunks


def split_page_docs_section_aware(
    page_docs: List[Document],
    *,
    default_chunk_size: int,
    default_chunk_overlap: int,
    profile_by_section: Dict[str, Dict[str, int]],
) -> List[Document]:
    """
    Split page docs into chunks with section-aware chunk sizes and sentence boundaries.
    """
    output: List[Document] = []
    for d in page_docs:
        meta = dict(d.metadata or {})
        section, debug = infer_section_with_debug(d.page_content)
        meta.setdefault("section", section or "unknown")
        meta["section_debug"] = debug

        profile = profile_by_section.get(meta["section"], {})
        lang = detect_language(d.page_content or "")
        base_chunk_size = default_chunk_size
        base_chunk_overlap = default_chunk_overlap
        if lang == "zh":
            base_chunk_size = int(default_chunk_size * 0.7)
            base_chunk_overlap = int(default_chunk_overlap * 1.3)
        elif lang == "mixed":
            base_chunk_size = int(default_chunk_size * 0.85)
            base_chunk_overlap = int(default_chunk_overlap * 1.1)
        chunk_size = int(profile.get("chunk_size", base_chunk_size))
        chunk_overlap = int(profile.get("chunk_overlap", base_chunk_overlap))

        sentences = split_sentences(d.page_content or "", min_len=8 if lang == "zh" else 6)
        for chunk_text in _chunk_sentences(sentences, chunk_size=chunk_size, overlap=chunk_overlap):
            output.append(Document(page_content=chunk_text, metadata=dict(meta)))

    return output

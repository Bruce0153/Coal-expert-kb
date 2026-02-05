from __future__ import annotations

"""Markdown-first hierarchical + semantic chunking.

Pipeline:
1) Parse markdown headings into section hierarchy (parent sections).
2) Build parent chunks with token budget while preserving heading path.
3) Build child chunks from parent content using semantic boundaries from
   sentence-level similarities, plus overlap and token limits.
4) Protect markdown tables, fenced code blocks, and block formulas as atomic
   units so they are never sentence-split.
"""

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from langchain_core.documents import Document

from .sentence_split import split_sentences
from .tokenizer import count_tokens

logger = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class ChunkingParams:
    max_parent_tokens: int = 1200
    max_child_tokens: int = 300
    overlap_tokens: int = 60
    similarity_threshold: float = 0.72
    heading_max_depth: int = 4
    embedding_backend: str = "local_st"


@dataclass
class SectionNode:
    heading_path: str
    level: int
    content: str


@dataclass
class AtomicUnit:
    text: str
    atomic: bool


def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()[:16]


def _build_parent_id(source_file: str, heading_path: str, section_start_anchor: int) -> str:
    payload = f"{source_file}|{heading_path}|{section_start_anchor}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def _build_child_id(parent_id: str, child_local_span: str, text: str) -> str:
    payload = f"{parent_id}|{child_local_span}|{_hash_text(text)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def parse_markdown_sections(markdown_text: str, *, heading_max_depth: int = 4) -> List[SectionNode]:
    sections: List[SectionNode] = []
    path_stack: dict[int, str] = {}
    current_heading = "(root)"
    current_level = 0
    buffer: List[str] = []

    for line in markdown_text.splitlines():
        m = _HEADING_RE.match(line)
        if m and len(m.group(1)) <= heading_max_depth:
            if buffer:
                sections.append(SectionNode(heading_path=current_heading, level=current_level, content="\n".join(buffer).strip()))
            level = len(m.group(1))
            path_stack[level] = m.group(2).strip()
            for k in list(path_stack.keys()):
                if k > level:
                    path_stack.pop(k)
            parts = [path_stack[k] for k in sorted(path_stack.keys())]
            current_heading = " > ".join(parts) if parts else "(root)"
            current_level = level
            buffer = []
        else:
            buffer.append(line)

    if buffer:
        sections.append(SectionNode(heading_path=current_heading, level=current_level, content="\n".join(buffer).strip()))
    return [s for s in sections if s.content]


def _extract_atomic_units(text: str) -> List[AtomicUnit]:
    lines = text.splitlines()
    units: List[AtomicUnit] = []
    para: List[str] = []
    in_code = False
    in_formula = False

    def flush_para() -> None:
        nonlocal para
        if para:
            units.append(AtomicUnit(text="\n".join(para).strip(), atomic=False))
            para = []

    i = 0
    while i < len(lines):
        line = lines[i]
        striped = line.strip()
        if striped.startswith("```"):
            flush_para()
            code_block = [line]
            in_code = not in_code
            i += 1
            while i < len(lines) and in_code:
                code_block.append(lines[i])
                if lines[i].strip().startswith("```"):
                    in_code = False
                    i += 1
                    break
                i += 1
            units.append(AtomicUnit(text="\n".join(code_block).strip(), atomic=True))
            continue

        if striped == "$$":
            flush_para()
            formula = [line]
            in_formula = True
            i += 1
            while i < len(lines) and in_formula:
                formula.append(lines[i])
                if lines[i].strip() == "$$":
                    in_formula = False
                    i += 1
                    break
                i += 1
            units.append(AtomicUnit(text="\n".join(formula).strip(), atomic=True))
            continue

        if "|" in line and i + 1 < len(lines) and re.search(r"\|?\s*:?-{3,}", lines[i + 1]):
            flush_para()
            table = [line, lines[i + 1]]
            i += 2
            while i < len(lines) and "|" in lines[i] and lines[i].strip():
                table.append(lines[i])
                i += 1
            units.append(AtomicUnit(text="\n".join(table).strip(), atomic=True))
            continue

        if not striped:
            flush_para()
            i += 1
            continue

        para.append(line)
        i += 1

    flush_para()
    return [u for u in units if u.text]


def _pack_units(units: Sequence[AtomicUnit], max_tokens: int) -> List[str]:
    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0
    for unit in units:
        tok = count_tokens(unit.text)
        if cur and cur_tokens + tok > max_tokens:
            chunks.append("\n\n".join(cur).strip())
            cur = []
            cur_tokens = 0
        cur.append(unit.text)
        cur_tokens += tok
    if cur:
        chunks.append("\n\n".join(cur).strip())
    return chunks


class _SentenceEmbedder:
    def __init__(self, backend: str = "local_st") -> None:
        self.backend = backend
        self.model = None
        self.existing = None
        if backend == "local_st":
            try:
                from sentence_transformers import SentenceTransformer

                self.model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as exc:
                logger.warning("local sentence-transformer unavailable, fallback lexical embeddings: %s", exc)
        elif backend == "existing_factory":
            try:
                from coal_kb.embeddings.factory import EmbeddingsConfig, make_embeddings

                self.existing = make_embeddings(EmbeddingsConfig())
            except Exception as exc:
                logger.warning("existing embedding factory unavailable, fallback lexical embeddings: %s", exc)

    def encode(self, texts: Sequence[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.model is not None:
            vectors = self.model.encode(list(texts), batch_size=32, show_progress_bar=False, normalize_embeddings=True)
            return [v.tolist() for v in vectors]
        if self.existing is not None:
            try:
                vectors = self.existing.embed_documents(list(texts))
                return [list(v) for v in vectors]
            except Exception:
                pass
        return [_lexical_embedding(t) for t in texts]


def _lexical_embedding(text: str, dim: int = 128) -> List[float]:
    vec = [0.0] * dim
    for token in re.findall(r"[\u4e00-\u9fff]|\w+", text.lower()):
        vec[hash(token) % dim] += 1.0
    norm = math.sqrt(sum(x * x for x in vec)) or 1.0
    return [x / norm for x in vec]


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _semantic_sentence_groups(sentences: List[str], similarity_threshold: float, embedding_backend: str) -> List[List[str]]:
    if len(sentences) <= 1:
        return [sentences] if sentences else []
    embedder = _SentenceEmbedder(backend=embedding_backend)
    vecs = embedder.encode(sentences)
    groups: List[List[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = _cosine(vecs[i - 1], vecs[i])
        if sim < similarity_threshold:
            groups.append([sentences[i]])
        else:
            groups[-1].append(sentences[i])
    return groups


def _pack_sentence_groups(groups: List[List[str]], max_tokens: int, overlap_tokens: int) -> List[str]:
    chunks: List[str] = []
    cur: List[str] = []
    cur_tokens = 0

    def overlap_tail(sentences: List[str]) -> List[str]:
        if overlap_tokens <= 0:
            return []
        tail: List[str] = []
        tok = 0
        for s in reversed(sentences):
            s_tok = count_tokens(s)
            if tok + s_tok > overlap_tokens:
                break
            tail.insert(0, s)
            tok += s_tok
        return tail

    for group in groups:
        group_text = " ".join(group).strip()
        gt = count_tokens(group_text)
        if cur:
            chunks.append(" ".join(cur).strip())
            cur = overlap_tail(cur)
            cur_tokens = count_tokens(" ".join(cur))
        if gt > max_tokens and len(group) > 1:
            for sent in group:
                st = count_tokens(sent)
                if cur and cur_tokens + st > max_tokens:
                    chunks.append(" ".join(cur).strip())
                    cur = overlap_tail(cur)
                    cur_tokens = count_tokens(" ".join(cur))
                cur.append(sent)
                cur_tokens += st
            continue
        if cur and cur_tokens + gt > max_tokens:
            chunks.append(" ".join(cur).strip())
            cur = overlap_tail(cur)
            cur_tokens = count_tokens(" ".join(cur))
        cur.append(group_text)
        cur_tokens += gt
    if cur:
        chunks.append(" ".join(cur).strip())
    return [c for c in chunks if c]


def split_docs_markdown_hierarchical_semantic(docs: Iterable[Document], params: ChunkingParams) -> List[Document]:
    out: List[Document] = []

    for doc in docs:
        meta = dict(doc.metadata or {})
        source_file = str(meta.get("source_file", "unknown"))
        doc_format = str(meta.get("format", "text"))
        sections = parse_markdown_sections(doc.page_content or "", heading_max_depth=params.heading_max_depth)
        if not sections:
            sections = [SectionNode(heading_path="(root)", level=0, content=doc.page_content or "")]

        for sec in sections:
            section_units = _extract_atomic_units(sec.content)
            parent_texts = _pack_units(section_units, params.max_parent_tokens)
            for parent_idx, parent_text in enumerate(parent_texts):
                p_start = parent_idx
                p_end = parent_idx + 1
                parent_id = _build_parent_id(source_file, sec.heading_path, p_start)
                parent_meta = {
                    **meta,
                    "chunk_id": parent_id,
                    "format": doc_format,
                    "heading_path": sec.heading_path,
                    "is_parent": True,
                    "parent_id": parent_id,
                    "position_start": p_start,
                    "position_end": p_end,
                }
                out.append(Document(page_content=parent_text, metadata=parent_meta))

                units = _extract_atomic_units(parent_text)
                child_groups: List[List[str]] = []
                for unit in units:
                    if unit.atomic:
                        child_groups.append([unit.text])
                    else:
                        sents = split_sentences(unit.text, min_len=6)
                        if sents:
                            child_groups.extend(
                                _semantic_sentence_groups(
                                    sents,
                                    similarity_threshold=params.similarity_threshold,
                                    embedding_backend=params.embedding_backend,
                                )
                            )
                child_texts = _pack_sentence_groups(
                    child_groups,
                    max_tokens=params.max_child_tokens,
                    overlap_tokens=params.overlap_tokens,
                )
                for child_idx, child_text in enumerate(child_texts):
                    c_id = _build_child_id(parent_id, f"{child_idx}:{child_idx + 1}", child_text)
                    child_meta = {
                        **meta,
                        "chunk_id": c_id,
                        "format": doc_format,
                        "heading_path": sec.heading_path,
                        "is_parent": False,
                        "parent_id": parent_id,
                        "position_start": child_idx,
                        "position_end": child_idx + 1,
                    }
                    out.append(Document(page_content=child_text, metadata=child_meta))

    return out

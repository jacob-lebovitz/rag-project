"""Token-aware recursive chunking with overlap.

Design considerations:
- We chunk at roughly a fixed *token* budget (via tiktoken) instead of raw
  character count. This keeps chunks predictable in size for the embedding
  model, which has token limits.
- We split *recursively* on a hierarchy of separators:
  paragraphs -> sentences -> words. This preserves semantic units wherever
  possible and only falls back to word-level splitting for extremely long
  run-on text.
- We add overlap between adjacent chunks so that answers spanning a
  boundary still get retrieved. Default overlap is 64 tokens.
- Page provenance is preserved: a chunk never crosses page boundaries so
  each chunk can cite a single page number.
"""
from __future__ import annotations

import re
import uuid
from typing import Iterable, List, Tuple

import tiktoken

from ..models import Chunk

# cl100k_base is a reasonable, generic tokenizer for size estimation.
_ENC = tiktoken.get_encoding("cl100k_base")


def _n_tokens(text: str) -> int:
    return len(_ENC.encode(text))


def _split_sentences(text: str) -> List[str]:
    # Lightweight sentence splitter — avoids an nltk download at runtime.
    # Splits on ., !, ? followed by whitespace + capital/digit; keeps punctuation.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])", text.strip())
    return [p for p in parts if p]


def _split_to_chunks(text: str, max_tokens: int, overlap: int) -> List[str]:
    """Recursive splitter: paragraphs -> sentences -> words."""
    if not text.strip():
        return []
    if _n_tokens(text) <= max_tokens:
        return [text.strip()]

    # 1) Try paragraphs
    paragraphs = [p for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) > 1:
        return _pack_units(paragraphs, max_tokens, overlap, join=" \n\n ")

    # 2) Try sentences
    sentences = _split_sentences(text)
    if len(sentences) > 1:
        return _pack_units(sentences, max_tokens, overlap, join=" ")

    # 3) Fallback: word-level
    words = text.split()
    return _pack_units(words, max_tokens, overlap, join=" ")


def _pack_units(
    units: List[str], max_tokens: int, overlap: int, join: str
) -> List[str]:
    """Greedily pack smaller units into chunks of at most ``max_tokens``.

    If a single unit is itself larger than max_tokens, it is re-split
    recursively on the next level of granularity.
    """
    chunks: List[str] = []
    buffer: List[str] = []
    buffer_tokens = 0

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if buffer:
            chunks.append(join.join(buffer).strip())
            buffer = []
            buffer_tokens = 0

    for unit in units:
        unit_tokens = _n_tokens(unit)

        # Oversized atomic unit — recurse
        if unit_tokens > max_tokens:
            flush()
            chunks.extend(_split_to_chunks(unit, max_tokens, overlap))
            continue

        if buffer_tokens + unit_tokens <= max_tokens:
            buffer.append(unit)
            buffer_tokens += unit_tokens
        else:
            flush()
            buffer.append(unit)
            buffer_tokens = unit_tokens

    flush()

    # Apply token overlap between adjacent chunks
    if overlap > 0 and len(chunks) > 1:
        chunks = _apply_overlap(chunks, overlap)

    return chunks


def _apply_overlap(chunks: List[str], overlap_tokens: int) -> List[str]:
    """Prepend the tail of chunk[i-1] onto chunk[i] for context continuity."""
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        prev = chunks[i - 1]
        prev_ids = _ENC.encode(prev)
        tail_ids = prev_ids[-overlap_tokens:]
        tail_text = _ENC.decode(tail_ids)
        out.append(f"{tail_text} {chunks[i]}")
    return out


def chunk_pages(
    pages: Iterable[str],
    *,
    doc_id: str,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
    start_chunk_id: int = 0,
) -> Tuple[List[Chunk], int]:
    """Convert page texts into ``Chunk`` objects.

    Returns the chunks and the next free chunk_id (for multi-doc ingestion).
    """
    chunks: List[Chunk] = []
    cid = start_chunk_id
    for page_idx, page_text in enumerate(pages, start=1):
        if not page_text.strip():
            continue
        for piece in _split_to_chunks(page_text, chunk_size, chunk_overlap):
            chunks.append(
                Chunk(
                    chunk_id=cid,
                    doc_id=doc_id,
                    filename=filename,
                    page=page_idx,
                    text=piece,
                )
            )
            cid += 1
    return chunks, cid


def new_doc_id() -> str:
    """Generate a short, URL-safe document id."""
    return uuid.uuid4().hex[:12]

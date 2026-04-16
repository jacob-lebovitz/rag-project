"""BM25 keyword search over the same chunk set.

BM25 complements dense retrieval by catching:
- Exact identifier / code / acronym matches that embeddings often miss.
- Rare words that don't cluster well in the embedding space.

We re-build the BM25 index from scratch on each ingestion — this keeps
implementation simple and is fast enough for documents up to tens of
thousands of chunks. For larger corpora, swap for an incremental scheme
(e.g. Tantivy / Elasticsearch).
"""
from __future__ import annotations

import re
import string
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from ..models import Chunk


_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "he", "her", "hers", "him", "his", "how", "i", "if", "in",
    "into", "is", "it", "its", "of", "on", "or", "our", "ours", "she", "so",
    "than", "that", "the", "their", "them", "they", "this", "to", "was", "we",
    "were", "what", "when", "where", "which", "who", "whom", "why", "will",
    "with", "you", "your", "yours",
}

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenizer with stopword removal."""
    toks = _TOKEN_RE.findall(text.lower())
    return [t for t in toks if t not in _STOPWORDS and len(t) > 1]


class KeywordStore:
    """In-memory BM25 store, rebuilt from the canonical chunk list."""

    def __init__(self) -> None:
        self._chunks: List[Chunk] = []
        self._bm25: BM25Okapi | None = None

    def rebuild(self, chunks: List[Chunk]) -> None:
        self._chunks = list(chunks)
        tokenized = [tokenize(c.text) for c in self._chunks]
        # BM25Okapi requires at least one non-empty doc
        if not tokenized or all(not t for t in tokenized):
            self._bm25 = None
            return
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, k: int) -> List[Tuple[Chunk, float]]:
        if self._bm25 is None or not self._chunks:
            return []
        q_toks = tokenize(query)
        if not q_toks:
            return []
        scores = self._bm25.get_scores(q_toks)
        # Top-k indices
        k = min(k, len(scores))
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self._chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]

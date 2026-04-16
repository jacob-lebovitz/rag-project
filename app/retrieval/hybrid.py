"""Hybrid retriever: dense + sparse, fused via Reciprocal Rank Fusion (RRF).

Why RRF?
- It does not require calibrating BM25 scores to cosine-similarity ranges.
- It is robust and well-documented for hybrid search.
- Formula: ``score(d) = Σ_i 1 / (k + rank_i(d))`` summed across all
  retriever lists that contain d. ``k`` defaults to 60 (Cormack et al.).

After RRF we truncate to ``top_k_final`` and return. An optional LLM-based
re-rank step can be plugged in on top.
"""
from __future__ import annotations

import logging
import threading
from functools import lru_cache
from typing import Dict, List, Tuple

import numpy as np

from ..config import settings
from ..generation import get_client
from ..models import Chunk
from .keyword_store import KeywordStore
from .vector_store import VectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Coordinates dense + sparse retrieval and fuses the results."""

    def __init__(self) -> None:
        self.vector_store = VectorStore(settings.index_dir)
        self.keyword_store = KeywordStore()
        self._lock = threading.Lock()
        # Warm keyword store from whatever is already persisted
        self.keyword_store.rebuild(self.vector_store.all_chunks())

    # ─────────────────── Ingestion ───────────────────
    def add_chunks(self, chunks: List[Chunk]) -> None:
        """Embed and add chunks to both stores."""
        if not chunks:
            return
        with self._lock:
            embeddings = get_client().embed([c.text for c in chunks])
            self.vector_store.add(chunks, embeddings)
            # Rebuild BM25 from full corpus
            self.keyword_store.rebuild(self.vector_store.all_chunks())
            logger.info(
                "Indexed %d chunks (total in store: %d)",
                len(chunks),
                self.vector_store.size,
            )

    @property
    def next_chunk_id(self) -> int:
        return self.vector_store.next_chunk_id

    @property
    def size(self) -> int:
        return self.vector_store.size

    def reset(self) -> None:
        with self._lock:
            self.vector_store.reset()
            self.keyword_store.rebuild([])

    # ─────────────────── Retrieval ───────────────────
    def search(
        self,
        query: str,
        *,
        top_k_final: int | None = None,
        top_k_vector: int | None = None,
        top_k_keyword: int | None = None,
    ) -> List[Tuple[Chunk, float]]:
        """Run hybrid retrieval and return fused top-k chunks."""
        if self.size == 0:
            return []

        top_k_final = top_k_final or settings.top_k_final
        top_k_vector = top_k_vector or settings.top_k_vector
        top_k_keyword = top_k_keyword or settings.top_k_keyword

        # Dense retrieval
        q_emb = get_client().embed([query])
        dense_hits = self.vector_store.search(q_emb, k=top_k_vector)

        # Sparse retrieval
        sparse_hits = self.keyword_store.search(query, k=top_k_keyword)

        fused = _reciprocal_rank_fusion(
            [dense_hits, sparse_hits], k=settings.rrf_k
        )

        return fused[:top_k_final]


# ────────────────────────── Utilities ──────────────────────────
def _reciprocal_rank_fusion(
    result_lists: List[List[Tuple[Chunk, float]]],
    k: int = 60,
) -> List[Tuple[Chunk, float]]:
    """Fuse multiple ranked lists into one via RRF.

    Returns a list of ``(chunk, rrf_score)`` sorted by RRF score desc.
    """
    rrf_scores: Dict[int, float] = {}
    chunk_by_id: Dict[int, Chunk] = {}

    for results in result_lists:
        for rank, (chunk, _score) in enumerate(results):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                k + rank + 1
            )
            chunk_by_id[chunk.chunk_id] = chunk

    fused = sorted(
        ((chunk_by_id[cid], score) for cid, score in rrf_scores.items()),
        key=lambda x: x[1],
        reverse=True,
    )
    return fused


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    """Process-wide singleton."""
    return HybridRetriever()

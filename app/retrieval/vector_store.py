"""FAISS-backed vector store with simple on-disk persistence.

Design choices:
- ``IndexFlatIP`` with L2-normalized vectors. With normalized vectors, inner
  product equals cosine similarity — which is what we want for semantic
  search on sentence-level embeddings. Flat index is exact; for up to a few
  hundred thousand chunks this is fast enough on CPU and avoids the tuning
  overhead of IVF/HNSW. The index can be swapped for ``IndexHNSWFlat`` later.
- We persist three files side-by-side:
    - ``faiss.index``      : the FAISS index
    - ``chunks.jsonl``     : the Chunk metadata (text, doc_id, page, ...)
    - ``meta.json``        : bookkeeping (next chunk_id, embedding dim)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import faiss
import numpy as np

from ..models import Chunk

logger = logging.getLogger(__name__)


def _l2_normalize(vecs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vecs / norms).astype(np.float32)


class VectorStore:
    """FAISS inner-product index over L2-normalized embeddings."""

    INDEX_FILE = "faiss.index"
    CHUNKS_FILE = "chunks.jsonl"
    META_FILE = "meta.json"

    def __init__(self, index_dir: Path, dim: int | None = None) -> None:
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._index: faiss.Index | None = None
        self._chunks: List[Chunk] = []
        self._next_chunk_id: int = 0
        self._dim: int | None = dim
        self._load()

    # ─────────────────────────── Public API ───────────────────────────
    @property
    def next_chunk_id(self) -> int:
        return self._next_chunk_id

    @property
    def size(self) -> int:
        return 0 if self._index is None else int(self._index.ntotal)

    def all_chunks(self) -> List[Chunk]:
        return list(self._chunks)

    def add(self, chunks: List[Chunk], embeddings: np.ndarray) -> None:
        """Append chunks + embeddings to the index and persist."""
        if not chunks:
            return
        if embeddings.shape[0] != len(chunks):
            raise ValueError("embeddings/chunks length mismatch")

        embeddings = _l2_normalize(embeddings)

        if self._index is None:
            self._dim = embeddings.shape[1]
            self._index = faiss.IndexFlatIP(self._dim)

        if embeddings.shape[1] != self._dim:
            raise ValueError(
                f"Embedding dim mismatch: index={self._dim}, incoming={embeddings.shape[1]}"
            )

        self._index.add(embeddings)
        self._chunks.extend(chunks)
        self._next_chunk_id = max(self._next_chunk_id, max(c.chunk_id for c in chunks) + 1)
        self._persist()

    def search(
        self, query_vec: np.ndarray, k: int
    ) -> List[Tuple[Chunk, float]]:
        """Return ``[(chunk, score)]`` sorted by descending similarity."""
        if self._index is None or self._index.ntotal == 0:
            return []
        if query_vec.ndim == 1:
            query_vec = query_vec[None, :]
        query_vec = _l2_normalize(query_vec.astype(np.float32))
        k = min(k, self._index.ntotal)
        scores, idxs = self._index.search(query_vec, k)
        out: List[Tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < 0 or idx >= len(self._chunks):
                continue
            out.append((self._chunks[idx], float(score)))
        return out

    def reset(self) -> None:
        """Wipe in-memory + on-disk state."""
        self._index = None
        self._chunks = []
        self._next_chunk_id = 0
        for fn in (self.INDEX_FILE, self.CHUNKS_FILE, self.META_FILE):
            p = self.index_dir / fn
            if p.exists():
                p.unlink()

    # ─────────────────────────── Persistence ──────────────────────────
    def _persist(self) -> None:
        assert self._index is not None
        faiss.write_index(self._index, str(self.index_dir / self.INDEX_FILE))

        with (self.index_dir / self.CHUNKS_FILE).open("w", encoding="utf-8") as f:
            for c in self._chunks:
                f.write(json.dumps(c.model_dump(), ensure_ascii=False) + "\n")

        meta = {"next_chunk_id": self._next_chunk_id, "dim": self._dim}
        (self.index_dir / self.META_FILE).write_text(json.dumps(meta))

    def _load(self) -> None:
        idx_path = self.index_dir / self.INDEX_FILE
        chunks_path = self.index_dir / self.CHUNKS_FILE
        meta_path = self.index_dir / self.META_FILE
        if not (idx_path.exists() and chunks_path.exists() and meta_path.exists()):
            return

        try:
            self._index = faiss.read_index(str(idx_path))
            with chunks_path.open("r", encoding="utf-8") as f:
                self._chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
            meta = json.loads(meta_path.read_text())
            self._next_chunk_id = int(meta.get("next_chunk_id", 0))
            self._dim = meta.get("dim")
            logger.info(
                "Loaded vector store: %d chunks, dim=%s", len(self._chunks), self._dim
            )
        except Exception as e:  # pragma: no cover - corrupted state
            logger.exception("Failed to load vector store: %s — starting fresh", e)
            self._index = None
            self._chunks = []
            self._next_chunk_id = 0

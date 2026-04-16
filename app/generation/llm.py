"""Thin wrapper around the Mistral AI SDK.

Exposes three operations:
- ``embed(texts)``: batch embed a list of strings (using ``mistral-embed``)
- ``chat(messages)``: chat completion (using ``mistral-small-latest`` by default)
- ``rewrite_query(question)``: LLM-driven query transformation

All calls are retried with exponential backoff on transient errors.
"""
from __future__ import annotations

import logging
import time
from functools import lru_cache
from typing import List

import numpy as np
from mistralai import Mistral

from ..config import settings

logger = logging.getLogger(__name__)

_EMBED_BATCH = 32  # Mistral embed endpoint handles batches efficiently
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 1.0  # seconds


class MistralClient:
    """Wrapper around the official Mistral SDK with retry + batching."""

    def __init__(
        self,
        api_key: str | None = None,
        chat_model: str | None = None,
        embed_model: str | None = None,
    ) -> None:
        api_key = api_key or settings.mistral_api_key
        if not api_key:
            raise RuntimeError(
                "MISTRAL_API_KEY is not configured. Set it in your environment or .env"
            )
        self._client = Mistral(api_key=api_key)
        self.chat_model = chat_model or settings.mistral_chat_model
        self.embed_model = embed_model or settings.mistral_embed_model

    # ────────────────────────── Embeddings ──────────────────────────
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an ``(N, D)`` float32 numpy array of embeddings."""
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)

        all_vectors: List[List[float]] = []
        for i in range(0, len(texts), _EMBED_BATCH):
            batch = texts[i : i + _EMBED_BATCH]
            resp = self._retry(
                lambda: self._client.embeddings.create(
                    model=self.embed_model,
                    inputs=batch,
                )
            )
            all_vectors.extend([d.embedding for d in resp.data])
        return np.asarray(all_vectors, dtype=np.float32)

    # ────────────────────────── Chat ────────────────────────────────
    def chat(
        self,
        messages: List[dict],
        *,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> str:
        resp = self._retry(
            lambda: self._client.chat.complete(
                model=self.chat_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        )
        return resp.choices[0].message.content or ""

    # ─────────────────────── Query rewriting ────────────────────────
    def rewrite_query(self, question: str) -> str:
        """Rewrite the user's question into a self-contained search query.

        The rewrite is kept deliberately conservative — we do not want to
        hallucinate new topics the user did not ask about.
        """
        system = (
            "You rewrite user questions into concise, self-contained search "
            "queries suitable for retrieving relevant passages from a document "
            "knowledge base. Preserve all named entities, numbers, and domain "
            "terms exactly. Do not add information that is not in the original "
            "question. Respond with ONLY the rewritten query, no preamble."
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": question.strip()},
        ]
        rewritten = self.chat(messages, temperature=0.0, max_tokens=128).strip()
        # Safety fallback: if the rewrite is empty or absurdly long, keep the original.
        if not rewritten or len(rewritten) > 4 * len(question) + 200:
            return question
        return rewritten

    # ───────────────────────── Internals ────────────────────────────
    def _retry(self, fn):
        last_exc: Exception | None = None
        for attempt in range(_MAX_RETRIES):
            try:
                return fn()
            except Exception as e:  # noqa: BLE001 — SDK raises many subclasses
                last_exc = e
                delay = _RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "Mistral call failed (attempt %d/%d): %s. Retrying in %.1fs",
                    attempt + 1,
                    _MAX_RETRIES,
                    e,
                    delay,
                )
                time.sleep(delay)
        assert last_exc is not None
        raise last_exc


@lru_cache(maxsize=1)
def get_client() -> MistralClient:
    """Process-wide singleton."""
    return MistralClient()

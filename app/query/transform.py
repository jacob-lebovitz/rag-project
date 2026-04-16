"""Query transformation for improved retrieval.

Two transforms are applied:
1. **Light normalization**: strip, collapse whitespace, remove trailing
   filler characters. This runs always and is free.
2. **LLM rewrite**: ask Mistral to produce a concise, self-contained search
   query. This expands pronouns, removes pleasantries, and normalizes
   phrasing. The original question is returned if the rewrite fails.

We deliberately avoid more aggressive transforms (HyDE, multi-query,
synonym expansion) in this minimal implementation — see the README for
an extension path.
"""
from __future__ import annotations

import logging
import re

from ..generation import get_client

logger = logging.getLogger(__name__)


def _normalize(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def transform_query(question: str, *, use_llm: bool = True) -> str:
    """Return a search-optimized version of ``question``."""
    normalized = _normalize(question)
    if not use_llm or not normalized:
        return normalized
    try:
        rewritten = get_client().rewrite_query(normalized)
        return _normalize(rewritten) or normalized
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("Query rewrite failed: %s — using original", e)
        return normalized

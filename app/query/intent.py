"""Intent detection: should we trigger a knowledge-base search?

Strategy — **rules first, LLM second**:
1. Cheap regex / length rules short-circuit the obvious cases:
   - Pure greetings ("hi", "hello", "thanks") → small_talk
   - Very short (< N tokens) non-question messages → small_talk
   - Presence of a question mark or a Wh-word or command verb → knowledge_base
2. Anything ambiguous falls through to a tiny LLM classification call.

Why not use the LLM for every query? Two reasons:
- Latency: rules resolve in microseconds.
- Cost: the majority of chit-chat is trivial and doesn't need an API call.
"""
from __future__ import annotations

import logging
import re
from enum import Enum

from ..config import settings
from ..generation import get_client

logger = logging.getLogger(__name__)


class QueryIntent(str, Enum):
    KNOWLEDGE_BASE = "knowledge_base"
    SMALL_TALK = "small_talk"


_GREETINGS = {
    "hi", "hello", "hey", "yo", "hiya", "howdy", "greetings",
    "good morning", "good afternoon", "good evening", "good night",
    "thanks", "thank you", "thx", "ty", "bye", "goodbye", "cya", "see you",
    "ok", "okay", "cool", "nice", "lol", "haha", "sup", "what's up",
    "how are you", "how r u",
}

_WH_WORDS = {"what", "why", "how", "when", "where", "who", "which", "whose", "whom"}
_COMMAND_VERBS = {
    "explain", "summarize", "summarise", "describe", "list", "compare",
    "define", "show", "tell", "give", "find", "search", "cite", "extract",
}

_TOKEN_RE = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _rule_based(question: str) -> QueryIntent | None:
    """Return a decision if a cheap rule fires, else None."""
    q = question.strip().lower()
    if not q:
        return QueryIntent.SMALL_TALK

    # Pure greeting (allow trailing punctuation / emoji)
    stripped = re.sub(r"[^\w\s']", "", q).strip()
    if stripped in _GREETINGS:
        return QueryIntent.SMALL_TALK

    toks = _tokens(q)

    # Too short and not question-shaped → small talk
    if len(toks) < settings.min_query_tokens and "?" not in q:
        if not (toks and toks[0] in _WH_WORDS | _COMMAND_VERBS):
            return QueryIntent.SMALL_TALK

    # Question shape → knowledge base
    if "?" in q:
        return QueryIntent.KNOWLEDGE_BASE
    if toks and toks[0] in _WH_WORDS | _COMMAND_VERBS:
        return QueryIntent.KNOWLEDGE_BASE

    return None


_LLM_SYSTEM = (
    "You are an intent classifier for a document Q&A system. "
    "Classify the user's message into EXACTLY one of:\n"
    "- knowledge_base: the user is asking a question or requesting information "
    "that should be answered from uploaded documents.\n"
    "- small_talk: greetings, acknowledgements, meta questions about the "
    "assistant, or anything that does not require document retrieval.\n"
    "Respond with ONLY the label, nothing else."
)


def _llm_classify(question: str) -> QueryIntent:
    try:
        label = get_client().chat(
            [
                {"role": "system", "content": _LLM_SYSTEM},
                {"role": "user", "content": question.strip()},
            ],
            temperature=0.0,
            max_tokens=8,
        ).strip().lower()
        if "knowledge" in label:
            return QueryIntent.KNOWLEDGE_BASE
        if "small" in label or "talk" in label:
            return QueryIntent.SMALL_TALK
    except Exception as e:  # pragma: no cover — defensive
        logger.warning("LLM intent classification failed: %s — defaulting to KB", e)
    # Default to knowledge_base when uncertain — false positives here just
    # waste a retrieval call; false negatives skip retrieval entirely.
    return QueryIntent.KNOWLEDGE_BASE


def detect_intent(question: str, *, use_llm_fallback: bool = True) -> QueryIntent:
    """Return the detected intent for ``question``."""
    decision = _rule_based(question)
    if decision is not None:
        return decision
    if use_llm_fallback:
        return _llm_classify(question)
    return QueryIntent.KNOWLEDGE_BASE

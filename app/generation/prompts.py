"""Prompt templates used for answer generation."""
from __future__ import annotations

from typing import List, Tuple

from ..models import Chunk


RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions using ONLY the provided context passages from a document knowledge base.

Rules:
- Ground every factual claim in the provided context. If the context does not contain the answer, say so explicitly: "I don't have that information in the provided documents."
- Cite sources inline using [n] where n is the passage number. You may cite multiple passages, e.g. [1][3].
- Be concise and direct. Do not repeat the question.
- Do not invent facts, numbers, or quotations that are not in the context.
- If the question is ambiguous, state your interpretation briefly, then answer.
"""


SMALL_TALK_SYSTEM_PROMPT = """You are a friendly assistant for a document Q&A system. The user's message does not require searching the knowledge base (e.g., a greeting or meta question).

Reply briefly and warmly. If the user seems to want to ask a document question, gently invite them to do so.
"""


def build_rag_user_prompt(question: str, retrieved: List[Tuple[Chunk, float]]) -> str:
    """Format the retrieved chunks into a prompt body."""
    blocks: List[str] = []
    for i, (chunk, score) in enumerate(retrieved, start=1):
        blocks.append(
            f"[{i}] (source: {chunk.filename}, page {chunk.page}, score {score:.3f})\n"
            f"{chunk.text}"
        )
    context = "\n\n".join(blocks) if blocks else "(no context retrieved)"
    return (
        f"Context passages:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer (with inline [n] citations):"
    )

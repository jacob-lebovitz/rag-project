"""Query endpoint: intent detection → rewrite → hybrid search → answer."""
from __future__ import annotations

import logging

from fastapi import APIRouter

from ..generation import get_client
from ..generation.prompts import (
    RAG_SYSTEM_PROMPT,
    SMALL_TALK_SYSTEM_PROMPT,
    build_rag_user_prompt,
)
from ..models import QueryRequest, QueryResponse, SourceChunk
from ..query import QueryIntent, detect_intent, transform_query
from ..retrieval import get_retriever

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    """Answer a user question, optionally using the knowledge base."""
    question = req.question.strip()

    # 1. Intent detection — skip retrieval for small talk
    intent = detect_intent(question)
    logger.info("Intent=%s  question=%r", intent.value, question)

    if intent == QueryIntent.SMALL_TALK:
        answer = get_client().chat(
            [
                {"role": "system", "content": SMALL_TALK_SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.6,
            max_tokens=256,
        )
        return QueryResponse(
            answer=answer.strip(),
            intent=intent.value,
            transformed_query=None,
            sources=[],
        )

    # 2. Query transformation
    search_query = transform_query(question)

    # 3. Hybrid retrieval
    retriever = get_retriever()
    retrieved = retriever.search(search_query, top_k_final=req.top_k)

    # 4. Answer generation
    if not retrieved:
        return QueryResponse(
            answer=(
                "I don't have any relevant information in the uploaded documents "
                "to answer that. Try ingesting some PDFs first, or rephrasing the question."
            ),
            intent=intent.value,
            transformed_query=search_query,
            sources=[],
        )

    user_prompt = build_rag_user_prompt(question, retrieved)
    answer = get_client().chat(
        [
            {"role": "system", "content": RAG_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=1024,
    )

    sources = (
        [
            SourceChunk(
                doc_id=chunk.doc_id,
                filename=chunk.filename,
                page=chunk.page,
                chunk_id=chunk.chunk_id,
                score=score,
                text=chunk.text,
            )
            for chunk, score in retrieved
        ]
        if req.include_sources
        else []
    )

    return QueryResponse(
        answer=answer.strip(),
        intent=intent.value,
        transformed_query=search_query,
        sources=sources,
    )

"""Pydantic request/response schemas."""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ───────────────────────── Ingestion ─────────────────────────
class IngestedFile(BaseModel):
    filename: str
    doc_id: str
    num_pages: int
    num_chunks: int


class IngestResponse(BaseModel):
    status: str = "ok"
    files: List[IngestedFile]
    total_chunks: int


# ───────────────────────── Querying ──────────────────────────
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    top_k: Optional[int] = Field(None, description="Override top-k for retrieval")
    include_sources: bool = True


class SourceChunk(BaseModel):
    doc_id: str
    filename: str
    page: int
    chunk_id: int
    score: float
    text: str


class QueryResponse(BaseModel):
    answer: str
    intent: str  # "knowledge_base" or "small_talk"
    transformed_query: Optional[str] = None
    sources: List[SourceChunk] = []


# ───────────────────────── Internal ──────────────────────────
class Chunk(BaseModel):
    """Internal chunk record stored in the index."""
    chunk_id: int
    doc_id: str
    filename: str
    page: int
    text: str

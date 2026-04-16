"""Ingestion endpoint: accept one or more PDF files and index them."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List

import aiofiles
from fastapi import APIRouter, File, HTTPException, UploadFile, status

from ..config import settings
from ..ingestion import chunk_pages, extract_pdf_pages
from ..ingestion.chunker import new_doc_id
from ..models import IngestedFile, IngestResponse
from ..retrieval import get_retriever

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/ingest", tags=["ingestion"])


@router.post("", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_files(files: List[UploadFile] = File(...)) -> IngestResponse:
    """Upload one or more PDF files for ingestion into the knowledge base."""
    if not files:
        raise HTTPException(400, "No files uploaded")

    retriever = get_retriever()
    next_cid = retriever.next_chunk_id

    ingested: List[IngestedFile] = []
    total_chunks = 0

    for upload in files:
        if not upload.filename:
            continue
        if not upload.filename.lower().endswith(".pdf"):
            raise HTTPException(
                400, f"Only PDF files are supported (got: {upload.filename})"
            )

        doc_id = new_doc_id()
        dest = settings.uploads_dir / f"{doc_id}__{Path(upload.filename).name}"

        async with aiofiles.open(dest, "wb") as f:
            while True:
                block = await upload.read(1024 * 1024)
                if not block:
                    break
                await f.write(block)

        try:
            pages, num_pages = extract_pdf_pages(dest)
        except Exception as e:
            logger.exception("PDF extraction failed for %s", upload.filename)
            # Clean up the uploaded file on failure
            dest.unlink(missing_ok=True)
            raise HTTPException(500, f"Failed to extract PDF: {e}")

        chunks, next_cid = chunk_pages(
            pages,
            doc_id=doc_id,
            filename=upload.filename,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            start_chunk_id=next_cid,
        )

        if not chunks:
            logger.warning("No text extracted from %s", upload.filename)
            ingested.append(
                IngestedFile(
                    filename=upload.filename,
                    doc_id=doc_id,
                    num_pages=num_pages,
                    num_chunks=0,
                )
            )
            continue

        retriever.add_chunks(chunks)
        total_chunks += len(chunks)

        ingested.append(
            IngestedFile(
                filename=upload.filename,
                doc_id=doc_id,
                num_pages=num_pages,
                num_chunks=len(chunks),
            )
        )
        logger.info("Ingested %s: %d pages, %d chunks", upload.filename, num_pages, len(chunks))

    return IngestResponse(files=ingested, total_chunks=total_chunks)


@router.delete("", status_code=status.HTTP_204_NO_CONTENT)
async def reset_index() -> None:
    """Wipe the entire index and all uploaded files. Useful for testing."""
    get_retriever().reset()
    if settings.uploads_dir.exists():
        shutil.rmtree(settings.uploads_dir)
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Index reset")

"""PDF ingestion: extraction + chunking."""
from .pdf_parser import extract_pdf_pages
from .chunker import chunk_pages

__all__ = ["extract_pdf_pages", "chunk_pages"]

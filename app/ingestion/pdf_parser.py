"""PDF text extraction.

Design considerations (see README for deeper discussion):
- We use ``pdfplumber`` as the primary extractor because it preserves layout
  reasonably well and handles most text-based PDFs robustly.
- We fall back to ``pypdf`` if pdfplumber fails on a given page, which makes
  ingestion resilient to malformed/partially corrupt files.
- Each page is emitted separately so downstream chunking can preserve page
  provenance for citation.
- Scanned (image-only) PDFs are out of scope here; they would require OCR
  (e.g. pytesseract) — see README for the extension path.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import List, Tuple

import pdfplumber
from pypdf import PdfReader

logger = logging.getLogger(__name__)


_WHITESPACE_RE = re.compile(r"[ \t]+")
_MULTI_NEWLINE_RE = re.compile(r"\n{3,}")


def _clean_text(text: str) -> str:
    """Collapse redundant whitespace while preserving paragraph breaks."""
    if not text:
        return ""
    # Normalize CR/LF
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of spaces/tabs
    text = _WHITESPACE_RE.sub(" ", text)
    # Collapse 3+ newlines to 2 (keep paragraph breaks)
    text = _MULTI_NEWLINE_RE.sub("\n\n", text)
    # De-hyphenate line-broken words: "exam-\nple" -> "example"
    text = re.sub(r"-\n(\w)", r"\1", text)
    return text.strip()


def _extract_with_pdfplumber(path: Path) -> List[str]:
    pages: List[str] = []
    with pdfplumber.open(str(path)) as pdf:
        for page in pdf.pages:
            try:
                txt = page.extract_text() or ""
            except Exception as e:  # pragma: no cover - defensive
                logger.warning("pdfplumber failed on page: %s", e)
                txt = ""
            pages.append(_clean_text(txt))
    return pages


def _extract_with_pypdf(path: Path) -> List[str]:
    reader = PdfReader(str(path))
    return [_clean_text(p.extract_text() or "") for p in reader.pages]


def extract_pdf_pages(path: Path) -> Tuple[List[str], int]:
    """Return ``(pages_text, num_pages)`` for the given PDF.

    The primary extractor is pdfplumber; on total failure we fall back to
    pypdf. If a page is empty after cleaning we still emit an empty string so
    page numbers align with the physical document.
    """
    path = Path(path)
    try:
        pages = _extract_with_pdfplumber(path)
    except Exception as e:
        logger.warning("pdfplumber failed for %s (%s); falling back to pypdf", path, e)
        pages = _extract_with_pypdf(path)

    # If everything came back empty, try the fallback once more
    if all(not p for p in pages):
        try:
            pages = _extract_with_pypdf(path)
        except Exception as e:
            logger.error("Both PDF extractors failed for %s: %s", path, e)

    return pages, len(pages)

"""
pdf_processor.py
-----------------
Responsible for two of the "hands-on concepts" this project demonstrates:

    1. PDF text extraction
    2. Document chunking

Design notes:
- Text is extracted PAGE BY PAGE (not as one giant blob) so that every
  chunk can keep track of exactly which page it came from. That page
  number is what later powers citations like "Source: paper.pdf, Page: 5".
- Chunking is implemented manually (simple sliding window over
  characters) rather than pulling in a heavy text-splitting library.
  This keeps the dependency list small and makes the chunking logic
  fully transparent/inspectable for a mini-project.
"""

from dataclasses import dataclass, field
from typing import List
import uuid

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    """A single chunk of document text plus the metadata needed for citations."""
    chunk_id: str
    text: str
    doc_name: str
    page_number: int
    chunk_index: int  # index of this chunk within its page

    def metadata(self) -> dict:
        return {
            "doc_name": self.doc_name,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
        }


class PDFProcessingError(Exception):
    """Raised when a PDF cannot be read or contains no extractable text."""


def extract_pages(file_bytes: bytes, doc_name: str) -> List[dict]:
    """
    Extract text from every page of a PDF.

    Returns a list of {"page_number": int, "text": str} dicts (1-indexed
    page numbers, which is the convention readers expect for citations).

    Raises PDFProcessingError for invalid/corrupted PDFs or files with no
    extractable text (e.g. pure image scans with no OCR layer).
    """
    try:
        reader = PdfReader(file_bytes)
    except (PdfReadError, Exception) as exc:  # pypdf can raise various errors
        raise PDFProcessingError(f"Could not read '{doc_name}': {exc}") from exc

    if reader.is_encrypted:
        try:
            reader.decrypt("")  # try an empty password before giving up
        except Exception:
            pass
        if reader.is_encrypted:
            raise PDFProcessingError(f"'{doc_name}' is password-protected and cannot be read.")

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append({"page_number": i, "text": text.strip()})

    if not any(p["text"] for p in pages):
        raise PDFProcessingError(
            f"No extractable text found in '{doc_name}'. "
            "It may be a scanned/image-only PDF with no text layer."
        )

    return pages


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split a block of text into overlapping character-based chunks.

    A simple sliding window: each chunk is `chunk_size` characters, and
    consecutive chunks overlap by `overlap` characters so that context
    near chunk boundaries isn't lost. We snap chunk boundaries to the
    nearest whitespace where possible to avoid slicing words in half.
    """
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Try to end the chunk at a space rather than mid-word.
        if end < text_len:
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break

        # Move the window forward, stepping back by `overlap` for context continuity.
        start = max(end - overlap, start + 1)

    return chunks


def process_pdf(file_bytes: bytes, doc_name: str) -> List[Chunk]:
    """
    Full pipeline for one PDF: extract pages -> chunk each page -> attach metadata.
    Returns a flat list of Chunk objects ready for embedding + storage.
    """
    pages = extract_pages(file_bytes, doc_name)

    all_chunks: List[Chunk] = []
    for page in pages:
        page_chunks = chunk_text(page["text"])
        for idx, chunk_str in enumerate(page_chunks):
            all_chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    text=chunk_str,
                    doc_name=doc_name,
                    page_number=page["page_number"],
                    chunk_index=idx,
                )
            )

    if not all_chunks:
        raise PDFProcessingError(f"'{doc_name}' produced no usable chunks after processing.")

    return all_chunks

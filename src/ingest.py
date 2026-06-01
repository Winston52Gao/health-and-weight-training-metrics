"""PDF ingestion and chunking utilities.

This module extracts text from PDFs and splits them into overlapping chunks
suitable for semantic search. Chunks include metadata (source, page range)
so retrieved results can be traced back to their source.
"""
from pathlib import Path
import fitz  # PyMuPDF
from typing import List, Dict, Any
import uuid


def extract_text_from_pdf(path: Path) -> List[Dict[str, Any]]:
    """Extract text by page from a PDF file.

    Returns a list of dicts: {"page": int, "text": str}
    """
    doc = fitz.open(path)
    pages = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text")
        pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def recursive_chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200, separators=None) -> List[str]:
    """Chunk text recursively by preferred separators.

    This tries larger separators first (\n\n, \n, ". ", " ") to avoid breaking
    sentences unnaturally. Returns list of text chunks with specified overlap.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    # Find a separator position to split near chunk_size
    for sep in separators:
        parts = text.split(sep)
        if len(parts) == 1:
            continue
        chunks = []
        current = parts[0]
        for part in parts[1:]:
            candidate = current + sep + part
            if len(candidate) >= chunk_size:
                chunks.append(candidate.strip())
                current = ""
            else:
                current = candidate
        if current:
            chunks.append(current.strip())

        # If chunks look reasonable, apply overlap and return
        if len(chunks) >= 1:
            merged = []
            for i, c in enumerate(chunks):
                if i == 0:
                    merged.append(c)
                else:
                    # prepend overlap from previous chunk
                    prev = merged[-1]
                    overlap_text = prev[-overlap:] if overlap > 0 else ""
                    merged.append((overlap_text + " " + c).strip())
            # final filter to ensure sizes
            return [m for m in merged if m]

    # Fallback: simple sliding window
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = max(end - overlap, end)
    return chunks


def ingest_pdfs(pdf_dir: Path, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Ingest all PDFs in a directory and return list of chunk metadata.

    Each chunk is a dict: {"id": str, "text": str, "source": str, "page": [start, end]}
    """
    pdf_dir = Path(pdf_dir)
    all_chunks = []
    for pdf in sorted(pdf_dir.glob("*.pdf")):
        pages = extract_text_from_pdf(pdf)
        # combine contiguous pages into a single large text then chunk
        full_text = "\n\n".join(p["text"] for p in pages)
        chunks = recursive_chunk_text(full_text, chunk_size=chunk_size, overlap=overlap)
        for i, c in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            meta = {"id": chunk_id, "text": c, "source": str(pdf), "chunk_index": i}
            all_chunks.append(meta)
    return all_chunks

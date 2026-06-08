"""Structure-aware PDF ingestion for RAG.

Provides functions:
- pdf_loader: read pages and return text per page
- toc_extractor: parse pages 2-3 to build TOC entries -> page ranges
- structure_parser: detect headings and assign paragraphs to sections
- chunk_builder: create coherent chunks from paragraphs respecting sections
- metadata_builder: assemble final chunk metadata

Design choices:
- Heuristic heading detection (ALL CAPS, numbered headings, short titles)
- TOC pages (2-3) are preferred for section labels and page ranges
- Overlap is by paragraphs (1-2) to preserve coherence
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import fitz
import re
import uuid
import difflib


def pdf_loader(pdf_path: Path) -> List[Dict[str, Any]]:
    """Load PDF and return list of pages as dicts: {page_number:int, text:str}.

    Page numbers are 1-based to match TOC entries.
    Images are ignored by using text extraction only.
    """
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text")
        pages.append({"page_number": i + 1, "text": text}
                     )
    doc.close()
    return pages


def toc_extractor(pages: List[Dict[str, Any]], toc_pages: List[int] = [2, 3]) -> List[Tuple[str, int]]:
    """Extract TOC entries from given pages (1-based indices).

    Returns ordered list of (title, start_page).
    Uses simple regex heuristics to find lines ending in a page number
    """
    entries: List[Tuple[str, int]] = []
    page_map = {p["page_number"]: p["text"] for p in pages}
    # combine text of requested toc pages
    toc_text = "\n\n".join(page_map.get(pg, "") for pg in toc_pages)
    if not toc_text.strip():
        return entries

    # process line by line
    lines = [l.strip() for l in toc_text.splitlines() if l.strip()]
    # regex: title ... page
    p_dot = re.compile(r"^(?P<title>.+?)\s+\.{2,}\s*(?P<page>\d+)$")
    p_space = re.compile(r"^(?P<title>.+?)\s+(?P<page>\d+)$")
    for line in lines:
        m = p_dot.match(line) or p_space.match(line)
        if m:
            title = m.group("title").strip()
            try:
                page = int(m.group("page"))
            except Exception:
                continue
            entries.append((title, page))

    # deduplicate and sort by page
    entries = sorted(entries, key=lambda x: x[1])
    # if empty, attempt to find numbered lines like '1. Introduction 5'
    if not entries:
        for line in lines:
            m = re.search(r"(?P<title>.+?)\s+(?P<page>\d+)$", line)
            if m:
                entries.append((m.group("title").strip(), int(m.group("page"))))
    return entries


def build_page_section_map(toc_entries: List[Tuple[str, int]], max_page: int) -> Dict[int, str]:
    """Given toc entries (title, start_page) produce mapping page->section name.

    Pages before first entry are unlabeled. The end of a section is next_start-1.
    """
    page_section = {}
    if not toc_entries:
        return page_section
    starts = toc_entries
    for i, (title, start) in enumerate(starts):
        end = (starts[i + 1][1] - 1) if i + 1 < len(starts) else max_page
        for p in range(start, end + 1):
            page_section[p] = title
    return page_section


def detect_headings_on_page(text: str) -> List[Tuple[str, int]]:
    """Return list of (heading_text, line_index) detected on the page text.

    Heuristics:
    - ALL CAPS lines of reasonable length
    - Numbered headings like '1.2 Title'
    - Short lines (<=6 words) followed by blank line
    """
    headings = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        s = line.strip()
        if not s:
            continue
        # numbered heading
        if re.match(r"^\d+(\.\d+)*\s+.+", s):
            headings.append((s, i))
            continue
        # ALL CAPS heuristic, ignore very short
        words = s.split()
        if len(s) > 4 and s.isupper() and len(words) <= 10:
            headings.append((s, i))
            continue
        # short line followed by blank or short next line
        if len(words) <= 6:
            # check next line existence
            if i + 1 < len(lines) and lines[i + 1].strip() == "":
                headings.append((s, i))
                continue
    return headings


def structure_parser(pages: List[Dict[str, Any]], toc_map: Dict[int, str]) -> List[Dict[str, Any]]:
    """Parse pages into paragraph-level items with inferred section labels.

    Returns list of paragraphs: {text, page, detected_heading (opt), section_name (opt), confidence}
    """
    paragraphs = []
    # prepare a list of unique TOC titles for fuzzy matching
    toc_titles = list(set(toc_map.values())) if toc_map else []

    # extract a flat list of paragraphs with page numbers and maintain
    # a current_section state within each page so sections can change
    # mid-page when a detected heading appears.
    for p in pages:
        page_no = p["page_number"]
        raw = p["text"]
        # split into paragraphs by blank lines
        parts = [s.strip() for s in re.split(r"\n\s*\n", raw) if s.strip()]
        # detect headings on the page
        headings = detect_headings_on_page(raw)
        heading_texts = [h[0] for h in headings]

        # initialize current section for this page from TOC mapping if present
        current_section: Optional[str] = toc_map.get(page_no)
        # confidence associated with the current_section source
        current_confidence: float = 0.9 if page_no in toc_map else 0.25

        for para in parts:
            para_entry = {"text": para, "page": page_no, "detected_heading": None, "section_name": None, "confidence": 0.0}
            # examine the first line to see if this paragraph is a heading
            first_line = para.splitlines()[0].strip()
            if first_line in heading_texts:
                # mark detected heading and update current_section
                para_entry["detected_heading"] = first_line
                # fuzzy match heading to TOC titles when available
                if toc_titles:
                    matches = difflib.get_close_matches(first_line, toc_titles, n=1, cutoff=0.6)
                    if matches:
                        current_section = matches[0]
                        current_confidence = 0.75
                    else:
                        current_section = first_line
                        current_confidence = 0.6
                else:
                    current_section = first_line
                    current_confidence = 0.5

            # assign the current section (which may have been initialized from
            # the page-level TOC or updated by a detected heading above)
            para_entry["section_name"] = current_section
            para_entry["confidence"] = current_confidence if current_section is not None else 0.25

            paragraphs.append(para_entry)

    return paragraphs


def chunk_builder(paragraphs: List[Dict[str, Any]], char_limit: int = 1000, overlap_paras: int = 1) -> List[Dict[str, Any]]:
    """Merge paragraphs into chunks respecting section boundaries and char limits.

    Returns list of chunks with page ranges and section_name.
    """
    chunks = []
    current_chunk = []
    current_section = None

    def flush_chunk():
        if not current_chunk:
            return None
        text = "\n\n".join(p["text"] for p in current_chunk)
        pages = [p["page"] for p in current_chunk]
        chunk = {
            "id": str(uuid.uuid4()),
            "text": text,
            "start_page": min(pages),
            "end_page": max(pages),
            "section_name": current_section,
            "paragraph_count": len(current_chunk),
            "paragraphs": current_chunk.copy(),
        }
        return chunk

    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        sec = p.get("section_name")
        # if starting new chunk
        if not current_chunk:
            current_chunk.append(p)
            current_section = sec
            i += 1
            continue

        # prefer to keep same section; if different and current chunk non-empty and would exceed small size, flush
        if sec != current_section and len(current_chunk) > 0:
            # if new paragraph small and current chunk short, allow mixing
            if len("\n\n".join(pp["text"] for pp in current_chunk)) < int(0.5 * char_limit):
                current_chunk.append(p)
                i += 1
                continue
            else:
                chunk = flush_chunk()
                if chunk:
                    chunks.append(chunk)
                # overlap: carry last few paragraphs into next
                overlap = current_chunk[-overlap_paras:] if overlap_paras > 0 else []
                current_chunk = overlap.copy()
                current_section = sec
                i += 0
                continue

        # if adding paragraph would exceed char limit, flush and start new
        combined_text = "\n\n".join(pp["text"] for pp in current_chunk + [p])
        if len(combined_text) > char_limit:
            chunk = flush_chunk()
            if chunk:
                chunks.append(chunk)
            # start new chunk with overlap
            overlap = current_chunk[-overlap_paras:] if overlap_paras > 0 else []
            current_chunk = overlap.copy()
            current_section = sec
            # if overlap copies the paragraph, avoid infinite loop
            if current_chunk and current_chunk[-1] is p:
                # paragraph already in overlap, skip adding
                i += 1
                continue
            else:
                # add paragraph to start new chunk
                current_chunk.append(p)
                i += 1
                continue
        else:
            current_chunk.append(p)
            i += 1

    # flush last
    last = None
    if current_chunk:
        last = flush_chunk()
        if last:
            chunks.append(last)

    # assign chunk_index and reduce paragraphs detail for storage
    for idx, c in enumerate(chunks):
        c["chunk_index"] = idx
        # compute a simple confidence as mean of paragraph confidence
        confs = [pp.get("confidence", 0.0) for pp in c["paragraphs"]]
        c["section_confidence"] = (sum(confs) / len(confs)) if confs else 0.0
        # remove full paragraph objects to keep metadata small; keep counts and sample pages
        c.pop("paragraphs")
    return chunks


def metadata_builder(chunks: List[Dict[str, Any]], source_file: Path) -> List[Dict[str, Any]]:
    """Prepare final metadata records per chunk with required fields.

    Fields: uuid (id), source filename, start_page, end_page, chunk_index, text, section_name, subsection (opt), confidence
    """
    out = []
    for c in chunks:
        rec = {
            "uuid": c.get("id"),
            "source": str(source_file),
            "start_page": c.get("start_page"),
            "end_page": c.get("end_page"),
            "chunk_index": c.get("chunk_index"),
            "text": c.get("text"),
            "section_name": c.get("section_name"),
            "subsection": None,
            "section_confidence": c.get("section_confidence", 0.0),
        }
        out.append(rec)
    return out


def process_pdf_structured(pdf_path: Path, chunk_char_limit: int = 1000, overlap_paras: int = 1) -> List[Dict[str, Any]]:
    """Full pipeline for a single PDF: load, extract TOC, parse structure, build chunks, return metadata list."""
    pages = pdf_loader(pdf_path)
    max_page = max(p["page_number"] for p in pages) if pages else 0
    toc_entries = toc_extractor(pages, toc_pages=[2, 3])
    toc_map = build_page_section_map(toc_entries, max_page)
    paragraphs = structure_parser(pages, toc_map)
    chunks = chunk_builder(paragraphs, char_limit=chunk_char_limit, overlap_paras=overlap_paras)
    metadata = metadata_builder(chunks, pdf_path)
    return metadata

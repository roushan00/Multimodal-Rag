"""PDF -> text blocks, markdown tables, figure crops and full-page renders (PyMuPDF)."""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

import pymupdf

from app.config import get_settings
from app.models import Chunk, FigureRef, PageContent, ParsedDoc

logger = logging.getLogger(__name__)


def doc_id_for(path: Path) -> str:
    return hashlib.sha1(path.read_bytes()).hexdigest()[:12]


def _overlaps(a: pymupdf.Rect, b: pymupdf.Rect, thresh: float = 0.5) -> bool:
    inter = a & b
    return not inter.is_empty and inter.get_area() / max(a.get_area(), 1e-6) >= thresh


def parse_pdf(path: str | Path) -> ParsedDoc:
    s = get_settings()
    path = Path(path)
    doc_id = doc_id_for(path)
    pages_dir = s.pages_dir / doc_id
    figs_dir = s.figures_dir / doc_id
    pages_dir.mkdir(parents=True, exist_ok=True)
    figs_dir.mkdir(parents=True, exist_ok=True)

    pdf = pymupdf.open(path)
    pages: list[PageContent] = []

    for pno, page in enumerate(pdf, start=1):
        pc = PageContent(page=pno)

        # 1) Full-page render (input for ColQwen and for the answering VLM)
        render = pages_dir / f"p{pno:04d}.png"
        page.get_pixmap(dpi=s.page_dpi).save(render)
        pc.render_path = str(render)

        # 2) Tables -> markdown; remember their areas so we don't duplicate as text
        table_rects: list[pymupdf.Rect] = []
        try:
            for tab in page.find_tables().tables:
                md = tab.to_markdown().strip()
                if md:
                    pc.tables_md.append(md)
                    table_rects.append(pymupdf.Rect(tab.bbox))
        except Exception:
            pass

        # 3) Text blocks outside tables
        blocks = page.get_text("blocks")
        for x0, y0, x1, y1, text, _, btype in blocks:
            if btype != 0 or not text.strip():
                continue
            r = pymupdf.Rect(x0, y0, x1, y1)
            if any(_overlaps(r, tr) for tr in table_rects):
                continue
            pc.text_blocks.append(" ".join(text.split()))

        # 4) Embedded raster images -> crops
        for idx, info in enumerate(page.get_images(full=True)):
            xref = info[0]
            for rect in page.get_image_rects(xref):
                if rect.width < s.min_image_px or rect.height < s.min_image_px:
                    continue
                out = figs_dir / f"p{pno:04d}_img{idx}.png"
                page.get_pixmap(dpi=int(s.page_dpi * 1.5), clip=rect).save(out)
                around = pymupdf.Rect(rect.x0 - 20, rect.y0 - 60, rect.x1 + 20, rect.y1 + 80)
                nearby = " ".join(page.get_text("text", clip=around).split())[:500]
                pc.figures.append(FigureRef(str(out), pno, tuple(rect), nearby))
                break

        pages.append(pc)

    pdf.close()
    return ParsedDoc(doc_id=doc_id, source=str(path), pages=pages)


# ── Semantic chunking ────────────────────────────────────────────────────

_HEADING_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*\.?\s+"           # 1. or 1.2.3
    r"|[A-Z][A-Z\s]{3,}$"           # ALL CAPS LINE
    r"|(?:Chapter|Section|Part)\s+\d"  # Chapter 1, Section 2
    r")",
    re.MULTILINE,
)


def _is_heading(block: str) -> bool:
    """Heuristic: short line that looks like a section heading."""
    return len(block) < 120 and bool(_HEADING_RE.match(block.strip()))


def _semantic_split(blocks: list[str], max_size: int, overlap: int) -> list[str]:
    """Group text blocks into chunks, preferring to split at paragraph/heading boundaries.

    Each block from PyMuPDF is already a visual paragraph. We merge consecutive
    blocks into chunks up to max_size, breaking at heading boundaries when possible.
    If a merged section exceeds max_size, fall back to sentence-boundary splitting.
    """
    if not blocks:
        return []

    # Group blocks into sections by headings
    sections: list[str] = []
    current: list[str] = []

    for block in blocks:
        if _is_heading(block) and current:
            sections.append("\n".join(current))
            current = [block]
        else:
            current.append(block)
    if current:
        sections.append("\n".join(current))

    # Merge small sections, split large ones
    chunks: list[str] = []
    buffer = ""

    for section in sections:
        candidate = f"{buffer}\n{section}".strip() if buffer else section

        if len(candidate) <= max_size:
            buffer = candidate
        else:
            # Flush buffer as a chunk
            if buffer:
                chunks.append(buffer)
            # If section itself is too large, split it by sentences
            if len(section) > max_size:
                chunks.extend(_split_sentences(section, max_size, overlap))
                buffer = ""
            else:
                buffer = section

    if buffer:
        chunks.append(buffer)

    return [c.strip() for c in chunks if c.strip()]


def _split_sentences(text: str, size: int, overlap: int) -> list[str]:
    """Fallback: split on sentence boundaries."""
    if len(text) <= size:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        cut = text.rfind(". ", start + size // 2, end)
        if cut != -1 and end < len(text):
            end = cut + 1
        out.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in out if c]


def to_chunks(doc: ParsedDoc) -> list[Chunk]:
    """Page-scoped chunks so every hit maps cleanly back to a page."""
    s = get_settings()
    chunks: list[Chunk] = []
    for p in doc.pages:
        for i, piece in enumerate(_semantic_split(p.text_blocks, s.chunk_size, s.chunk_overlap)):
            chunks.append(Chunk(f"{doc.doc_id}-p{p.page}-t{i}", doc.doc_id, p.page, "text", piece))
        for i, md in enumerate(p.tables_md):
            chunks.append(Chunk(f"{doc.doc_id}-p{p.page}-tb{i}", doc.doc_id, p.page, "table", md))
        for i, fig in enumerate(p.figures):
            content = fig.caption or fig.nearby_text
            if content:
                chunks.append(
                    Chunk(f"{doc.doc_id}-p{p.page}-f{i}", doc.doc_id, p.page, "figure",
                          f"[Figure] {content}", image_path=fig.path)
                )
    return chunks

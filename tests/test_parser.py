"""Builds a tiny PDF with text, a table and an image, then checks the parser output."""
import io

import fitz
from PIL import Image

from app.config import get_settings
from app.ingest.pdf_parser import parse_pdf, to_chunks


def _make_pdf(path):
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Quarterly report. Revenue grew strongly in the north region.", fontsize=11)

    # simple 3x3 ruled table
    x0, y0, cw, rh = 72, 120, 120, 24
    cells = [["Region", "Q2", "Q3"], ["North", "10", "14"], ["South", "8", "9"]]
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            rect = fitz.Rect(x0 + c * cw, y0 + r * rh, x0 + (c + 1) * cw, y0 + (r + 1) * rh)
            page.draw_rect(rect, color=(0, 0, 0), width=0.8)
            page.insert_text((rect.x0 + 6, rect.y0 + 16), val, fontsize=10)

    img = Image.new("RGB", (300, 200), (30, 120, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    page.insert_image(fitz.Rect(72, 260, 372, 460), stream=buf.getvalue())
    page.insert_text((72, 480), "Figure 1: Revenue by region", fontsize=10)
    doc.save(path)


def test_parse(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "data_dir", tmp_path / "data")
    pdf = tmp_path / "t.pdf"
    _make_pdf(pdf)

    doc = parse_pdf(pdf)
    p = doc.pages[0]
    assert "Revenue grew" in " ".join(p.text_blocks)
    assert p.tables_md and "North" in p.tables_md[0]
    assert len(p.figures) == 1 and "Figure 1" in p.figures[0].nearby_text
    assert (tmp_path / "data" / "pages" / doc.doc_id / "p0001.png").exists()

    kinds = {c.kind for c in to_chunks(doc)}
    assert kinds == {"text", "table", "figure"}
    # table cell text should not leak into plain text chunks
    assert not any("South" in c.content for c in to_chunks(doc) if c.kind == "text")

from __future__ import annotations

import time
from pathlib import Path

from app.config import get_settings
from app.embed import text_embedder
from app.ingest.captioner import caption_figures
from app.ingest.pdf_parser import parse_pdf, to_chunks
from app.store import qdrant_store as store


def ingest_pdf(path: str | Path) -> dict:
    s = get_settings()
    t0 = time.perf_counter()
    doc = parse_pdf(path)
    store.ensure_collections(text_embedder.dim(), s.enable_visual)
    store.delete_doc(doc.doc_id)  # idempotent re-ingest

    n_caps = caption_figures(doc) if s.caption_images else 0

    # Path A: text / tables / figure captions -> dense vectors
    chunks = to_chunks(doc)
    if chunks:
        vecs = text_embedder.embed_passages([c.content for c in chunks])
        store.upsert_chunks(chunks, vecs, doc.source)

    # Path B: page images -> ColQwen2 multivectors
    n_pages_visual = 0
    if s.enable_visual:
        from app.embed import visual_embedder
        pages = [(p.page, p.render_path) for p in doc.pages]
        mvs = visual_embedder.embed_pages([p for _, p in pages])
        store.upsert_pages(doc.doc_id, doc.source, pages, mvs)
        n_pages_visual = len(pages)

    return {
        "doc_id": doc.doc_id,
        "source": doc.source,
        "pages": len(doc.pages),
        "chunks": len(chunks),
        "tables": sum(len(p.tables_md) for p in doc.pages),
        "figures": sum(len(p.figures) for p in doc.pages),
        "captions": n_caps,
        "visual_pages": n_pages_visual,
        "seconds": round(time.perf_counter() - t0, 1),
    }

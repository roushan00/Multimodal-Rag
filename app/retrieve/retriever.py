"""Three retrieval modes, all returning ranked *pages* plus supporting text chunks."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

from app.config import get_settings
from app.embed import text_embedder
from app.retrieve.fusion import dedupe_keep_order, rrf
from app.retrieve.reranker import rerank
from app.store import qdrant_store as store

logger = logging.getLogger(__name__)

Mode = Literal["text", "visual", "hybrid"]
PageKey = tuple[str, int]  # (doc_id, page)


@dataclass
class PageHit:
    doc_id: str
    page: int
    score: float
    source: str = ""
    render_path: str = ""
    chunks: list[dict] = field(default_factory=list)


@dataclass
class Retrieval:
    mode: Mode
    pages: list[PageHit]
    debug: dict = field(default_factory=dict)


def _text_hits(q: str, k: int, doc_ids):
    return store.search_text(text_embedder.embed_query(q), k, doc_ids)


def _visual_hits(q: str, k: int, doc_ids):
    from app.embed import visual_embedder
    return store.search_pages(visual_embedder.embed_query(q), k, doc_ids)


def retrieve(q: str, mode: Mode = "hybrid", top_pages: int = 4, k_text: int = 20, k_visual: int = 8,
             doc_ids: list[str] | None = None, rrf_k: int = 60, rerank: bool = True) -> Retrieval:
    s = get_settings()
    if mode != "text" and not s.enable_visual:
        mode = "text"

    text_hits = _text_hits(q, k_text, doc_ids) if mode in ("text", "hybrid") else []
    vis_hits = _visual_hits(q, k_visual, doc_ids) if mode in ("visual", "hybrid") else []

    # Rank pages per path. Path A: a page's rank = rank of its best chunk.
    text_rank: list[PageKey] = dedupe_keep_order((h.payload["doc_id"], h.payload["page"]) for h in text_hits)
    vis_rank: list[PageKey] = [(h.payload["doc_id"], h.payload["page"]) for h in vis_hits]

    if mode == "text":
        fused = [(p, 1.0 / (i + 1)) for i, p in enumerate(text_rank)]
    elif mode == "visual":
        fused = [(p, float(h.score)) for p, h in zip(vis_rank, vis_hits)]
    else:
        fused = rrf([text_rank, vis_rank], k=rrf_k)

    meta = {(h.payload["doc_id"], h.payload["page"]): h.payload for h in vis_hits}
    by_page: dict[PageKey, list[dict]] = {}
    for h in text_hits:
        key = (h.payload["doc_id"], h.payload["page"])
        by_page.setdefault(key, []).append({**h.payload, "score": float(h.score)})

    pages: list[PageHit] = []
    for (doc_id, pno), score in fused[:top_pages]:
        m = meta.get((doc_id, pno), {})
        chunks = by_page.get((doc_id, pno), [])
        render = m.get("render_path") or str(s.pages_dir / doc_id / f"p{pno:04d}.png")
        source = m.get("source") or (chunks[0]["source"] if chunks else "")

        # Rerank chunks within each page for better ordering in the prompt
        if rerank and chunks and s.rerank_model:
            chunks = _rerank_chunks(q, chunks)

        pages.append(PageHit(doc_id, pno, float(score), source, render, chunks))

    return Retrieval(mode, pages, debug={"text_rank": text_rank[:10], "visual_rank": vis_rank[:10]})


def _rerank_chunks(query: str, chunks: list[dict]) -> list[dict]:
    from app.retrieve.reranker import rerank as do_rerank
    return do_rerank(query, chunks)

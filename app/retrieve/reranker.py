"""Cross-encoder reranker for text chunks.

Uses sentence-transformers CrossEncoder (runs on CPU, ~100ms for 20 pairs).
Falls back gracefully if the model can't be loaded.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _model():
    try:
        from sentence_transformers import CrossEncoder
        m = CrossEncoder(get_settings().rerank_model)
        logger.info("Loaded reranker: %s", get_settings().rerank_model)
        return m
    except Exception:
        logger.warning("Could not load reranker model, skipping reranking")
        return None


def rerank(query: str, chunks: list[dict], top_k: int | None = None) -> list[dict]:
    """Re-score chunks using a cross-encoder. Returns chunks sorted by rerank score."""
    model = _model()
    if model is None or not chunks:
        return chunks

    pairs = [(query, c["content"]) for c in chunks]
    scores = model.predict(pairs)

    for c, s in zip(chunks, scores):
        c["rerank_score"] = float(s)

    ranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    if top_k:
        ranked = ranked[:top_k]
    return ranked

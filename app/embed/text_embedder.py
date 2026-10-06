from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import get_settings

# BGE v1.5 recommends this instruction for queries (not for passages)
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _model() -> SentenceTransformer:
    return SentenceTransformer(get_settings().text_embed_model)


def dim() -> int:
    return _model().get_embedding_dimension()


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    return _model().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                           show_progress_bar=len(texts) > 64)


def embed_query(text: str) -> np.ndarray:
    prefix = BGE_QUERY_PREFIX if "bge" in get_settings().text_embed_model.lower() else ""
    return _model().encode(prefix + text, normalize_embeddings=True)

"""ColQwen2 late-interaction embeddings: one 128-d vector per image patch / query token.

Score(query, page) = sum over query tokens of max cosine-sim to any page patch (MaxSim).
Qdrant computes this natively with multivector + MAX_SIM.
"""
from __future__ import annotations

from functools import lru_cache

from PIL import Image

from app.config import get_settings


@lru_cache
def _load():
    import torch
    from colpali_engine.models import ColQwen2, ColQwen2Processor

    name = get_settings().visual_embed_model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    model = ColQwen2.from_pretrained(name, torch_dtype=dtype, device_map=device).eval()
    processor = ColQwen2Processor.from_pretrained(name)
    return model, processor, torch


DIM = 128


def embed_pages(image_paths: list[str], batch_size: int = 4) -> list[list[list[float]]]:
    model, processor, torch = _load()
    out: list[list[list[float]]] = []
    for i in range(0, len(image_paths), batch_size):
        imgs = [Image.open(p).convert("RGB") for p in image_paths[i:i + batch_size]]
        batch = processor.process_images(imgs).to(model.device)
        with torch.no_grad():
            embs = model(**batch)                       # (B, n_patches, 128), padded
        mask = batch["attention_mask"].bool()
        for e, m in zip(embs, mask):
            out.append(e[m].float().cpu().tolist())     # drop padding vectors
    return out


def embed_query(text: str) -> list[list[float]]:
    model, processor, torch = _load()
    batch = processor.process_queries([text]).to(model.device)
    with torch.no_grad():
        emb = model(**batch)[0]
    mask = batch["attention_mask"][0].bool()
    return emb[mask].float().cpu().tolist()

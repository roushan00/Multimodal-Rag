from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from functools import lru_cache

from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.models import Chunk

logger = logging.getLogger(__name__)


def _pid(key: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


@lru_cache
def client() -> QdrantClient:
    return QdrantClient(url=get_settings().qdrant_url)


def ensure_collections(text_dim: int, with_visual: bool) -> None:
    s, c = get_settings(), client()
    if not c.collection_exists(s.text_collection):
        c.create_collection(
            s.text_collection,
            vectors_config=models.VectorParams(size=text_dim, distance=models.Distance.COSINE),
        )
        c.create_payload_index(s.text_collection, "doc_id", models.PayloadSchemaType.KEYWORD)
    if with_visual and not c.collection_exists(s.page_collection):
        c.create_collection(
            s.page_collection,
            vectors_config=models.VectorParams(
                size=128,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(comparator=models.MultiVectorComparator.MAX_SIM),
            ),
        )
        c.create_payload_index(s.page_collection, "doc_id", models.PayloadSchemaType.KEYWORD)


def upsert_chunks(chunks: list[Chunk], vectors, source: str) -> None:
    pts = [
        models.PointStruct(
            id=_pid(ch.chunk_id),
            vector=list(map(float, v)),
            payload={"chunk_id": ch.chunk_id, "doc_id": ch.doc_id, "page": ch.page, "kind": ch.kind,
                     "content": ch.content, "image_path": ch.image_path, "source": source},
        )
        for ch, v in zip(chunks, vectors)
    ]
    client().upsert(get_settings().text_collection, points=pts, wait=True)


def upsert_pages(doc_id: str, source: str, pages: list[tuple[int, str]], multivecs) -> None:
    pts = [
        models.PointStruct(
            id=_pid(f"{doc_id}-page-{pno}"),
            vector=mv,
            payload={"doc_id": doc_id, "page": pno, "render_path": path, "source": source},
        )
        for (pno, path), mv in zip(pages, multivecs)
    ]
    for i in range(0, len(pts), 8):
        client().upsert(get_settings().page_collection, points=pts[i:i + 8], wait=True)


def _flt(doc_ids: list[str] | None):
    if not doc_ids:
        return None
    return models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchAny(any=doc_ids))])


def search_text(qvec, k: int, doc_ids: list[str] | None = None):
    return client().query_points(get_settings().text_collection, query=list(map(float, qvec)),
                                 limit=k, query_filter=_flt(doc_ids), with_payload=True).points


def search_pages(qmulti, k: int, doc_ids: list[str] | None = None):
    return client().query_points(get_settings().page_collection, query=qmulti,
                                 limit=k, query_filter=_flt(doc_ids), with_payload=True).points


def delete_doc(doc_id: str) -> None:
    s, f = get_settings(), _flt([doc_id])
    for col in (s.text_collection, s.page_collection):
        if client().collection_exists(col):
            client().delete(col, points_selector=models.FilterSelector(filter=f))
    logger.info("Deleted doc %s from Qdrant", doc_id)


def list_docs() -> list[dict]:
    """List all unique documents across collections with page/chunk counts."""
    s, c = get_settings(), client()
    docs: dict[str, dict] = {}

    # Scan text_chunks for doc metadata
    if c.collection_exists(s.text_collection):
        offset = None
        while True:
            result = c.scroll(s.text_collection, limit=100, offset=offset,
                              with_payload=["doc_id", "source", "page", "kind"])
            points, offset = result
            for pt in points:
                did = pt.payload["doc_id"]
                if did not in docs:
                    docs[did] = {"doc_id": did, "source": pt.payload.get("source", ""),
                                 "pages": set(), "chunks": 0, "tables": 0, "figures": 0}
                docs[did]["pages"].add(pt.payload["page"])
                docs[did]["chunks"] += 1
                kind = pt.payload.get("kind", "text")
                if kind == "table":
                    docs[did]["tables"] += 1
                elif kind == "figure":
                    docs[did]["figures"] += 1
            if offset is None:
                break

    # Scan pages collection for visual page count
    if c.collection_exists(s.page_collection):
        page_counts: dict[str, int] = defaultdict(int)
        offset = None
        while True:
            result = c.scroll(s.page_collection, limit=100, offset=offset,
                              with_payload=["doc_id", "source"])
            points, offset = result
            for pt in points:
                did = pt.payload["doc_id"]
                page_counts[did] += 1
                if did not in docs:
                    docs[did] = {"doc_id": did, "source": pt.payload.get("source", ""),
                                 "pages": set(), "chunks": 0, "tables": 0, "figures": 0}
            if offset is None:
                break
        for did, cnt in page_counts.items():
            docs[did]["visual_pages"] = cnt

    # Convert page sets to counts
    out = []
    for d in docs.values():
        d["pages"] = len(d["pages"])
        d.setdefault("visual_pages", 0)
        out.append(d)
    return out

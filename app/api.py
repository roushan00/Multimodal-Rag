from __future__ import annotations

import logging
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from app.config import get_settings
from app.generate.answerer import answer, answer_stream, build_content, SYSTEM
from app.ingest.pipeline import ingest_pdf
from app.retrieve.retriever import retrieve
from app.store import qdrant_store as store

logger = logging.getLogger(__name__)

app = FastAPI(title="Multimodal RAG")

# ── In-memory job tracker for async ingestion ────────────────────────────
_jobs: dict[str, dict] = {}


def _run_ingest(job_id: str, path: Path):
    _jobs[job_id]["status"] = "running"
    try:
        result = ingest_pdf(path)
        _jobs[job_id].update(status="done", result=result)
        logger.info("Ingest job %s done: %s", job_id, result["doc_id"])
    except Exception as e:
        _jobs[job_id].update(status="failed", error=str(e))
        logger.exception("Ingest job %s failed", job_id)


# ── Request models ───────────────────────────────────────────────────────

class QueryIn(BaseModel):
    question: str
    mode: Literal["text", "visual", "hybrid"] = "hybrid"
    context: Literal["text", "images", "both"] = "both"
    top_pages: int = 4
    doc_ids: list[str] | None = None
    retrieve_only: bool = False
    stream: bool = False
    rerank: bool = True


# ── Endpoints ────────────────────────────────────────────────────────────

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    """Upload a PDF. Ingestion runs in background; returns a job_id to poll."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Upload a PDF")
    dest_dir = get_settings().data_dir / "pdfs"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / Path(file.filename).name
    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "queued", "file": file.filename, "started": time.time()}
    thread = threading.Thread(target=_run_ingest, args=(job_id, dest), daemon=True)
    thread.start()
    return {"job_id": job_id, "status": "queued", "file": file.filename}


@app.get("/ingest/{job_id}")
def ingest_status(job_id: str):
    """Poll ingestion job status."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Job not found")
    return job


@app.post("/query")
def query(q: QueryIn):
    """Retrieve pages and generate an answer. Use stream=true for SSE."""
    r = retrieve(q.question, mode=q.mode, top_pages=q.top_pages, doc_ids=q.doc_ids,
                 rerank=q.rerank)
    pages = [
        {"doc_id": p.doc_id, "page": p.page, "score": p.score, "source": p.source,
         "chunks": [{"kind": c["kind"], "content": c["content"][:300], "score": c["score"]} for c in p.chunks]}
        for p in r.pages
    ]
    out = {"mode": r.mode, "pages": pages, "debug": r.debug}

    if q.retrieve_only:
        return out

    if q.stream:
        content = build_content(q.question, r, q.context)

        async def event_gen():
            # Send retrieval metadata first
            yield {"event": "retrieval", "data": out}
            # Stream answer tokens
            for chunk in answer_stream(content):
                yield {"event": "token", "data": chunk}
            yield {"event": "done", "data": ""}

        return EventSourceResponse(event_gen())

    out["answer"] = answer(q.question, r, q.context)
    return out


@app.get("/documents")
def list_docs():
    """List all ingested documents with page/chunk counts."""
    return store.list_docs()


@app.delete("/documents/{doc_id}")
def delete_doc(doc_id: str):
    """Delete a document and all its vectors from Qdrant."""
    store.delete_doc(doc_id)
    # Clean up rendered pages and figures from disk
    s = get_settings()
    for d in (s.pages_dir / doc_id, s.figures_dir / doc_id):
        if d.exists():
            shutil.rmtree(d)
    return {"deleted": doc_id}


@app.get("/health")
def health():
    return {"ok": True}

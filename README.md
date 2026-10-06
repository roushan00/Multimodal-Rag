# Multimodal RAG over PDFs (learning build)

Ask questions about PDFs that contain text, **tables, charts and figures**, and get answers grounded in the actual pages.

This repo implements **two retrieval strategies side by side** so you can learn how each one behaves, plus a fusion of the two and an eval harness to measure them.

## Architecture

```
                 ┌──────────────── INGEST ────────────────┐
 PDF ──► PyMuPDF ├─ text blocks ──► chunker ─┐            │
                 ├─ tables ──► markdown ─────┼─► BGE (local) ──► Qdrant: text_chunks (dense)
                 ├─ images ──► VLM caption ──┘            │
                 └─ page render (PNG) ──► ColQwen2 (local) ──► Qdrant: pages (multivector, MaxSim)
                                                          │
                 ┌──────────────── QUERY ─────────────────┐
 question ──┬─► BGE ──► text_chunks top-k ──┐             │
            └─► ColQwen2 ──► pages top-k ───┴─► RRF fusion (page level)
                                                  │
                         page PNGs + text chunks ─┴─► Claude (hosted VLM) ──► answer + [doc p.N] citations
```

**Path A – "parse then embed" (text-first).** Extract text, convert tables to markdown, caption images with a VLM, then embed everything as text. Cheap at query time, debuggable, but loses layout and depends on parser/caption quality.

**Path B – "embed the page" (vision-first, ColPali family).** Render every page as an image and embed it with a late-interaction vision model (ColQwen2: ~1 vector per image patch, 128-d). Retrieval uses MaxSim. Skips OCR/parsing entirely and handles charts and complex layouts well; costs more storage and GPU.

**Fusion.** Both paths vote for *pages*; Reciprocal Rank Fusion merges the rankings. The generator receives the page images *and* the matching text chunks.

**Hybrid deployment.** Embeddings run locally (BGE + ColQwen2); captioning and answering use a hosted VLM (Anthropic API). Swap `app/llm.py` to point at a local VLM later if you need fully on-prem.

## Tech choices

| Concern | Choice | Why |
|---|---|---|
| PDF parsing | PyMuPDF (`find_tables`, `get_images`) | Fast, no extra services. Upgrade: Docling/Unstructured |
| Text embeddings | `BAAI/bge-small-en-v1.5` (384-d) | Runs on CPU; swap for bge-m3 for multilingual |
| Visual embeddings | `vidore/colqwen2-v1.0` via `colpali-engine` | Strong on ViDoRe benchmark; needs GPU (~8 GB) |
| Vector DB | Qdrant | Native multivector + MaxSim, so ColPali needs no custom code |
| Generator / captioner | Claude via Anthropic API | Accepts multiple images per request |
| API | FastAPI | `/ingest`, `/query` |

## Quickstart

```bash
docker compose up -d                  # Qdrant on :6333
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt       # add requirements-visual.txt on a GPU box
cp .env.example .env                  # set ANTHROPIC_API_KEY, model names

python -m scripts.ingest data/pdfs/*.pdf
python -m scripts.ask "What was Q3 revenue by region?" --mode hybrid
uvicorn app.api:app --reload          # http://localhost:8000/docs
python -m scripts.eval eval/questions.jsonl
```

No GPU? Set `ENABLE_VISUAL=false`; Path A still works end-to-end.

## Learning milestones

1. **Parse & inspect** – run `scripts/inspect_pdf.py` on 3 messy PDFs. Look at what PyMuPDF misses (merged table cells, text inside charts, scanned pages).
2. **Path A baseline** – ingest, ask 10 questions, note failures. Toggle `CAPTION_IMAGES` and measure what captions add.
3. **Path B** – enable ColQwen2. Compare top-5 pages for the same questions. Visualise token-level MaxSim (which patches matched which query tokens) — the best way to understand late interaction.
4. **Eval** – write 30+ questions in `eval/questions.jsonl` with the gold page. Compare Recall@k / MRR for `text`, `visual`, `hybrid`.
5. **Generation quality** – compare answers using (a) text chunks only, (b) page images only, (c) both. Check citation accuracy.
6. **Stretch** – binary quantisation / token pooling to shrink ColQwen storage; re-ranking with a VLM; table-QA with a SQL path (reuse your Text-to-SQL ideas).

## Experiments worth logging

- Chunk size 256 vs 512 vs 1024 tokens for Path A.
- Image size limits for ColQwen (`max_num_visual_tokens`) vs recall and latency.
- RRF `k` (60 default) and per-path weights.
- Storage per page: dense (384 floats) vs multivector (~700×128 floats).

## Layout

```
app/
  config.py              settings from .env
  ingest/pdf_parser.py   text, tables, images, page renders
  ingest/captioner.py    VLM captions for figures
  ingest/pipeline.py     orchestrates parse → embed → upsert
  embed/text_embedder.py BGE
  embed/visual_embedder.py ColQwen2
  store/qdrant_store.py  two collections, search helpers
  retrieve/retriever.py  text / visual / hybrid (RRF)
  generate/answerer.py   builds the multimodal prompt, calls the LLM
  llm.py                 Anthropic client wrapper
  api.py                 FastAPI
scripts/  ingest.py, ask.py, eval.py, inspect_pdf.py
eval/     questions.example.jsonl
tests/    test_parser.py, test_fusion.py
```

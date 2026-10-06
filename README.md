# Multimodal RAG over PDFs

Ask questions about PDFs that contain text, **tables, charts and figures**, and get answers grounded in the actual pages.

Two retrieval strategies side by side (text + visual), RRF fusion, cross-encoder reranking, and a comprehensive eval harness to measure everything.

## Architecture

```
                 ┌──────────────── INGEST ────────────────┐
 PDF ──► PyMuPDF ├─ text blocks ──► semantic chunker ─┐   │
                 ├─ tables ──► markdown ──────────────┼─► BGE (local) ──► Qdrant: text_chunks (384-d dense)
                 ├─ images ──► Gemini caption (rich) ─┘   │
                 └─ page render (PNG) ──► ColQwen2 (local) ──► Qdrant: pages (128-d multivector, MaxSim)

                 ┌──────────────── QUERY ─────────────────┐
 question ──┬─► BGE ──► text_chunks top-20 ──┐            │
            └─► ColQwen2 ──► pages top-8 ────┴─► RRF fusion (page level)
                                                   │
                                    cross-encoder rerank (per-page chunks)
                                                   │
                              page PNGs + text ────┴─► Gemini 2.5 Flash ──► answer + [doc p.N] citations
```

**Path A – "parse then embed" (text-first).** Extract text, convert tables to markdown, caption images with a VLM, then embed everything as text. Cheap at query time, debuggable, but loses layout and depends on parser/caption quality.

**Path B – "embed the page" (vision-first, ColPali family).** Render every page as an image and embed it with a late-interaction vision model (ColQwen2: ~1 vector per image patch, 128-d). Retrieval uses MaxSim. Skips OCR/parsing entirely and handles charts and complex layouts well; costs more storage and GPU.

**Fusion.** Both paths vote for *pages*; Reciprocal Rank Fusion merges the rankings. Cross-encoder reranking reorders chunks within each page. The generator receives the page images *and* the matching text chunks.

## Tech choices

| Concern | Choice | Why |
|---|---|---|
| PDF parsing | PyMuPDF (`find_tables`, `get_images`) | Fast, no extra services |
| Text embeddings | `BAAI/bge-small-en-v1.5` (384-d) | Runs on CPU; swap for bge-m3 for multilingual |
| Visual embeddings | `vidore/colqwen2-v1.0` via `colpali-engine` | Strong on ViDoRe benchmark; needs GPU (~8 GB) |
| Vector DB | Qdrant | Native multivector + MaxSim |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Fast CPU cross-encoder, reranks chunks per page |
| Generator / captioner | Gemini 2.5 Flash via Vertex AI | Accepts multiple images per request, fast |
| Chunking | Semantic (heading-aware) | Splits on section/paragraph boundaries, falls back to sentence-level |
| API | FastAPI | Async ingest, SSE streaming, document management |

## Quickstart

```bash
docker compose up -d                  # Qdrant on :6333
uv venv && source .venv/bin/activate  # or: python -m venv .venv
uv pip install -r requirements.txt    # add requirements-visual.txt on a GPU box
cp .env.example .env                  # set GCP_PROJECT, GOOGLE_APPLICATION_CREDENTIALS

python -m scripts.ingest data/pdfs/*.pdf
python -m scripts.ask "What was Q3 revenue by region?" --mode hybrid
uvicorn app.api:app --reload          # http://localhost:8000/docs
python -m scripts.eval                # full eval with 27 questions
```

No GPU? Set `ENABLE_VISUAL=false`; Path A still works end-to-end.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/ingest` | Upload PDF, runs async background ingestion. Returns `job_id` |
| `GET` | `/ingest/{job_id}` | Poll ingestion job status |
| `POST` | `/query` | Retrieve + answer. Supports `stream=true` (SSE), `rerank=true/false` |
| `GET` | `/documents` | List all ingested documents with page/chunk counts |
| `DELETE` | `/documents/{doc_id}` | Remove document from Qdrant and disk |
| `GET` | `/health` | Health check |

### Query example

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the Transformer architecture?", "mode": "hybrid", "top_pages": 3}'
```

### Streaming example

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the oil price forecasts?", "stream": true}'
```

## Evaluation

Two evaluation frameworks included:

### Custom eval (`scripts/eval.py`)
- **Retrieval**: Recall@k, MRR, Hit Rate, Context Precision
- **Answer quality**: Faithfulness (LLM-as-judge, image-aware), Citation Accuracy, Correctness
- **Anti-hallucination**: Hallucination rate tracking

```bash
python -m scripts.eval                          # full eval
python -m scripts.eval --retrieval-only         # skip LLM-based answer eval
python -m scripts.eval --modes text hybrid      # specific modes
```

### DeepEval integration (`scripts/eval_deepeval.py`)
- Faithfulness, Answer Relevancy, Contextual Precision/Recall, Hallucination
- Two-step process to avoid OOM with ColQwen2

```bash
python -m scripts.eval_deepeval --build --mode text    # step 1: build test cases
python -m scripts.eval_deepeval --evaluate             # step 2: run deepeval metrics
```

### Eval results (27 questions, hybrid mode)

| Metric | Score |
|--------|-------|
| Faithfulness | 0.947 |
| Hallucination Rate | 7.4% |
| Correctness | 0.841 |
| Retrieval R@1 / MRR | 0.815 / 0.889 |

DeepEval scores: Faithfulness 0.96, Answer Relevancy 0.88, Contextual Recall 0.98

## Key improvements implemented

1. **Semantic chunking** — heading-aware splitting instead of fixed character windows
2. **Cross-encoder reranking** — chunks reranked within each page before LLM generation
3. **Rich figure captioning** — 250-word detailed captions with data points, axes, trends
4. **Anti-hallucination prompt** — strict 7-rule system prompt, context-only answers
5. **Async ingestion** — background thread with job_id polling
6. **SSE streaming** — retrieval metadata first, then answer tokens
7. **Document management** — list, delete endpoints
8. **Image-aware eval judge** — faithfulness judge receives page images, not just text

## Layout

```
app/
  config.py                settings from .env (Vertex AI, Qdrant, models)
  llm.py                   Vertex AI Gemini wrapper with streaming
  models.py                ParsedDoc, PageContent, Chunk dataclasses
  api.py                   FastAPI (async ingest, streaming, doc mgmt)
  ingest/
    pdf_parser.py           PyMuPDF parsing + semantic chunking
    captioner.py            rich figure captioning via Gemini
    pipeline.py             orchestrates parse → caption → embed → upsert
  embed/
    text_embedder.py        BGE small en v1.5
    visual_embedder.py      ColQwen2 multivector
  store/
    qdrant_store.py         two collections, search, list_docs, delete
  retrieve/
    retriever.py            text / visual / hybrid (RRF) + reranking
    fusion.py               Reciprocal Rank Fusion
    reranker.py             cross-encoder reranker
  generate/
    answerer.py             multimodal prompt builder + streaming
scripts/
  ingest.py                 CLI batch PDF ingestion
  ask.py                    CLI single question
  eval.py                   custom eval framework
  eval_deepeval.py          DeepEval integration
  test_queries.py           retrieval smoke test
  inspect_pdf.py            PDF parsing inspector
eval/
  questions.jsonl           27 ground-truth questions
  questions.example.jsonl   example format
tests/
  test_parser.py            parser unit tests
  test_fusion.py            RRF fusion unit tests
```

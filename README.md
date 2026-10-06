<p align="center">
  <h1 align="center">📄 Multimodal RAG over PDFs</h1>
  <p align="center">
    <em>Ask questions about PDFs with text, tables, charts & figures — get grounded answers with page citations.</em>
  </p>
  <p align="center">
    <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python&logoColor=white" alt="Python">
    <img src="https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white" alt="FastAPI">
    <img src="https://img.shields.io/badge/Gemini_2.5-Flash-4285F4?logo=google&logoColor=white" alt="Gemini">
    <img src="https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?logo=qdrant&logoColor=white" alt="Qdrant">
    <img src="https://img.shields.io/badge/Faithfulness-0.96-brightgreen" alt="Faithfulness">
    <img src="https://img.shields.io/badge/Hallucination-7.4%25-yellow" alt="Hallucination Rate">
  </p>
</p>

---

## 🎯 What is this?

A **dual-path retrieval** system that combines text-based and vision-based search over PDF documents, fuses the results, reranks them, and generates grounded answers with page-level citations.

> **Two retrieval strategies, one fused answer** — learn how each behaves, then measure them with a built-in eval harness.

---

## 🏗️ Architecture

### High-Level Flow

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Upload    │────►│   Parse &    │────►│   Embed &    │────►│    Store     │
│    PDF      │     │   Extract    │     │   Caption    │     │  in Qdrant   │
└─────────────┘     └──────────────┘     └──────────────┘     └──────────────┘
                                                                     │
┌─────────────┐     ┌──────────────┐     ┌──────────────┐           │
│   Answer    │◄────│   Rerank &   │◄────│   Retrieve   │◄──────────┘
│  + Cite     │     │   Generate   │     │   & Fuse     │
└─────────────┘     └──────────────┘     └──────────────┘
```

### Detailed Ingestion Pipeline

```
                          ┌─────────────────────────────────────────┐
                          │            INGESTION                    │
                          └─────────────────────────────────────────┘

                    ┌──────────────────────────────────────────────────┐
                    │                   PyMuPDF                        │
  ┌──────┐         │                                                  │
  │      │         │  ┌─────────────┐   ┌──────────────┐              │
  │ PDF  │────────►│  │ Text Blocks │──►│  Semantic    │──┐           │
  │      │         │  └─────────────┘   │  Chunker    │  │           │
  └──────┘         │                    └──────────────┘  │           │
                   │  ┌─────────────┐                     │  ┌─────┐  │   ┌─────────────────────────┐
                   │  │   Tables    │──► Markdown ────────┼─►│ BGE │──┼──►│ Qdrant: text_chunks     │
                   │  └─────────────┘                     │  │384-d│  │   │ (dense cosine vectors)  │
                   │                                      │  └─────┘  │   └─────────────────────────┘
                   │  ┌─────────────┐   ┌──────────────┐  │           │
                   │  │   Figures   │──►│ Gemini 2.5   │──┘           │
                   │  └─────────────┘   │ Rich Caption │              │
                   │                    │ (250 words)  │              │
                   │                    └──────────────┘              │
                   │                                                  │
                   │  ┌─────────────┐   ┌──────────────┐              │   ┌─────────────────────────┐
                   │  │ Page Render │──►│  ColQwen2    │──────────────┼──►│ Qdrant: pages           │
                   │  │   (PNG)     │   │ 128-d multi  │              │   │ (multivector, MaxSim)   │
                   │  └─────────────┘   └──────────────┘              │   └─────────────────────────┘
                   └──────────────────────────────────────────────────┘
```

### Detailed Query Pipeline

```
                          ┌─────────────────────────────────────────┐
                          │              QUERY                      │
                          └─────────────────────────────────────────┘

                   ┌──────────────┐
  ┌──────────┐     │     BGE      │     ┌──────────────────┐
  │          │────►│  embed query │────►│ Search top-20    │──┐
  │ Question │     └──────────────┘     │ text_chunks      │  │
  │          │                          └──────────────────┘  │   ┌────────────┐
  │          │     ┌──────────────┐                            ├──►│    RRF     │
  │          │────►│  ColQwen2    │     ┌──────────────────┐  │   │  Fusion    │
  │          │     │  embed query │────►│ Search top-8     │──┘   │ (page lvl) │
  └──────────┘     └──────────────┘     │ pages (MaxSim)   │      └─────┬──────┘
                                        └──────────────────┘            │
                                                                        ▼
                                                                 ┌────────────┐
                   ┌──────────────┐     ┌──────────────────┐     │  Cross-    │
                   │   Gemini     │◄────│  Top-5 pages     │◄────│  Encoder   │
                   │  2.5 Flash   │     │  images + chunks │     │  Rerank    │
                   └──────┬───────┘     └──────────────────┘     └────────────┘
                          │
                          ▼
                   ┌──────────────┐
                   │   Answer     │
                   │ with [doc    │
                   │  p.N] cites  │
                   └──────────────┘
```

---

## 🔀 Two Retrieval Paths

<table>
<tr>
<td width="50%">

### 🔤 Path A — Text-First
1. Extract text blocks via PyMuPDF
2. Convert tables to markdown
3. Caption figures with Gemini (250-word rich descriptions)
4. Semantic chunking (heading-aware)
5. Embed with BGE → dense vectors

**Pros:** Cheap, debuggable, fast queries
**Cons:** Loses layout, depends on parser quality

</td>
<td width="50%">

### 🖼️ Path B — Vision-First (ColPali)
1. Render each page as PNG
2. Embed with ColQwen2 (~700 patch vectors per page)
3. Store as multivectors in Qdrant
4. Retrieve via MaxSim (late interaction)

**Pros:** Handles charts, complex layouts, no parsing needed
**Cons:** More storage, needs GPU (~8 GB)

</td>
</tr>
</table>

**🔗 Fusion:** Both paths vote for *pages* → Reciprocal Rank Fusion merges rankings → Cross-encoder reranks chunks → Gemini generates answer from page images + text.

---

## ⚡ Tech Stack

| Component | Technology | Details |
|:----------|:-----------|:--------|
| 📄 PDF Parsing | **PyMuPDF** | `find_tables`, `get_images`, text extraction |
| 🔤 Text Embeddings | **BGE small en v1.5** | 384-d dense, runs on CPU |
| 🖼️ Visual Embeddings | **ColQwen2 v1.0** | 128-d multivector, MaxSim, needs GPU |
| 🗄️ Vector Database | **Qdrant** | Native multivector + MaxSim support |
| 🔄 Reranker | **ms-marco-MiniLM** | Cross-encoder, reranks per page |
| 🤖 Generator | **Gemini 2.5 Flash** | Vertex AI, multimodal, streaming |
| ✂️ Chunking | **Semantic** | Heading-aware, paragraph boundaries |
| 🌐 API | **FastAPI** | Async ingest, SSE streaming |

---

## 🚀 Quickstart

```bash
# 1. Start vector database
docker compose up -d                  # Qdrant on :6333

# 2. Setup Python environment
uv venv && source .venv/bin/activate  # or: python -m venv .venv
uv pip install -r requirements.txt    # add requirements-visual.txt on GPU box

# 3. Configure
cp .env.example .env                  # set GCP_PROJECT, GOOGLE_APPLICATION_CREDENTIALS

# 4. Ingest PDFs
python -m scripts.ingest data/pdfs/*.pdf

# 5. Ask questions
python -m scripts.ask "What was Q3 revenue by region?" --mode hybrid

# 6. Start API server
uvicorn app.api:app --reload          # http://localhost:8000/docs

# 7. Run evaluation
python -m scripts.eval                # full eval with 27 questions
```

> 💡 **No GPU?** Set `ENABLE_VISUAL=false` in `.env` — Path A (text-only) works end-to-end on CPU.

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|:------:|:---------|:------------|
| `POST` | `/ingest` | Upload PDF → async background ingestion → returns `job_id` |
| `GET` | `/ingest/{job_id}` | Poll ingestion job status (`queued` → `running` → `done`) |
| `POST` | `/query` | Retrieve + answer. Options: `stream`, `rerank`, `mode`, `top_pages` |
| `GET` | `/documents` | List all ingested docs with page/chunk/table/figure counts |
| `DELETE` | `/documents/{doc_id}` | Remove document from Qdrant and disk |
| `GET` | `/health` | Health check |

<details>
<summary><b>📝 Query example</b></summary>

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the Transformer architecture?",
    "mode": "hybrid",
    "top_pages": 3,
    "rerank": true
  }'
```
</details>

<details>
<summary><b>📡 Streaming example (SSE)</b></summary>

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the oil price forecasts for 2025?",
    "stream": true
  }'

# Response:
# event: retrieval
# data: {"mode": "hybrid", "pages": [...], "debug": {...}}
#
# event: token
# data: The baseline Brent oil price
#
# event: token
# data: is projected to average $73/bbl
#
# event: done
# data:
```
</details>

---

## 📊 Evaluation Results

Evaluated on **27 ground-truth questions** across 3 PDF types (research paper, financial report, AI index).

### Retrieval Metrics

| Mode | R@1 | R@3 | R@5 | MRR | Hit Rate |
|:-----|:---:|:---:|:---:|:---:|:--------:|
| 🔤 Text | 0.778 | 0.963 | **1.000** | 0.880 | **1.000** |
| 🖼️ Visual | 0.667 | 0.963 | 0.963 | 0.802 | 0.963 |
| 🔗 **Hybrid** | **0.815** | 0.963 | 0.963 | **0.889** | 0.963 |

### Answer Quality

| Metric | Score | |
|:-------|:-----:|:-|
| Faithfulness | **0.947** | ✅ Claims grounded in context |
| Hallucination Rate | **7.4%** | ✅ Down from 48.1% baseline |
| Correctness | **0.841** | ✅ Matches gold answers |
| Citation Precision | **0.741** | ✅ Cites correct pages |

### DeepEval Scores

| Metric | Score | Pass Rate |
|:-------|:-----:|:---------:|
| Faithfulness | 0.96 | 96.3% |
| Answer Relevancy | 0.88 | 81.5% |
| Contextual Precision | 0.73 | 63.0% |
| Contextual Recall | 0.98 | 96.3% |
| Hallucination | 0.96 | 96.3% |

### What Reduced Hallucinations (48% → 7.4%)

```
Baseline (48.1%)
    │
    ├── + Strict anti-hallucination prompt (7 rules)
    ├── + Rich figure captions (120→250 words)
    ├── + k_text 12→20 (more context chunks)
    ├── + top_pages 4→5
    └── + Image-aware eval judge
    │
    ▼
Final (7.4%)
```

---

## 🔧 Key Features

<table>
<tr>
<td>

**🧩 Semantic Chunking**
Heading-aware splitting instead of fixed character windows. Respects paragraph and section boundaries.

**🔄 Cross-Encoder Reranking**
Chunks reranked within each page using `ms-marco-MiniLM` before LLM generation.

**🖼️ Rich Figure Captioning**
250-word detailed captions with data points, axis labels, trends, and key takeaways.

**🛡️ Anti-Hallucination**
Strict 7-rule system prompt. Model answers ONLY from provided context.

</td>
<td>

**⚡ Async Ingestion**
Background thread with job_id polling. Server stays responsive during ingestion.

**📡 SSE Streaming**
Retrieval metadata sent first, then answer tokens stream incrementally.

**📁 Document Management**
List all docs, delete by ID. Cleans up Qdrant vectors and disk files.

**📊 Dual Eval Framework**
Custom eval (image-aware faithfulness) + DeepEval integration (5 metrics).

</td>
</tr>
</table>

---

## 📁 Project Structure

```
multimodal-rag/
│
├── app/                          # Core application
│   ├── config.py                 # Settings from .env (Vertex AI, Qdrant, models)
│   ├── llm.py                    # Vertex AI Gemini wrapper + streaming
│   ├── models.py                 # ParsedDoc, PageContent, Chunk dataclasses
│   ├── api.py                    # FastAPI (async ingest, streaming, doc mgmt)
│   │
│   ├── ingest/                   # PDF processing
│   │   ├── pdf_parser.py         # PyMuPDF parsing + semantic chunking
│   │   ├── captioner.py          # Rich figure captioning via Gemini
│   │   └── pipeline.py           # Orchestrates parse → caption → embed → upsert
│   │
│   ├── embed/                    # Embedding models
│   │   ├── text_embedder.py      # BGE small en v1.5 (384-d)
│   │   └── visual_embedder.py    # ColQwen2 multivector (128-d)
│   │
│   ├── store/                    # Vector database
│   │   └── qdrant_store.py       # Two collections, search, list, delete
│   │
│   ├── retrieve/                 # Retrieval & fusion
│   │   ├── retriever.py          # Text / visual / hybrid (RRF) + reranking
│   │   ├── fusion.py             # Reciprocal Rank Fusion
│   │   └── reranker.py           # Cross-encoder reranker
│   │
│   └── generate/                 # Answer generation
│       └── answerer.py           # Multimodal prompt builder + streaming
│
├── scripts/                      # CLI tools
│   ├── ingest.py                 # Batch PDF ingestion
│   ├── ask.py                    # Single question CLI
│   ├── eval.py                   # Custom eval framework
│   ├── eval_deepeval.py          # DeepEval integration (2-step)
│   ├── test_queries.py           # Retrieval smoke test
│   └── inspect_pdf.py            # PDF parsing inspector
│
├── eval/                         # Evaluation data
│   ├── questions.jsonl           # 27 ground-truth questions
│   └── questions.example.jsonl   # Example format
│
├── tests/                        # Unit tests
│   ├── test_parser.py            # Parser tests
│   └── test_fusion.py            # RRF fusion tests
│
├── docker-compose.yml            # Qdrant container
├── requirements.txt              # Core dependencies
├── requirements-visual.txt       # GPU dependencies (ColQwen2)
└── .env.example                  # Environment template
```

---

## 📄 License

This is a learning project. Use it however you like.

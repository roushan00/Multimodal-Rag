"""Quick retrieval test across all ingested PDFs."""
import warnings
import os

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

from app.retrieve.retriever import retrieve

queries = [
    ("How many AI patents were filed globally?", "hybrid"),
    ("What are the BLEU scores for machine translation?", "hybrid"),
    ("Global commodity price trends", "hybrid"),
    ("Which country leads in AI research publications?", "hybrid"),
    ("What is multi-head attention?", "text"),
    ("GDP growth forecast 2024", "visual"),
]

for q, mode in queries:
    print(f"Q: {q}  [mode={mode}]")
    r = retrieve(q, mode=mode, top_pages=3)
    for p in r.pages:
        src = os.path.basename(p.source)
        print(f"  -> {src} p.{p.page} | score={p.score:.4f} | chunks={len(p.chunks)}")
        if p.chunks:
            snippet = p.chunks[0]["content"][:100]
            print(f"     ({p.chunks[0]['kind']}) {snippet}...")
    print()

"""python -m scripts.ask "question" --mode hybrid --context both"""
import argparse
from pathlib import Path

from app.generate.answerer import answer
from app.retrieve.retriever import retrieve

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", default="hybrid", choices=["text", "visual", "hybrid"])
    ap.add_argument("--context", default="both", choices=["text", "images", "both"])
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--no-answer", action="store_true", help="only show retrieval")
    a = ap.parse_args()

    r = retrieve(a.question, mode=a.mode, top_pages=a.top)
    print(f"\nRetrieved pages ({r.mode}):")
    for p in r.pages:
        kinds = ",".join(sorted({c["kind"] for c in p.chunks})) or "-"
        print(f"  {Path(p.source).name} p.{p.page:<4} score={p.score:.4f} chunks={kinds}")
    print(f"  text rank:   {r.debug['text_rank'][:5]}")
    print(f"  visual rank: {r.debug['visual_rank'][:5]}")
    if not a.no_answer:
        print("\n" + answer(a.question, r, a.context))

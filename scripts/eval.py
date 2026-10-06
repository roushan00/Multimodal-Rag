"""Comprehensive RAG evaluation: retrieval metrics + answer quality + hallucination detection.

Metrics:
  Retrieval:  Recall@k, MRR, Hit Rate, Context Precision
  Answer:     Faithfulness, Citation Accuracy, Correctness
  Anti-hallucination: Faithfulness score (LLM-as-judge)

Usage:
  python -m scripts.eval                           # full eval
  python -m scripts.eval --retrieval-only          # skip LLM-based answer eval
  python -m scripts.eval --modes text hybrid       # specific modes
  python -m scripts.eval --no-rerank               # compare without reranking
"""
import argparse
import json
import logging
import re
import time
from collections import defaultdict
from pathlib import Path

from app.config import get_settings
from app.generate.answerer import answer, build_content, SYSTEM
from app.llm import complete, image_block
from app.retrieve.retriever import retrieve

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

KS = (1, 3, 5)

# ── LLM-as-judge prompts ────────────────────────────────────────────────

FAITHFULNESS_PROMPT = """You are an expert fact-checker evaluating a RAG system's answer for faithfulness.

You are given CONTEXT consisting of retrieved text passages AND page images from documents,
plus the ANSWER to evaluate. Your job is to check whether every claim in the answer is
supported by this context.

IMPORTANT judging rules:
- A claim is SUPPORTED if it can be verified from EITHER the text passages OR the page images.
- Numbers visible in charts, tables, or figures in the page images count as supporting evidence.
- Paraphrased or rephrased information still counts as SUPPORTED if the meaning is preserved.
- Only mark a claim UNSUPPORTED if it introduces genuinely new facts not present anywhere
  in the provided text or images.
- Do NOT mark a claim as UNSUPPORTED just because it uses different wording than the context.

TEXT CONTEXT:
{context}

(Page images are provided above this text.)

ANSWER:
{answer}

Evaluate:
1. List each factual claim in the answer (number them).
2. For each claim, state whether it is SUPPORTED or UNSUPPORTED by the context (text OR images).
3. Give a faithfulness score from 0.0 to 1.0 (proportion of supported claims).

Respond in this exact JSON format:
{{"claims": [{{"claim": "...", "verdict": "SUPPORTED|UNSUPPORTED"}}], "score": 0.0}}"""

CORRECTNESS_PROMPT = """Compare the GENERATED answer against the GOLD (reference) answer.

QUESTION: {question}
GOLD ANSWER: {gold_answer}
GENERATED ANSWER: {generated_answer}

Score the generated answer from 0.0 to 1.0:
- 1.0: Fully correct, covers all key facts from gold answer
- 0.7-0.9: Mostly correct, minor omissions
- 0.4-0.6: Partially correct, some key facts missing or wrong
- 0.1-0.3: Mostly wrong but has some relevant info
- 0.0: Completely wrong or irrelevant

Respond in this exact JSON format:
{{"score": 0.0, "reason": "brief explanation"}}"""


def _parse_json_from_llm(text: str) -> dict:
    """Extract JSON from LLM response, handling markdown code blocks."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        text = match.group(1)
    elif not text.startswith("{"):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            text = match.group(0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM JSON: %s", text[:200])
        return {}


# ── Retrieval evaluation ────────────────────────────────────────────────

def eval_retrieval(rows: list[dict], modes: tuple[str, ...], rerank: bool) -> dict:
    """Compute Recall@k, MRR, Hit Rate, Context Precision per mode."""
    results = {}

    for mode in modes:
        agg = defaultdict(list)
        for row in rows:
            r = retrieve(row["q"], mode=mode, top_pages=max(KS), rerank=rerank)
            ranked = [(Path(p.source).name, p.page) for p in r.pages]
            gold = {(row["source"], pg) for pg in row["pages"]}

            # First hit position (1-indexed)
            first = next((i for i, x in enumerate(ranked, 1) if x in gold), None)

            # Context precision: how many retrieved pages are actually relevant?
            relevant_count = sum(1 for x in ranked[:max(KS)] if x in gold)
            ctx_precision = relevant_count / min(len(ranked), max(KS)) if ranked else 0.0

            for bucket in ("all", row.get("type", "untyped")):
                agg[(bucket, "mrr")].append(1.0 / first if first else 0.0)
                agg[(bucket, "hit_rate")].append(1.0 if first else 0.0)
                agg[(bucket, "ctx_precision")].append(ctx_precision)
                for k in KS:
                    agg[(bucket, f"R@{k}")].append(1.0 if first and first <= k else 0.0)

        results[mode] = {key: round(sum(v) / len(v), 3) for key, v in agg.items()}

    return results


# ── Answer quality evaluation (LLM-as-judge) ────────────────────────────

def eval_answer_quality(rows: list[dict], mode: str, rerank: bool) -> list[dict]:
    """Evaluate faithfulness, citation accuracy, and correctness for each question."""
    results = []

    for i, row in enumerate(rows):
        logger.warning("Evaluating answer %d/%d: %s", i + 1, len(rows), row["q"][:50])
        r = retrieve(row["q"], mode=mode, top_pages=5, rerank=rerank)

        # Generate answer
        generated = answer(row["q"], r, "both")

        # Build context for faithfulness judge: page images + text chunks
        faith_content: list[dict] = []
        context_str = ""
        for p in r.pages:
            src = Path(p.source).name
            # Include page image so judge can verify visual claims
            if Path(p.render_path).exists():
                faith_content.append({"type": "text", "text": f"[Page image: {src} p.{p.page}]"})
                faith_content.append(image_block(p.render_path))
            context_str += f"\n[{src} p.{p.page}]:\n"
            for c in p.chunks:
                context_str += f"  {c['content'][:500]}\n"

        # 1. Faithfulness (anti-hallucination) — with page images
        faith_content.append({"type": "text", "text": FAITHFULNESS_PROMPT.format(
            context=context_str, answer=generated)})
        faith_resp = complete(faith_content, max_tokens=800)
        faith_data = _parse_json_from_llm(faith_resp)
        faithfulness = faith_data.get("score", 0.0)
        claims = faith_data.get("claims", [])

        # 2. Citation accuracy
        cited_pages = set(re.findall(r"\[.*?p\.(\d+)\]", generated))
        gold_pages = {str(p) for p in row["pages"]}
        cited_correct = len(cited_pages & gold_pages)
        citation_precision = cited_correct / len(cited_pages) if cited_pages else 0.0
        citation_recall = cited_correct / len(gold_pages) if gold_pages else 0.0

        # 3. Correctness (if gold answer provided)
        correctness = 0.0
        correctness_reason = ""
        if row.get("gold_answer"):
            corr_resp = complete(
                [{"type": "text", "text": CORRECTNESS_PROMPT.format(
                    question=row["q"], gold_answer=row["gold_answer"], generated_answer=generated)}],
                max_tokens=300,
            )
            corr_data = _parse_json_from_llm(corr_resp)
            correctness = corr_data.get("score", 0.0)
            correctness_reason = corr_data.get("reason", "")

        result = {
            "question": row["q"],
            "type": row.get("type", "untyped"),
            "faithfulness": round(faithfulness, 3),
            "unsupported_claims": [c["claim"] for c in claims if c.get("verdict") == "UNSUPPORTED"],
            "citation_precision": round(citation_precision, 3),
            "citation_recall": round(citation_recall, 3),
            "correctness": round(correctness, 3),
            "correctness_reason": correctness_reason,
            "answer_preview": generated[:200],
        }
        results.append(result)

    return results


# ── Report formatting ────────────────────────────────────────────────────

def print_retrieval_report(results: dict):
    """Print retrieval metrics table."""
    print("\n" + "=" * 70)
    print("RETRIEVAL METRICS")
    print("=" * 70)

    metrics = [f"R@{k}" for k in KS] + ["mrr", "hit_rate", "ctx_precision"]
    header = f"{'mode':<8}{'bucket':<12}" + "".join(f"{m:>14}" for m in metrics)
    print(header)
    print("-" * len(header))

    for mode, res in results.items():
        buckets = sorted({b for b, _ in res})
        for b in buckets:
            row = f"{mode:<8}{b:<12}"
            for m in metrics:
                val = res.get((b, m), 0)
                row += f"{val:>14.3f}"
            print(row)
    print()


def print_answer_report(results: list[dict]):
    """Print answer quality metrics."""
    print("\n" + "=" * 70)
    print("ANSWER QUALITY METRICS")
    print("=" * 70)

    # Per-question details
    for r in results:
        status = "OK" if r["faithfulness"] >= 0.9 else "WARN" if r["faithfulness"] >= 0.7 else "FAIL"
        print(f"\n[{status}] {r['question'][:60]}...")
        print(f"  faithfulness={r['faithfulness']:.2f}  correctness={r['correctness']:.2f}  "
              f"cite_prec={r['citation_precision']:.2f}  cite_rec={r['citation_recall']:.2f}")
        if r["unsupported_claims"]:
            print(f"  UNSUPPORTED: {r['unsupported_claims']}")
        if r["correctness_reason"]:
            print(f"  correctness_reason: {r['correctness_reason']}")

    # Aggregates
    print("\n" + "-" * 70)
    print("AGGREGATE SCORES")
    print("-" * 70)

    by_type = defaultdict(list)
    for r in results:
        by_type["all"].append(r)
        by_type[r["type"]].append(r)

    header = f"{'bucket':<12}{'faith':>10}{'correct':>10}{'cite_prec':>12}{'cite_rec':>12}{'halluc_rate':>14}"
    print(header)

    for bucket in sorted(by_type):
        items = by_type[bucket]
        faith = sum(r["faithfulness"] for r in items) / len(items)
        correct = sum(r["correctness"] for r in items) / len(items)
        cite_p = sum(r["citation_precision"] for r in items) / len(items)
        cite_r = sum(r["citation_recall"] for r in items) / len(items)
        halluc = sum(1 for r in items if r["faithfulness"] < 0.9) / len(items)
        print(f"{bucket:<12}{faith:>10.3f}{correct:>10.3f}{cite_p:>12.3f}{cite_r:>12.3f}{halluc:>14.3f}")

    print()


def save_results(retrieval_results: dict, answer_results: list[dict] | None, output_path: str):
    """Save full results to JSON."""
    data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "retrieval": {mode: {f"{b}_{m}": v for (b, m), v in res.items()}
                      for mode, res in retrieval_results.items()},
    }
    if answer_results:
        data["answer_quality"] = answer_results
        data["summary"] = {
            "avg_faithfulness": round(sum(r["faithfulness"] for r in answer_results) / len(answer_results), 3),
            "avg_correctness": round(sum(r["correctness"] for r in answer_results) / len(answer_results), 3),
            "hallucination_rate": round(
                sum(1 for r in answer_results if r["faithfulness"] < 0.9) / len(answer_results), 3),
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(data, indent=2))
    print(f"Results saved to {output_path}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Evaluate RAG pipeline")
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--modes", nargs="+", default=["text", "visual", "hybrid"])
    ap.add_argument("--retrieval-only", action="store_true", help="Skip LLM-based answer eval")
    ap.add_argument("--answer-mode", default="hybrid", help="Mode to use for answer quality eval")
    ap.add_argument("--no-rerank", action="store_true", help="Disable reranking")
    ap.add_argument("--output", default="eval/results.json", help="Output JSON path")
    args = ap.parse_args()

    s = get_settings()
    if not s.enable_visual:
        args.modes = [m for m in args.modes if m != "visual"]

    rows = [json.loads(l) for l in Path(args.questions).read_text().splitlines() if l.strip()]
    print(f"Loaded {len(rows)} questions from {args.questions}")
    rerank = not args.no_rerank

    # Retrieval eval
    print(f"\nRunning retrieval eval (modes={args.modes}, rerank={rerank})...")
    retrieval_results = eval_retrieval(rows, tuple(args.modes), rerank)
    print_retrieval_report(retrieval_results)

    # Answer quality eval
    answer_results = None
    if not args.retrieval_only:
        print(f"Running answer quality eval (mode={args.answer_mode}, {len(rows)} questions)...")
        answer_results = eval_answer_quality(rows, args.answer_mode, rerank)
        print_answer_report(answer_results)

    save_results(retrieval_results, answer_results, args.output)


if __name__ == "__main__":
    main()

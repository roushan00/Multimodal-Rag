"""RAG evaluation using DeepEval metrics.

Two-step process to avoid OOM:
  Step 1: python -m scripts.eval_deepeval --build    (retrieve + answer, save to JSON)
  Step 2: python -m scripts.eval_deepeval --evaluate  (run deepeval metrics on saved JSON)

Or run both:
  python -m scripts.eval_deepeval
"""
import argparse
import asyncio
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

CASES_PATH = "eval/deepeval_cases.json"


# ── Step 1: Build test cases (loads RAG models) ──────────────────────────

def build_and_save(questions_path: str, mode: str, rerank: bool):
    from app.generate.answerer import answer
    from app.retrieve.retriever import retrieve

    rows = [json.loads(l) for l in Path(questions_path).read_text().splitlines() if l.strip()]
    cases = []

    for i, row in enumerate(rows):
        logger.warning("Building case %d/%d: %s", i + 1, len(rows), row["q"][:50])
        r = retrieve(row["q"], mode=mode, top_pages=5, rerank=rerank)
        generated = answer(row["q"], r, "both")

        retrieval_context = []
        for p in r.pages:
            src = Path(p.source).name
            for c in p.chunks:
                retrieval_context.append(f"[{src} p.{p.page}] ({c['kind']}) {c['content']}")

        cases.append({
            "input": row["q"],
            "actual_output": generated,
            "expected_output": row.get("gold_answer", ""),
            "retrieval_context": retrieval_context,
        })

    Path(CASES_PATH).write_text(json.dumps(cases, indent=2))
    print(f"Saved {len(cases)} test cases to {CASES_PATH}")


# ── Step 2: Run DeepEval metrics (no RAG models needed) ─────────────────

def run_deepeval(threshold: float):
    import os
    os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "secrets/vertex-key.json")

    from google import genai
    from google.genai import types

    from deepeval import evaluate
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        HallucinationMetric,
    )
    from deepeval.models import DeepEvalBaseLLM
    from deepeval.test_case import LLMTestCase

    from app.config import get_settings

    class GeminiVertexModel(DeepEvalBaseLLM):
        def __init__(self, model_name: str = "gemini-2.5-flash"):
            super().__init__(model=model_name)
            s = get_settings()
            self._client = genai.Client(
                vertexai=True,
                project=s.gcp_project,
                location=s.gcp_location,
            )
            self._model_name = model_name

        def load_model(self):
            return self

        def get_model_name(self) -> str:
            return self._model_name

        def generate(self, prompt: str, schema=None) -> str:
            config = types.GenerateContentConfig(
                max_output_tokens=4096,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            )
            if schema:
                config.response_mime_type = "application/json"
                config.response_schema = schema
            response = self._client.models.generate_content(
                model=self._model_name, contents=prompt, config=config,
            )
            return response.text or ""

        async def a_generate(self, prompt: str, schema=None) -> str:
            return await asyncio.to_thread(self.generate, prompt, schema)

        def generate_with_schema(self, prompt: str, schema=None) -> str:
            return self.generate(prompt, schema)

        async def a_generate_with_schema(self, prompt: str, schema=None) -> str:
            return await asyncio.to_thread(self.generate, prompt, schema)

    # Load saved cases
    cases_data = json.loads(Path(CASES_PATH).read_text())
    print(f"Loaded {len(cases_data)} test cases from {CASES_PATH}")

    test_cases = [
        LLMTestCase(
            input=c["input"],
            actual_output=c["actual_output"],
            expected_output=c["expected_output"],
            retrieval_context=c["retrieval_context"],
            context=[c["expected_output"]],
        )
        for c in cases_data
    ]

    gemini = GeminiVertexModel(model_name=get_settings().llm_model)

    metrics = [
        FaithfulnessMetric(threshold=threshold, model=gemini),
        AnswerRelevancyMetric(threshold=threshold, model=gemini),
        ContextualPrecisionMetric(threshold=threshold, model=gemini),
        ContextualRecallMetric(threshold=threshold, model=gemini),
        HallucinationMetric(threshold=threshold, model=gemini),
    ]

    print("Running DeepEval evaluation...")
    results = evaluate(test_cases=test_cases, metrics=metrics)
    print("\nDeepEval evaluation complete.")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="DeepEval RAG evaluation")
    ap.add_argument("--questions", default="eval/questions.jsonl")
    ap.add_argument("--mode", default="hybrid", choices=["text", "visual", "hybrid"])
    ap.add_argument("--no-rerank", action="store_true")
    ap.add_argument("--threshold", type=float, default=0.7)
    ap.add_argument("--build", action="store_true", help="Only build test cases (step 1)")
    ap.add_argument("--evaluate", action="store_true", help="Only run deepeval (step 2)")
    args = ap.parse_args()

    rerank = not args.no_rerank

    if args.evaluate:
        run_deepeval(args.threshold)
    elif args.build:
        build_and_save(args.questions, args.mode, rerank)
    else:
        build_and_save(args.questions, args.mode, rerank)
        run_deepeval(args.threshold)


if __name__ == "__main__":
    main()

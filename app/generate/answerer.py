from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Literal

from app.llm import complete, image_block, stream
from app.retrieve.retriever import Retrieval

Context = Literal["text", "images", "both"]

SYSTEM = (
    "You are a precise document QA assistant. Your ONLY source of truth is the provided "
    "document pages and text extracts below. Follow these rules strictly:\n"
    "\n"
    "1. NEVER use your own knowledge. If the answer is not in the provided context, say "
    "   \"The provided documents do not contain this information.\" Do not guess or infer "
    "   beyond what is explicitly stated.\n"
    "2. Every factual claim MUST be supported by the provided context. Do not add facts, "
    "   numbers, or details that are not present in the extracts or visible in the page images.\n"
    "3. Cite every claim as [<filename> p.<page>]. Only cite pages that are provided to you.\n"
    "4. When reading tables, report exact numbers as shown. Do not round or approximate.\n"
    "5. When reading charts, describe only what is visually present — axis values, data points, "
    "   trends, and labels. Do not extrapolate beyond the chart's data range.\n"
    "6. If the context is ambiguous or incomplete, state that explicitly rather than filling "
    "   in gaps with assumptions.\n"
    "7. Keep your answer concise and factual. Prefer shorter, accurate answers over longer, "
    "   speculative ones."
)


def build_content(question: str, r: Retrieval, context: Context = "both") -> list[dict]:
    content: list[dict] = []
    for p in r.pages:
        label = f"{Path(p.source).name or p.doc_id} p.{p.page}"
        content.append({"type": "text", "text": f"=== Page: [{label}] ==="})
        if context in ("images", "both") and Path(p.render_path).exists():
            content.append(image_block(p.render_path))
        if context in ("text", "both") and p.chunks:
            extracts = "\n\n".join(f"({c['kind']}) {c['content']}" for c in p.chunks)
            content.append({"type": "text", "text": f"Extracted text for [{label}]:\n{extracts}"})
    content.append({"type": "text", "text": f"Question: {question}"})
    return content


def answer(question: str, r: Retrieval, context: Context = "both") -> str:
    if not r.pages:
        return "No relevant pages found."
    return complete(build_content(question, r, context), system=SYSTEM, max_tokens=1200)


def answer_stream(content: list[dict]) -> Iterator[str]:
    """Stream answer tokens given pre-built content."""
    yield from stream(content, system=SYSTEM, max_tokens=1200)

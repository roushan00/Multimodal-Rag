"""Thin wrapper over Google Vertex AI Gemini API with image support.

Uses a service account JSON key for authentication.
"""
from __future__ import annotations

import base64
import logging
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from google import genai
from google.genai import types

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _client() -> genai.Client:
    s = get_settings()
    return genai.Client(
        vertexai=True,
        project=s.gcp_project,
        location=s.gcp_location,
    )


def image_block(path: str | Path) -> dict:
    data = base64.standard_b64encode(Path(path).read_bytes()).decode()
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def _to_gemini_parts(content: list[dict]) -> list[types.Part]:
    """Convert our internal content format to Gemini parts."""
    parts: list[types.Part] = []
    for block in content:
        if block.get("type") == "text":
            parts.append(types.Part.from_text(text=block["text"]))
        elif block.get("type") == "image":
            src = block["source"]
            parts.append(types.Part.from_bytes(
                data=base64.standard_b64decode(src["data"]),
                mime_type=src["media_type"],
            ))
    return parts


def _config(system: str, max_tokens: int) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=system or None,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_budget=0),
    )


def complete(content: list[dict], *, model: str | None = None, system: str = "",
             max_tokens: int = 1024) -> str:
    mdl = model or get_settings().llm_model
    parts = _to_gemini_parts(content)
    response = _client().models.generate_content(
        model=mdl, contents=parts, config=_config(system, max_tokens),
    )
    return response.text or ""


def stream(content: list[dict], *, model: str | None = None, system: str = "",
           max_tokens: int = 1024) -> Iterator[str]:
    """Yield text chunks as they arrive from the model."""
    mdl = model or get_settings().llm_model
    parts = _to_gemini_parts(content)
    for chunk in _client().models.generate_content_stream(
        model=mdl, contents=parts, config=_config(system, max_tokens),
    ):
        if chunk.text:
            yield chunk.text

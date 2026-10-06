"""Turn figures into retrievable text. This is what makes Path A 'multimodal'."""
import logging

from app.config import get_settings
from app.llm import complete, image_block
from app.models import ParsedDoc

logger = logging.getLogger(__name__)

PROMPT = (
    "You are a precise document figure analyst. Analyze this figure extracted from a "
    "document and produce a detailed, searchable description. Include ALL of the following:\n"
    "\n"
    "1. FIGURE TYPE: What kind of figure is this? (bar chart, line chart, pie chart, "
    "   scatter plot, flow diagram, architecture diagram, table, photograph, etc.)\n"
    "2. TITLE: The exact title or caption if visible.\n"
    "3. AXES & LABELS: For charts — exact axis labels, units, scale ranges. "
    "   For diagrams — key components and their labels.\n"
    "4. DATA POINTS: List ALL specific numbers, percentages, and values visible in the "
    "   figure. For charts with multiple series, list values for each series. Read bar "
    "   heights, line endpoints, pie slice percentages exactly as shown.\n"
    "5. TRENDS & COMPARISONS: Describe the overall trend (increasing, decreasing, flat), "
    "   peaks, troughs, and notable comparisons between data series.\n"
    "6. KEY TAKEAWAY: The single most important insight from this figure.\n"
    "\n"
    "Be exhaustive with numbers — a user should be able to answer quantitative questions "
    "about this figure using only your description. Max 250 words, no preamble.\n"
    "\n"
    "Text near the figure (may help with context):\n{nearby}"
)


def caption_figures(doc: ParsedDoc) -> int:
    s = get_settings()
    n = 0
    for page in doc.pages:
        for fig in page.figures:
            try:
                fig.caption = complete(
                    [image_block(fig.path), {"type": "text", "text": PROMPT.format(nearby=fig.nearby_text or "-")}],
                    model=s.caption_model,
                    max_tokens=500,
                ).strip()
                n += 1
                logger.info("Captioned figure on page %d: %s", page.page, fig.path)
            except Exception as e:
                logger.warning("Caption failed for %s: %s", fig.path, e)
    return n

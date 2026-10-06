from dataclasses import dataclass, field
from typing import Literal

ChunkKind = Literal["text", "table", "figure"]


@dataclass
class FigureRef:
    path: str
    page: int
    bbox: tuple[float, float, float, float]
    nearby_text: str = ""        # text around the figure, helps captioning
    caption: str = ""


@dataclass
class PageContent:
    page: int                    # 1-based
    text_blocks: list[str] = field(default_factory=list)
    tables_md: list[str] = field(default_factory=list)
    figures: list[FigureRef] = field(default_factory=list)
    render_path: str = ""        # full-page PNG for the visual path + generation


@dataclass
class ParsedDoc:
    doc_id: str
    source: str
    pages: list[PageContent]


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page: int
    kind: ChunkKind
    content: str
    image_path: str | None = None

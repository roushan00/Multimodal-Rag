from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_application_credentials: str = "secrets/vertex-key.json"
    gcp_project: str = "vertex-ocr-492906"
    gcp_location: str = "us-central1"
    llm_model: str = "gemini-2.5-flash"
    caption_model: str = "gemini-2.5-flash"

    qdrant_url: str = "http://localhost:6333"
    text_collection: str = "text_chunks"
    page_collection: str = "pages"

    text_embed_model: str = "BAAI/bge-small-en-v1.5"
    visual_embed_model: str = "vidore/colqwen2-v1.0"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    enable_visual: bool = True
    caption_images: bool = True

    data_dir: Path = Path("data")
    chunk_size: int = 800        # characters
    chunk_overlap: int = 120
    page_dpi: int = 110
    min_image_px: int = 120      # skip icons/logos smaller than this on both sides

    @property
    def pages_dir(self) -> Path:
        return self.data_dir / "pages"

    @property
    def figures_dir(self) -> Path:
        return self.data_dir / "figures"


@lru_cache
def get_settings() -> Settings:
    return Settings()

"""Application configuration loaded from environment variables."""
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central configuration. Values can be overridden via .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Mistral
    mistral_api_key: str = ""
    mistral_chat_model: str = "mistral-small-latest"
    mistral_embed_model: str = "mistral-embed"

    # Storage
    data_dir: Path = Path("./data")
    index_dir: Path = Path("./data/index")
    uploads_dir: Path = Path("./data/uploads")

    # Chunking
    chunk_size: int = 512
    chunk_overlap: int = 64

    # Retrieval
    top_k_vector: int = 10
    top_k_keyword: int = 10
    top_k_final: int = 5
    rrf_k: int = 60  # Reciprocal Rank Fusion constant

    # Intent detection
    min_query_tokens: int = 3

    def ensure_dirs(self) -> None:
        """Create required directories if missing."""
        for d in (self.data_dir, self.index_dir, self.uploads_dir):
            d.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_dirs()

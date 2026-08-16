from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "development"
    log_level: str = "INFO"
    database_url: str = "postgresql+psycopg://copilot:copilot@localhost:5432/copilot"

    provider_mode: Literal["openai", "gemini"] = "openai"

    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-large"
    openai_generation_model: str = "gpt-4o-mini"

    gemini_api_key: str | None = None
    gemini_embedding_model: str = "gemini-embedding-001"
    gemini_generation_model: str = "gemini-2.5-flash"

    embedding_dimensions: int = 3072

    retrieval_vector_k: int = 20
    retrieval_lexical_k: int = 20
    retrieval_fused_k: int = 10
    retrieval_final_k: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()

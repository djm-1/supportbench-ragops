from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    database_url: str = "sqlite:///./supportbench.db"
    supportbench_data_dir: Optional[str] = None
    use_real_models: bool = False
    groq_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-2.5-flash"
    use_pinecone: bool = False
    pinecone_api_key: str = ""
    pinecone_index: str = "supportbench-ragops-integrated"
    pinecone_namespace: str = "customer-support-v1"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_metric: str = "cosine"
    pinecone_batch_size: int = 96
    pinecone_embedding_mode: str = "integrated"
    pinecone_embed_model: str = "llama-text-embed-v2"
    pinecone_text_field: str = "chunk_text"
    openai_api_key: str = ""
    gemini_api_key: str = ""
    embedding_model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    embedding_batch_size: int = 64
    default_answer_model: str = "openai_primary"
    eval_judge_provider: str = "openai"
    eval_judge_model: str = "gpt-5.4-nano"
    eval_judge_prompt_version: str = "answer-judge-v1"

    model_config = SettingsConfigDict(
        env_file=(
            str(Path(__file__).resolve().parents[2] / ".env"),
            str(Path(__file__).resolve().parents[1] / ".env"),
        ),
        extra="ignore",
    )

    @property
    def data_dir(self) -> Path:
        if self.supportbench_data_dir:
            return Path(self.supportbench_data_dir)
        return Path(__file__).resolve().parents[2] / "data"


settings = Settings()

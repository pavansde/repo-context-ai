"""Centralized, validated application configuration."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        env_ignore_empty=True,
    )

    # Application
    app_name: str = "RepoContext AI"
    app_env: Literal[
        "development",
        "testing",
        "production",
    ] = "development"

    debug: bool = False
    log_level: str = "INFO"
    api_host: str = "[IP_ADDRESS]"
    api_port: int = Field(default=8000, ge=1, le=65535)
    # API authentication
    api_key: SecretStr = SecretStr("")

    # Model provider configuration
    llm_provider: Literal["openai", "ollama"] = "ollama"
    llm_model: str = "gpt-4.1-mini"

    llm_fallback_provider: Literal[
        "openai",
        "ollama",
    ] | None = None

    llm_fallback_model: str | None = None

    openai_api_key: SecretStr = SecretStr("")
    openai_base_url: str | None = None

    # Ollama connection
    ollama_host: str = "localhost"
    ollama_port: int = Field(
        default=11434,
        ge=1,
        le=65535,
    )

    ollama_chat_model: str = "qwen2.5-coder:7b"

    ollama_timeout: int = Field(
        default=120,
        gt=0,
    )

    ollama_num_ctx: int = Field(
        default=8192,
        gt=0,
    )

    # Embedding provider
    embedding_provider: Literal[
        "openai",
        "ollama",
    ] = "ollama"

    embedding_model: str = "nomic-embed-text"

    llm_temperature: float = Field(
        default=0.0,
        ge=0,
        le=2,
    )

    # GitHub
    github_token: SecretStr = SecretStr("")

    github_clone_timeout: int = Field(
        default=120,
        gt=0,
    )

    max_repository_size_mb: int = Field(
        default=500,
        gt=0,
    )

    # RAG
    chunk_size: int = Field(
        default=1200,
        gt=0,
    )

    chunk_overlap: int = Field(
        default=200,
        ge=0,
    )

    retriever_top_k: int = Field(
        default=5,
        gt=0,
    )

    vector_store_path: Path = Path("./data/chroma")

    collection_name: str = "repo_context"

    # Repository cache
    repository_cache_path: Path = Path(
        "./data/repositories"
    )

    # Repository registry
    repositories: str = ""

    # Memory
    session_ttl_minutes: int = Field(
        default=60,
        gt=0,
    )

    checkpoint_db_path: Path = Path(
        "./data/checkpoints.db"
    )

    # LangSmith
    langsmith_tracing: bool = True

    langsmith_api_key: SecretStr = SecretStr("")

    langsmith_project: str = "repo-context-ai"

    langsmith_endpoint: str = (
        "https://api.smith.langchain.com"
    )

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(
        cls,
        value: int,
        info,
    ):
        """Ensure overlap is smaller than the chunk size."""

        chunk_size = info.data.get(
            "chunk_size",
            1200,
        )

        if value >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size"
            )

        return value

    def prepare_directories(self) -> None:
        """Create local directories required by the application."""

        self.vector_store_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.repository_cache_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.checkpoint_db_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

    def validate_runtime(self) -> None:
        """Validate configuration needed to start the application."""

        if (
            self.llm_provider == "openai"
            or self.embedding_provider == "openai"
            or self.llm_fallback_provider == "openai"
        ):
            if not self.openai_api_key.get_secret_value():
                raise ValueError(
                    "OPENAI_API_KEY is required for the configured OpenAI provider"
                )

        if self.app_env == "production":
            if not self.api_key.get_secret_value():
                raise ValueError(
                    "API_KEY is required in production"
                )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached settings instance."""

    return Settings()
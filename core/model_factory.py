
"""Provider-independent model construction for RepoContext AI."""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from core.config import Settings, get_settings
from core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)

Provider = Literal["openai", "ollama"]


def get_ollama_base_url(settings: Settings) -> str:
    """Build the Ollama HTTP endpoint from configured host and port."""
    host = settings.ollama_host.strip().rstrip("/")

    if not host:
        raise ConfigurationError("OLLAMA_HOST cannot be empty")

    # Accept either an IP address or hostname, but not a URL here.
    if "://" in host or "/" in host:
        raise ConfigurationError(
            "OLLAMA_HOST must be a hostname or IP address, "
            "not a URL or path"
        )

    return f"http://{host}:{settings.ollama_port}"


def create_chat_model(
    provider: Provider | None = None,
    model_name: str | None = None,
    settings: Settings | None = None,
) -> BaseChatModel:
    """Create a LangChain chat model for the selected provider.

    Args:
        provider: 'openai' or 'ollama'. Defaults to configured provider.
        model_name: Optional model override.
        settings: Application settings.

    Returns:
        A LangChain BaseChatModel implementation.

    Raises:
        ConfigurationError: If provider configuration is invalid.
    """
    settings = settings or get_settings()
    selected_provider = provider or settings.llm_provider

    if selected_provider == "openai":
        api_key = settings.openai_api_key.get_secret_value()

        if not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is required for OpenAI"
            )

        kwargs = {
            "model": model_name or settings.llm_model,
            "temperature": settings.llm_temperature,
            "api_key": api_key,
        }

        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url

        return ChatOpenAI(**kwargs)

    if selected_provider == "ollama":
        return ChatOllama(
            model=model_name or settings.ollama_chat_model,
            base_url=get_ollama_base_url(settings),
            temperature=settings.llm_temperature,
            num_ctx=settings.ollama_num_ctx,
            client_kwargs={"timeout": settings.ollama_timeout},
            validate_model_on_init=False,
        )

    raise ConfigurationError(
        f"Unsupported chat model provider: {selected_provider}"
    )


def create_embedding_model(
    provider: Provider | None = None,
    model_name: str | None = None,
    settings: Settings | None = None,
) -> Embeddings:
    """Create a provider-specific embeddings implementation.

    The same embedding provider and model must be used consistently
    for indexing and retrieval within a vector collection.
    """
    settings = settings or get_settings()
    selected_provider = provider or settings.embedding_provider
    selected_model = model_name or settings.embedding_model

    if selected_provider == "openai":
        api_key = settings.openai_api_key.get_secret_value()

        if not api_key:
            raise ConfigurationError(
                "OPENAI_API_KEY is required for OpenAI embeddings"
            )

        kwargs = {
            "model": selected_model,
            "api_key": api_key,
        }

        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url

        return OpenAIEmbeddings(**kwargs)

    if selected_provider == "ollama":
        return OllamaEmbeddings(
            model=selected_model,
            base_url=get_ollama_base_url(settings),
            client_kwargs={"timeout": settings.ollama_timeout},
        )

    raise ConfigurationError(
        f"Unsupported embedding provider: {selected_provider}"
    )


def create_chat_model_with_fallbacks(
    settings: Settings | None = None,
) -> BaseChatModel:
    """Create the primary chat model with a secondary fallback."""
    settings = settings or get_settings()

    primary = create_chat_model(
        provider=settings.llm_provider,
        settings=settings,
    )

    fallback_provider = settings.llm_fallback_provider
    fallback_model_name = settings.llm_fallback_model

    if not fallback_provider or not fallback_model_name:
        return primary

    fallback = create_chat_model(
        provider=fallback_provider,
        model_name=fallback_model_name,
        settings=settings,
    )

    logger.info(
        "Configured chat model fallback: %s -> %s",
        settings.llm_provider,
        fallback_provider,
    )

    return primary.with_fallbacks([fallback])
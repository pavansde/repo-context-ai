"""Unit tests for reliability and failure scenarios.

These tests cover configuration validation, model factory guards,
repository registry error handling, RAG indexing edge cases,
inspect_file security constraints, and the reranker fix.
They do NOT require a live LLM or Ollama instance.
"""

from pathlib import Path

import pytest

from core.config import Settings, get_settings
from core.exceptions import ConfigurationError, IndexingError
from core.model_factory import get_ollama_base_url
from core.repository_registry import get_repository_registry


# ---------------------------------------------------------------------------
# Configuration validation
# ---------------------------------------------------------------------------


def test_production_requires_api_key():
    """validate_runtime must raise if API_KEY is empty in production."""
    settings = get_settings().model_copy(
        update={
            "app_env": "production",
            "api_key": __import__("pydantic").SecretStr(""),
        }
    )
    with pytest.raises(ValueError, match="API_KEY is required in production"):
        settings.validate_runtime()


def test_production_accepts_non_empty_api_key():
    """validate_runtime must not raise when API_KEY is set in production."""
    settings = get_settings().model_copy(
        update={
            "app_env": "production",
            "api_key": __import__("pydantic").SecretStr("some-secret"),
            "llm_provider": "ollama",
            "embedding_provider": "ollama",
            "llm_fallback_provider": None,
        }
    )
    # Should not raise.
    settings.validate_runtime()


def test_chunk_overlap_must_be_less_than_chunk_size():
    """chunk_overlap >= chunk_size must be rejected at construction time."""
    with pytest.raises(ValueError, match="chunk_overlap must be smaller"):
        Settings(
            chunk_size=500,
            chunk_overlap=500,
        )


def test_openai_provider_requires_api_key():
    """validate_runtime must raise when provider is openai but key is empty."""
    settings = get_settings().model_copy(
        update={
            "llm_provider": "openai",
            "openai_api_key": __import__("pydantic").SecretStr(""),
        }
    )
    with pytest.raises(ValueError, match="OPENAI_API_KEY is required"):
        settings.validate_runtime()


# ---------------------------------------------------------------------------
# Ollama base URL construction
# ---------------------------------------------------------------------------


def test_get_ollama_base_url_builds_http_url():
    settings = get_settings().model_copy(
        update={"ollama_host": "localhost", "ollama_port": 11434}
    )
    assert get_ollama_base_url(settings) == "http://localhost:11434"


def test_get_ollama_base_url_rejects_url_scheme():
    settings = get_settings().model_copy(
        update={"ollama_host": "http://localhost", "ollama_port": 11434}
    )
    with pytest.raises(ConfigurationError, match="hostname or IP address"):
        get_ollama_base_url(settings)


def test_get_ollama_base_url_rejects_empty_host():
    settings = get_settings().model_copy(
        update={"ollama_host": "   ", "ollama_port": 11434}
    )
    with pytest.raises(ConfigurationError, match="OLLAMA_HOST cannot be empty"):
        get_ollama_base_url(settings)


# ---------------------------------------------------------------------------
# Repository registry error handling
# ---------------------------------------------------------------------------


def test_registry_rejects_malformed_entry_without_separator():
    """An entry without ':' must raise ConfigurationError."""
    settings = get_settings().model_copy(
        update={"repositories": "no-colon-here"}
    )
    with pytest.raises(
        ConfigurationError,
        match="Invalid REPOSITORIES entry",
    ):
        get_repository_registry(settings)


def test_registry_rejects_empty_repository_id(tmp_path: Path):
    """An entry with an empty ID must raise ConfigurationError."""
    settings = get_settings().model_copy(
        update={"repositories": f":{tmp_path}"}
    )
    with pytest.raises(
        ConfigurationError,
        match="Repository ID cannot be empty",
    ):
        get_repository_registry(settings)


def test_registry_parses_multiple_entries(tmp_path: Path):
    """Registry should accept comma-separated entries."""
    repo_a = tmp_path / "a"
    repo_b = tmp_path / "b"
    repo_a.mkdir()
    repo_b.mkdir()

    settings = get_settings().model_copy(
        update={"repositories": f"repo-a:{repo_a},repo-b:{repo_b}"}
    )

    registry = get_repository_registry(settings)
    assert registry.contains("repo-a")
    assert registry.contains("repo-b")


# ---------------------------------------------------------------------------
# RAG indexing
# ---------------------------------------------------------------------------


def test_index_repository_rejects_empty_repository_id(tmp_path: Path):
    """index_repository must raise IndexingError for empty repository_id."""
    from rag import index_repository

    settings = get_settings().model_copy(
        update={
            "vector_store_path": tmp_path / "chroma",
            "collection_name": "empty_id_test",
        }
    )

    repo = tmp_path / "repo"
    repo.mkdir()

    with pytest.raises(IndexingError, match="repository_id cannot be empty"):
        index_repository(
            repository_path=repo,
            repository_id="   ",
            settings=settings,
        )


def test_index_repository_rejects_missing_directory(tmp_path: Path):
    """index_repository must raise IndexingError when the path doesn't exist."""
    from rag import index_repository

    settings = get_settings().model_copy(
        update={
            "vector_store_path": tmp_path / "chroma",
            "collection_name": "missing_dir_test",
        }
    )

    with pytest.raises(IndexingError, match="does not exist"):
        index_repository(
            repository_path=tmp_path / "does-not-exist",
            repository_id="missing-repo",
            settings=settings,
        )


# ---------------------------------------------------------------------------
# inspect_file security constraints (no LLM required — tested via tools module)
# ---------------------------------------------------------------------------


def test_inspect_file_rejects_path_traversal(tmp_path: Path):
    """inspect_file must reject paths that escape the repository root."""
    from core.config import get_settings
    from agent.tools import create_repository_tools

    settings = get_settings().model_copy(
        update={
            "vector_store_path": tmp_path / "chroma",
            "collection_name": "traversal_test",
        }
    )

    repo = tmp_path / "repo"
    repo.mkdir()

    tools = create_repository_tools(
        repository_path=repo,
        repository_id="traversal-test",
        settings=settings,
    )

    inspect_file = next(t for t in tools if t.name == "inspect_file")

    with pytest.raises(ValueError, match="must remain inside the repository"):
        inspect_file.invoke({"relative_path": "../../etc/passwd"})


def test_inspect_file_rejects_missing_file(tmp_path: Path):
    """inspect_file must raise FileNotFoundError for non-existent files."""
    from agent.tools import create_repository_tools

    settings = get_settings().model_copy(
        update={
            "vector_store_path": tmp_path / "chroma",
            "collection_name": "missing_file_test",
        }
    )

    repo = tmp_path / "repo"
    repo.mkdir()

    tools = create_repository_tools(
        repository_path=repo,
        repository_id="missing-file-test",
        settings=settings,
    )

    inspect_file = next(t for t in tools if t.name == "inspect_file")

    with pytest.raises(FileNotFoundError):
        inspect_file.invoke({"relative_path": "does_not_exist.py"})


def test_inspect_file_rejects_oversized_file(tmp_path: Path):
    """inspect_file must reject files exceeding the 100 KB size limit."""
    from agent.tools import create_repository_tools, MAX_INSPECT_BYTES

    settings = get_settings().model_copy(
        update={
            "vector_store_path": tmp_path / "chroma",
            "collection_name": "oversized_test",
        }
    )

    repo = tmp_path / "repo"
    repo.mkdir()

    # Write a file slightly larger than the limit.
    large_file = repo / "large.py"
    large_file.write_bytes(b"x" * (MAX_INSPECT_BYTES + 1))

    tools = create_repository_tools(
        repository_path=repo,
        repository_id="oversized-test",
        settings=settings,
    )

    inspect_file = next(t for t in tools if t.name == "inspect_file")

    with pytest.raises(ValueError, match="exceeds the maximum"):
        inspect_file.invoke({"relative_path": "large.py"})


def test_create_repository_tools_rejects_missing_path(tmp_path: Path):
    """create_repository_tools must raise ValueError for non-existent paths."""
    from agent.tools import create_repository_tools

    settings = get_settings().model_copy(
        update={
            "vector_store_path": tmp_path / "chroma",
            "collection_name": "missing_path_test",
        }
    )

    with pytest.raises(ValueError, match="existing directory"):
        create_repository_tools(
            repository_path=tmp_path / "does-not-exist",
            repository_id="bad-path-test",
            settings=settings,
        )


# ---------------------------------------------------------------------------
# Reranker: validate the fix for unreachable code
# ---------------------------------------------------------------------------


def test_reranker_returns_candidates_on_invalid_json():
    """rerank_sources must fall back to original order for non-JSON output."""
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    mock_response.content = "this is not json"

    with patch(
        "evaluation.reranker.create_chat_model_with_fallbacks"
    ) as mock_factory:
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_factory.return_value = mock_model

        from evaluation.reranker import rerank_sources

        sources = ["rag.py", "agent/tools.py"]
        result = rerank_sources("What does the agent do?", sources)

    assert result == sources


def test_reranker_returns_candidates_on_invalid_permutation():
    """rerank_sources must fall back when the LLM invents new file names."""
    import json
    from unittest.mock import MagicMock, patch

    mock_response = MagicMock()
    # LLM invented an extra file not in the candidates list.
    mock_response.content = json.dumps(["rag.py", "invented/file.py"])

    with patch(
        "evaluation.reranker.create_chat_model_with_fallbacks"
    ) as mock_factory:
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_factory.return_value = mock_model

        from evaluation.reranker import rerank_sources

        sources = ["rag.py", "agent/tools.py"]
        result = rerank_sources("What does the agent do?", sources)

    assert result == sources


def test_reranker_returns_reranked_on_valid_permutation():
    """rerank_sources must return the LLM's ordering when it is a valid permutation."""
    import json
    from unittest.mock import MagicMock, patch

    sources = ["rag.py", "agent/tools.py"]
    reranked = ["agent/tools.py", "rag.py"]

    mock_response = MagicMock()
    mock_response.content = json.dumps(reranked)

    with patch(
        "evaluation.reranker.create_chat_model_with_fallbacks"
    ) as mock_factory:
        mock_model = MagicMock()
        mock_model.invoke.return_value = mock_response
        mock_factory.return_value = mock_model

        from evaluation.reranker import rerank_sources

        result = rerank_sources("What does the agent do?", sources)

    assert result == reranked


# ---------------------------------------------------------------------------
# Health endpoints
# ---------------------------------------------------------------------------


def test_health_endpoint_returns_ok():
    """The /health endpoint must return 200 without app state."""
    from fastapi.testclient import TestClient
    from app import app

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_readiness_endpoint_returns_ready_after_startup():
    """The /health/ready endpoint must return 200 after app startup."""
    from fastapi.testclient import TestClient
    from app import app

    with TestClient(app) as client:
        response = client.get("/health/ready")

    assert response.status_code == 200
    assert response.json()["status"] == "ready"


def test_readiness_endpoint_returns_503_without_state():
    """The /health/ready endpoint must return 503 when app state is missing."""
    from fastapi.testclient import TestClient
    from fastapi import FastAPI

    # Create a bare app with the ready route but no state.
    bare_app = FastAPI()

    from app import ready
    bare_app.add_api_route("/health/ready", ready)

    client = TestClient(bare_app, raise_server_exceptions=False)
    response = client.get("/health/ready")

    assert response.status_code == 503


# ---------------------------------------------------------------------------
# Indexing: test that tests/ and evaluation/ are excluded
# ---------------------------------------------------------------------------


def test_ignored_directories_includes_tests_and_evaluation():
    """The IGNORED_DIRECTORIES set must include 'tests' and 'evaluation'."""
    from rag import IGNORED_DIRECTORIES

    assert "tests" in IGNORED_DIRECTORIES
    assert "evaluation" in IGNORED_DIRECTORIES


def test_test_files_are_not_indexed(tmp_path: Path):
    """Files inside a 'tests/' subdirectory must not be included in indexing."""
    from rag import load_repository_documents

    repo = tmp_path / "repo"
    repo.mkdir()

    # A real source file.
    (repo / "app.py").write_text("def main(): pass\n", encoding="utf-8")

    # A test file that must be excluded.
    tests_dir = repo / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_app.py").write_text(
        "def test_main(): assert True\n", encoding="utf-8"
    )

    docs = load_repository_documents(repo, "test-exclusion-repo")

    sources = [doc.metadata["file_path"] for doc in docs]
    assert "app.py" in sources
    assert not any("tests/" in s for s in sources), (
        f"Test files should not be indexed, but found: {sources}"
    )

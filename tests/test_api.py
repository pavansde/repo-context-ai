from pathlib import Path

from fastapi.testclient import TestClient

from app import app
from core.config import get_settings
from rag import index_repository

def test_chat_requires_authentication():
    with TestClient(app) as client:
        response = client.post(
        "/api/v1/chat",
            json={
                "repository_id": "repo-context-ai",
                "message": "Hello",
                "thread_id": "auth-test",
            },
        )

        assert response.status_code == 401

def test_chat_accepts_valid_api_key():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            headers={
                "Authorization": "Bearer dev-secret-key"
            },
            json={
                "repository_id": "repo-context-ai",
                "message": "Hello",
                "thread_id": "auth-test",
            },
        )

        assert response.status_code == 200

def test_chat_rejects_invalid_api_key():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            headers={
                "Authorization": "Bearer wrong-key"
            },
            json={
                "repository_id": "repo-context-ai",
                "message": "Hello",
                "thread_id": "auth-test",
            },
        )

        assert response.status_code == 401

def test_chat_stream_requires_authentication():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/stream",
            json={
                "repository_id": "repo-context-ai",
                "message": "Hello",
                "thread_id": "auth-stream-test",
            },
        )

        assert response.status_code == 401

def test_chat_endpoint():
    settings = get_settings().model_copy(
        update={
            "vector_store_path": Path("data/test_api_chroma"),
            "repository_cache_path": Path(
                "data/test_api_repositories"
            ),
            "collection_name": "api_test",
            "repositories": (
                "api-test-repo:"
                "D:/Personal/repo-context-ai"
            ),
        }
    )

    app.dependency_overrides[get_settings] = (
        lambda: settings
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/chat",
                headers={
                    "Authorization": "Bearer dev-secret-key"
                },
                json={
                    "repository_id": "api-test-repo",
                    "message": (
                        "Explain the RAG pipeline. "
                        "Mention the source file."
                    ),
                    "thread_id": "api-test-thread",
                },
            )

        assert response.status_code == 200

        data = response.json()

        assert data["thread_id"] == "api-test-thread"
        assert data["answer"]

    finally:
        app.dependency_overrides.clear()

def test_chat_stream_endpoint():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat/stream",
            headers={
                "Authorization": "Bearer dev-secret-key"
            },
            json={
                "repository_id": "repo-context-ai",
                "message": "Find the main application entry point.",
                "thread_id": "stream-test",
            },
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith(
            "text/event-stream"
        )

        assert "type" in response.text
        assert "done" in response.text

def test_chat_rejects_unauthorized_repository():
    settings = get_settings().model_copy(
        update={
            "repositories": (
                "allowed-repo:"
                "D:/Personal/repo-context-ai"
            ),
        }
    )

    app.dependency_overrides[get_settings] = (
        lambda: settings
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/chat",
                headers={
                    "Authorization": "Bearer dev-secret-key"
                },
                json={
                    "repository_id": "unauthorized-repo",
                    "message": "Hello",
                    "thread_id": "authorization-test",
                },
            )

        assert response.status_code == 403
        assert response.json()["detail"] == (
            "Access to this repository is not authorized"
        )

    finally:
        app.dependency_overrides.clear()

def test_chat_stream_rejects_unauthorized_repository():
    settings = get_settings().model_copy(
        update={
            "repositories": (
                "allowed-repo:"
                "D:/Personal/repo-context-ai"
            ),
        }
    )

    app.dependency_overrides[get_settings] = (
        lambda: settings
    )

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/chat/stream",
                headers={
                    "Authorization": "Bearer dev-secret-key"
                },
                json={
                    "repository_id": "unauthorized-repo",
                    "message": "Hello",
                    "thread_id": "authorization-stream-test",
                },
            )

        assert response.status_code == 403

    finally:
        app.dependency_overrides.clear()

def test_chat_returns_request_id():
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/chat",
            headers={
                "Authorization": "Bearer dev-secret-key"
            },
            json={
                "repository_id": "repo-context-ai",
                "message": "Hello",
                "thread_id": "request-id-test",
            },
        )

        assert response.status_code == 200
        assert response.headers.get("X-Request-ID")
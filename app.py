"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from api.routes import router
from core.app_state import AppState
from core.config import get_settings
from core.logging_config import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown."""

    settings = get_settings()

    configure_logging(settings.log_level)

    settings.prepare_directories()
    settings.validate_runtime()

    app_state = AppState(settings)
    app_state.startup()

    app.state.repo_context = app_state

    try:
        yield
    finally:
        app_state.shutdown()


app = FastAPI(
    title="RepoContext AI",
    description=(
        "Repository-aware GenAI assistant "
        "using RAG and LangGraph."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)


@app.get("/health")
def health():
    """Liveness check endpoint."""

    return {
        "status": "ok",
        "service": "repo-context-ai",
    }


@app.get("/health/ready")
def ready(request: Request):
    """Readiness check endpoint.

    Returns 503 if the application state has not been fully initialized.
    Useful for container orchestrators that distinguish starting from ready.
    """

    app_state = getattr(request.app.state, "repo_context", None)

    if app_state is None or app_state.checkpointer is None:
        raise HTTPException(
            status_code=503,
            detail="Service is not ready",
        )

    return {
        "status": "ready",
        "service": "repo-context-ai",
    }
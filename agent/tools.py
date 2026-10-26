"""Repository tools available to the LangGraph agent."""

import logging
import time
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.tools import tool

from core.config import Settings, get_settings
from rag import create_repository_retriever

logger = logging.getLogger(__name__)

MAX_INSPECT_BYTES = 100_000


def create_repository_tools(
    repository_path: str | Path,
    repository_id: str,
    settings: Settings | None = None,
):
    """Create tools scoped to a single repository."""

    settings = settings or get_settings()

    root = Path(repository_path).resolve()

    if not root.exists() or not root.is_dir():
        raise ValueError(
            "Repository path must be an existing directory"
        )

    retriever = create_repository_retriever(
        repository_id=repository_id,
        settings=settings,
    )

    @tool
    def search_repository(query: str) -> list[Document]:
        """Search the repository for relevant code."""

        start_time = time.perf_counter()

        logger.info(
            "Repository search started | repository_id=%s",
            repository_id,
        )

        try:
            documents = retriever.invoke(query)

            duration_ms = (
                time.perf_counter() - start_time
            ) * 1000

            logger.info(
                "Repository search completed | "
                "repository_id=%s results=%d "
                "duration_ms=%.2f",
                repository_id,
                len(documents),
                duration_ms,
            )

            return documents

        except Exception:
            logger.exception(
                "Repository search failed | "
                "repository_id=%s",
                repository_id,
            )
            raise

    @tool
    def inspect_file(relative_path: str) -> str:
        """Inspect a repository file using its relative path."""

        start_time = time.perf_counter()

        logger.info(
            "File inspection started | "
            "repository_id=%s",
            repository_id,
        )

        try:
            requested_path = (
                root / relative_path
            ).resolve()

            # Prevent path traversal outside the repository.
            try:
                requested_path.relative_to(root)
            except ValueError as exc:
                raise ValueError(
                    "File path must remain inside the repository"
                ) from exc

            if not requested_path.exists():
                raise FileNotFoundError(
                    f"File not found: {relative_path}"
                )

            if not requested_path.is_file():
                raise ValueError(
                    f"Path is not a file: {relative_path}"
                )

            file_size = requested_path.stat().st_size

            if file_size > MAX_INSPECT_BYTES:
                raise ValueError(
                    "File exceeds the maximum inspection size "
                    f"of {MAX_INSPECT_BYTES} bytes"
                )

            content = requested_path.read_text(
                encoding="utf-8"
            )

            duration_ms = (
                time.perf_counter() - start_time
            ) * 1000

            logger.info(
                "File inspection completed | "
                "repository_id=%s duration_ms=%.2f",
                repository_id,
                duration_ms,
            )

            return content

        except Exception:
            logger.exception(
                "File inspection failed | "
                "repository_id=%s",
                repository_id,
            )
            raise

    return [
        search_repository,
        inspect_file,
    ]
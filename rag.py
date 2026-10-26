
"""Repository-aware RAG pipeline for RepoContext AI.

This module loads source files, creates code-aware chunks, indexes them
in Chroma, retrieves relevant chunks with metadata filtering, and
exposes an LCEL question-answering chain.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any, Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.model_factory import (
    create_embedding_model,
    create_chat_model_with_fallbacks,
)

from core.config import Settings, get_settings
from core.exceptions import IndexingError, RetrievalError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# 1. Source file configuration
# ---------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".py", ".pyi",
    ".js", ".jsx", ".ts", ".tsx",
    ".java", ".kt", ".go", ".rs",
    ".c", ".h", ".cpp", ".hpp",
    ".cs", ".rb", ".php",
    ".swift", ".scala", ".sql",
    ".html", ".css", ".scss",
    ".json", ".yaml", ".yml",
    ".toml", ".md", ".rst",
    ".sh", ".dockerfile",
}

IGNORED_DIRECTORIES = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    "dist",
    "build",
    "vendor",
    ".idea",
    ".vscode",
    "data",
    "coverage",
    ".next",
    "tests",
    "evaluation",
}

IGNORED_FILES = {
    "evaluation/results.json",
    "evaluation/dataset.py",
    "evaluation/runner.py",
    "evaluation/metrics.py",
}

MAX_FILE_SIZE_BYTES = 1_000_000


def _is_supported_file(path: Path) -> bool:
    """Return whether a file should be included in repository indexing."""

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        return False

    if not path.is_file():
        return False

    if path.as_posix().endswith("/lock") or path.name.lower() == "lock":
        return False

    return True


# ---------------------------------------------------------------------
# 2. Load repository files as LangChain Documents
# ---------------------------------------------------------------------

def load_repository_documents(
    repository_path: str | Path,
    repository_id: str,
) -> list[Document]:
    """Load supported text files from a local repository.

    Each Document contains source text and metadata identifying its
    repository, relative file path, language, and content hash.

    Args:
        repository_path: Root directory of a previously acquired repo.
        repository_id: Stable identifier used to isolate repositories.

    Returns:
        A list of LangChain Document objects.

    Raises:
        IndexingError: If the repository is missing or cannot be read.
    """
    root = Path(repository_path).resolve()

    if not root.exists() or not root.is_dir():
        raise IndexingError(
            f"Repository directory does not exist: {root}"
        )

    documents: list[Document] = []

    try:
        for path in root.rglob("*"):
            # Do not read files inside generated or dependency folders.
            relative_path = path.relative_to(root)
            if relative_path.as_posix() in IGNORED_FILES:
                continue
            if any(
                part in IGNORED_DIRECTORIES
                for part in relative_path.parts
            ):
                continue

            if not _is_supported_file(path):
                continue

            try:
                if path.stat().st_size > MAX_FILE_SIZE_BYTES:
                    logger.warning(
                        "Skipping oversized file: %s", relative_path
                    )
                    continue

                # Reject binary or non-UTF-8 source files.
                content = path.read_text(
                    encoding="utf-8",
                    errors="strict",
                )

                if not content.strip():
                    continue

                content_hash = hashlib.sha256(
                    content.encode("utf-8")
                ).hexdigest()

                documents.append(
                    Document(
                        page_content=content,
                        metadata={
                            "repository_id": repository_id,
                            "file_path": relative_path.as_posix(),
                            "extension": path.suffix.lower(),
                            "content_hash": content_hash,
                            "source": relative_path.as_posix(),
                        },
                    )
                )

            except (OSError, UnicodeDecodeError) as exc:
                logger.warning(
                    "Skipping unreadable file %s: %s",
                    relative_path,
                    exc,
                )

    except OSError as exc:
        raise IndexingError(
            f"Failed to scan repository: {exc}"
        ) from exc

    if not documents:
        logger.warning(
            "No supported source files found in %s", root
        )

    logger.info(
        "Loaded %d documents for repository %s",
        len(documents),
        repository_id,
    )

    return documents


# ---------------------------------------------------------------------
# 3. Code-aware text splitting
# ---------------------------------------------------------------------

def create_code_splitter(
    settings: Settings | None = None,
) -> RecursiveCharacterTextSplitter:
    """Create a recursive text splitter with code-aware boundaries.

    The separators prefer logical source-code boundaries before
    falling back to smaller character-level separators.
    """
    settings = settings or get_settings()

    separators = [
        "\nclass ",
        "\ndef ",
        "\nasync def ",
        "\nfunction ",
        "\nexport ",
        "\n\n",
        "\n",
        " ",
        "",
    ]

    return RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=separators,
        keep_separator="start",
        add_start_index=True,
    )


def split_documents(
    documents: Iterable[Document],
    settings: Settings | None = None,
) -> list[Document]:
    """Split source documents into overlapping chunks.

    Chunk metadata is copied from the original Document, with an
    additional start_index indicating the approximate source offset.
    """
    splitter = create_code_splitter(settings)
    chunks = splitter.split_documents(list(documents))

    logger.info(
        "Split documents into %d chunks", len(chunks)
    )

    return chunks


# ---------------------------------------------------------------------
# 4. Embeddings and Chroma vector store
# ---------------------------------------------------------------------

def create_embeddings(
    settings: Settings | None = None,
) -> Embeddings:
    """Create the configured embedding provider."""
    settings = settings or get_settings()

    try:
        return create_embedding_model(settings=settings)
    except Exception as exc:
        raise IndexingError(
            f"Could not initialize embeddings: {exc}"
        ) from exc


def create_vector_store(
    settings: Settings | None = None,
) -> Chroma:
    """Create or connect to the persistent Chroma collection."""
    settings = settings or get_settings()
    settings.prepare_directories()

    return Chroma(
        collection_name=settings.collection_name,
        embedding_function=create_embeddings(settings),
        persist_directory=str(settings.vector_store_path),
    )


def _chunk_id(document: Document) -> str:
    """Generate a deterministic ID for a code chunk."""
    identity = "|".join(
        [
            str(document.metadata["repository_id"]),
            str(document.metadata["file_path"]),
            str(document.metadata.get("start_index", 0)),
            str(document.metadata["content_hash"]),
        ]
    )

    return hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()



def index_repository(
    repository_path: str | Path,
    repository_id: str,
    settings: Settings | None = None,
) -> int:
    """Index a repository and remove obsolete chunks after upserting."""
    settings = settings or get_settings()

    if not repository_id.strip():
        raise IndexingError("repository_id cannot be empty")

    try:
        documents = load_repository_documents(
            repository_path,
            repository_id,
        )
        chunks = split_documents(documents, settings)

        vector_store = create_vector_store(settings)

        # Find the chunks already stored for this repository.
        existing = vector_store.get(
            where={"repository_id": repository_id},
            include=["metadatas"],
        )
        old_ids = set(existing.get("ids", []))

        # Generate deterministic IDs for the current chunks.
        ids = [_chunk_id(chunk) for chunk in chunks]
        current_ids = set(ids)

        # Add or update current chunks first.
        if chunks:
            vector_store.add_documents(
                documents=chunks,
                ids=ids,
            )

        # Delete chunks that are no longer part of the repository.
        stale_ids = old_ids - current_ids
        if stale_ids:
            vector_store.delete(ids=list(stale_ids))

        logger.info(
            "Indexed %d chunks and removed %d stale chunks for %s",
            len(chunks),
            len(stale_ids),
            repository_id,
        )

        return len(chunks)

    except IndexingError:
        raise
    except Exception as exc:
        logger.exception(
            "Repository indexing failed for %s",
            repository_id,
        )
        raise IndexingError(
            f"Failed to index repository: {exc}"
        ) from exc


# ---------------------------------------------------------------------
# 5. Metadata-filtered retrieval
# ---------------------------------------------------------------------

def create_repository_retriever(
    repository_id: str,
    settings: Settings | None = None,
    file_path: str | None = None,
):
    """Create a similarity retriever scoped to one repository.

    An optional relative file path narrows retrieval to one file.
    """
    settings = settings or get_settings()

    if not repository_id.strip():
        raise RetrievalError("repository_id cannot be empty")

    vector_store = create_vector_store(settings)

    conditions: list[dict[str, Any]] = [
        {"repository_id": repository_id}
    ]

    if file_path:
        conditions.append({"file_path": file_path})

    metadata_filter: dict[str, Any]
    if len(conditions) == 1:
        metadata_filter = conditions[0]
    else:
        metadata_filter = {"$and": conditions}

    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": settings.retriever_top_k,
            "filter": metadata_filter,
        },
    )


# ---------------------------------------------------------------------
# 6. LCEL retrieval-augmented generation chain
# ---------------------------------------------------------------------

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are RepoContext AI, a codebase analysis assistant.

Answer the user's question using the provided repository context.
Do not invent source files, function names, or code behavior.
If the context does not contain enough evidence, say so clearly.

When possible:
- Identify the relevant file paths.
- Explain the code behavior.
- Distinguish observed facts from suggestions.
- Include concise code excerpts only when useful.

Repository context:
{context}""",
        ),
        (
            "human",
            "{question}",
        ),
    ]
)


def format_documents(
    documents: list[Document],
) -> str:
    """Format retrieved documents with source labels."""
    sections: list[str] = []

    for document in documents:
        file_path = document.metadata.get(
            "file_path", "unknown"
        )
        start_index = document.metadata.get(
            "start_index", "unknown"
        )

        sections.append(
            f"Source: {file_path} "
            f"(start index: {start_index})\n"
            f"{document.page_content}"
        )

    return "\n\n---\n\n".join(sections)


def create_rag_chain(
    repository_id: str,
    settings: Settings | None = None,
    file_path: str | None = None,
):
    """Build a repository-scoped LCEL RAG chain.

    Input:
        {"question": "..."}
    Output:
        Plain-text answer from the configured chat model.
    """
    settings = settings or get_settings()

    retriever = create_repository_retriever(
        repository_id=repository_id,
        settings=settings,
        file_path=file_path,
    )

    resilient_model = create_chat_model_with_fallbacks(
        settings=settings
    )

    # RunnableParallel retrieves context while preserving the question.
    retrieval_stage = RunnableParallel(
        {
            "context": (
                RunnableLambda(
                    lambda inputs: inputs["question"]
                )
                | retriever
                | RunnableLambda(format_documents)
            ),
            "question": RunnablePassthrough(),
        }
    )

    # LCEL: retrieval -> prompt -> model -> output parser.
    return (
        retrieval_stage
        | RAG_PROMPT
        | resilient_model
        | StrOutputParser()
    )


def ask_repository(
    question: str,
    repository_id: str,
    settings: Settings | None = None,
    file_path: str | None = None,
) -> str:
    """Answer a repository question using the RAG chain."""
    if not question.strip():
        raise RetrievalError("Question cannot be empty")

    try:
        chain = create_rag_chain(
            repository_id=repository_id,
            settings=settings,
            file_path=file_path,
        )

        return chain.invoke({"question": question})

    except RetrievalError:
        raise
    except Exception as exc:
        logger.exception(
            "RAG query failed for repository %s",
            repository_id,
        )
        raise RetrievalError(
            f"Failed to answer repository question: {exc}"
        ) from exc
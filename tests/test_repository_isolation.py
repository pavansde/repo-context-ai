
from pathlib import Path

from core.config import get_settings
from rag import (
    create_repository_retriever,
    create_vector_store,
    index_repository,
)


def test_repository_isolation(tmp_path: Path):
    """Ensure retrieval from one repository excludes another."""
    settings = get_settings().model_copy(
        update={
            "vector_store_path": tmp_path / "chroma",
            "repository_cache_path": tmp_path / "repositories",
            "collection_name": "isolation_test",
        }
    )

    repo_a = tmp_path / "repo_a"
    repo_b = tmp_path / "repo_b"
    repo_a.mkdir()
    repo_b.mkdir()

    (repo_a / "database.py").write_text(
        """
        # Repository A
        def connect_database():
            return "DATABASE_A_SECRET"
        """,
        encoding="utf-8",
    )

    (repo_b / "database.py").write_text(
        """
        # Repository B
        def connect_database():
            return "DATABASE_B_SECRET"
        """,
        encoding="utf-8",
    )

    repository_a_id = "isolation-repo-a"
    repository_b_id = "isolation-repo-b"

    try:
        count_a = index_repository(
            repo_a, repository_a_id, settings
        )
        count_b = index_repository(
            repo_b, repository_b_id, settings
        )

        assert count_a > 0
        assert count_b > 0

        retriever_a = create_repository_retriever(
            repository_a_id, settings
        )
        retriever_b = create_repository_retriever(
            repository_b_id, settings
        )

        results_a = retriever_a.invoke("connect database")
        results_b = retriever_b.invoke("connect database")

        assert results_a, "Repository A returned no results"
        assert results_b, "Repository B returned no results"

        # Every retrieved chunk must belong to its requested repo.
        assert all(
            doc.metadata["repository_id"] == repository_a_id
            for doc in results_a
        )
        assert all(
            doc.metadata["repository_id"] == repository_b_id
            for doc in results_b
        )

        content_a = "\n".join(
            doc.page_content for doc in results_a
        )
        content_b = "\n".join(
            doc.page_content for doc in results_b
        )

        assert "DATABASE_A_SECRET" in content_a
        assert "DATABASE_B_SECRET" not in content_a

        assert "DATABASE_B_SECRET" in content_b
        assert "DATABASE_A_SECRET" not in content_b

    finally:
        # Keep this test isolated even if an assertion fails.
        vector_store = create_vector_store(settings)
        vector_store.delete(
            where={"repository_id": repository_a_id}
        )
        vector_store.delete(
            where={"repository_id": repository_b_id}
        )

from pathlib import Path
from tempfile import TemporaryDirectory

from core.config import get_settings
from rag import index_repository, create_repository_retriever


def main():
    settings = get_settings()
    repository_id = "reindex-test"

    with TemporaryDirectory() as temp_dir:
        repo = Path(temp_dir)
        source = repo / "database.py"

        source.write_text(
            'DATABASE_URL = "sqlite:///old.db"\n',
            encoding="utf-8",
        )

        first_count = index_repository(
            repo, repository_id, settings
        )
        assert first_count > 0

        source.write_text(
            'DATABASE_URL = "sqlite:///new.db"\n',
            encoding="utf-8",
        )

        second_count = index_repository(
            repo, repository_id, settings
        )
        assert second_count > 0

        retriever = create_repository_retriever(
            repository_id, settings
        )
        results = retriever.invoke("database URL")

        combined = "\n".join(
            doc.page_content for doc in results
        )

        assert "new.db" in combined
        assert "old.db" not in combined

        # Deleting the source file and re-indexing should
        # remove the repository's old chunks.
        source.unlink()
        empty_count = index_repository(
            repo, repository_id, settings
        )
        assert empty_count == 0

        results_after_delete = retriever.invoke("database URL")
        assert not results_after_delete

    print("Re-indexing tests passed.")


if __name__ == "__main__":
    main()
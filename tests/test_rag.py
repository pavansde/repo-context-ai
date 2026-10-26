from pathlib import Path
from tempfile import TemporaryDirectory

from core.config import get_settings
from rag import index_repository, ask_repository


def main():
    settings = get_settings()

    with TemporaryDirectory() as temp_dir:
        repo_path = Path(temp_dir)

        (repo_path / "database.py").write_text(
            """
DATABASE_URL = "sqlite:///app.db"

def get_database_url():
    return DATABASE_URL
""",
            encoding="utf-8",
        )

        repository_id = "repo-context-ai"

        chunks = index_repository(
            repository_path=repo_path,
            repository_id=repository_id,
            settings=settings,
        )

        print(f"Indexed chunks: {chunks}")

        answer = ask_repository(
            question="Where is the database URL configured?",
            repository_id=repository_id,
            settings=settings,
        )

        print("\nAnswer:")
        print(answer)


if __name__ == "__main__":
    main()
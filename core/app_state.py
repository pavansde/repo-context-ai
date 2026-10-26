import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph import create_repository_agent
from core.config import Settings


class AppState:
    """Owns application-wide resources."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.connection: sqlite3.Connection | None = None
        self.checkpointer: SqliteSaver | None = None

    def startup(self) -> None:
        """Initialize application resources."""

        checkpoint_db = Path(
            self.settings.checkpoint_db_path
        )

        checkpoint_db.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.connection = sqlite3.connect(
            str(checkpoint_db),
            check_same_thread=False,
        )

        self.checkpointer = SqliteSaver(
            self.connection
        )

    def shutdown(self) -> None:
        """Release application resources."""

        if self.connection is not None:
            self.connection.close()

        self.connection = None
        self.checkpointer = None

    def create_agent(
        self,
        repository_path: Path,
        repository_id: str,
    ):
        """Create an agent using the application checkpointer."""

        if self.checkpointer is None:
            raise RuntimeError(
                "Application state has not been started."
            )

        return create_repository_agent(
            repository_path=repository_path,
            repository_id=repository_id,
            settings=self.settings,
            checkpointer=self.checkpointer,
        )
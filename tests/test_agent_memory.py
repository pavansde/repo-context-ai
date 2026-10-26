from pathlib import Path
import sqlite3

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from agent.graph import create_repository_agent
from core.config import get_settings
from rag import index_repository


def test_sqlite_agent_persistence(tmp_path: Path):

    settings = get_settings().model_copy(
        update={
            "vector_store_path": tmp_path / "chroma",
            "repository_cache_path": tmp_path / "repositories",
            "collection_name": "sqlite_memory_test",
            "checkpoint_db_path": tmp_path / "checkpoints.db",
        }
    )

    repo = tmp_path / "sample_repo"
    repo.mkdir()

    (repo / "auth.py").write_text(
        """
def authenticate_user(username, password):
    if not username or not password:
        return False

    return check_credentials(username, password)
""",
        encoding="utf-8",
    )

    repository_id = "sqlite-memory-test"

    index_repository(
        repository_path=repo,
        repository_id=repository_id,
        settings=settings,
    )

    thread_config = {
        "configurable": {
            "thread_id": "sqlite-test-thread"
        }
    }

    # ---------------------------------------------------------
    # Shared SQLite checkpointer
    # ---------------------------------------------------------

    connection = sqlite3.connect(
        str(settings.checkpoint_db_path),
        check_same_thread=False,
    )

    checkpointer = SqliteSaver(connection)

    # ---------------------------------------------------------
    # Agent instance 1
    # ---------------------------------------------------------

    agent_1 = create_repository_agent(
        repository_path=repo,
        repository_id=repository_id,
        settings=settings,
        checkpointer=checkpointer,
    )

    first_result = agent_1.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Find authenticate_user and tell me "
                        "which file contains it."
                    )
                )
            ]
        },
        config=thread_config,
    )

    first_answer = first_result["messages"][-1].content

    assert "auth.py" in first_answer.lower()

    # ---------------------------------------------------------
    # Agent instance 2
    #
    # Same SQLite checkpointer + same thread ID.
    # ---------------------------------------------------------

    agent_2 = create_repository_agent(
        repository_path=repo,
        repository_id=repository_id,
        settings=settings,
        checkpointer=checkpointer,
    )

    second_result = agent_2.invoke(
        {
            "messages": [
                HumanMessage(
                    content="What function were we discussing?"
                )
            ]
        },
        config=thread_config,
    )

    second_answer = second_result["messages"][-1].content

    assert "authenticate_user" in second_answer.lower()

    connection.close()

from pathlib import Path

from langchain_core.messages import HumanMessage

from core.config import get_settings
from rag import index_repository
from agent.graph import create_repository_agent


def test_repository_agent(tmp_path: Path):
    settings = get_settings().model_copy(
        update={
            "vector_store_path": tmp_path / "chroma",
            "repository_cache_path": tmp_path / "repositories",
            "collection_name": "agent_test",
        }
    )

    repo = tmp_path / "sample_repo"
    repo.mkdir()

    (repo / "auth.py").write_text(
        '''
def authenticate_user(username, password):
    """Authenticate a user using the account database."""
    if not username or not password:
        return False
    return check_credentials(username, password)
''',
        encoding="utf-8",
    )

    repository_id = "agent-test-repo"

    index_repository(
        repository_path=repo,
        repository_id=repository_id,
        settings=settings,
    )

    agent = create_repository_agent(
        repository_path=repo,
        repository_id=repository_id,
        settings=settings,
    )

    result = agent.invoke(
        {
            "messages": [
                HumanMessage(
                    content="How does authenticate_user work? "
                            "Mention the source file."
                )
            ]
        },
        config={"configurable": {"thread_id": "agent-test-1"}},
    )

    answer = result["messages"][-1].content
    print("\nAgent answer:\n", answer)

    assert isinstance(answer, str)
    assert answer.strip()
    assert "auth.py" in answer.lower()


if __name__ == "__main__":
    test_repository_agent(Path("data/agent_test").resolve())
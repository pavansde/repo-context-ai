from langchain_core.tools import tool
from core.model_factory import create_chat_model
from core.config import get_settings


@tool
def search_repository(query: str) -> str:
    """Search the repository for code matching a query."""
    return f"Found authenticate_user in auth.py for query: {query}"


def test_ollama_native_tool_calling():
    settings = get_settings()

    model = create_chat_model(settings=settings).bind_tools(
    [search_repository]
)

    response = model.invoke(
        "Use search_repository to find authenticate_user."
    )

    print("Content:", response.content)
    print("Tool calls:", response.tool_calls)

    assert response.tool_calls, (
        "The model returned no structured tool calls. "
        "Check Ollama model and LangChain integration support."
    )
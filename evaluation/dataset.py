"""Ground-truth dataset for RepoContext AI evaluation."""

from typing import TypedDict


class EvaluationSample(TypedDict):
    question: str
    ground_truth: str
    expected_sources: list[str]


EVALUATION_DATASET: list[EvaluationSample] = [
    {
        "question": (
            "Where is the repository indexing pipeline implemented?"
        ),
        "ground_truth": (
            "The repository indexing pipeline is implemented "
            "in rag.py."
        ),
        "expected_sources": ["rag.py"],
    },
    {
        "question": (
            "Which file creates the repository tools used by the agent?"
        ),
        "ground_truth": (
            "The repository tools are created in agent/tools.py."
        ),
        "expected_sources": ["agent/tools.py"],
    },
    {
        "question": (
            "Where is the LangGraph repository agent constructed?"
        ),
        "ground_truth": (
            "The repository agent graph is constructed in agent/graph.py."
        ),
        "expected_sources": ["agent/graph.py"],
    },
    {
        "question": (
            "Where is the FastAPI application and its lifespan configured?"
        ),
        "ground_truth": (
            "The FastAPI application and lifespan are configured in app.py."
        ),
        "expected_sources": ["app.py"],
    },
    {
        "question": (
            "Which component validates API Bearer keys?"
        ),
        "ground_truth": (
            "The API key validation dependency is implemented in api/auth.py."
        ),
        "expected_sources": ["api/auth.py"],
    },
    {
        "question": (
            "What is the maximum file size allowed by inspect_file?"
        ),
        "ground_truth": (
            "The inspect_file tool limits inspection to 100,000 bytes."
        ),
        "expected_sources": ["agent/tools.py"],
    },
    {
        "question": (
            "Which database-backed checkpointer persists agent conversation state?"
        ),
        "ground_truth": (
            "The application uses LangGraph's SQLite checkpointer, "
            "SqliteSaver, for persistent conversation state."
        ),
        "expected_sources": ["core/app_state.py"],
    },
    {
        "question": (
            "What is the company's production revenue for the current quarter?"
        ),
        "ground_truth": (
            "This information cannot be determined from the repository "
            "unless relevant revenue data is present in its files."
        ),
        "expected_sources": [],
    },
]
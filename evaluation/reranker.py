"""LLM-based reranking for retrieval evaluation."""

import json
import logging

from core.config import Settings, get_settings
from core.model_factory import create_chat_model_with_fallbacks

logger = logging.getLogger(__name__)


def rerank_sources(
    question: str,
    sources: list[str],
    settings: Settings | None = None,
) -> list[str]:
    """Rank candidate source paths by relevance to a question."""

    if not sources:
        return []

    settings = settings or get_settings()
    model = create_chat_model_with_fallbacks(settings)

    candidates = list(dict.fromkeys(sources))

    prompt = f"""
Rank the following repository files by how directly they answer
the question. Use only the supplied file paths and do not invent files.

Question:
{question}

Candidate files:
{json.dumps(candidates)}

Return only a JSON array containing every candidate file exactly once,
ordered from most relevant to least relevant.
"""

    response = model.invoke(prompt)
    content = response.content

    if not isinstance(content, str):
        logger.warning("Reranker returned non-text output; keeping original order")
        return candidates

    try:
        ranked = json.loads(content)
    except json.JSONDecodeError:
        logger.warning("Reranker returned invalid JSON; keeping original order")
        return candidates

    # Validate the response is a valid permutation of candidate paths.
    if (
        not isinstance(ranked, list)
        or not all(isinstance(item, str) for item in ranked)
        or len(ranked) != len(candidates)
        or set(ranked) != set(candidates)
    ):
        logger.warning(
            "Reranker returned invalid candidates; keeping original order"
        )
        return candidates

    return ranked
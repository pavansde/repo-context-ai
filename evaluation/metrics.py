"""Retrieval evaluation metrics."""

from collections.abc import Sequence

def precision_at_k(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
    k: int,
) -> float:
    """Fraction of the top-k result slots that are relevant."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    expected = set(expected_sources)
    top_k = list(retrieved_sources[:k])

    relevant_count = sum(
        source in expected
        for source in top_k
    )

    return relevant_count / k


def recall_at_k(
    retrieved_sources: Sequence[str],
    expected_sources: Sequence[str],
    k: int,
) -> float:
    """Fraction of expected sources found in the top-k results."""

    if k <= 0:
        raise ValueError("k must be greater than zero")

    expected = set(expected_sources)

    # No relevant sources are expected for abstention questions.
    if not expected:
        return 0.0

    top_k = set(retrieved_sources[:k])

    relevant_count = len(top_k & expected)

    return relevant_count / len(expected)
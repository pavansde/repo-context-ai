import pytest

from evaluation.metrics import precision_at_k, recall_at_k


def test_precision_at_k():
    retrieved = ["rag.py", "agent/tools.py", "README.md"]
    expected = ["rag.py", "agent/graph.py"]

    assert precision_at_k(retrieved, expected, 3) == pytest.approx(
        1 / 3
    )


def test_recall_at_k():
    retrieved = ["rag.py", "agent/tools.py", "agent/graph.py"]
    expected = ["rag.py", "agent/graph.py"]

    assert recall_at_k(retrieved, expected, 3) == 1.0


def test_recall_at_k_partial():
    retrieved = ["rag.py", "README.md"]
    expected = ["rag.py", "agent/graph.py"]

    assert recall_at_k(retrieved, expected, 2) == 0.5


def test_empty_retrieval():
    assert precision_at_k([], ["rag.py"], 5) == 0.0
    assert recall_at_k([], ["rag.py"], 5) == 0.0


def test_no_expected_sources():
    assert recall_at_k(["rag.py"], [], 5) == 0.0


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k(k):
    with pytest.raises(ValueError):
        precision_at_k(["rag.py"], ["rag.py"], k)

    with pytest.raises(ValueError):
        recall_at_k(["rag.py"], ["rag.py"], k)
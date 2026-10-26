"""Tests for the evaluation dataset."""

from evaluation.dataset import EVALUATION_DATASET


def test_evaluation_dataset_is_not_empty():
    assert EVALUATION_DATASET


def test_each_sample_has_required_fields():
    for sample in EVALUATION_DATASET:
        assert sample["question"].strip()
        assert sample["ground_truth"].strip()
        assert isinstance(sample["expected_sources"], list)


def test_evaluation_questions_are_unique():
    questions = [
        sample["question"]
        for sample in EVALUATION_DATASET
    ]

    assert len(questions) == len(set(questions))


def test_dataset_contains_an_abstention_case():
    assert any(
        not sample["expected_sources"]
        for sample in EVALUATION_DATASET
    )
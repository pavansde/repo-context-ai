from pathlib import Path

import pytest

from core.config import get_settings
from core.exceptions import ConfigurationError
from core.repository_registry import get_repository_registry


def test_repository_registry_resolves_registered_repository(
    tmp_path: Path,
):
    repository = tmp_path / "sample_repo"
    repository.mkdir()

    settings = get_settings().model_copy(
        update={
            "repositories": (
                f"sample-repo:{repository}"
            ),
        }
    )

    registry = get_repository_registry(settings)

    resolved = registry.resolve("sample-repo")

    assert resolved == repository.resolve()


def test_repository_registry_rejects_unknown_repository(
    tmp_path: Path,
):
    settings = get_settings().model_copy(
        update={
            "repositories": "",
        }
    )

    registry = get_repository_registry(settings)

    with pytest.raises(
        ConfigurationError,
        match="Unknown repository",
    ):
        registry.resolve("does-not-exist")


def test_repository_registry_rejects_missing_repository(
    tmp_path: Path,
):
    missing_repository = tmp_path / "missing"

    settings = get_settings().model_copy(
        update={
            "repositories": (
                f"missing-repo:{missing_repository}"
            ),
        }
    )

    registry = get_repository_registry(settings)

    with pytest.raises(
        ConfigurationError,
        match="Repository does not exist",
    ):
        registry.resolve("missing-repo")
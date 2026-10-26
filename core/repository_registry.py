"""Trusted repository registry."""

from pathlib import Path

from core.config import Settings
from core.exceptions import ConfigurationError


class RepositoryRegistry:
    """Maps trusted repository IDs to server-side paths."""

    def __init__(
        self,
        repositories: dict[str, Path],
    ):
        self._repositories = repositories

    def contains(
        self,
        repository_id: str,
    ) -> bool:
        """Return whether a repository is registered."""

        return repository_id in self._repositories

    def resolve(
        self,
        repository_id: str,
    ) -> Path:
        """Resolve a registered repository path."""

        repository_path = self._repositories.get(
            repository_id
        )

        if repository_path is None:
            raise ConfigurationError(
                f"Unknown repository: {repository_id}"
            )

        if not repository_path.exists():
            raise ConfigurationError(
                f"Repository does not exist: {repository_id}"
            )

        if not repository_path.is_dir():
            raise ConfigurationError(
                f"Repository path is not a directory: "
                f"{repository_id}"
            )

        return repository_path


def get_repository_registry(
    settings: Settings,
) -> RepositoryRegistry:
    """Build the repository registry from settings."""

    repositories: dict[str, Path] = {}

    if not settings.repositories:
        return RepositoryRegistry(repositories)

    for entry in settings.repositories.split(","):
        entry = entry.strip()

        if ":" not in entry:
            raise ConfigurationError(
                f"Invalid REPOSITORIES entry (expected 'id:path'): {entry!r}"
            )

        repository_id, repository_path = entry.split(
            ":",
            1,
        )

        if not repository_id.strip():
            raise ConfigurationError(
                f"Repository ID cannot be empty in entry: {entry!r}"
            )

        repositories[repository_id.strip()] = (
            Path(
                repository_path.strip()
            ).resolve()
        )

    return RepositoryRegistry(repositories)
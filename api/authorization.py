from fastapi import HTTPException

from core.config import Settings
from core.repository_registry import RepositoryRegistry


def authorize_repository(
    repository_id: str,
    registry: RepositoryRegistry,
) -> None:
    """Verify that the repository is registered and accessible."""

    if not registry.contains(repository_id):
        raise HTTPException(
            status_code=403,
            detail="Access to this repository is not authorized",
        )
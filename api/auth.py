"""API authentication."""

import secrets

from fastapi import Header, HTTPException

from core.config import get_settings


def require_api_key(
    authorization: str | None = Header(default=None),
):
    """Validate the Bearer API key."""

    settings = get_settings()

    expected_key = settings.api_key.get_secret_value()

    if not expected_key:
        raise HTTPException(
            status_code=500,
            detail="API authentication is not configured",
        )

    if not authorization:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    scheme, _, token = authorization.partition(" ")

    if scheme.lower() != "bearer" or not token:
        raise HTTPException(
            status_code=401,
            detail="Invalid Authorization header",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    if not secrets.compare_digest(
        token,
        expected_key,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={
                "WWW-Authenticate": "Bearer"
            },
        )

    return True
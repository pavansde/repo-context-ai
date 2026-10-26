"""Pytest test suite configuration and environment setup."""

import os

# Ensure safe defaults are set for test runs even if .env is absent (e.g. CI runners)
os.environ.setdefault("API_KEY", "dev-secret-key")
os.environ.setdefault("REPOSITORIES", "repo-context-ai:.")
os.environ.setdefault("APP_ENV", "testing")


"""Application-specific exception definitions."""


class RepoContextError(Exception):
    """Base exception for application errors."""


class RepositoryError(RepoContextError):
    """Raised when repository access or processing fails."""


class RepositoryTooLargeError(RepositoryError):
    """Raised when a repository exceeds configured limits."""


class IndexingError(RepoContextError):
    """Raised when code ingestion or indexing fails."""


class RetrievalError(RepoContextError):
    """Raised when relevant code cannot be retrieved."""


class AgentExecutionError(RepoContextError):
    """Raised when agent execution fails."""


class ConfigurationError(RepoContextError):
    """Raised when application configuration is invalid."""
"""Custom exception hierarchy for the project.

Follow the project's error handling standards: specific exceptions, no bare excepts,
and clear exception types for production-grade error handling.
"""
from typing import Optional


class ProjectException(Exception):
    """Base exception for the project."""
    def __init__(self, message: str, *, context: Optional[dict] = None) -> None:
        super().__init__(message)
        self.context = context or {}


class DataLoadingError(ProjectException):
    pass


class RetrievalError(ProjectException):
    pass


class EmbeddingError(ProjectException):
    pass


class LLMError(ProjectException):
    pass


class DatabaseError(ProjectException):
    pass


class APIError(ProjectException):
    pass


class ValidationError(ProjectException):
    pass


# RAG / domain specific
class VectorStoreError(ProjectException):
    pass


class RerankerError(ProjectException):
    pass


class QueryRewriteError(ProjectException):
    pass


class LLMGenerationError(ProjectException):
    pass

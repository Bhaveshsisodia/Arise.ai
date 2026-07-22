"""Custom exception hierarchy for the project."""
from typing import Any, Optional


class ProjectException(Exception):
    """Base exception carrying an API-friendly status code and error code."""

    error_code = "PROJECT_ERROR"
    status_code = 500

    def __init__(
        self,
        message: str,
        *,
        context: Optional[dict[str, Any]] = None,
        error_code: Optional[str] = None,
        status_code: Optional[int] = None,
    ) -> None:
        super().__init__(message)
        self.context = context or {}
        self.error_code = error_code or self.error_code
        self.status_code = status_code or self.status_code

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.error_code,
            "message": str(self),
            "context": self.context,
        }


class DataLoadingError(ProjectException):
    error_code = "DATA_LOADING_ERROR"
    status_code = 500


class RetrievalError(ProjectException):
    error_code = "RETRIEVAL_ERROR"
    status_code = 500


class EmbeddingError(ProjectException):
    error_code = "EMBEDDING_ERROR"
    status_code = 500


class LLMError(ProjectException):
    error_code = "LLM_ERROR"
    status_code = 502


class DatabaseError(ProjectException):
    error_code = "DATABASE_ERROR"
    status_code = 503


class APIError(ProjectException):
    error_code = "API_ERROR"
    status_code = 500


class ValidationError(ProjectException):
    error_code = "VALIDATION_ERROR"
    status_code = 400


# RAG / domain specific
class VectorStoreError(ProjectException):
    error_code = "VECTOR_STORE_ERROR"
    status_code = 503


class RerankerError(ProjectException):
    error_code = "RERANKER_ERROR"
    status_code = 500


class QueryRewriteError(ProjectException):
    error_code = "QUERY_REWRITE_ERROR"
    status_code = 500


class LLMGenerationError(ProjectException):
    error_code = "LLM_GENERATION_ERROR"
    status_code = 502


class GuardrailViolationError(ProjectException):
    error_code = "GUARDRAIL_VIOLATION"
    status_code = 400

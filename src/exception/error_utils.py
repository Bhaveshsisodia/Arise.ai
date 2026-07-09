"""Helpers for logging and re-raising project exceptions with context."""
import traceback
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type

from .custom_exception import ProjectException


logger = logging.getLogger("arise.errors")


def _find_user_frame(tb_exc: traceback.TracebackException) -> Optional[traceback.FrameSummary]:
    project_root = Path(__file__).resolve().parents[2]
    for frame in reversed(tb_exc.stack):
        try:
            p = Path(frame.filename).resolve()
        except Exception:
            continue
        if str(p).startswith(str(project_root)):
            return frame
    # fallback
    return tb_exc.stack[-1] if tb_exc.stack else None


def get_detailed_error(error: Exception) -> Dict[str, Any]:
    """Return a structured dictionary with detailed error info.

    Keys:
        - error_type
        - message
        - file
        - function
        - line
        - stack_trace
        - suggestion (optional)
    """
    tb_exc = traceback.TracebackException.from_exception(error)
    user_frame = _find_user_frame(tb_exc)

    file = user_frame.filename if user_frame else None
    func = user_frame.name if user_frame else None
    line = user_frame.lineno if user_frame else None

    stack = "".join(tb_exc.format())

    info = {
        "error_type": type(error).__name__,
        "message": str(error),
        "file": file,
        "function": func,
        "line": line,
        "stack_trace": stack,
    }
    return info


def format_detailed_error(info: Dict[str, Any]) -> str:
    """Render the detailed error dict into a human-readable string."""
    lines = []
    lines.append(f"Error Type: {info.get('error_type')}")
    lines.append(f"Message: {info.get('message')}")
    lines.append("")
    lines.append(f"File: {info.get('file')}")
    lines.append(f"Function: {info.get('function')}")
    lines.append(f"Line: {info.get('line')}")
    lines.append("")
    lines.append("Stack Trace:")
    lines.append(info.get("stack_trace", ""))
    return "\n".join(lines)


def raise_with_context(
    exc_type: Type[ProjectException],
    original_exc: Exception,
    message: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
    context: Optional[Dict[str, Any]] = None,
) -> None:
    """Log the original exception with details and raise the new exception chaining the original.

    Example:
        raise_with_context(RetrievalError, e, "Hybrid retrieval failed")
    """
    info = get_detailed_error(original_exc)
    if context:
        info["context"] = context
    formatted = format_detailed_error(info)
    if logger is None:
        logger = logging.getLogger("arise.errors")
    logger.error("%s", formatted)
    raise exc_type(message or str(original_exc), context=info) from original_exc


__all__ = ["get_detailed_error", "format_detailed_error", "raise_with_context"]

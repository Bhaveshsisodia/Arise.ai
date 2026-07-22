"""Deterministic guardrails around the RAG pipeline.

This module keeps the first integration lightweight:
- input validation before routing/retrieval
- output validation before API return

It also exposes clean extension points for Guardrails AI / Hub validators.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.config import CFG
from src.exception.custom_exception import GuardrailViolationError
from src.utils.logger import pipeline_logger as logger

try:  # pragma: no cover - optional dependency until installed in runtime env
    from guardrails import Guard
except Exception:  # pragma: no cover - guardrails-ai is optional during local dev
    Guard = None


_CITATION_PATTERN = re.compile(r"\[\d+\]")


@dataclass(frozen=True)
class GuardrailDecision:
    text: str
    metadata: dict[str, Any]


class GuardrailRuntime:
    def __init__(self) -> None:
        cfg = CFG.get("guardrails", {})
        self.enabled = bool(cfg.get("enabled", False))
        self.fail_closed = bool(cfg.get("fail_closed", True))
        self.input_cfg = cfg.get("input", {})
        self.output_cfg = cfg.get("output", {})
        self._guardrails_sdk_available = Guard is not None

        if self.enabled and not self._guardrails_sdk_available:
            logger.warning(
                "guardrails-ai dependency not importable; using deterministic in-app guardrails only"
            )

    def validate_input(self, question: str) -> GuardrailDecision:
        if not self.enabled:
            return GuardrailDecision(text=question, metadata={"guardrails_enabled": False})

        normalized = (question or "").strip()
        min_chars = int(self.input_cfg.get("min_chars", 1))
        max_chars = int(self.input_cfg.get("max_chars", 5000))

        if len(normalized) < min_chars:
            raise GuardrailViolationError(
                "Question is too short to process safely.",
                context={"stage": "input", "min_chars": min_chars},
            )
        if len(normalized) > max_chars:
            raise GuardrailViolationError(
                "Question exceeds the allowed length.",
                context={"stage": "input", "max_chars": max_chars},
            )

        if bool(self.input_cfg.get("block_prompt_injection", True)):
            self._check_patterns(
                normalized,
                self.input_cfg.get("block_patterns", []),
                stage="input",
                message="Question blocked by input guardrails.",
            )

        return GuardrailDecision(
            text=normalized,
            metadata={
                "guardrails_enabled": True,
                "guardrails_sdk_available": self._guardrails_sdk_available,
            },
        )

    def validate_output(self, answer: str, *, datasource: str) -> GuardrailDecision:
        if not self.enabled:
            return GuardrailDecision(text=answer, metadata={"guardrails_enabled": False})

        normalized = (answer or "").strip()

        self._check_patterns(
            normalized,
            self.output_cfg.get("blocked_patterns", []),
            stage="output",
            message="Answer blocked by output guardrails.",
        )

        exempt = str(
            self.output_cfg.get(
                "citation_exempt_response",
                "Information not available in retrieved documents.",
            )
        ).strip()
        require_citations = bool(self.output_cfg.get("require_citations_for_retrieval", True))

        if datasource == "retrieval" and require_citations and normalized and normalized != exempt:
            if not _CITATION_PATTERN.search(normalized):
                raise GuardrailViolationError(
                    "Retrieved answers must contain at least one source citation.",
                    context={"stage": "output", "datasource": datasource},
                )

        return GuardrailDecision(
            text=normalized,
            metadata={
                "guardrails_enabled": True,
                "guardrails_sdk_available": self._guardrails_sdk_available,
                "datasource": datasource,
            },
        )

    def validate_stream_chunk(self, chunk: str) -> str:
        if not self.enabled:
            return chunk

        normalized = (chunk or "").strip()
        if not normalized:
            return chunk

        self._check_patterns(
            normalized,
            self.output_cfg.get("blocked_patterns", []),
            stage="output_stream",
            message="Streaming chunk blocked by output guardrails.",
        )
        return chunk

    @staticmethod
    def _check_patterns(text: str, patterns: list[str], *, stage: str, message: str) -> None:
        for pattern in patterns:
            if re.search(pattern, text):
                raise GuardrailViolationError(
                    message,
                    context={"stage": stage, "matched_pattern": pattern},
                )


guardrail_runtime = GuardrailRuntime()

import logging
import json
import os
from pathlib import Path
from logging import Logger


LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
LOG_TO_FILES = os.getenv("LOG_TO_FILES", "false").strip().lower() in {"1", "true", "yes", "on"}
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_AS_JSON = os.getenv("LOG_AS_JSON", "true").strip().lower() in {"1", "true", "yes", "on"}

if LOG_TO_FILES:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "lineno": record.lineno,
            "message": record.getMessage(),
        }
        # include any extra keys passed via LoggerAdapter
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            )
        }
        if extras:
            payload["extra"] = extras
        return json.dumps(payload, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    """Human-readable structured log lines for .log files.

    Format: 2026-06-22T12:00:00Z INFO arise.pipeline module: event - key1=val key2=val
    """

    def format(self, record: logging.LogRecord) -> str:
        timestamp = self.formatTime(record, self.datefmt)
        base = f"{timestamp} {record.levelname} {record.name} {record.module}: {record.getMessage()}"
        # include extras (structured details)
        extras = {
            k: v
            for k, v in record.__dict__.items()
            if k not in (
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
            )
        }
        if extras:
            # render extras as key=repr(value)
            kv = " ".join(f"{k}={repr(v)}" for k, v in extras.items())
            return f"{base} - {kv}"
        return base


def _make_logger(name: str, file_name: str) -> Logger:
    lg = logging.getLogger(name)
    lg.setLevel(LOG_LEVEL)
    lg.propagate = False

    # avoid duplicate handlers
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in lg.handlers):
        sh = logging.StreamHandler()
        sh.setLevel(LOG_LEVEL)
        sh.setFormatter(JSONFormatter() if LOG_AS_JSON else logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        lg.addHandler(sh)

    if not LOG_TO_FILES:
        return lg

    # file handler with JSON formatter
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "_is_structured", False) for h in lg.handlers):
        fh = logging.FileHandler(LOG_DIR / file_name, encoding="utf-8")
        fh.setLevel(LOG_LEVEL)
        fh.setFormatter(JSONFormatter())
        fh._is_structured = True
        lg.addHandler(fh)

    plain_name = Path(file_name).with_suffix(".log")
    if not any(isinstance(h, logging.FileHandler) and getattr(h, "_is_plain", False) for h in lg.handlers):
        ph = logging.FileHandler(LOG_DIR / plain_name, encoding="utf-8")
        ph.setLevel(LOG_LEVEL)
        ph.setFormatter(PlainFormatter(datefmt="%Y-%m-%dT%H:%M:%S"))
        ph._is_plain = True
        lg.addHandler(ph)

    return lg


# Two specialized loggers: pipeline and evaluation
pipeline_logger = _make_logger("arise.pipeline", "pipeline.jsonl")
eval_logger = _make_logger("arise.eval", "evaluation.jsonl")


def get_pipeline_logger() -> Logger:
    return pipeline_logger


def get_eval_logger() -> Logger:
    return eval_logger


def pipeline_event(event: str, **details) -> None:
    """Log a structured pipeline event to the pipeline logger."""
    pipeline_logger.info(event, extra={"event": event, **details})


def eval_event(event: str, **details) -> None:
    """Log a structured evaluation event to the eval logger."""
    eval_logger.info(event, extra={"event": event, **details})


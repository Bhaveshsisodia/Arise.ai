from dotenv import load_dotenv
import json
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.chain import ask_with_sources, ask_stream_with_sources
from src.exception.custom_exception import APIError, ProjectException, ValidationError
from src.utils.logger import pipeline_logger as logger
from src.utils.redis_cache import get_redis_cache
from src.vector_db import collection
load_dotenv()

@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Application startup initiated")
    if os.getenv("ARISE_PRELOAD_MODELS", "false").strip().lower() in {"1", "true", "yes", "on"}:
        try:
            from src.chain import build_chain_components
            build_chain_components(False)
            logger.info("Application models preloaded successfully")
        except Exception as exc:
            logger.exception("Application preload failed: %s", exc)
            raise
    yield


app = FastAPI(title="Arise RAG API", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


class QuestionRequest(BaseModel):
    question: str


class ConversationTurn(BaseModel):
    role: str
    content: str


class SourceItem(BaseModel):
    source_id: int | None = None
    document_title: str
    section: str
    pages: str
    relevance: float | None = None
    cross_encoder_score: float | None = None
    llm_rank: int | None = None
    snippet: str
    content: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    status: str = "ok"


class ChatRequest(QuestionRequest):
    session_id: str | None = None
    history: list[ConversationTurn] = Field(default_factory=list)


def _validate_question(question: str) -> str:
    normalized = (question or "").strip()
    if not normalized:
        raise ValidationError("Question cannot be empty", context={"field": "question"})
    return normalized


@app.exception_handler(ProjectException)
async def handle_project_exception(_: Request, exc: ProjectException):
    logger.error("Project exception: %s", exc)
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "error": exc.to_dict()},
    )


@app.exception_handler(RequestValidationError)
async def handle_request_validation(_: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content={
            "status": "error",
            "error": {
                "code": "REQUEST_VALIDATION_ERROR",
                "message": "Request validation failed",
                "context": {"details": exc.errors()},
            },
        },
    )


@app.exception_handler(Exception)
async def handle_unexpected_exception(_: Request, exc: Exception):
    logger.exception("Unhandled application exception: %s", exc)
    api_error = APIError("Internal server error")
    return JSONResponse(
        status_code=api_error.status_code,
        content={"status": "error", "error": api_error.to_dict()},
    )


@app.get("/health")
def health():
    return {"status": "ok", "message": "Arise RAG API is running"}


@app.get("/ready")
def ready():
    checks = {"mongodb": "ok", "redis": "ok"}

    try:
        collection.database.client.admin.command("ping")
    except Exception as exc:
        logger.warning("MongoDB readiness check failed: %s", exc)
        checks["mongodb"] = "error"

    cache = get_redis_cache()
    if cache.enabled and cache.client is not None:
        try:
            cache.client.ping()
        except Exception as exc:
            logger.warning("Redis readiness check failed: %s", exc)
            checks["redis"] = "error"
    else:
        checks["redis"] = "disabled"

    if checks["mongodb"] != "ok":
        return JSONResponse(
            status_code=503,
            content={"status": "error", "checks": checks},
        )

    return {"status": "ok", "checks": checks}


@app.get("/chat")
def chat_page(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})


@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: ChatRequest):
    response = ask_with_sources(
        _validate_question(payload.question),
        session_id=payload.session_id,
        history=[turn.model_dump() for turn in payload.history],
        use_llm_reranker=True,
    )
    return AnswerResponse(**response)


@app.post("/ask/stream")
def ask_question_stream(payload: ChatRequest):
    question = _validate_question(payload.question)

    def stream_generator():
        try:
            for event in ask_stream_with_sources(
                question,
                session_id=payload.session_id,
                history=[turn.model_dump() for turn in payload.history],
                use_llm_reranker=True,
            ):
                yield json.dumps(event) + "\n"
        except ProjectException as exc:
            yield json.dumps({"type": "error", "error": exc.to_dict()}) + "\n"
        except Exception as exc:
            logger.exception("Unhandled streaming exception: %s", exc)
            yield json.dumps(
                {"type": "error", "error": APIError("Internal server error").to_dict()}
            ) + "\n"

    return StreamingResponse(
        stream_generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-transform"},
    )


if __name__ == "__main__":
    import uvicorn

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload_enabled = os.getenv("ARISE_RELOAD", "false").strip().lower() in {"1", "true", "yes", "on"}
    uvicorn.run("app:app", host=host, port=port, reload=reload_enabled)

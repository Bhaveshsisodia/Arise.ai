from dotenv import load_dotenv
import json

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from src.chain import ask_with_sources, ask_stream, ask_stream_with_sources
load_dotenv()

app = FastAPI(title="Arise RAG API", version="1.0.0")
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


@app.get("/health")
def health():
    return {"status": "ok", "message": "Arise RAG API is running"}


@app.get("/chat")
def chat_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: ChatRequest):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        response = ask_with_sources(
            payload.question,
            session_id=payload.session_id,
            history=[turn.model_dump() for turn in payload.history],
            use_llm_reranker=True,
        )
        return AnswerResponse(**response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/ask/stream")
def ask_question_stream(payload: ChatRequest):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    def stream_generator():
        try:
            for event in ask_stream_with_sources(
                payload.question,
                session_id=payload.session_id,
                history=[turn.model_dump() for turn in payload.history],
                use_llm_reranker=True,
            ):
                yield json.dumps(event) + "\n"
        except Exception as exc:
            yield json.dumps({"type": "error", "error": str(exc)}) + "\n"

    return StreamingResponse(
        stream_generator(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-transform"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

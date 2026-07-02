from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from src.chain import ask_with_sources, build_chain_components

load_dotenv()

app = FastAPI(title="Arise RAG API", version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
build_chain_components(use_llm_reranker=True)


class QuestionRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    section: str
    pages: str
    cross_encoder_score: float | None = None
    llm_rank: int | None = None
    snippet: str


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceItem]
    status: str = "ok"


@app.get("/health")
def health():
    return {"status": "ok", "message": "Arise RAG API is running"}


@app.get("/chat")
def chat_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask", response_model=AnswerResponse)
def ask_question(payload: QuestionRequest):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        response = ask_with_sources(payload.question, use_llm_reranker=True)
        return AnswerResponse(**response)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

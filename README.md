# Arise Chatbot

FastAPI-based RAG assistant for Indian power sector regulatory documents. The app combines hybrid retrieval, optional reranking, source-backed answer generation, streaming responses, and a browser chat UI.

## Problem It Solves

Regulatory teams in the Indian power sector work with long, dense documents such as tariff petitions, ARR filings, true-up orders, and regulations. Finding a specific figure, justification, section reference, or filing assumption often requires manually scanning hundreds of pages across tables, narrative sections, and document metadata.

This project reduces that manual effort by turning the document set into a searchable assistant. Instead of reading petitions page by page, users can ask questions like:

- What ARR has JUSNL projected for FY 2025-26?
- What employee expenses were claimed?
- Why was a petition filed?
- Which regulation or section supports this claim?

In practice, it helps solve these concrete problems:

- Slow manual review of large regulatory filings
- Difficulty locating exact numbers across tables and narrative text
- Repetitive analyst work in ARR, tariff, and compliance reviews
- Weak traceability when answers are not tied back to source sections
- Time lost switching between search, document reading, and note-taking

The intended outcome is faster regulatory analysis with grounded answers and visible source evidence.

## What The Project Does

The system is designed for questions over tariff petitions, ARR filings, true-up orders, regulations, and related power-sector documents. It supports:

- Natural-language question answering over indexed documents
- Hybrid retrieval with vector search and MongoDB text search
- Optional metadata-aware filtering extracted from the query
- Cross-encoder reranking and optional LLM listwise reranking
- Conversation-aware chat with session memory
- Streaming answers with cited source snippets
- Evaluation workflows through LangSmith

## Current Stack

- API/UI: FastAPI, Jinja2 templates, vanilla JS
- LLM provider: Groq
- Default LLM in config: `openai/gpt-oss-20b`
- Embeddings: `BAAI/bge-large-en-v1.5`
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Database: MongoDB with vector search and text search
- Cache: Redis with exact and semantic cache support
- Tracing/Evaluation: LangSmith

## Retrieval Pipeline

The live code follows this flow:

1. Route the question to either `chat` or `retrieval`
2. Rewrite the question with chat history when needed
3. Extract metadata filters from the query using the LLM
4. Run hybrid retrieval:
   - MongoDB vector search
   - MongoDB text search
   - Reciprocal Rank Fusion (RRF)
5. Apply MMR for diversity if enabled
6. Apply cross-encoder reranking if enabled
7. Apply LLM listwise reranking if enabled
8. Build grounded context blocks
9. Generate the final answer with citations such as `[1]`, `[2]`

## Features In The Current Codebase

- `POST /ask` returns a final answer plus structured sources
- `POST /ask/stream` streams NDJSON events for incremental UI rendering
- `GET /chat` serves the browser chat interface
- `GET /health` returns a health response
- Session-aware chat memory is supported through `session_id`
- Structured application errors are returned with API-friendly payloads
- Logs are written under `src/logs/`

## Project Structure

```text
.
|-- app.py
|-- config/
|   `-- config.yaml
|-- src/
|   |-- chain.py
|   |-- config.py
|   |-- generator.py
|   |-- metadata_filter.py
|   |-- query_optimizer.py
|   |-- reranker.py
|   |-- retriever.py
|   |-- router.py
|   |-- vector_db.py
|   |-- evaluation_lang/
|   |   |-- dataset.py
|   |   |-- evaluators.py
|   |   |-- llm_loader.py
|   |   |-- run_eval.py
|   |   `-- test_dataset.json
|   |-- exception/
|   `-- utils/
|-- static/
|   |-- css/chat.css
|   `-- js/chat.js
|-- templates/
|   `-- index.html
|-- requirements.txt
`-- setup.py
```

## Requirements

- Python 3.10+
- MongoDB collection containing:
  - `text`
  - `embedding`
  - `metadata`
- MongoDB vector index matching `config/config.yaml`
- MongoDB text index for keyword search
- Groq API key

Optional:

- Redis for caching
- LangSmith API key for tracing
- OpenRouter key for the evaluation helper that uses `ChatOpenAI`

## Environment Variables

Create a `.env` file in the project root with the values you use:

```env
MONGO_URI=
GROQ_API_KEY=

# Optional runtime tracing
LANGCHAIN_API_KEY=
LANGCHAIN_PROJECT=

# Optional Redis cache
REDIS_URL=
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_DB=0

# Optional evaluation providers
OPEN_ROUTER_API_KEY=
GOOGLE_API_KEY=
GEMINI_API=
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

You can also install the package locally:

```bash
pip install -e .
```

## Running The App

Start the FastAPI server:

```bash
python app.py
```

Default local URLs:

- API: `http://127.0.0.1:8000`
- Chat UI: `http://127.0.0.1:8000/chat`
- Health check: `http://127.0.0.1:8000/health`
- Readiness check: `http://127.0.0.1:8000/ready`

## Production Deployment

Production deployment artifacts are included in the repo:

- [Dockerfile](/D:/python scripts/Generative AI/arise_chatbot/Dockerfile)
- [.env.example](/D:/python scripts/Generative AI/arise_chatbot/.env.example)
- [azure-container-app.template.yaml](/D:/python scripts/Generative AI/arise_chatbot/azure-container-app.template.yaml)
- [docs/production-deployment.md](/D:/python scripts/Generative AI/arise_chatbot/docs/production-deployment.md)
- [.github/workflows/deploy-container-app.yml](/D:/python scripts/Generative AI/arise_chatbot/.github/workflows/deploy-container-app.yml)

## API Examples

### `POST /ask`

Request:

```json
{
  "question": "What ARR has JUSNL projected for FY 2025-26?",
  "session_id": "demo-session",
  "history": [
    {
      "role": "user",
      "content": "We are discussing the latest tariff petition."
    }
  ]
}
```

Response shape:

```json
{
  "answer": "...",
  "sources": [
    {
      "source_id": 1,
      "document_title": "Source Document",
      "section": "5.11 ARR for FY 2025-26",
      "pages": "52-53",
      "relevance": 0.9123,
      "cross_encoder_score": 0.9123,
      "llm_rank": 1,
      "snippet": "...",
      "content": "..."
    }
  ],
  "status": "ok"
}
```

### `POST /ask/stream`

Returns `application/x-ndjson` with events like:

```json
{"type":"delta","text":"partial answer"}
{"type":"sources","sources":[...]}
```

## Configuration

Primary runtime settings live in [config/config.yaml](/D:/python%20scripts/Generative%20AI/arise_chatbot/config/config.yaml).

Notable options:

- MongoDB database, collection, and vector index name
- Embedding model and dimensions
- Groq model name and temperature
- Retrieval stage sizes
- MMR enablement and weighting
- Cross-encoder and LLM reranker toggles
- Query rewriting strategy
- Redis cache settings

## Evaluation

Evaluation utilities live in [src/evaluation_lang/run_eval.py](/D:/python%20scripts/Generative%20AI/arise_chatbot/src/evaluation_lang/run_eval.py).

Run:

```bash
python src/evaluation_lang/run_eval.py
```

Notes:

- The evaluation script creates or reuses a LangSmith dataset from `src/evaluation_lang/test_dataset.json`
- It builds the same retriever stack used by the app
- It may require `LANGCHAIN_API_KEY` and `OPEN_ROUTER_API_KEY` depending on the evaluator setup

## Notes About Data Indexing

This repository expects documents to already be present in MongoDB. The current repo does not include a production ingestion/indexing script, so any existing indexing workflow should ensure:

- chunk text is stored in `text`
- embedding vectors are stored in `embedding`
- document metadata is stored under `metadata.*`
- the configured vector index and text index exist

## License

MIT

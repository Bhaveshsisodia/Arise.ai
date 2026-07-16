"""
chain.py - LCEL chain assembly + LangSmith tracing.
"""

import json
import os
from functools import lru_cache
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langsmith import traceable

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - optional dependency at runtime
    SentenceTransformer = None
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc
else:
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = None

from src.config import CFG
from src.exception.custom_exception import (
    EmbeddingError,
    LLMGenerationError,
    ProjectException,
    RetrievalError,
    ValidationError,
)
from src.exception.error_utils import raise_with_context
from src.generator import build_context, get_llm, output_parser
from src.router import QueryRouter
from src.reranker import CrossEncoderReranker, LLMListwiseReranker
from src.retriever import MongoHybridRetriever
from src.utils.logger import pipeline_logger as logger
from src.utils.redis_cache import build_cache_key, get_redis_cache
from src.utils.session_store import RedisBackedChatMessageHistory
from src.vector_db import collection
import re

load_dotenv()

_SEMANTIC_CACHE_THRESHOLD = float(CFG.get("redis", {}).get("semantic_similarity_threshold", 0.92))
_SEMANTIC_CACHE_ENABLED = bool(CFG.get("redis", {}).get("semantic_enabled", True))
_MAX_HISTORY_TURNS = 8


def setup_langsmith() -> None:
    """Enable LangSmith tracing when credentials are available."""
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "jusnl-rag")


def _normalize_output(output: Any) -> str:
    if isinstance(output, str):
        text = output
    else:
        text = getattr(output, "content", str(output))
    # normalize citation style: keep [n] only, remove internal range markers like †L1-L4
    text = re.sub(r"\[(\d+)†L\d+-L\d+\]", r"[\1]", text)
    text = re.sub(r"†L\d+-L\d+", "", text)
    return text


def _apply_ce_reranker(inputs: Dict[str, Any], reranker: CrossEncoderReranker) -> List[Document]:
    return reranker.invoke(
        {"query": inputs["question"], "documents": inputs["documents"]}
    )


def _apply_llm_reranker(inputs: Dict[str, Any], reranker: LLMListwiseReranker) -> List[Document]:
    return reranker.invoke(
        {"query": inputs["question"], "documents": inputs["documents"]}
    )


def _normalize_history(history: List[Dict[str, Any]] | None) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for item in history or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()
        if role not in {"user", "assistant"} or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized[-_MAX_HISTORY_TURNS:]


def _format_history(history: List[Dict[str, str]]) -> str:
    if not history:
        return ""
    lines = []
    for turn in history:
        speaker = "User" if turn["role"] == "user" else "Assistant"
        lines.append(f"{speaker}: {turn['content']}")
    return "\n".join(lines)


def _get_session_history(session_id: str) -> BaseChatMessageHistory:
    return RedisBackedChatMessageHistory(session_id)


def _append_session_turn(session_id: str | None, question: str, answer: str) -> None:
    if not session_id:
        return
    history = _get_session_history(session_id)
    history.add_user_message(question)
    history.add_ai_message(answer)


def _messages_to_history(messages: List[Any]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for message in messages:
        content = getattr(message, "content", "")
        if not content:
            continue
        if isinstance(message, HumanMessage):
            normalized.append({"role": "user", "content": str(content)})
        elif isinstance(message, AIMessage):
            normalized.append({"role": "assistant", "content": str(content)})
    return normalized[-_MAX_HISTORY_TURNS:]


def _get_effective_history(
    history: List[Dict[str, Any]] | None,
    session_id: str | None,
) -> List[Dict[str, str]]:
    if session_id:
        stored_history = _messages_to_history(_get_session_history(session_id).messages)
        if stored_history:
            return stored_history
    return _normalize_history(history)


def _to_langchain_history(history: List[Dict[str, str]]) -> List[Any]:
    messages: List[Any] = []
    for turn in history:
        if turn["role"] == "user":
            messages.append(HumanMessage(content=turn["content"]))
        else:
            messages.append(AIMessage(content=turn["content"]))
    return messages


def _build_chat_prompt(question: str, history: List[Dict[str, str]]) -> Any:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. "
                "Use the conversation history from this session as working memory when it is relevant. "
                "For normal conversation such as greetings, thanks, or general chat, respond naturally. "
                "Only rely on conversation history when the user asks about something said earlier or refers to prior messages. "
                "If the user asks for something that should be in prior messages but it is not present, say so plainly.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    ).format_prompt(
        chat_history=_to_langchain_history(history),
        input=question,
    )


def _build_rag_answer_prompt(question: str, history: List[Dict[str, str]], context: str) -> Any:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a power sector regulatory analyst specialising in Indian electricity regulation. "
                "Use only the provided context to answer the user's question. "
                "If the answer is not in the context, say exactly: "
                "\"Information not available in retrieved documents.\" "
                "When using facts from the context, cite source blocks like [1], [2], [3].",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "Context:\n{context}\n\nQuestion:\n{input}"),
        ]
    ).format_prompt(
        chat_history=_to_langchain_history(history),
        context=context,
        input=question,
    )


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _is_history_question(question: str) -> bool:
    tokens = set(_tokenize(question))
    if not tokens:
        return False

    memory_terms = {
        "remember", "recall", "earlier", "before", "previous", "prior",
        "history", "conversation", "message", "messages", "question",
        "questions", "asked", "said", "tell", "told", "name",
    }
    self_reference_terms = {"i", "me", "my", "mine"}
    return bool(tokens & memory_terms) and bool(tokens & self_reference_terms)


def _select_history_context(question: str, history: List[Dict[str, str]], limit: int = 4) -> List[Dict[str, str]]:
    question_tokens = set(_tokenize(question))
    ranked: List[tuple[float, Dict[str, str]]] = []

    for idx, turn in enumerate(history):
        turn_tokens = set(_tokenize(turn["content"]))
        overlap = len(question_tokens & turn_tokens)
        recency_bonus = (idx + 1) / max(len(history), 1)
        role_bonus = 0.15 if turn["role"] == "user" else 0.0
        score = overlap + recency_bonus + role_bonus
        if turn["role"] == "user" and {"last", "previous", "earlier"} & question_tokens:
            score += 0.5
        ranked.append((score, turn))

    ranked.sort(key=lambda item: item[0], reverse=True)
    selected = [turn for score, turn in ranked if score > 0][:limit]
    if not selected:
        selected = history[-limit:]
    return selected


def _build_history_answer_prompt(question: str, history: List[Dict[str, str]]) -> Any:
    relevant_history = _select_history_context(question, history)
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are answering a question about the current conversation only. "
                "Use only the conversation history provided. "
                "Never claim memory is unavailable if the answer appears in the history. "
                "If the answer is not present in the history, say so plainly and briefly.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    ).format_prompt(
        chat_history=_to_langchain_history(relevant_history),
        input=question,
    )


def _answer_from_history(llm, question: str, history: List[Dict[str, str]]) -> str | None:
    if not history:
        return None
    if not _is_history_question(question):
        return None
    try:
        prompt = _build_history_answer_prompt(question, history)
        answer = _normalize_output(llm.invoke(prompt)).strip()
        return answer or None
    except Exception:
        return None

def _build_retriever_history_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Given the conversation history and the latest user question, "
                "rewrite the latest question into a standalone search query for retrieving relevant regulatory documents. "
                "Do not answer the question. Return only the rewritten search query.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )


def _contextualize_question(llm, question: str, history: List[Dict[str, str]]) -> str:
    if not history:
        return question
    prompt = _build_retriever_history_prompt().format_prompt(
        chat_history=_to_langchain_history(history),
        input=question,
    )
    try:
        rewritten = _normalize_output(llm.invoke(prompt)).strip()
        return rewritten or question
    except Exception:
        return question


def _route_question(question: str, router: QueryRouter) -> str:
    try:
        return router.route(question)
    except Exception:
        return "retrieval"


def _prepare_response_payload(
    question: str,
    *,
    history: List[Dict[str, Any]] | None = None,
    session_id: str | None = None,
    use_llm_reranker: bool = False,
) -> Dict[str, Any]:
    if not (question or "").strip():
        raise ValidationError(
            "Question cannot be empty",
            context={"field": "question"},
        )

    components = build_chain_components(use_llm_reranker)
    normalized_history = _get_effective_history(history, session_id)
    router = QueryRouter(llm=components["llm"])
    datasource = _route_question(question, router)
    retrieval_question = _contextualize_question(components["llm"], question, normalized_history)

    if datasource == "chat":
        documents: List[Document] = []
    else:
        try:
            documents = components["retrieval_engine"].invoke(retrieval_question)
            if components["ce_reranker"] is not None:
                documents = _apply_ce_reranker(
                    {"question": retrieval_question, "documents": documents},
                    components["ce_reranker"],
                )
            if use_llm_reranker and components["llm_reranker"] is not None:
                documents = _apply_llm_reranker(
                    {"question": retrieval_question, "documents": documents},
                    components["llm_reranker"],
                )
        except ProjectException:
            raise
        except Exception as exc:
            raise_with_context(
                RetrievalError,
                exc,
                "Failed to prepare retrieval response payload",
                context={"question": question, "retrieval_question": retrieval_question},
            )

    return {
        "llm": components["llm"],
        "datasource": datasource,
        "documents": documents,
        "context": build_context(documents),
        "question": question,
        "history": normalized_history,
        "session_id": session_id,
    }


def _build_sources(documents: List[Document], max_sources: int) -> List[Dict[str, Any]]:
    sources = []
    for i, doc in enumerate(documents[:max_sources], start=1):
        meta = doc.metadata
        document_title = (
            meta.get("document_title")
            or meta.get("document")
            or meta.get("document_name")
            or meta.get("file_name")
            or meta.get("document_type")
            or "Source Document"
        )
        sources.append(
            {
                "source_id": i,
                "document_title": document_title,
                "section": meta.get("section_heading", "Unknown"),
                "pages": f"{meta.get('page_start', '?')}-{meta.get('page_end', '?')}",
                "relevance": round(float(meta.get("cross_encoder_score", 0) or 0), 4),
                "cross_encoder_score": meta.get("cross_encoder_score"),
                "llm_rank": meta.get("llm_rank"),
                "snippet": doc.page_content[:250].replace("\n", " "),
                "content": doc.page_content,
            }
        )
    return sources


@lru_cache(maxsize=1)
def _get_embedder() -> SentenceTransformer:
    model_name = CFG["embedding"]["model"]
    logger.info("Loading embedding model: %s", model_name)
    if SentenceTransformer is None:
        raise EmbeddingError(
            "sentence-transformers could not be imported; embedding model is unavailable.",
            context={"model_name": model_name},
        )
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        raise_with_context(
            EmbeddingError,
            exc,
            "Failed to load embedding model",
            context={"model_name": model_name},
        )


@lru_cache(maxsize=4)
def build_chain_components(use_llm_reranker: bool = False) -> Dict[str, Any]:
    """Build the chain once and reuse heavy components across requests."""
    setup_langsmith()

    use_mmr_cfg = CFG["retrieval"].get("use_mmr", True)
    use_ce_cfg = CFG.get("reranker", {}).get("use_ce", True)
    use_llm_cfg = CFG.get("reranker", {}).get("use_llm", False)
    context_compression = CFG["retrieval"].get("context_compression", False)

    try:
        embedder = _get_embedder()
        llm = get_llm()

        retriever = MongoHybridRetriever(
            embedder=embedder,
            llm=llm,
            collection=collection,
            use_mmr=use_mmr_cfg,
        )
    except ProjectException:
        raise
    except Exception as exc:
        raise_with_context(
            RetrievalError,
            exc,
            "Failed to build chain components",
        )

    retrieval_engine = retriever

    # Router: decide whether to run retrieval or a chat-only flow
    router = QueryRouter(llm=llm)

    def _route_and_retrieve(question: str):
        try:
            ds = router.route(question)
        except Exception:
            ds = "retrieval"
        if ds == "retrieval":
            return retrieval_engine.invoke(question)
        # chat route: return empty list (no documents)
        return []

    ce_reranker = CrossEncoderReranker() if use_ce_cfg else None
    llm_reranker = LLMListwiseReranker(llm=llm) if use_llm_cfg else None

    logger.info(
        "build_chain | embed_model=%s use_mmr=%s use_ce=%s use_llm=%s llm_model=%s context_compression=%s",
        CFG["embedding"]["model"],
        use_mmr_cfg,
        use_ce_cfg,
        use_llm_cfg,
        CFG["llm"]["model"],
        context_compression,
    )

    stage1 = RunnablePassthrough.assign(
        documents=RunnableLambda(
            lambda x: _route_and_retrieve(x["question"])
        ).with_config({"run_name": "Stage1-RetrievalOrChat"}),
        datasource=RunnableLambda(lambda x: router.route(x["question"]))
    )

    if ce_reranker is not None:
        stage2 = RunnablePassthrough.assign(
            documents=RunnableLambda(
                lambda x: _apply_ce_reranker(x, ce_reranker)
            ).with_config({"run_name": "Stage2-CrossEncoder"})
        )
    else:
        stage2 = RunnablePassthrough()

    if use_llm_reranker and llm_reranker is not None:
        stage3 = RunnablePassthrough.assign(
            documents=RunnableLambda(
                lambda x: _apply_llm_reranker(x, llm_reranker)
            ).with_config({"run_name": "Stage3-LLMReranker"})
        )
    else:
        stage3 = RunnablePassthrough()

    def _generate_using_llm(inputs: Dict[str, Any]):
        question = inputs["question"]
        datasource = inputs.get("datasource", "retrieval")
        try:
            if datasource == "chat":
                # simple chat prompt for conversational queries
                chat_prompt = ChatPromptTemplate.from_template(
                    """
You are a helpful assistant.

Question:
{question}
"""
                )
                prompt = chat_prompt.format_prompt(question=question)
                resp = llm.invoke(prompt)
            else:
                context = build_context(inputs.get("documents", []))
                prompt = _build_rag_answer_prompt(question, [], context)
                resp = llm.invoke(prompt)
            return resp
        except Exception as e:
            logger.exception("LLM generation failed: %s", e)
            raise LLMGenerationError(
                "Failed to generate answer from language model",
                context={"datasource": datasource, "question": question},
            ) from e

    generation_chain = (
        RunnableLambda(_generate_using_llm)
        | output_parser
        | RunnableLambda(_normalize_output)
    )

    response_chain = (
        stage1
        | stage2
        | stage3
        | RunnablePassthrough.assign(
            context=RunnableLambda(lambda x: build_context(x["documents"]))
        )
        | RunnablePassthrough.assign(answer=generation_chain)
    ).with_config({"run_name": "JUSNL-RAG-Pipeline"})

    answer_chain = response_chain | RunnableLambda(lambda x: x["answer"])

    chat_prompt_chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a helpful assistant. "
                    "Use the conversation history from this session as working memory when it is relevant. "
                    "For normal conversation such as greetings, thanks, or general chat, respond naturally. "
                    "Only rely on conversation history when the user asks about something said earlier or refers to prior messages. "
                    "If the user asks for something that should be in prior messages but it is not present, say so plainly.",
                ),
                MessagesPlaceholder("history"),
                ("human", "{question}"),
            ]
        )
        | llm
        | output_parser
        | RunnableLambda(_normalize_output)
    )

    rag_prompt_chain = (
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are a power sector regulatory analyst specialising in Indian electricity regulation. "
                    "Use only the provided context to answer the user's question. "
                    "If the answer is not in the context, say exactly: "
                    "\"Information not available in retrieved documents.\" "
                    "When using facts from the context, cite source blocks like [1], [2], [3].",
                ),
                MessagesPlaceholder("history"),
                ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
            ]
        )
        | llm
        | output_parser
        | RunnableLambda(_normalize_output)
    )

    return {
        "chain": answer_chain,
        "answer_chain": answer_chain,
        "response_chain": response_chain,
        "retrieval_engine": retrieval_engine,
        "ce_reranker": ce_reranker,
        "llm_reranker": llm_reranker,
        "llm": llm,
        "chat_with_history": RunnableWithMessageHistory(
            chat_prompt_chain,
            _get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        ),
        "rag_with_history": RunnableWithMessageHistory(
            rag_prompt_chain,
            _get_session_history,
            input_messages_key="question",
            history_messages_key="history",
        ),
    }


def build_chain(use_llm_reranker: bool = False):
    """Return the answer-only runnable."""
    return build_chain_components(use_llm_reranker)["chain"]


def ask_with_sources(
    question: str,
    history: List[Dict[str, Any]] | None = None,
    session_id: str | None = None,
    use_llm_reranker: bool = False,
    max_sources: int = 5,
) -> Dict[str, Any]:
    if not (question or "").strip():
        raise ValidationError("Question cannot be empty", context={"field": "question"})

    cache = get_redis_cache()
    normalized_question = (question or "").strip()
    normalized_history = _get_effective_history(history, session_id)
    history_cache_fragment = json.dumps(normalized_history, ensure_ascii=True, sort_keys=True)
    cache_variant = build_cache_key(
        "answer_variant",
        history_cache_fragment,
        use_llm_reranker,
        max_sources,
    )
    cache_key = build_cache_key(
        "answer",
        normalized_question,
        history_cache_fragment,
        use_llm_reranker,
        max_sources,
    )

    cached_answer = cache.get_json("answer", cache_key)
    if cached_answer is not None:
        logger.info("Redis answer cache hit for question: %s", normalized_question)
        _append_session_turn(session_id, question, cached_answer.get("answer", ""))
        return cached_answer

    embedder = _get_embedder()
    try:
        question_embedding = embedder.encode(normalized_question, normalize_embeddings=True).tolist()
    except Exception as exc:
        raise_with_context(
            EmbeddingError,
            exc,
            "Failed to encode question for answer caching",
            context={"question": normalized_question},
        )
    if _SEMANTIC_CACHE_ENABLED:
        cached_answer = cache.get_semantic_json(
            "answer_semantic",
            question_embedding,
            min_similarity=_SEMANTIC_CACHE_THRESHOLD,
            variant=cache_variant,
        )
        if cached_answer is not None:
            logger.info("Redis semantic answer cache hit for question: %s", normalized_question)
            _append_session_turn(session_id, question, cached_answer.get("answer", ""))
            return cached_answer

    result = _prepare_response_payload(
        question,
        history=normalized_history,
        session_id=session_id,
        use_llm_reranker=use_llm_reranker,
    )
    payload = {
        "answer": _normalize_output(
            _generate_answer_text(
                llm=result["llm"],
                question=result["question"],
                history=result["history"],
                session_id=result["session_id"],
                datasource=result["datasource"],
                context=result["context"],
            )
        ),
        "sources": _build_sources(result["documents"], max_sources),
    }

    cache.set_json("answer", cache_key, payload, ttl=1800)
    if _SEMANTIC_CACHE_ENABLED:
        cache.set_semantic_json(
            "answer_semantic",
            cache_key,
            query_text=normalized_question,
            query_vector=question_embedding,
            value=payload,
            ttl=1800,
            variant=cache_variant,
        )
    return payload


def ask_stream(question: str, session_id: str | None = None, use_llm_reranker: bool = False):
    result = _prepare_response_payload(
        question,
        history=None,
        session_id=session_id,
        use_llm_reranker=use_llm_reranker,
    )
    yield from _stream_answer_text(
        llm=result["llm"],
        question=result["question"],
        history=result["history"],
        session_id=result["session_id"],
        datasource=result["datasource"],
        context=result["context"],
    )


def ask_stream_with_sources(
    question: str,
    history: List[Dict[str, Any]] | None = None,
    session_id: str | None = None,
    use_llm_reranker: bool = False,
    max_sources: int = 5,
):
    result = _prepare_response_payload(
        question,
        history=history,
        session_id=session_id,
        use_llm_reranker=use_llm_reranker,
    )

    for chunk in _stream_answer_text(
        llm=result["llm"],
        question=result["question"],
        history=result["history"],
        session_id=result["session_id"],
        datasource=result["datasource"],
        context=result["context"],
    ):
        yield {"type": "delta", "text": chunk}

    yield {"type": "sources", "sources": _build_sources(result["documents"], max_sources)}


@traceable(name="JUSNL-RAG-Pipeline")
def ask(question: str, chain=None) -> str:
    """Ask a question and return a grounded answer."""
    chain = chain or build_chain()

    try:
        answer = chain.invoke({"question": question})
    except ProjectException:
        raise
    except Exception as exc:
        logger.exception("Chain invocation failed: %s", exc)
        raise LLMGenerationError(
            "Chain invocation failed",
            context={"question": question},
        ) from exc

    if answer is None:
        logger.warning("Chain returned None for question: %s", question)
        return "Chain returned no output (None) - check logs"

    return answer


def _generate_answer_text(
    *,
    llm,
    question: str,
    history: List[Dict[str, str]],
    session_id: str | None,
    datasource: str,
    context: str,
) -> str:
    components = build_chain_components(False)
    if session_id:
        chain = components["chat_with_history"] if datasource == "chat" else components["rag_with_history"]
        payload = {"question": question}
        if datasource != "chat":
            payload["context"] = context
        return chain.invoke(payload, config={"configurable": {"session_id": session_id}})

    if datasource == "chat":
        prompt = _build_chat_prompt(question, history)
    else:
        prompt = _build_rag_answer_prompt(question, history, context)

    try:
        response = llm.invoke(prompt)
    except Exception as exc:
        raise_with_context(
            LLMGenerationError,
            exc,
            "Failed to generate final answer",
            context={"datasource": datasource, "question": question},
        )
    return output_parser.invoke(response)


def _stream_answer_text(
    *,
    llm,
    question: str,
    history: List[Dict[str, str]],
    session_id: str | None,
    datasource: str,
    context: str,
):
    components = build_chain_components(False)
    if session_id:
        chain = components["chat_with_history"] if datasource == "chat" else components["rag_with_history"]
        payload = {"question": question}
        if datasource != "chat":
            payload["context"] = context
        for chunk in chain.stream(payload, config={"configurable": {"session_id": session_id}}):
            normalized = _normalize_output(chunk)
            if normalized:
                yield normalized
        return

    if datasource == "chat":
        prompt = _build_chat_prompt(question, history)
    else:
        prompt = _build_rag_answer_prompt(question, history, context)

    try:
        for chunk in llm.stream(prompt):
            text = getattr(chunk, "content", chunk)
            if text is None:
                continue
            normalized = _normalize_output(text)
            if normalized:
                yield normalized
    except Exception as exc:
        raise_with_context(
            LLMGenerationError,
            exc,
            "Failed to stream final answer",
            context={"datasource": datasource, "question": question},
        )

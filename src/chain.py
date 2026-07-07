"""
chain.py - LCEL chain assembly + LangSmith tracing.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.chain_extract import (
    LLMChainExtractor,
)
from langsmith import traceable

try:
    from sentence_transformers import SentenceTransformer
except Exception as exc:  # pragma: no cover - optional dependency at runtime
    SentenceTransformer = None
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = exc
else:
    _SENTENCE_TRANSFORMERS_IMPORT_ERROR = None

from src.config import CFG
from src.generator import RAG_PROMPT, build_context, get_llm, output_parser
from src.router import QueryRouter
from src.reranker import CrossEncoderReranker, LLMListwiseReranker
from src.retriever import MongoHybridRetriever
from src.utils.logger import pipeline_logger as logger
from src.utils.redis_cache import build_cache_key, get_redis_cache
from src.vector_db import collection
import re

load_dotenv()

_SEMANTIC_CACHE_THRESHOLD = float(CFG.get("redis", {}).get("semantic_similarity_threshold", 0.92))
_SEMANTIC_CACHE_ENABLED = bool(CFG.get("redis", {}).get("semantic_enabled", True))


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
    return SentenceTransformer(model_name)


@lru_cache(maxsize=4)
def build_chain_components(use_llm_reranker: bool = False) -> Dict[str, Any]:
    """Build the chain once and reuse heavy components across requests."""
    setup_langsmith()

    use_mmr_cfg = CFG["retrieval"].get("use_mmr", True)
    use_ce_cfg = CFG.get("reranker", {}).get("use_ce", True)
    use_llm_cfg = CFG.get("reranker", {}).get("use_llm", False)
    context_compression = CFG["retrieval"].get("context_compression", False)

    embedder = _get_embedder()
    llm = get_llm()

    retriever = MongoHybridRetriever(
        embedder=embedder,
        llm=llm,
        collection=collection,
        use_mmr=use_mmr_cfg,
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
                prompt = RAG_PROMPT.format_prompt(context=context, question=question)
                resp = llm.invoke(prompt)
            return resp
        except Exception as e:
            logger.exception("LLM generation failed: %s", e)
            raise

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

    return {
        "chain": answer_chain,
        "answer_chain": answer_chain,
        "response_chain": response_chain,
        "retrieval_engine": retrieval_engine,
        "ce_reranker": ce_reranker,
        "llm_reranker": llm_reranker,
    }


def build_chain(use_llm_reranker: bool = False):
    """Return the answer-only runnable."""
    return build_chain_components(use_llm_reranker)["chain"]


def ask_with_sources(
    question: str, use_llm_reranker: bool = False, max_sources: int = 5
) -> Dict[str, Any]:
    cache = get_redis_cache()
    normalized_question = (question or "").strip()
    cache_variant = build_cache_key("answer_variant", use_llm_reranker, max_sources)
    cache_key = build_cache_key("answer", normalized_question, use_llm_reranker, max_sources)

    cached_answer = cache.get_json("answer", cache_key)
    if cached_answer is not None:
        logger.info("Redis answer cache hit for question: %s", normalized_question)
        return cached_answer

    embedder = _get_embedder()
    question_embedding = embedder.encode(normalized_question, normalize_embeddings=True).tolist()
    if _SEMANTIC_CACHE_ENABLED:
        cached_answer = cache.get_semantic_json(
            "answer_semantic",
            question_embedding,
            min_similarity=_SEMANTIC_CACHE_THRESHOLD,
            variant=cache_variant,
        )
        if cached_answer is not None:
            logger.info("Redis semantic answer cache hit for question: %s", normalized_question)
            return cached_answer

    response_chain = build_chain_components(use_llm_reranker)["response_chain"]
    result = response_chain.invoke({"question": question})
    payload = {
        "answer": result["answer"],
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


def ask_stream(question: str, use_llm_reranker: bool = False):
    answer_chain = build_chain_components(use_llm_reranker)["answer_chain"]
    for chunk in answer_chain.stream({"question": question}):
        if chunk is None:
            continue

        if isinstance(chunk, str):
            yield _normalize_output(chunk)
            continue

        if isinstance(chunk, dict):
            if "answer" in chunk and chunk["answer"] is not None:
                yield _normalize_output(chunk["answer"])
                continue
            if "content" in chunk and chunk["content"] is not None:
                yield _normalize_output(chunk["content"])
                continue
            continue

        if hasattr(chunk, "content"):
            content = getattr(chunk, "content")
            if content is not None:
                yield _normalize_output(content)
                continue

        if hasattr(chunk, "answer"):
            answer = getattr(chunk, "answer")
            if answer is not None:
                yield _normalize_output(answer)
                continue

        yield _normalize_output(str(chunk))


def ask_stream_with_sources(
    question: str,
    use_llm_reranker: bool = False,
    max_sources: int = 5,
):
    response_chain = build_chain_components(use_llm_reranker)["response_chain"]
    documents = None

    for chunk in response_chain.stream({"question": question}):
        if chunk is None:
            continue

        if isinstance(chunk, dict):
            if documents is None and "documents" in chunk:
                documents = chunk["documents"]

            if "answer" in chunk and chunk["answer"] is not None:
                yield {"type": "delta", "text": _normalize_output(chunk["answer"])}
                continue
            if "content" in chunk and chunk["content"] is not None:
                yield {"type": "delta", "text": _normalize_output(chunk["content"])}
                continue

            # Ignore internal routing/context metadata chunks that are not actual answer text.
            if set(chunk.keys()) <= {"question", "datasource", "documents", "context"}:
                continue

            # If the dict contains unknown fields, only emit if it has usable text.
            if "text" in chunk and isinstance(chunk["text"], str):
                yield {"type": "delta", "text": _normalize_output(chunk["text"])}
                continue
            continue

        if isinstance(chunk, str):
            yield {"type": "delta", "text": _normalize_output(chunk)}
            continue

        if hasattr(chunk, "content"):
            content = getattr(chunk, "content")
            if content is not None:
                yield {"type": "delta", "text": _normalize_output(content)}
                continue

        if hasattr(chunk, "answer"):
            answer = getattr(chunk, "answer")
            if answer is not None:
                yield {"type": "delta", "text": _normalize_output(answer)}
                continue

        # Ignore all other chunks that are not answer content.
        continue

    yield {"type": "sources", "sources": _build_sources(documents or [], max_sources)}


@traceable(name="JUSNL-RAG-Pipeline")
def ask(question: str, chain=None) -> str:
    """Ask a question and return a grounded answer."""
    chain = chain or build_chain()

    try:
        answer = chain.invoke({"question": question})
    except Exception as exc:
        logger.exception("Chain invocation failed: %s", exc)
        return f"Chain error: {type(exc).__name__}: {exc}"

    if answer is None:
        logger.warning("Chain returned None for question: %s", question)
        return "Chain returned no output (None) - check logs"

    return answer

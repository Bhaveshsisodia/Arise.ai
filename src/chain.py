"""
chain.py - LCEL chain assembly + LangSmith tracing.
"""

import os
from functools import lru_cache
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors.chain_extract import (
    LLMChainExtractor,
)
from langsmith import traceable
from sentence_transformers import SentenceTransformer

from src.config import CFG
from src.generator import RAG_PROMPT, build_context, get_llm, output_parser
from src.reranker import CrossEncoderReranker, LLMListwiseReranker
from src.retriever import MongoHybridRetriever
from src.utils.logger import pipeline_logger as logger
from src.vector_db import collection

load_dotenv()


def setup_langsmith() -> None:
    """Enable LangSmith tracing when credentials are available."""
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "jusnl-rag")


def _normalize_output(output: Any) -> str:
    if isinstance(output, str):
        return output
    return getattr(output, "content", str(output))


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
    for doc in documents[:max_sources]:
        meta = doc.metadata
        sources.append(
            {
                "section": meta.get("section_heading", "Unknown"),
                "pages": f"{meta.get('page_start', '?')}-{meta.get('page_end', '?')}",
                "cross_encoder_score": meta.get("cross_encoder_score"),
                "llm_rank": meta.get("llm_rank"),
                "snippet": doc.page_content[:250].replace("\n", " "),
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
            lambda x: retrieval_engine.invoke(x["question"])
        ).with_config({"run_name": "Stage1-Retrieval"})
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

    generation_chain = (
        RunnableLambda(lambda x: {"context": x["context"], "question": x["question"]})
        | RAG_PROMPT
        | llm
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
    response_chain = build_chain_components(use_llm_reranker)["response_chain"]
    result = response_chain.invoke({"question": question})
    return {
        "answer": result["answer"],
        "sources": _build_sources(result["documents"], max_sources),
    }


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

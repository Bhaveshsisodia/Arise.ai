"""
chain.py — LCEL chain assembly + LangSmith tracing.

This is the core of the refactor. Your existing pipeline:

    hybrid_retrieve() → mmr() → stage2_crossencoder_rerank()
    → stage3_llm_listwise_rerank() → build_context() → llm

Becomes a composable LCEL chain:

    retriever | ce_reranker | llm_reranker | prompt | llm | parser

Why LCEL?
  - Every | step is a Runnable — composable, streamable, traceable
  - LangSmith automatically traces every step when env vars are set
  - .stream() works out of the box for token-by-token streaming
  - Easy to swap any component (e.g. different retriever or LLM)

Usage:
    from src.chain import build_chain, ask

    chain = build_chain()
    answer = ask("What employee expenses has JUSNL projected?", chain)
"""

import os
from typing import List, Dict, Any

from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

from langchain_classic.retrievers import ContextualCompressionRetriever

# 2. Import your chosen compressor from langchain-community or langchain-classic
from langchain_classic.retrievers.document_compressors.chain_extract import LLMChainExtractor
from src.config import CFG
from src.vector_db import collection
from src.retriever import MongoHybridRetriever
from src.reranker import CrossEncoderReranker, LLMListwiseReranker
from src.generator import get_llm, build_context, RAG_PROMPT, output_parser
from src.utils.logger import pipeline_logger as logger

load_dotenv()


# ============================================================
# LANGSMITH TRACING SETUP
# Just set these two env vars → every chain call is traced.
# No other code changes needed.
#
# Add to your .env:
#   LANGCHAIN_TRACING_V2=true
#   LANGCHAIN_API_KEY=your_langsmith_key
#   LANGCHAIN_PROJECT=jusnl-rag
# ============================================================

def setup_langsmith():
    """
    Activates LangSmith tracing if env vars are present.
    Safe to call even if LangSmith is not configured —
    it will just skip tracing silently.
    """
    if os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"]     = os.getenv(
            "LANGCHAIN_PROJECT", "jusnl-rag"
        )
        print("✅ LangSmith tracing enabled")
    else:
        print("ℹ️  LangSmith not configured — set LANGCHAIN_API_KEY to enable tracing")


# ============================================================
# RERANKER ADAPTER
# The rerankers expect {"query": str, "documents": List[Document]}
# but the LCEL chain passes state as a dict.
# This adapter extracts the right fields.
# ============================================================

def _apply_ce_reranker(input: Dict[str, Any], reranker: CrossEncoderReranker) -> List[Document]:
    return reranker.invoke({
        "query":     input["question"],
        "documents": input["documents"],
    })

def _apply_llm_reranker(input: Dict[str, Any], reranker: LLMListwiseReranker) -> List[Document]:
    return reranker.invoke({
        "query":     input["question"],
        "documents": input["documents"],
    })


# ============================================================
# CHAIN BUILDER
# ============================================================

def build_chain(use_llm_reranker: bool = True):
    """
    Assembles the full 3-stage RAG chain using LCEL.

    Pipeline:
        Input: {"question": str}
            ↓
        Meta Data Filtering
            ↓
        retriever        → List[Document]  (Stage 1: hybrid RRF + MMR)
            ↓
        ce_reranker      → List[Document]  (Stage 2: cross-encoder)
            ↓
        llm_reranker     → List[Document]  (Stage 3: LLM listwise)
            ↓
        build_context    → str             (format docs into context)
            ↓
        RAG_PROMPT       → ChatPromptValue (insert context + question)
            ↓
        llm              → AIMessage       (generate answer)
            ↓
        output_parser    → str             (extract content)

    Args:
        use_llm_reranker: set False to skip Stage 3 (faster, saves LLM call)

    Returns:
        Runnable LCEL chain
    """
    setup_langsmith()

    # ── Load models ───────────────────────────────────────────
    print("Loading embedding model...")
    embedder = SentenceTransformer(CFG["embedding"]["model"])
    print(f"✅ Embedder loaded | dim={embedder.get_sentence_embedding_dimension()}")

    llm = get_llm()
    print(f"✅ LLM loaded | model={CFG['llm']['model']}")

    # ── Instantiate components ────────────────────────────────
    # read toggles from config
    use_mmr_cfg = CFG["retrieval"].get("use_mmr", True)
    use_ce_cfg = CFG.get("reranker", {}).get("use_ce", True)
    use_llm_cfg = CFG.get("reranker", {}).get("use_llm", True)
    cont_comp_cfg = CFG["retrieval"].get("context_compression", True)

    logger.info(
        "build_chain | embed_model=%s use_mmr=%s use_ce=%s use_llm=%s llm_model=%s context_compression=%s",
        CFG["embedding"]["model"],
        use_mmr_cfg,
        use_ce_cfg,
        use_llm_cfg,
        CFG["llm"]["model"],
        cont_comp_cfg

    )
    from src.utils.logger import pipeline_event
    pipeline_event(
        "chain.build",
        embed_model=CFG["embedding"]["model"],
        use_mmr=use_mmr_cfg,
        use_ce=use_ce_cfg,
        use_llm=use_llm_cfg,
        llm_model=CFG["llm"]["model"],
        context_compression = cont_comp_cfg
    )

    retriever = MongoHybridRetriever(
        embedder=embedder,
        llm=llm,
        collection=collection,
        use_mmr=use_mmr_cfg,
    )

# OR if using another compressor (e.g., LLMLingua):
# from langchain_community.document_compressors import LLMLinguaCompressor

# 3. Setup your workflow

    compressor = LLMChainExtractor.from_llm(llm)

    # 4. Wrap your base vector store retriever
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=retriever
    )
    retrieval_engine = (
    compression_retriever
    if cont_comp_cfg
    else retriever)

    logger.info(
    "Retriever Used: %s",
    retrieval_engine.__class__.__name__)


    ce_reranker = CrossEncoderReranker() if use_ce_cfg else None
    llm_reranker = LLMListwiseReranker(llm=llm) if use_llm_cfg else None

    # ── LCEL chain definition ─────────────────────────────────
    #
    # RunnablePassthrough() keeps "question" flowing through
    # while we add "documents" and "context" at each step.
    #
    # Step by step:
    #   1. {"question": q} → retriever → {"question": q, "documents": [...]}
    #   2. ce_reranker re-scores documents
    #   3. llm_reranker globally re-orders documents
    #   4. build_context formats docs → context string
    #   5. RAG_PROMPT inserts context + question
    #   6. llm generates answer
    #   7. output_parser extracts string

    # Stage 1: retrieve
    stage1 = RunnablePassthrough.assign(
        documents = RunnableLambda(lambda x: retrieval_engine.invoke(x["question"])).with_config({"run_name": "Stage1-Retrieval"})
    )

    # Stage 2: cross-encoder rerank (optional)
    if ce_reranker is not None:
        stage2 = RunnablePassthrough.assign(
            documents=RunnableLambda(
                lambda x: _apply_ce_reranker(x, ce_reranker)
            ).with_config({"run_name": "Stage2-CrossEncoder"})
        )
    else:
        stage2 = RunnablePassthrough()

    # Stage 3: LLM listwise rerank (optional via config)
    final_use_llm = use_llm_reranker and (llm_reranker is not None)
    if final_use_llm:
        stage3 = RunnablePassthrough.assign(
            documents=RunnableLambda(
                lambda x: _apply_llm_reranker(x, llm_reranker)
            ).with_config({"run_name": "Stage3-LLMReranker"})
        )
    else:
        stage3 = RunnablePassthrough()

    # Context formatting + generation
    # Ensure the final output is a plain string regardless of LLM message object.
    # Some LLM implementations return AIMessage-like objects; normalise them
    # to a string before returning so downstream callers and LangSmith see
    # consistent output instead of `null`.
    generation = (
        RunnablePassthrough.assign(
            context = RunnableLambda(lambda x: build_context(x["documents"]))
        )
        | RAG_PROMPT
        | llm
        | output_parser
        | RunnableLambda(lambda out: out if isinstance(out, str) else getattr(out, "content", str(out)))
    )

    # Full chain
    chain = stage1 | stage2 | stage3 | generation

    return chain.with_config({
    "run_name": "JUSNL-RAG-Pipeline"   # ← this becomes the parent in LangSmith
})


# ============================================================
# PUBLIC ask() FUNCTION
# ============================================================
from langsmith import traceable

@traceable(name="JUSNL-RAG-Pipeline")
def ask(question: str, chain=None) -> str:
    """
    Ask a question and get an answer from the full RAG pipeline.

    Args:
        question: natural language question
        chain:    pre-built LCEL chain (builds one if not provided)

    Returns:
        str — LLM answer grounded in retrieved documents
    """
    if chain is None:
        chain = build_chain()

    print(f"\nQuestion: {question}")
    print("=" * 60)

    try:
        answer = chain.invoke({"question": question})
    except Exception as e:
        logger.exception("Chain invocation failed: %s", e)
        print("Chain invocation raised an exception:", type(e).__name__, e)
        return f"Chain error: {type(e).__name__}: {e}"

    # Defensive diagnostics: ensure callers see something useful rather than None
    if answer is None:
        logger.warning("Chain returned None for question: %s", question)
        print("Chain returned None — check LangSmith trace for step errors.")
        return "Chain returned no output (None) — check logs"

    print("\nAnswer:")
    print(repr(answer))
    return answer


# ============================================================
# STREAMING VERSION
# Works out of the box because every step is a Runnable.
# No extra code needed — LCEL handles it automatically.
# ============================================================
@traceable(name="JUSNL-RAG-Pipeline-Stream")

def ask_stream(question: str, chain=None):
    """
    Same as ask() but streams tokens as they arrive.
    Useful for Streamlit or FastAPI streaming endpoints.
    """
    if chain is None:
        chain = build_chain()

    print(f"\nQuestion: {question}")
    print("=" * 60)

    for chunk in chain.stream({"question": question}):
        print(chunk, end="", flush=True)

    try:
        answer = chain.invoke({"question": question})
    except Exception as e:
        logger.exception("Chain invocation failed: %s", e)
        print("Chain invocation raised an exception:", type(e).__name__, e)
        return f"Chain error: {type(e).__name__}: {e}"

    return answer
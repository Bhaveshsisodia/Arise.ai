"""
retriever.py — Hybrid RRF retrieval + MMR diversity.

Implements LangChain's BaseRetriever interface so it
plugs directly into LCEL chains as a Runnable.

Stages:
  1. Vector search ($vectorSearch on MongoDB Atlas)
  2. BM25 text search ($text on MongoDB)
  3. RRF fusion of both ranked lists
  4. MMR for diversity (optional)

Usage:
    from src.retriever import MongoHybridRetriever
    retriever = MongoHybridRetriever(embedder=embedder, llm=llm, collection=collection)
    docs = retriever.invoke("What employee expenses has JUSNL projected?")
"""

import numpy as np
from collections import defaultdict
from typing import List , Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun

try:
    from sentence_transformers import util
except Exception as exc:  # pragma: no cover - optional dependency at runtime
    util = None
    _SENTENCE_TRANSFORMERS_UTIL_IMPORT_ERROR = exc
else:
    _SENTENCE_TRANSFORMERS_UTIL_IMPORT_ERROR = None

from src.config import CFG
from src.exception.custom_exception import (
    DatabaseError,
    EmbeddingError,
    QueryRewriteError,
    RetrievalError,
    ValidationError,
)
from src.exception.error_utils import raise_with_context
from src.utils.logger import pipeline_logger as logger, pipeline_event
from src.metadata_filter import get_query_filters
from src.query_optimizer import hyde_rewrite, multi_query_rewrite, stepback_rewrite
from src.utils.redis_cache import build_cache_key, get_redis_cache

_SEMANTIC_CACHE_THRESHOLD = float(CFG.get("redis", {}).get("semantic_similarity_threshold", 0.92))
_SEMANTIC_CACHE_ENABLED = bool(CFG.get("redis", {}).get("semantic_enabled", True))

# ============================================================
# HELPER: dict → LangChain Document
# LangChain chains expect Document objects (page_content + metadata)
# ============================================================
_embedding_cache: dict = {}

def _get_embedding(text: str, embedder) -> list:
    """
    Returns cached embedding if available, otherwise computes and caches.
    Key: (text, id(embedder)) — safe across multiple embedder instances.
    """
    cache_key = (text, id(embedder))
    if cache_key not in _embedding_cache:
        try:
            _embedding_cache[cache_key] = embedder.encode(
                text, normalize_embeddings=True
            ).tolist()
        except Exception as exc:
            raise_with_context(
                EmbeddingError,
                exc,
                "Failed to compute query embedding",
                context={"text_preview": text[:120]},
            )
    return _embedding_cache[cache_key]


def clear_embedding_cache():
    """Call between sessions to free memory."""
    _embedding_cache.clear()


# ============================================================
# HELPER: dict → LangChain Document
# ============================================================

def _to_document(doc: dict) -> Document:
    return Document(
        page_content=doc["text"],
        metadata={
            **doc.get("metadata", {}),
            "rrf_score":           doc.get("rrf_score", 0),
            "mmr_score":           doc.get("mmr_score", 0),
            "cross_encoder_score": doc.get("cross_encoder_score", 0),
            "_id":                 str(doc.get("_id", "")),
            "_embedding":          doc.get("embedding", []),
        }
    )


# ============================================================
# CORE: Vector search helper
# Shared by all retriever types — avoids code duplication.
# ============================================================

def _vector_search(
    search_vector: list,
    collection,
    mongo_filter: dict,
    vec_candidates: int,
    limit: int,
) -> list:
    """
    Runs $vectorSearch with an optional metadata filter.
    Falls back to no-filter if filter returns 0 results.
    """
    vector_config = {
        "index":         CFG["mongodb"]["vector_index"],
        "path":          "embedding",
        "queryVector":   search_vector,
        "numCandidates": vec_candidates,
        "limit":         limit,
    }
    if mongo_filter:
        vector_config["filter"] = mongo_filter

    project = {"$project": {
        "_id": 1, "text": 1, "metadata": 1, "embedding": 1,
        "vec_score": {"$meta": "vectorSearchScore"}
    }}

    try:
        results = list(collection.aggregate([{"$vectorSearch": vector_config}, project]))
    except Exception as exc:
        raise_with_context(
            DatabaseError,
            exc,
            "MongoDB vector search failed",
            context={"limit": limit, "num_candidates": vec_candidates},
        )

    # Fallback: retry without filter if nothing returned
    if not results and mongo_filter:
        no_filter = {k: v for k, v in vector_config.items() if k != "filter"}
        try:
            results = list(collection.aggregate([{"$vectorSearch": no_filter}, project]))
        except Exception as exc:
            raise_with_context(
                DatabaseError,
                exc,
                "MongoDB vector search fallback failed",
                context={"limit": limit, "num_candidates": vec_candidates},
            )

    return results


# ============================================================
# CORE: RRF fusion helper
# Merges multiple ranked lists into one fused ranking.
# ============================================================

def _rrf_fuse(
    ranked_lists: List[List[dict]],
    weights: Optional[List[float]] = None,
    top_k: int = 50,
    k: int = 60,
) -> List[dict]:
    """
    Reciprocal Rank Fusion across multiple result lists.

    Args:
        ranked_lists: list of result lists, each already ranked
        weights:      per-list weights (default: all 1.0)
        top_k:        how many to return
        k:            RRF constant (60 is standard)

    Returns:
        Fused, re-ranked list with combined_rrf_score added
    """
    if weights is None:
        weights = [1.0] * len(ranked_lists)

    rrf_scores = defaultdict(float)
    docs = {}

    for result_list, weight in zip(ranked_lists, weights):
        for rank, doc in enumerate(result_list, start=1):
            doc_id = str(doc["_id"])
            rrf_scores[doc_id] += weight / (k + rank)
            if doc_id not in docs:
                docs[doc_id] = doc

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    results = []
    for doc_id, score in ranked[:top_k]:
        doc = docs[doc_id]
        doc["rrf_score"] = round(score, 6)
        results.append(doc)

    return results


# ============================================================
# PURE FUNCTIONS: standalone, testable, reusable
# ============================================================

def hybrid_retrieve(
    query:          str,
    embedder,
    llm,
    collection,
    top_k:          int = None,
    vec_candidates: int = None,
    text_limit:     int = None,
) -> List[dict]:
    """
    Baseline: hybrid RRF (vector + BM25 text search).

    Returns raw dicts with rrf_score added.
    """
    cfg            = CFG["retrieval"]
    top_k          = top_k          or cfg["stage1_k"]
    vec_candidates = vec_candidates or cfg["vec_candidates"]
    text_limit     = text_limit     or cfg["text_limit"]

    use_vector = cfg.get("use_vector", True)
    use_text = cfg.get("use_text", True)

    pipeline_event(
        "hybrid_retrieve.start",
        use_vector=use_vector,
        use_text=use_text,
        top_k=top_k,
        vec_candidates=vec_candidates,
        text_limit=text_limit,
    )

    mongo_filter = get_query_filters(query, llm, collection)

    vector_results = []
    text_results = []

    # Vector search (optional)
    if use_vector:
        query_embedding = _get_embedding(query, embedder)
        vector_results = _vector_search(
            query_embedding, collection, mongo_filter, vec_candidates, text_limit
        )

    # BM25 text search (optional)
    if use_text:
        text_query = {"$text": {"$search": query}}
        if mongo_filter:
            text_query.update(mongo_filter)

        try:
            text_results = list(
                collection.find(
                    text_query,
                    {"text": 1, "metadata": 1, "embedding": 1, "text_score": {"$meta": "textScore"}}
                )
                .sort([("text_score", {"$meta": "textScore"})])
                .limit(text_limit)
            )
        except Exception as exc:
            raise_with_context(
                DatabaseError,
                exc,
                "MongoDB text search failed",
                context={"query": query, "text_limit": text_limit},
            )

    if not (use_vector or use_text):
        raise ValidationError(
            "Both vector and text search are disabled in config; enable at least one of 'use_vector' or 'use_text'."
        )

    pipeline_event(
        "hybrid_retrieve.results",
        vector_count=len(vector_results),
        text_count=len(text_results),
    )

    ranked_lists = [rl for rl in (vector_results, text_results) if rl]
    if not ranked_lists:
        pipeline_event("hybrid_retrieve.empty")
        return []

    fused = _rrf_fuse(ranked_lists, top_k=top_k)
    pipeline_event("hybrid_retrieve.fused", fused_count=len(fused))
    return fused


def mmr(
    query:        str,
    documents:    List[dict],
    embedder,
    lambda_param: float = None,
    top_k:        int   = None,
) -> List[dict]:
    """
    Maximal Marginal Relevance for diversity — vectorized.

    Uses numpy matrix operations instead of Python loops.
    Speed: O(n) matrix ops vs O(n²) loop → ~60x faster on 50 docs.

    lambda_param=1.0 → pure relevance, 0.0 → pure diversity.
    """
    cfg          = CFG["retrieval"]
    lambda_param = lambda_param or cfg["mmr_lambda"]
    top_k        = top_k        or cfg["stage2_k"]

    pipeline_event("mmr.start", lambda_param=lambda_param, top_k=top_k, input_docs=len(documents))

    if not documents:
        pipeline_event("mmr.empty")
        return documents

    # Pre-compute ALL embeddings as a single matrix — one numpy op
    query_vec  = np.array(_get_embedding(query, embedder))                                               # shape: (dim,)

    doc_matrix = np.array(
        [doc["embedding"] for doc in documents]
    )                                               # shape: (n_docs, dim)

    # Relevance scores: dot product of query vs all docs (vectorized)
    # Works because embeddings are already normalized → cosine = dot product
    relevance_scores = doc_matrix @ query_vec       # shape: (n_docs,)

    selected_indices  = []
    candidate_indices = list(range(len(documents)))

    while len(selected_indices) < top_k and candidate_indices:
        if not selected_indices:
            # First pick: highest relevance, no diversity penalty yet
            mmr_scores = relevance_scores[candidate_indices]
        else:
            # Diversity: max similarity to ANY already-selected doc
            # selected_matrix shape: (n_selected, dim)
            selected_matrix = doc_matrix[selected_indices]

            # candidate_matrix shape: (n_candidates, dim)
            candidate_matrix = doc_matrix[candidate_indices]

            # similarity_matrix shape: (n_candidates, n_selected)
            # each row = similarities of one candidate to all selected docs
            similarity_matrix = candidate_matrix @ selected_matrix.T

            # Max similarity to any selected doc (worst-case diversity penalty)
            max_similarity = similarity_matrix.max(axis=1)  # shape: (n_candidates,)

            # MMR score = relevance - diversity_penalty
            mmr_scores = (
                lambda_param * relevance_scores[candidate_indices]
                - (1 - lambda_param) * max_similarity
            )

        best_local_idx = int(np.argmax(mmr_scores))
        best_doc_idx   = candidate_indices.pop(best_local_idx)

        documents[best_doc_idx]["mmr_score"] = float(
            relevance_scores[best_doc_idx]
        )
        selected_indices.append(best_doc_idx)

    selected = [documents[i] for i in selected_indices]
    pipeline_event("mmr.selected", selected_count=len(selected))
    return selected


# ============================================================
# LANGCHAIN RETRIEVER CLASS
# Wraps hybrid_retrieve + mmr into a BaseRetriever so it
# plugs into LCEL chains with .invoke() / .stream().
#
# Why BaseRetriever?
#   LangChain LCEL chains expect Runnable objects.
#   BaseRetriever implements Runnable automatically —
#   you just define _get_relevant_documents().
# ============================================================

class MongoHybridRetriever(BaseRetriever):
    """
    LangChain-compatible retriever wrapping:
      hybrid_retrieve() → mmr() → List[Document]

    Implements BaseRetriever so it's directly composable
    in LCEL chains:

        chain = retriever | reranker | prompt | llm
    """

    embedder:   object   # SentenceTransformer
    llm:        object   # LangChain LLM
    collection: object   # pymongo Collection
    use_mmr:    bool = CFG["retrieval"].get("use_mmr", True)

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:

        cfg = CFG["retrieval"]
        original_query = query

        cfg_query = CFG.get('query_rewriting', {})
        strategy = cfg_query.get('default_strategy', 'stepback')
        pipeline_event(
            "query_rewrite.strategy",
            strategy=strategy,
            allow_hyde=cfg_query.get('allow_hyde', True),
            allow_multi=cfg_query.get('allow_multi', True),
            allow_stepback=cfg_query.get('allow_stepback', True),
        )

        try:
            if strategy == "hyde" and cfg_query.get('allow_hyde', True):
                query = hyde_rewrite(query, self.llm)
            elif strategy == "multi" and cfg_query.get('allow_multi', True):
                query = multi_query_rewrite(query, self.llm)
            elif strategy == "stepback" and cfg_query.get('allow_stepback', True):
                query = stepback_rewrite(query, self.llm)
        except QueryRewriteError as exc:
            logger.warning("Query rewrite failed; using original query: %s", exc)
            query = original_query

        pipeline_event(
            "query_rewrite.attempt",
            original_query=original_query,
            rewritten_query=query,
            strategy=strategy,
            rewritten=query != original_query,
        )

        if query != original_query:
            logger.info("Query rewritten | strategy=%s", strategy)

        cache = get_redis_cache()
        cache_variant = build_cache_key("retrieval_variant", cfg["stage1_k"], self.use_mmr)
        cache_key = build_cache_key("retrieval", query, cfg["stage1_k"], self.use_mmr)
        cached_candidates = cache.get_json("retrieval", cache_key)
        if cached_candidates is not None:
            logger.info("Redis retrieval cache hit for query: %s", query)
            return [_to_document(doc) for doc in cached_candidates]

        query_embedding = _get_embedding(query, self.embedder)
        if _SEMANTIC_CACHE_ENABLED:
            cached_candidates = cache.get_semantic_json(
                "retrieval_semantic",
                query_embedding,
                min_similarity=_SEMANTIC_CACHE_THRESHOLD,
                variant=cache_variant,
            )
            if cached_candidates is not None:
                logger.info("Redis semantic retrieval cache hit for query: %s", query)
                return [_to_document(doc) for doc in cached_candidates]

        # Stage 1: Hybrid RRF
        try:
            candidates = hybrid_retrieve(
                query=query,
                embedder=self.embedder,
                llm=self.llm,
                collection=self.collection,
                top_k=cfg["stage1_k"],
            )
        except Exception as exc:
            if isinstance(exc, (DatabaseError, EmbeddingError, RetrievalError, ValidationError)):
                raise
            raise_with_context(
                RetrievalError,
                exc,
                "Hybrid retrieval failed",
                context={"query": query},
            )

        # Optional: MMR diversity
        if self.use_mmr and candidates:
            try:
                candidates = mmr(
                    query=query,
                    documents=candidates,
                    embedder=self.embedder,
                    top_k=cfg["stage1_k"],
                )
            except Exception as exc:
                if isinstance(exc, EmbeddingError):
                    raise
                raise_with_context(
                    RetrievalError,
                    exc,
                    "MMR reranking failed",
                    context={"query": query, "candidate_count": len(candidates)},
                )

        if candidates:
            cache.set_json("retrieval", cache_key, candidates, ttl=1800)
            if _SEMANTIC_CACHE_ENABLED:
                cache.set_semantic_json(
                    "retrieval_semantic",
                    cache_key,
                    query_text=query,
                    query_vector=query_embedding,
                    value=candidates,
                    ttl=1800,
                    variant=cache_variant,
                )

        # Convert raw dicts → LangChain Documents
        return [_to_document(doc) for doc in candidates]

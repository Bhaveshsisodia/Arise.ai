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
from typing import List

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from sentence_transformers import util

from src.config import CFG
from src.metadata_filter import get_query_filters
from src.query_optimizer import hyde_rewrite, multi_query_rewrite, stepback_rewrite

# ============================================================
# HELPER: dict → LangChain Document
# LangChain chains expect Document objects (page_content + metadata)
# ============================================================

def _to_document(doc: dict) -> Document:
    return Document(
        page_content=doc["text"],
        metadata={
            **doc.get("metadata", {}),
            "rrf_score":            doc.get("rrf_score", 0),
            "mmr_score":            doc.get("mmr_score", 0),
            "cross_encoder_score":  doc.get("cross_encoder_score", 0),
            "_id":                  str(doc.get("_id", "")),
            "_embedding":           doc.get("embedding", []),
        }
    )


# ============================================================
# PURE FUNCTION: Hybrid RRF retrieval
# Kept as a standalone function so it's testable independently.
# ============================================================

def hybrid_retrieve(
    query:          str,
    embedder,
    llm,
    collection,
    top_k:          int  = None,
    vec_candidates: int  = None,
    text_limit:     int  = None,
) -> List[dict]:
    """
    Stage 1: Hybrid RRF retrieval.

    Runs vector search + BM25 text search in parallel,
    then fuses their rankings using Reciprocal Rank Fusion.

    Args:
        query:          user question
        embedder:       SentenceTransformer model
        llm:            LangChain LLM (for metadata filter extraction)
        collection:     pymongo Collection
        top_k:          number of results to return (default from config)
        vec_candidates: numCandidates for $vectorSearch (default from config)
        text_limit:     BM25 result limit (default from config)

    Returns:
        List of raw dicts with rrf_score added
    """
    cfg = CFG["retrieval"]
    top_k          = top_k          or cfg["stage1_k"]
    vec_candidates = vec_candidates or cfg["vec_candidates"]
    text_limit     = text_limit     or cfg["text_limit"]
    k              = cfg["rrf_k"]

    # ── Metadata filter ────────────────────────────────────────
    mongo_filter = get_query_filters(query, llm, collection)

    # ── Query embedding ────────────────────────────────────────
    query_embedding = embedder.encode(
        query, normalize_embeddings=True
    ).tolist()

    # ── Vector search ──────────────────────────────────────────
    vector_config = {
        "index":        CFG["mongodb"]["vector_index"],
        "path":         "embedding",
        "queryVector":  query_embedding,
        "numCandidates": vec_candidates,
        "limit":        text_limit,
    }
    if mongo_filter:
        vector_config["filter"] = mongo_filter

    vector_pipeline = [
        {"$vectorSearch": vector_config},
        {"$project": {
            "_id": 1, "text": 1, "metadata": 1, "embedding": 1,
            "vec_score": {"$meta": "vectorSearchScore"}
        }}
    ]
    vector_results = list(collection.aggregate(vector_pipeline))

    # Fallback: if filter returns nothing, retry without filter
    if not vector_results and mongo_filter:
        no_filter_config = {k: v for k, v in vector_config.items() if k != "filter"}
        vector_results = list(collection.aggregate([
            {"$vectorSearch": no_filter_config},
            {"$project": {
                "_id": 1, "text": 1, "metadata": 1, "embedding": 1,
                "vec_score": {"$meta": "vectorSearchScore"}
            }}
        ]))

    # ── BM25 text search ───────────────────────────────────────
    text_query = {"$text": {"$search": query}}
    if mongo_filter:
        text_query.update(mongo_filter)

    text_results = list(
        collection.find(
            text_query,
            {"text": 1, "metadata": 1, "embedding": 1,
             "text_score": {"$meta": "textScore"}}
        )
        .sort([("text_score", {"$meta": "textScore"})])
        .limit(text_limit)
    )

    # ── RRF fusion ─────────────────────────────────────────────
    rrf_scores = defaultdict(float)
    docs = {}

    for rank, doc in enumerate(vector_results, start=1):
        doc_id = str(doc["_id"])
        rrf_scores[doc_id] += 1 / (k + rank)
        docs[doc_id] = doc

    for rank, doc in enumerate(text_results, start=1):
        doc_id = str(doc["_id"])
        rrf_scores[doc_id] += 1 / (k + rank)
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
# PURE FUNCTION: MMR diversity
# ============================================================

def mmr(
    query:       str,
    documents:   List[dict],
    embedder,
    lambda_param: float = None,
    top_k:        int   = None,
) -> List[dict]:
    """
    Maximal Marginal Relevance for diversity.

    Balances relevance to the query with diversity among
    selected documents. lambda_param=1.0 → pure relevance,
    lambda_param=0.0 → pure diversity.
    """
    cfg          = CFG["retrieval"]
    lambda_param = lambda_param or cfg["mmr_lambda"]
    top_k        = top_k        or cfg["stage2_k"]

    query_embedding = embedder.encode(
        query, normalize_embeddings=True
    ).tolist()

    selected   = []
    candidates = documents.copy()

    while len(selected) < top_k and candidates:
        mmr_scores = []

        for doc in candidates:
            relevance = util.cos_sim(
                query_embedding, doc["embedding"]
            ).item()

            diversity = max(
                (util.cos_sim(doc["embedding"], s["embedding"]).item()
                 for s in selected),
                default=0
            )

            score = lambda_param * relevance - (1 - lambda_param) * diversity
            mmr_scores.append(score)

        best_idx = int(np.argmax(mmr_scores))
        best_doc = candidates.pop(best_idx)
        best_doc["mmr_score"] = float(mmr_scores[best_idx])
        selected.append(best_doc)

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
    use_mmr:    bool = True

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None,
    ) -> List[Document]:

        cfg = CFG["retrieval"]

        cfg_query=CFG['query_rewriting']
        if cfg_query['default_strategy'] == "hyde":
            query = hyde_rewrite(query , self.llm)
        elif cfg_query['default_strategy'] == "multi":
            query = multi_query_rewrite(query , self.llm)
        else:
            query = stepback_rewrite(query , self.llm)

        print("multi:")






        # Stage 1: Hybrid RRF
        candidates = hybrid_retrieve(
            query      = query,
            embedder   = self.embedder,
            llm        = self.llm,
            collection = self.collection,
            top_k      = cfg["stage1_k"],
        )

        # Optional: MMR diversity
        if self.use_mmr and candidates:
            candidates = mmr(
                query    = query,
                documents= candidates,
                embedder = self.embedder,
                top_k    = cfg["stage1_k"],
            )

        # Convert raw dicts → LangChain Documents
        return [_to_document(doc) for doc in candidates]
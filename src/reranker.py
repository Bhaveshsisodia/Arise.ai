"""
reranker.py — Two-stage reranking as LangChain Runnables.

Stage 2: Cross-encoder reranking (bi-encoder → cross-encoder)
Stage 3: LLM listwise reranking (pairwise → global ordering)

Both are implemented as Runnable so they plug into LCEL chains.

Usage:
    from src.reranker import CrossEncoderReranker, LLMListwiseReranker
    reranker = CrossEncoderReranker()
    docs = reranker.invoke({"query": query, "documents": docs})
"""

import re
import json
from typing import List, Dict, Any

from langchain_core.runnables import Runnable
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

from src.config import CFG
from src.utils.logger import pipeline_logger as logger, pipeline_event


# ============================================================
# STAGE 2: Cross-encoder reranker
# ============================================================

class CrossEncoderReranker(Runnable):
    """
    Re-scores each (query, document) pair using a cross-encoder.

    Unlike a bi-encoder, the cross-encoder sees the query and
    document TOGETHER in a single forward pass:
        [CLS] query [SEP] document [SEP] → relevance score

    This is much more accurate than cosine similarity but slower,
    which is why we only run it on top-k candidates from Stage 1.

    Input:  {"query": str, "documents": List[Document]}
    Output: List[Document]  (re-ordered, cross_encoder_score added)
    """

    def __init__(self, top_k: int = None):
        self.top_k = top_k or CFG["retrieval"]["stage2_k"]
        self.model = CrossEncoder(CFG["reranker"]["model"])

    def invoke(
        self,
        input: Dict[str, Any],
        config=None,
        **kwargs,
    ) -> List[Document]:

        query = input["query"]
        documents = input["documents"]

        pipeline_event("reranker.ce.invoke", query_len=len(query or ""), docs=len(documents), top_k=self.top_k)

        if not documents:
            return documents

        # Build (query, text) pairs — cross-encoder needs both together
        pairs  = [(query, doc.page_content) for doc in documents]
        scores = self.model.predict(pairs)

        # Attach score to document metadata for inspection
        for doc, score in zip(documents, scores):
            doc.metadata["cross_encoder_score"] = round(float(score), 4)

        # Sort highest score first
        reranked = sorted(
            documents,
            key=lambda d: d.metadata["cross_encoder_score"],
            reverse=True
        )

        pipeline_event("reranker.ce.results", reranked_count=len(reranked))
        return reranked[: self.top_k]


# ============================================================
# STAGE 3: LLM listwise reranker
# ============================================================

class LLMListwiseReranker(Runnable):
    """
    Asks the LLM to globally rank all Stage 2 candidates at once.

    Unlike cross-encoder (pairwise), the LLM sees all documents
    together and reasons about relative relevance:
    "Doc 3 is better than Doc 1 because it contains specific figures."

    Includes a sanity check fallback: if the LLM's top pick has a
    below-median cross-encoder score, we trust the cross-encoder
    order instead.

    Input:  {"query": str, "documents": List[Document]}
    Output: List[Document]  (re-ordered, llm_rank added)
    """

    def __init__(self, llm, top_k: int = None):
        self.llm   = llm
        self.top_k = top_k or CFG["retrieval"]["stage3_k"]

    def invoke(
        self,
        input: Dict[str, Any],
        config=None,
        **kwargs,
    ) -> List[Document]:

        query = input["query"]
        documents = input["documents"]

        pipeline_event("reranker.llm.invoke", docs=len(documents), top_k=self.top_k)

        if not documents or len(documents) <= self.top_k:
            return documents[:self.top_k]

        # Build numbered doc list for the prompt
        docs_text = ""
        for i, doc in enumerate(documents):
            section  = doc.metadata.get("section_heading", "Unknown")
            snippet  = doc.page_content[:400].replace("\n", " ")
            ce_score = doc.metadata.get("cross_encoder_score", 0)
            docs_text += (
                f"[DOC {i+1}] (relevance score: {ce_score:.2f})\n"
                f"Section: {section}\n"
                f"Text: {snippet}...\n\n"
            )

        prompt = f"""You are a retrieval expert ranking document chunks for a RAG system.

USER QUERY: {query}

DOCUMENTS (already roughly ranked by a cross-encoder, scores shown):
{docs_text}

Task: Re-rank these documents by how directly and specifically they answer the query.
Prefer documents that contain EXACT numerical data, projections, or figures relevant to the query.

Rules:
- Return ONLY valid JSON, nothing else
- Use the exact document numbers shown as [DOC N]
- Every document number must appear exactly once

JSON format: {{"ranking": [2, 1, 4, 3, 5]}}
JSON:"""

        try:
            from langchain_core.messages import HumanMessage
            response      = self.llm.invoke([HumanMessage(content=prompt)])
            response_text = response.content.strip()

            # Strict JSON extraction
            json_match = re.search(
                r'\{"ranking":\s*\[[\d,\s]+\]\}', response_text
            )
            if not json_match:
                return self._fallback(documents)

            ranking = json.loads(json_match.group()).get("ranking", [])

            # Validate indices
            if not all(1 <= r <= len(documents) for r in ranking):
                return self._fallback(documents)
            if len(set(ranking)) != len(ranking):
                return self._fallback(documents)

            reordered = [documents[r - 1] for r in ranking]

            # Sanity check: LLM top pick should have above-median CE score
            ce_scores  = [d.metadata.get("cross_encoder_score", 0) for d in documents]
            median_ce  = sorted(ce_scores)[len(ce_scores) // 2]
            top_ce     = reordered[0].metadata.get("cross_encoder_score", 0)

            if top_ce < median_ce:
                return self._fallback(documents)

            # Tag with llm_rank
            for i, doc in enumerate(reordered):
                doc.metadata["llm_rank"] = i + 1

            return reordered[:self.top_k]

        except Exception:
            return self._fallback(documents)

    def _fallback(self, documents: List[Document]) -> List[Document]:
        """Return cross-encoder order when LLM reranking fails."""
        return documents[:self.top_k]
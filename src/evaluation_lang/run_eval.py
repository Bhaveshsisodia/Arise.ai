# # 1) ensure project root is importable
# import sys
# import os
# import os

# from pathlib import Path

# # Get the current folder path
# current_folder = Path.cwd()
# print("Current:", current_folder)

# # Get the path of the folder one level up
# parent_folder = current_folder.parent
# print("Parent:", parent_folder)

# # If you actually need to change your working directory to it:
# import os
# os.chdir(parent_folder)

# # 2) imports
# from sentence_transformers import SentenceTransformer
# from src.config import CFG
# from src.vector_db import collection
# from src.generator import get_llm
# from src.retriever import MongoHybridRetriever
# from src.evaluation_lang.evaluators import *
# from src.evaluation_lang.llm_loader import llm_loader
# # 3) build components
# from langsmith import traceable

# from langsmith import Client
# from src.evaluation_lang.dataset import creating_dataset

# filename = CFG["Evaluation"]["eval_dataset"]
# Dataset_name = CFG["Evaluation"]["eval_name"]
# creating_dataset(filename,Dataset_name)
# embedder = SentenceTransformer(CFG["embedding"]["model"])
# llm = get_llm()

# # 4) instantiate retriever
# retriever = MongoHybridRetriever(
#     embedder=embedder,
#     llm=llm,
#     collection=collection,
#     use_mmr=True
# )

# ## add Decorator

# # Add decorator so this function is traced in LangSmith
# @traceable()
# def rag_bot(question: str) -> dict:
#     # LangChain retriever will be automatically traced
#     docs = retriever.invoke(question)
#     docs_string = "".join(doc.page_content for doc in docs)
#     instructions = f"""You are a helpful assistant who is good at analyzing source information and answering questions.
#        Use the following source documents to answer the user's questions.
#        If you don't know the answer, just say that you don't know.
#        Use three sentences maximum and keep the answer concise.

# <context>
# {docs_string}
# </context>"""

#     print("Number of docs:", len(docs))
#     print("Characters:", len(docs_string))
#     print("Approx tokens:", len(docs_string) // 4)
#     # langchain ChatModel will be automatically traced
#     ai_msg = llm.invoke([
#             {"role": "system", "content": instructions},
#             {"role": "user", "content": question},
#         ],
#     )
#     return {"answer": ai_msg.content, "documents": docs}

# correctness_eval = CorrectnessEvaluator(
#     llm=llm_loader("OpenAI"),
#     model_name="OpenAI"
# )

# relevance_eval = RelevanceEvaluator(
#     llm=llm_loader("OpenAI"),
#     model_name="OpenAI"
# )

# groundedness_eval = GroundednessEvaluator(
#     llm=llm_loader("OpenAI"),
#     model_name="OpenAI"
# )

# retrieval_relevance_eval = RetrievalRelevanceEvaluator(
#     llm=llm_loader("OpenAI"),
#     model_name="OpenAI"
# )

# def target(inputs: dict) -> dict:
#     return rag_bot(inputs["question"])


# client_ls = Client()

# experiment_results = client_ls.evaluate(
#     target,
#     data=CFG["Evaluation"]["eval_name"],
#     evaluators=[correctness_eval, relevance_eval, groundedness_eval, retrieval_relevance_eval],
#     experiment_prefix="rag-doc-relevance",
#     metadata={"version": "LCEL context, OpenAI_chatgptoss"},
# )

# src/evaluation_lang/run_eval.py

import sys
from pathlib import Path
from langsmith import Client, traceable
from sentence_transformers import SentenceTransformer

# Ensure project root is importable (project root is two parents above this file)
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.config import CFG
from src.vector_db import collection
from src.generator import get_llm
from src.retriever import MongoHybridRetriever
from src.evaluation_lang.evaluators import *
from src.evaluation_lang.llm_loader import llm_loader
from src.utils.logger import eval_logger as logger, eval_event
from src.reranker import CrossEncoderReranker, LLMListwiseReranker


class EvalRunner:
    def __init__(self, project_root: Path = project_root):
        self.project_root = project_root
        self.cfg = CFG
        self.client = Client()
        self._build_components()

    def _build_components(self):
        filename = self.cfg["Evaluation"]["eval_dataset"]
        dataset_name = self.cfg["Evaluation"]["eval_name"]

        # create dataset (side-effect)
        from src.evaluation_lang.dataset import creating_dataset
        creating_dataset(filename, dataset_name)

        # embeddings and llm
        self.embedder = SentenceTransformer(self.cfg["embedding"]["model"])
        self.llm = get_llm()

        # retriever
        use_mmr_cfg = self.cfg.get("retrieval", {}).get("use_mmr", True)
        logger.info(
            "EvalRunner building components | embed_model=%s use_mmr=%s",
            self.cfg["embedding"]["model"],
            use_mmr_cfg,
        )
        eval_event(
            "evalrunner.build",
            embed_model=self.cfg["embedding"]["model"],
            use_mmr=use_mmr_cfg,
        )
        self.retriever = MongoHybridRetriever(
            embedder=self.embedder,
            llm=self.llm,
            collection=collection,
            use_mmr=use_mmr_cfg,
        )

        # Rerankers (Stage 2: cross-encoder, Stage 3: LLM listwise)
        use_ce = self.cfg.get("reranker", {}).get("use_ce", True)
        use_llm_rerank = self.cfg.get("reranker", {}).get("use_llm", True)

        try:
            self.ce_reranker = CrossEncoderReranker() if use_ce else None
        except Exception:
            logger.exception("Failed to instantiate CrossEncoderReranker")
            self.ce_reranker = None

        try:
            self.llm_reranker = LLMListwiseReranker(llm=self.llm) if use_llm_rerank else None
        except Exception:
            logger.exception("Failed to instantiate LLMListwiseReranker")
            self.llm_reranker = None

        # evaluators
        self.correctness_eval = CorrectnessEvaluator(
            llm=llm_loader("OpenAI"), model_name="OpenAI"
        )
        self.relevance_eval = RelevanceEvaluator(
            llm=llm_loader("OpenAI"), model_name="OpenAI"
        )
        self.groundedness_eval = GroundednessEvaluator(
            llm=llm_loader("OpenAI"), model_name="OpenAI"
        )
        self.retrieval_relevance_eval = RetrievalRelevanceEvaluator(
            llm=llm_loader("OpenAI"), model_name="OpenAI"
        )

    @traceable()
    def rag_bot(self, question: str) -> dict:
        docs = self.retriever.invoke(question)

        # Get full candidate list from retriever; we'll apply rerankers first
        # and only truncate to the final `stage3_k` afterwards so both
        # Cross-Encoder and LLM listwise rerankers see the intended input set.
        docs_list = list(docs)
        docs_string = "".join(doc.page_content for doc in docs_list)

        # Stage 2: Cross-encoder reranking (optional, controlled by config)
        if getattr(self, "ce_reranker", None) is not None and docs_list:
            try:
                logger.info("Applying CrossEncoderReranker: docs=%s", len(docs_list))
                docs_list = self.ce_reranker.invoke({"query": question, "documents": docs_list})
                eval_event("rag_bot.ce_reranked", docs=len(docs_list))
            except Exception:
                logger.exception("Cross-encoder reranking failed — continuing with original order")

        # Stage 3: LLM listwise reranking (optional, controlled by config)
        if getattr(self, "llm_reranker", None) is not None and docs_list:
            try:
                logger.info("Applying LLMListwiseReranker: docs=%s", len(docs_list))
                docs_list = self.llm_reranker.invoke({"query": question, "documents": docs_list})
                eval_event("rag_bot.llm_reranked", docs=len(docs_list))
            except Exception:
                logger.exception("LLM listwise reranking failed — falling back to cross-encoder order")

        # After reranking, enforce final document count (stage3_k) so the
        # prompted context sent to the LLM is bounded. This preserves the
        # intent of running rerankers on the larger candidate set but keeps
        # the model input size controlled.
        doc_limit = self.cfg.get("retrieval", {}).get("stage3_k", 5)
        if len(docs_list) > doc_limit:
            logger.info(
                "Truncating documents after reranking: original=%s, keep=%s",
                len(docs_list),
                doc_limit,
            )
            eval_event(
                "rag_bot.truncated_after_rerank",
                original_docs=len(docs_list),
                truncated_docs=doc_limit,
            )
            docs_list = docs_list[:doc_limit]

        # Rebuild context after reranking and truncation
        docs_string = "".join(doc.page_content for doc in docs_list)

        instructions = f"""You are a helpful assistant who is good at analyzing source information and answering questions.
Use the following source documents to answer the user's questions. If you don't know the answer, say you don't know. Use three sentences maximum and be concise.

<context>
{docs_string}
</context>"""

        # Call LLM with defensive error handling so evaluators always get a stable response
        try:
            ai_msg = self.llm.invoke(
                [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": question},
                ]
            )
            answer = getattr(ai_msg, "content", str(ai_msg))
        except Exception as e:
            logger.exception("LLM invocation failed: %s", e)
            answer = f"LLM error: {type(e).__name__}: {e}"

        return {"answer": answer, "documents": docs_list}

    def target(self, inputs: dict) -> dict:
        return self.rag_bot(inputs["question"])

    def run(self, experiment_prefix: str = "rag-doc-relevance-Baseline"):
        results = self.client.evaluate(
            self.target,
            data=self.cfg["Evaluation"]["eval_name"],
            evaluators=[
                self.correctness_eval,
                self.relevance_eval,
                self.groundedness_eval,
                self.retrieval_relevance_eval,
            ],
            experiment_prefix=experiment_prefix,
            metadata={"version": "LCEL context, OpenAI_chatgptoss"},
        )
        return results


if __name__ == "__main__":
    # run from project root so relative resources resolve predictably
    Path.cwd()  # optional check
    runner = EvalRunner()
    runner.run()
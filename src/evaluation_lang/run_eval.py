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
#     llm=llm_loader("gemini"),
#     model_name="gemini"
# )

# relevance_eval = RelevanceEvaluator(
#     llm=llm_loader("gemini"),
#     model_name="gemini"
# )

# groundedness_eval = GroundednessEvaluator(
#     llm=llm_loader("gemini"),
#     model_name="gemini"
# )

# retrieval_relevance_eval = RetrievalRelevanceEvaluator(
#     llm=llm_loader("gemini"),
#     model_name="gemini"
# )

# def target(inputs: dict) -> dict:
#     return rag_bot(inputs["question"])


# client_ls = Client()

# experiment_results = client_ls.evaluate(
#     target,
#     data=CFG["Evaluation"]["eval_name"],
#     evaluators=[correctness_eval, relevance_eval, groundedness_eval, retrieval_relevance_eval],
#     experiment_prefix="rag-doc-relevance",
#     metadata={"version": "LCEL context, gemini_chatgptoss"},
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

        # evaluators
        self.correctness_eval = CorrectnessEvaluator(
            llm=llm_loader("gemini"), model_name="gemini"
        )
        self.relevance_eval = RelevanceEvaluator(
            llm=llm_loader("gemini"), model_name="gemini"
        )
        self.groundedness_eval = GroundednessEvaluator(
            llm=llm_loader("gemini"), model_name="gemini"
        )
        self.retrieval_relevance_eval = RetrievalRelevanceEvaluator(
            llm=llm_loader("gemini"), model_name="gemini"
        )

    @traceable()
    def rag_bot(self, question: str) -> dict:
        docs = self.retriever.invoke(question)
        docs_string = "".join(doc.page_content for doc in docs)
        instructions = f"""You are a helpful assistant who is good at analyzing source information and answering questions.
Use the following source documents to answer the user's questions. If you don't know the answer, say you don't know. Use three sentences maximum and be concise.

<context>
{docs_string}
</context>"""

        ai_msg = self.llm.invoke(
            [
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},
            ]
        )
        return {"answer": ai_msg.content, "documents": docs}

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
            metadata={"version": "LCEL context, gemini_chatgptoss"},
        )
        return results


if __name__ == "__main__":
    # run from project root so relative resources resolve predictably
    Path.cwd()  # optional check
    runner = EvalRunner()
    runner.run()
"""
usage_in_colab.py
-----------------
Drop this in a Colab cell to use the new modular chain.
Assumes you've cloned the project and installed requirements.

Setup in Colab:
    !git clone https://github.com/yourusername/jusnl-rag
    %cd jusnl-rag
    !pip install -r requirements.txt

Add secrets in Colab (left sidebar → key icon):
    MONGO_URI          = your MongoDB Atlas connection string
    GROQ_API_KEY       = your Groq API key
    LANGCHAIN_API_KEY  = your LangSmith API key (optional)
    LANGCHAIN_PROJECT  = jusnl-rag
"""

import os
from dotenv import load_dotenv
load_dotenv()
# Load secrets from Colab's secure secrets store
os.environ["MONGO_URI"]         = os.getenv("MONGO_URI")
os.environ["GROQ_API_KEY"]      = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")  # optional

# ── Build the chain (loads models, connects to MongoDB) ────────
from src.chain import build_chain,  ask_stream  , ask

chain = build_chain(use_llm_reranker=True)

# # ── Ask a question ─────────────────────────────────────────────
answer = ask(
    "What employee expenses has JUSNL projected?",
    chain
)

# ── Streaming version ──────────────────────────────────────────
ask_stream(
    "What employee expenses has JUSNL projected?",
    chain
)

# ── Access intermediate results (for debugging) ────────────────
# The chain returns the final string answer,
# but you can inspect individual stages:

from src.retriever import MongoHybridRetriever
from src.vector_db import collection
from sentence_transformers import SentenceTransformer
from src.generator import get_llm

embedder = SentenceTransformer("BAAI/bge-large-en-v1.5")
llm      = get_llm()

retriever = MongoHybridRetriever(
    embedder=embedder, llm=llm, collection=collection
)
docs = retriever.invoke("What employee expenses has JUSNL projected?")

print(f"Retrieved {len(docs)} documents")
for i, doc in enumerate(docs[:3], 1):
    print(f"\n[{i}] {doc.metadata.get('section_heading')}")
    print(f"     RRF score: {doc.metadata.get('rrf_score')}")
    print(f"     {doc.page_content[:150]}...")

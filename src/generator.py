"""
generator.py — Context builder + prompt template + LLM setup.

Converts retrieved documents into a formatted context string,
wraps it with a prompt, and connects to the LLM.

Usage:
    from src.generator import build_context, build_prompt, get_llm
"""

import os
from typing import List

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from dotenv import load_dotenv

from src.config import CFG

load_dotenv()


# ============================================================
# LLM
# ============================================================

def get_llm() -> ChatGroq:
    """
    Returns a configured ChatGroq LLM instance.
    Model and temperature come from config/config.yaml.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("GROQ_API_KEY not found in environment / .env file")

    return ChatGroq(
        api_key    = api_key,
        model_name = CFG["llm"]["model"],
        temperature= CFG["llm"]["temperature"],
    )


# ============================================================
# CONTEXT BUILDER
# Converts List[Document] → formatted string for the prompt.
# ============================================================

def build_context(documents: List[Document]) -> str:
    """
    Formats retrieved documents into a numbered context block.

    Each block shows:
    - Rank (so LLM knows ordering was intentional)
    - Section heading
    - Page range
    - Text

    Returns:
        str — formatted context ready to insert into the prompt
    """
    blocks = []
    for i, doc in enumerate(documents, 1):
        meta    = doc.metadata
        section = meta.get("section_heading", "Unknown Section")
        pages   = f"{meta.get('page_start', '?')}-{meta.get('page_end', '?')}"
        ce      = meta.get("cross_encoder_score", "")
        ce_str  = f" | relevance: {ce:.2f}" if ce else ""

        blocks.append(
            f"[{i}] Section: {section} | Pages: {pages}{ce_str}\n"
            f"{doc.page_content}"
        )

    return "\n\n---\n\n".join(blocks)


# ============================================================
# PROMPT TEMPLATE
# Using LangChain ChatPromptTemplate so it's traceable
# in LangSmith and composable in LCEL.
# ============================================================

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a power sector regulatory analyst specialising in Indian electricity regulation.

Use ONLY the provided context to answer the question.
If the answer is not in the context, say:
"Information not available in retrieved documents."

For every answer provide:
1. Direct answer with specific figures/data where available
2. Sources:
   - Section: [section heading]
   - Pages: [page range]
   - Relevance score: [score if available]

Context:
{context}

Question:
{question}
""")


# ============================================================
# OUTPUT PARSER
# Extracts the string content from the LLM message object.
# ============================================================

output_parser = StrOutputParser()
from typing import Literal

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate

from src.generator import get_llm


class RouteQuery(BaseModel):
    """Route the user query to the correct pipeline."""

    datasource: Literal["chat", "retrieval"] = Field(
        ...,
        description="""
        Choose:
        - chat: Greetings, thanks, farewell, general conversation, identity questions.
        - retrieval: Questions requiring information from regulatory documents.
        """,
    )


SYSTEM_PROMPT = """
You are an expert query router for a Power Sector Regulatory Assistant.

Your task is to determine whether a user's question requires searching
the regulatory document database.

Return ONLY one datasource: 'chat' or 'retrieval'.
"""

PROMPT = ChatPromptTemplate.from_messages(
    [("system", SYSTEM_PROMPT), ("human", "{question}")]
)


class QueryRouter:
    def __init__(self, llm=None):
        # llm is a LangChain LLM instance (ChatGroq in this repo)
        self.llm = llm or get_llm()
        # Use structured output to get a validated response
        # method and strictness can be tuned in the future
        try:
            self.structured_llm = self.llm.with_structured_output(
                RouteQuery, method="json_schema", strict=True
            )
        except Exception:
            # Fallback to raw llm if structured output not supported
            self.structured_llm = None

    def route(self, question: str) -> str:
        """Return 'chat' or 'retrieval'. Falls back to 'retrieval' on error."""
        if self.structured_llm is None:
            # Best-effort simple heuristic fallback
            q = question.strip().lower()
            if any(w in q for w in ("hi", "hello", "thanks", "thank", "bye", "who are you", "what can you do")):
                return "chat"
            return "retrieval"

        try:
            out = PROMPT | self.structured_llm
            res = out.invoke({"question": question})
            # pydantic model instance or dict-like
            if hasattr(res, "datasource"):
                return res.datasource
            if isinstance(res, dict) and "datasource" in res:
                return res["datasource"]
        except Exception:
            pass

        return "retrieval"

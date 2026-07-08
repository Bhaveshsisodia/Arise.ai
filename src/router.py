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

Choose 'chat' for:
- greetings, thanks, farewells, general conversation
- questions about the current conversation or prior messages
- questions about the user's name, preferences, or details already stated in chat
- prompts like "what did I say", "what is my name", "remember this", "who am I"

Choose 'retrieval' only when the answer should come from regulatory documents.

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

    def _tokenize(self, text: str) -> set[str]:
        import re
        return set(re.findall(r"[a-z0-9]+", (text or "").lower()))

    def _heuristic_route(self, question: str) -> str:
        q = question.strip().lower()
        tokens = self._tokenize(question)

        greeting_patterns = (
            "hi",
            "hello",
            "thanks",
            "thank",
            "bye",
            "who are you",
            "what can you do",
        )
        if any(pattern in q for pattern in greeting_patterns):
            return "chat"

        memory_terms = {
            "remember", "recall", "earlier", "before", "previous", "prior",
            "history", "conversation", "message", "messages", "question",
            "questions", "asked", "said", "tell", "told", "name",
        }
        self_reference_terms = {"i", "me", "my", "mine"}
        if (tokens & memory_terms) and (tokens & self_reference_terms):
            return "chat"

        if "my name is" in q or "who am i" in q:
            return "chat"

        return "retrieval"

    def route(self, question: str) -> str:
        """Return 'chat' or 'retrieval'. Falls back to 'retrieval' on error."""
        heuristic = self._heuristic_route(question)
        if heuristic == "chat":
            return "chat"

        if self.structured_llm is None:
            return heuristic

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

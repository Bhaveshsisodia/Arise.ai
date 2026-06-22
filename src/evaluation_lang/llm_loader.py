from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


def llm_loader(provider):
    """
    Returns an initialized LLM for evaluation.
    """

    if provider == "gemini":
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0
        )

    elif provider == "groq":
        return ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0
        )

    elif provider == "llama8b":
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )

    else:
        raise ValueError(f"Unsupported provider: {provider}")
from dotenv import load_dotenv
import os

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# Load .env into the environment when this module is imported
load_dotenv()


def llm_loader(provider):
    """
    Returns an initialized LLM for evaluation.
    """

    if provider == "gemini":
        # Ensure either GOOGLE_API_KEY or GEMINI_API is set for the Google client
        google_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API")
        print(google_key)
        if google_key:
            os.environ["GOOGLE_API_KEY"] = google_key

        return ChatGoogleGenerativeAI(
            model="gemini-2.5-pro",
            temperature=0
        )

    elif provider == "groq":
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key

        return ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0
        )

    elif provider == "llama8b":
        # If this provider also needs a GROQ key or other key, prefer GROQ
        groq_key = os.getenv("GROQ_API_KEY")
        if groq_key:
            os.environ["GROQ_API_KEY"] = groq_key

        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0
        )

    elif provider=="OpenAI":
        open_router_api_key = os.getenv("OPEN_ROUTER_API_KEY")
        if open_router_api_key:
            os.environ['OPEN_ROUTER_API_KEY'] = open_router_api_key

        return ChatOpenAI(
                    base_url="https://openrouter.ai/api/v1",
                    api_key=os.getenv("OPEN_ROUTER_API_KEY"),
                    model="qwen/qwen3-32b" #deepseek/deepseek-r1:free
#                     cohere/north-mini-code:free
# nvidia/nemotron-3.5-content-safety:free
# nvidia/nemotron-3-ultra-550b-a55b:free
# nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free
# poolside/laguna-xs.2:free
# poolside/laguna-m.1:free
# google/gemma-4-26b-a4b-it:free
# google/gemma-4-31b-it:free
# nvidia/nemotron-3-super-120b-a12b:free
# liquid/lfm-2.5-1.2b-thinking:free
# liquid/lfm-2.5-1.2b-instruct:free
# nvidia/nemotron-3-nano-30b-a3b:free
# nvidia/nemotron-nano-12b-v2-vl:free
# qwen/qwen3-next-80b-a3b-instruct:free
# nvidia/nemotron-nano-9b-v2:free
# openai/gpt-oss-120b:free
# openai/gpt-oss-20b:free
# qwen/qwen3-coder:free
# cognitivecomputations/dolphin-mistral-24b-venice-edition:free
# meta-llama/llama-3.3-70b-instruct:free
# meta-llama/llama-3.2-3b-instruct:free
# nousresearch/hermes-3-llama-3.1-405b:free
                )

    else:
        raise ValueError(f"Unsupported provider: {provider}")
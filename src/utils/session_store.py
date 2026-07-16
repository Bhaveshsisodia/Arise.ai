import json
from dataclasses import dataclass
from typing import Dict, List

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from src.config import CFG
from src.utils.logger import pipeline_logger as logger
from src.utils.redis_cache import get_redis_cache


def _message_to_dict(message: BaseMessage) -> dict[str, str]:
    if isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    else:
        role = "assistant"
    return {"role": role, "content": str(getattr(message, "content", ""))}


def _dict_to_message(item: dict[str, str]) -> BaseMessage | None:
    role = str(item.get("role", "")).strip().lower()
    content = str(item.get("content", "")).strip()
    if not content:
        return None
    if role == "user":
        return HumanMessage(content=content)
    if role == "assistant":
        return AIMessage(content=content)
    return None


@dataclass
class SessionHistoryBackend:
    ttl_seconds: int
    namespace: str = "session_history"

    def __post_init__(self) -> None:
        self._fallback_store: Dict[str, List[dict[str, str]]] = {}

    def _read_redis(self, session_id: str) -> List[dict[str, str]] | None:
        cache = get_redis_cache()
        if not cache.enabled or cache.client is None:
            return None
        raw = cache.get(self.namespace, session_id)
        if raw is None:
            return []
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError):
            logger.warning("Invalid session history payload in Redis for session_id=%s", session_id)
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def load(self, session_id: str) -> List[dict[str, str]]:
        redis_messages = self._read_redis(session_id)
        if redis_messages is not None:
            return redis_messages
        return list(self._fallback_store.get(session_id, []))

    def save(self, session_id: str, items: List[dict[str, str]]) -> None:
        cache = get_redis_cache()
        payload = json.dumps(items, ensure_ascii=True)
        if cache.enabled and cache.client is not None:
            if cache.set(self.namespace, session_id, payload, ttl=self.ttl_seconds):
                return
            logger.warning("Failed to persist session history to Redis for session_id=%s", session_id)
        self._fallback_store[session_id] = list(items)

    def clear(self, session_id: str) -> None:
        cache = get_redis_cache()
        if cache.enabled and cache.client is not None:
            cache.delete(self.namespace, session_id)
        self._fallback_store.pop(session_id, None)


_SESSION_BACKEND = SessionHistoryBackend(
    ttl_seconds=int(CFG.get("redis", {}).get("ttl", 1800))
)


class RedisBackedChatMessageHistory(BaseChatMessageHistory):
    def __init__(self, session_id: str):
        self.session_id = session_id

    @property
    def messages(self) -> List[BaseMessage]:
        items = _SESSION_BACKEND.load(self.session_id)
        messages: List[BaseMessage] = []
        for item in items:
            message = _dict_to_message(item)
            if message is not None:
                messages.append(message)
        return messages

    def add_messages(self, messages: List[BaseMessage]) -> None:
        existing = [_message_to_dict(message) for message in self.messages]
        existing.extend(_message_to_dict(message) for message in messages if getattr(message, "content", ""))
        _SESSION_BACKEND.save(self.session_id, existing)

    def clear(self) -> None:
        _SESSION_BACKEND.clear(self.session_id)

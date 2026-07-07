import hashlib
import json
import math
import os
from datetime import date, datetime
from typing import Any, Optional

from dotenv import load_dotenv

from src.config import CFG
from src.utils.logger import pipeline_logger as logger

load_dotenv()

try:
    from bson import ObjectId
except Exception:  # pragma: no cover - optional dependency
    ObjectId = None

try:
    import redis
except ImportError:  # pragma: no cover - optional dependency
    redis = None


class RedisCache:
    """Small Redis wrapper with JSON serialization and graceful fallback."""

    def __init__(self, enabled: Optional[bool] = None, ttl: Optional[int] = None):
        self.enabled = enabled if enabled is not None else bool(CFG.get("redis", {}).get("enabled", True))
        self.ttl = ttl if ttl is not None else int(CFG.get("redis", {}).get("ttl", 900))
        self.prefix = str(CFG.get("redis", {}).get("prefix", "arise"))
        self._client = None
        self._disabled_reason = None

        if not self.enabled:
            self._disabled_reason = "disabled by config"
            return

        if redis is None:
            self._disabled_reason = "redis package not installed"
            self.enabled = False
            return

        redis_url = os.getenv("REDIS_URL") or CFG.get("redis", {}).get("url")
        try:
            if redis_url:
                self._client = redis.from_url(redis_url, decode_responses=True)
            else:
                host = str(CFG.get("redis", {}).get("host", os.getenv("REDIS_HOST", "localhost")))
                port = int(CFG.get("redis", {}).get("port", os.getenv("REDIS_PORT", 6379)))
                db = int(CFG.get("redis", {}).get("db", os.getenv("REDIS_DB", 0)))
                self._client = redis.Redis(host=host, port=port, db=db, decode_responses=True)

            self._client.ping()
        except Exception as exc:  # pragma: no cover - environment-specific
            self._disabled_reason = str(exc)
            self.enabled = False
            logger.warning("Redis unavailable, caching disabled: %s", exc)

    @property
    def client(self):
        return self._client

    def _build_key(self, namespace: str, key: str) -> str:
        return f"{self.prefix}:{namespace}:{key}"

    def get(self, namespace: str, key: str) -> Optional[str]:
        if not self.enabled or not self.client:
            return None
        try:
            return self.client.get(self._build_key(namespace, key))
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.warning("Redis GET failed: %s", exc)
            return None

    def get_json(self, namespace: str, key: str) -> Optional[Any]:
        value = self.get(namespace, key)
        if value is None:
            return None
        try:
            return json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return None

    def set(self, namespace: str, key: str, value: str, ttl: Optional[int] = None) -> bool:
        if not self.enabled or not self.client:
            return False
        try:
            self.client.setex(self._build_key(namespace, key), ttl or self.ttl, value)
            return True
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.warning("Redis SET failed: %s", exc)
            return False

    def _json_default(self, value: Any) -> Any:
        if ObjectId is not None and isinstance(value, ObjectId):
            return str(value)
        if value.__class__.__name__ == "ObjectId":
            return str(value)
        if isinstance(value, (datetime, date)):
            return value.isoformat()
        if hasattr(value, "tolist"):
            try:
                return value.tolist()
            except Exception:
                pass
        raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")

    def _json_safe(self, value: Any) -> Any:
        """Recursively normalize Mongo/numpy/date values before Redis caching."""
        if isinstance(value, dict):
            return {str(k): self._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._json_safe(item) for item in value]
        try:
            return self._json_default(value)
        except TypeError:
            return value

    def set_json(self, namespace: str, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        try:
            payload = json.dumps(self._json_safe(value), default=self._json_default)
            return self.set(namespace, key, payload, ttl=ttl)
        except TypeError as exc:  # pragma: no cover - runtime safety
            logger.warning("Redis JSON serialization failed: %s", exc)
            return False

    def delete(self, namespace: str, key: str) -> bool:
        if not self.enabled or not self.client:
            return False
        try:
            return bool(self.client.delete(self._build_key(namespace, key)))
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.warning("Redis DELETE failed: %s", exc)
            return False

    def clear_prefix(self, namespace: str) -> int:
        if not self.enabled or not self.client:
            return 0
        pattern = f"{self.prefix}:{namespace}:*"
        try:
            keys = list(self.client.scan_iter(match=pattern))
            return int(self.client.delete(*keys)) if keys else 0
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.warning("Redis clear_prefix failed: %s", exc)
            return 0

    def _semantic_similarity(self, left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return -1.0
        dot = sum(float(a) * float(b) for a, b in zip(left, right))
        left_norm = math.sqrt(sum(float(a) * float(a) for a in left))
        right_norm = math.sqrt(sum(float(b) * float(b) for b in right))
        if left_norm == 0 or right_norm == 0:
            return -1.0
        return dot / (left_norm * right_norm)

    def get_semantic_json(
        self,
        namespace: str,
        query_vector: list[float],
        *,
        min_similarity: float,
        variant: Optional[str] = None,
    ) -> Optional[Any]:
        if not self.enabled or not self.client:
            return None

        pattern = self._build_key(namespace, "*")
        best_payload = None
        best_score = min_similarity

        try:
            for key in self.client.scan_iter(match=pattern):
                raw = self.client.get(key)
                if not raw:
                    continue
                try:
                    entry = json.loads(raw)
                except (TypeError, json.JSONDecodeError):
                    continue

                if not isinstance(entry, dict) or "value" not in entry or "embedding" not in entry:
                    continue
                if variant is not None and entry.get("variant") != variant:
                    continue

                score = self._semantic_similarity(query_vector, entry.get("embedding") or [])
                if score >= best_score:
                    best_score = score
                    best_payload = entry.get("value")

            if best_payload is not None:
                logger.info(
                    "Redis semantic cache hit | namespace=%s variant=%s similarity=%.4f",
                    namespace,
                    variant,
                    best_score,
                )
            return best_payload
        except Exception as exc:  # pragma: no cover - runtime safety
            logger.warning("Redis semantic GET failed: %s", exc)
            return None

    def set_semantic_json(
        self,
        namespace: str,
        key: str,
        *,
        query_text: str,
        query_vector: list[float],
        value: Any,
        ttl: Optional[int] = None,
        variant: Optional[str] = None,
    ) -> bool:
        payload = {
            "query_text": query_text,
            "variant": variant,
            "embedding": query_vector,
            "value": value,
        }
        return self.set_json(namespace, key, payload, ttl=ttl)


def build_cache_key(namespace: str, *parts: Any) -> str:
    payload = "|".join(str(part) for part in [namespace, *parts])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_CACHE_INSTANCE: Optional[RedisCache] = None


def get_redis_cache() -> RedisCache:
    global _CACHE_INSTANCE
    if _CACHE_INSTANCE is None:
        _CACHE_INSTANCE = RedisCache()
    return _CACHE_INSTANCE

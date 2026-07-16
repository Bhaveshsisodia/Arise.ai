"""
vectordb.py — MongoDB connection.

Centralises the MongoClient so every other module
just imports `collection` instead of re-connecting.

Usage:
    from src.database import collection
sdfs"""

import os
from pymongo import MongoClient
from dotenv import load_dotenv
from src.config import CFG
from src.exception.custom_exception import DatabaseError
from src.exception.error_utils import raise_with_context

load_dotenv()

_MONGO_URI = os.environ.get("MONGO_URI") or os.environ.get("MONGOURI")
if not _MONGO_URI:
    raise DatabaseError(
        "MONGO_URI not found in environment / .env file",
        context={"env_var": "MONGO_URI", "fallback_env_var": "MONGOURI"},
    )

try:
    _client = MongoClient(_MONGO_URI)
    _db = _client[CFG["mongodb"]["database"]]
    collection = _db[CFG["mongodb"]["collection"]]
except DatabaseError:
    raise
except Exception as exc:
    raise_with_context(
        DatabaseError,
        exc,
        "Failed to initialize MongoDB collection",
        context={
            "database": CFG["mongodb"]["database"],
            "collection": CFG["mongodb"]["collection"],
        },
    )

"""
vectordb.py — MongoDB connection.

Centralises the MongoClient so every other module
just imports `collection` instead of re-connecting.

Usage:
    from src.database import collection
"""

import os
from pymongo import MongoClient
from dotenv import load_dotenv
from src.config import CFG

load_dotenv()

_MONGO_URI = os.environ.get("MONGO_URI")
if not _MONGO_URI:
    raise EnvironmentError("MONGO_URI not found in environment / .env file")

_client = MongoClient(_MONGO_URI)
_db     = _client[CFG["mongodb"]["database"]]

collection = _db[CFG["mongodb"]["collection"]]
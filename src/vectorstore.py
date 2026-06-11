from typing import Iterable, List, Dict, Optional
import os

try:
    import numpy as np
except Exception:  # pragma: no cover - optional dependency
    np = None


class PineconeVectorStore:
    """Minimal Pinecone helper for upserting and querying chunks.

    Requires `pinecone-client` to be installed and PINECONE_API_KEY + PINECONE_ENV
    set in environment or passed to `init`.
    """

    def __init__(self, index_name: str, dimension: int, api_key: Optional[str] = None, environment: Optional[str] = None):
        self.index_name = index_name
        self.dimension = dimension
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        self.environment = environment or os.getenv("PINECONE_ENV")
        self._client = None
        self._index = None

    def init(self):
        import pinecone

        pinecone.init(api_key=self.api_key, environment=self.environment)
        self._client = pinecone
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(self.index_name, dimension=self.dimension)
        self._index = pinecone.Index(self.index_name)

    def upsert_chunks(self, chunks: Iterable[Dict], embedder, batch_size: int = 32):
        if self._index is None:
            self.init()

        texts = [c["text"] for c in chunks]
        ids = [str(i) for i, _ in enumerate(chunks)]
        # compute embeddings in batches via embedder
        embeddings = embedder.encode(texts, show_progress_bar=False)

        to_upsert = []
        for cid, emb, chunk in zip(ids, embeddings, chunks):
            meta = chunk.copy()
            meta.pop("text", None)
            to_upsert.append((cid, emb.tolist() if hasattr(emb, "tolist") else emb, meta))

        # Pinecone accepts batches of tuples (id, vector, metadata)
        for i in range(0, len(to_upsert), batch_size):
            self._index.upsert(vectors=to_upsert[i : i + batch_size])

    def query(self, query_text: str, embedder, top_k: int = 5):
        if self._index is None:
            self.init()
        q_emb = embedder.encode([query_text])[0]
        resp = self._index.query(vector=q_emb.tolist() if hasattr(q_emb, "tolist") else q_emb, top_k=top_k, include_metadata=True)
        return resp.get("matches", [])


class MongoVectorStore:
    """Simple MongoDB-backed vector store.

    This stores each chunk as a document with an `embedding` numeric list field.
    For search this implementation falls back to an in-Python nearest-neighbor search
    (works without Atlas Vector Search). For production, use MongoDB Atlas Vector Search.
    """

    def __init__(self, uri: str = "mongodb://localhost:27017", db_name: str = "vectors", collection_name: str = "chunks"):
        from pymongo import MongoClient

        self.client = MongoClient(uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]

    def upsert_chunks(self, chunks: Iterable[Dict], embedder, batch_size: int = 32):
        texts = [c["text"] for c in chunks]
        embeddings = embedder.encode(texts, show_progress_bar=False)

        ops = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            doc = chunk.copy()
            doc["embedding"] = emb.tolist() if hasattr(emb, "tolist") else emb
            doc["_id"] = doc.get("id") or str(i)
            ops.append(doc)

        # bulk upsert (replace_one with upsert)
        from pymongo import ReplaceOne

        requests = [ReplaceOne({"_id": d["_id"]}, d, upsert=True) for d in ops]
        if requests:
            self.collection.bulk_write(requests, ordered=False)

    def count(self) -> int:
        return int(self.collection.count_documents({}))

    def search(self, query_text: str, embedder, top_k: int = 5):
        if np is None:
            raise RuntimeError("numpy is required for vector math in MongoVectorStore.search")
        q_emb = embedder.encode([query_text])[0]
        # load all embeddings (small datasets only)
        cursor = self.collection.find({}, {"embedding": 1, "text": 1, "_id": 1, "metadata": 1})
        docs = list(cursor)
        if not docs:
            return []

        embs = np.array([d.get("embedding") for d in docs], dtype=float)
        q = np.array(q_emb, dtype=float)
        # cosine similarity
        embs_norm = embs / np.linalg.norm(embs, axis=1, keepdims=True)
        q_norm = q / np.linalg.norm(q)
        sims = embs_norm.dot(q_norm)
        idx = np.argsort(-sims)[:top_k]
        results = []
        for i in idx:
            d = docs[int(i)]
            results.append({"_id": d.get("_id"), "score": float(sims[i]), "doc": d})
        return results


def upsert_chunks_if_empty(store, chunks: List[Dict], embedder, batch_size: int = 32):
    """Upsert chunks into provided store only if the store appears empty.

    `store` may be an instance of `PineconeVectorStore` or `MongoVectorStore` (or any
    object exposing `count()` and `upsert_chunks()` / `upsert_chunks_if_empty`).
    """
    # try count() if present
    try:
        cnt = store.count()
    except Exception:
        cnt = None

    if cnt is None or cnt == 0:
        # call upsert_chunks on store
        store.upsert_chunks(chunks, embedder, batch_size=batch_size)
        return True
    return False

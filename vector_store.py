"""
vector_store.py
----------------
Handles the "ChromaDB" hands-on concept: storing chunk text, embeddings,
and metadata, and supporting semantic search across multiple documents.

A thin wrapper class keeps all ChromaDB-specific calls in one place, so
the rest of the app doesn't need to know anything about Chroma's API.
"""

from typing import List, Dict, Any
import chromadb

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from pdf_processor import Chunk


class VectorStore:
    def __init__(self, persist_dir: str = CHROMA_PERSIST_DIR, collection_name: str = CHROMA_COLLECTION_NAME):
        # PersistentClient writes to disk so the knowledge base survives
        # across Streamlit re-runs within the same session/server.
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._collection_name = collection_name
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Add a batch of chunks (with their pre-computed embeddings) to the store."""
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata() for c in chunks],
        )

    def query(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """
        Run a semantic search and return a list of results, each with
        the chunk text, its metadata, and a distance score (lower = more similar).
        """
        if self.count() == 0:
            return []

        top_k = min(top_k, self.count())
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for doc_text, meta, dist, chunk_id in zip(docs, metas, dists, ids):
            hits.append({
                "chunk_id": chunk_id,
                "text": doc_text,
                "metadata": meta,
                "distance": dist,
            })
        return hits

    def count(self) -> int:
        return self._collection.count()

    def list_documents(self) -> List[str]:
        """Return the distinct document names currently stored."""
        if self.count() == 0:
            return []
        data = self._collection.get(include=["metadatas"])
        names = {m["doc_name"] for m in data["metadatas"]}
        return sorted(names)

    def reset(self) -> None:
        """Delete all data and recreate an empty collection (used by 'Clear knowledge base')."""
        self._client.delete_collection(self._collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

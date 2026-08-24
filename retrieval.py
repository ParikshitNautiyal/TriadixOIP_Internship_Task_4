"""
retrieval.py
------------
Handles the "semantic search" hands-on concept: turning a user question
into an embedding, searching ChromaDB, and filtering down to genuinely
relevant chunks.
"""

from typing import List, Dict, Any

from config import TOP_K, MAX_RELEVANT_DISTANCE
from embeddings import embed_query
from vector_store import VectorStore


def retrieve_relevant_chunks(
    question: str,
    store: VectorStore,
    top_k: int = TOP_K,
    max_distance: float = MAX_RELEVANT_DISTANCE,
) -> List[Dict[str, Any]]:
    """
    Given a natural-language question, return the most semantically
    relevant chunks from the vector store.

    Steps:
      1. Embed the question with the same local embedding model used for chunks.
      2. Query ChromaDB for the top_k nearest chunks (cosine distance).
      3. Drop chunks whose distance exceeds `max_distance` -- these are
         probably not actually relevant, and including them tends to
         encourage the LLM to answer from general knowledge instead of
         correctly saying "not found in the documents".
    """
    question = question.strip()
    if not question:
        return []

    query_vector = embed_query(question)
    hits = store.query(query_vector, top_k=top_k)

    relevant = [h for h in hits if h["distance"] <= max_distance]
    return relevant

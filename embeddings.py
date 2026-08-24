"""
embeddings.py
-------------
Handles the "text embeddings" hands-on concept.

Uses a local Sentence-Transformers model so that embedding generation is
free and doesn't require network calls (aligned with the project's goal
of avoiding unnecessary API usage). The model is loaded once and cached.
"""

from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedding_model() -> SentenceTransformer:
    """
    Load (and cache) the local embedding model.

    lru_cache ensures the model is only loaded into memory once per
    process, even though Streamlit re-runs the script on every interaction.
    """
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of text strings. Returns a list of float vectors."""
    if not texts:
        return []
    model = get_embedding_model()
    vectors = model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
    return vectors.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query string (e.g. the user's question)."""
    model = get_embedding_model()
    vector = model.encode([query], show_progress_bar=False, convert_to_numpy=True)
    return vector[0].tolist()

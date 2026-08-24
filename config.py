"""
config.py
---------
Central configuration for the AI PDF Knowledge Assistant.

Loads settings from environment variables (via a .env file) and exposes
them as simple constants that the rest of the app imports from.
Keeping all tunable knobs in one place makes the RAG pipeline easy to
understand and experiment with.
"""

import os
from dotenv import load_dotenv

# Load variables from a local .env file (if present) into the environment.
load_dotenv()

# ---------------------------------------------------------------------------
# LLM (Groq) settings
# ---------------------------------------------------------------------------
# Groq is used only for the final answer-generation step. Retrieval and
# embeddings are fully local, so the LLM is the only paid/networked call.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# ---------------------------------------------------------------------------
# Embedding model (local, via sentence-transformers)
# ---------------------------------------------------------------------------
# all-MiniLM-L6-v2 is small, fast on CPU, and good enough for a mini-project.
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------------
# Chunk size / overlap are measured in characters for simplicity and
# predictability (no tokenizer dependency needed just for chunking).
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# ---------------------------------------------------------------------------
# Vector store (ChromaDB) settings
# ---------------------------------------------------------------------------
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
CHROMA_COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "pdf_knowledge_base")

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "4"))

# A cosine-distance cutoff used as a lightweight relevance guard. Chunks
# with a distance ABOVE this value are treated as "not relevant enough"
# and excluded from the context (helps the app say "not found" instead of
# hallucinating from unrelated chunks). Tune this if using a different
# embedding model.
MAX_RELEVANT_DISTANCE = float(os.getenv("MAX_RELEVANT_DISTANCE", "0.9"))

# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "50"))

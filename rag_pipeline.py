"""
rag_pipeline.py
----------------
Handles the "RAG" and "question answering over documents" hands-on
concepts: building a grounded prompt from retrieved chunks and calling
the LLM (Groq) to produce a final, cited answer.

This is the ONLY place in the app that makes a network call to an LLM.
Embeddings and retrieval are fully local; the LLM is invoked once per
user question, exactly when an answer needs to be generated.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

from groq import Groq

from config import GROQ_API_KEY, GROQ_MODEL

NOT_FOUND_PHRASE = "I could not find this information in the uploaded documents."

SYSTEM_PROMPT = f"""You are a document question-answering assistant.

Rules you MUST follow:
1. Answer ONLY using the information present in the provided context chunks.
2. Do NOT use outside/general knowledge, even if you know the answer.
3. If the context does not contain enough information to answer the
   question, respond with exactly: "{NOT_FOUND_PHRASE}" (you may add a
   short reason, but keep that sentence).
4. Be concise and factual. Do not speculate.
5. When possible, mention which source(s) support each part of your answer,
   referring to them by their [Source N] label.
"""


@dataclass
class RAGResult:
    answer: str
    sources: List[Dict[str, Any]]  # deduplicated list of {"doc_name", "page_number"}
    grounded: bool  # False if the model reported "not found"


def _build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """Turn retrieved chunks into a numbered context block the LLM can cite by number."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk["metadata"]
        lines.append(
            f"[Source {i}] (Document: {meta['doc_name']}, Page: {meta['page_number']})\n{chunk['text']}"
        )
    return "\n\n".join(lines)


def _dedupe_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    sources = []
    for chunk in chunks:
        meta = chunk["metadata"]
        key = (meta["doc_name"], meta["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append({"doc_name": meta["doc_name"], "page_number": meta["page_number"]})
    return sources


def generate_answer(question: str, retrieved_chunks: List[Dict[str, Any]]) -> RAGResult:
    """
    Build a grounded prompt from retrieved_chunks and call the LLM.

    If no chunks were retrieved (e.g. nothing relevant enough was found),
    we short-circuit and skip the LLM call entirely -- there's nothing
    grounded to answer from, and this saves an unnecessary API call.
    """
    if not retrieved_chunks:
        return RAGResult(answer=NOT_FOUND_PHRASE, sources=[], grounded=False)

    if not GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY is not set. Add it to your .env file (see .env.example)."
        )

    context_block = _build_context_block(retrieved_chunks)
    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Answer the question using only the context above."
    )

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=800,
    )
    answer_text = response.choices[0].message.content.strip()

    grounded = NOT_FOUND_PHRASE.lower() not in answer_text.lower()
    sources = _dedupe_sources(retrieved_chunks) if grounded else []

    return RAGResult(answer=answer_text, sources=sources, grounded=grounded)

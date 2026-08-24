"""
testing.py
----------
Handles the "response testing" hands-on concept.

Provides a small, dependency-free test runner that exercises the full
pipeline (retrieval -> RAG -> citations) against either a predefined set
of sample questions or questions the user types in. This is intentionally
simple (no pytest/CI setup) since the goal is to let the user SEE how
well retrieval, grounding, and citations are working -- not to build a
production test suite.
"""

from dataclasses import dataclass
from typing import List, Dict, Any

from retrieval import retrieve_relevant_chunks
from rag_pipeline import generate_answer, RAGResult
from vector_store import VectorStore

# A few generic sample questions that tend to be useful for sanity-checking
# any document set. Users can add their own from the UI as well.
DEFAULT_TEST_QUESTIONS = [
    "What is the main topic of this document?",
    "Summarize the key points in a few sentences.",
    "What conclusions or results are mentioned?",
    "Does the document mention any limitations or future work?",
    "What is the capital of France?",  # deliberately unrelated -> should trigger "not found"
]


@dataclass
class TestCaseResult:
    question: str
    num_chunks_retrieved: int
    top_distance: float
    answer: str
    sources: List[Dict[str, Any]]
    grounded: bool


def run_test_suite(questions: List[str], store: VectorStore) -> List[TestCaseResult]:
    """
    Run each question through retrieval + RAG and collect diagnostics.

    This surfaces, per question:
      - how many chunks were retrieved (semantic retrieval working?)
      - the best (lowest) distance score (how relevant was the top match?)
      - the generated answer and whether it was grounded or a "not found"
      - the citations attached (citation correctness)
    """
    results: List[TestCaseResult] = []

    for question in questions:
        question = question.strip()
        if not question:
            continue

        chunks = retrieve_relevant_chunks(question, store)
        top_distance = min((c["distance"] for c in chunks), default=float("inf"))

        rag_result: RAGResult = generate_answer(question, chunks)

        results.append(
            TestCaseResult(
                question=question,
                num_chunks_retrieved=len(chunks),
                top_distance=round(top_distance, 4) if chunks else None,
                answer=rag_result.answer,
                sources=rag_result.sources,
                grounded=rag_result.grounded,
            )
        )

    return results

"""
app.py
------
Streamlit UI for the AI PDF Knowledge Assistant.

This file only handles presentation/orchestration. All the real logic
(PDF parsing, chunking, embeddings, vector storage, retrieval, RAG) lives
in their own modules and is simply called from here.

Run with:  streamlit run app.py
"""

import io
import streamlit as st

from config import MAX_UPLOAD_MB, TOP_K
from pdf_processor import process_pdf, PDFProcessingError
from embeddings import embed_texts
from vector_store import VectorStore
from retrieval import retrieve_relevant_chunks
from rag_pipeline import generate_answer
from testing import run_test_suite, DEFAULT_TEST_QUESTIONS

st.set_page_config(page_title="AI PDF Knowledge Assistant", page_icon="📄", layout="wide")


# ---------------------------------------------------------------------------
# Session state / resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_store() -> VectorStore:
    """Cache the VectorStore connection across Streamlit re-runs."""
    return VectorStore()


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of (question, answer, sources)

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()  # names already added to the vector store this session

store = get_store()


# ---------------------------------------------------------------------------
# Sidebar: upload + document management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("📄 Documents")

    uploaded_files = st.file_uploader(
        "Upload one or more PDFs",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Max {MAX_UPLOAD_MB} MB per file.",
    )

    process_clicked = st.button("Process documents", type="primary", use_container_width=True)

    if process_clicked:
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            for f in uploaded_files:
                if f.name in st.session_state.processed_files:
                    st.info(f"'{f.name}' was already processed. Skipping.")
                    continue

                size_mb = len(f.getvalue()) / (1024 * 1024)
                if size_mb > MAX_UPLOAD_MB:
                    st.error(f"'{f.name}' is {size_mb:.1f} MB, which exceeds the {MAX_UPLOAD_MB} MB limit.")
                    continue

                with st.spinner(f"Processing '{f.name}'..."):
                    try:
                        chunks = process_pdf(io.BytesIO(f.getvalue()), f.name)
                        embeddings = embed_texts([c.text for c in chunks])
                        store.add_chunks(chunks, embeddings)
                        st.session_state.processed_files.add(f.name)
                        st.success(f"'{f.name}' processed: {len(chunks)} chunks added.")
                    except PDFProcessingError as e:
                        st.error(str(e))
                    except Exception as e:
                        st.error(f"Unexpected error processing '{f.name}': {e}")

    st.divider()

    doc_names = store.list_documents()
    st.subheader(f"Knowledge base ({len(doc_names)} docs, {store.count()} chunks)")
    if doc_names:
        for name in doc_names:
            st.markdown(f"- {name}")
    else:
        st.caption("No documents processed yet.")

    if st.button("🗑️ Clear knowledge base", use_container_width=True):
        store.reset()
        st.session_state.processed_files = set()
        st.session_state.chat_history = []
        st.rerun()


# ---------------------------------------------------------------------------
# Main area: tabs for Q&A and Testing
# ---------------------------------------------------------------------------
st.title("📄 AI PDF Knowledge Assistant")
st.caption("Upload PDFs, then ask questions. Answers are grounded in your documents with citations.")

tab_chat, tab_test, tab_about = st.tabs(["💬 Ask Questions", "🧪 Test the System", "ℹ️ How it works"])

# --- Tab 1: Chat / Q&A -------------------------------------------------------
with tab_chat:
    if store.count() == 0:
        st.info("Upload and process at least one PDF from the sidebar to get started.")

    for question, answer, sources in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)
            if sources:
                with st.expander("📚 Sources"):
                    for s in sources:
                        st.markdown(f"- **{s['doc_name']}** — Page {s['page_number']}")

    question = st.chat_input("Ask a question about your uploaded documents...")
    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                try:
                    chunks = retrieve_relevant_chunks(question, store, top_k=TOP_K)
                    result = generate_answer(question, chunks)
                    st.markdown(result.answer)
                    if result.sources:
                        with st.expander("📚 Sources"):
                            for s in result.sources:
                                st.markdown(f"- **{s['doc_name']}** — Page {s['page_number']}")
                    st.session_state.chat_history.append((question, result.answer, result.sources))
                except RuntimeError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Something went wrong generating the answer: {e}")

# --- Tab 2: Testing -----------------------------------------------------------
with tab_test:
    st.subheader("Test the RAG pipeline")
    st.caption(
        "Run a batch of questions through retrieval + generation to sanity-check "
        "semantic search, grounding, and citation correctness."
    )

    default_text = "\n".join(DEFAULT_TEST_QUESTIONS)
    test_questions_raw = st.text_area(
        "Test questions (one per line)",
        value=default_text,
        height=160,
    )

    if st.button("▶️ Run tests"):
        if store.count() == 0:
            st.warning("Process at least one document before running tests.")
        else:
            questions = [q for q in test_questions_raw.splitlines() if q.strip()]
            with st.spinner(f"Running {len(questions)} test question(s)..."):
                results = run_test_suite(questions, store)

            for r in results:
                status = "✅ Grounded" if r.grounded else "⚠️ Not found / ungrounded"
                with st.expander(f"{status} — {r.question}"):
                    st.markdown(f"**Chunks retrieved:** {r.num_chunks_retrieved}")
                    if r.top_distance is not None:
                        st.markdown(f"**Best match distance:** {r.top_distance} (lower = more similar)")
                    st.markdown(f"**Answer:** {r.answer}")
                    if r.sources:
                        st.markdown("**Sources:**")
                        for s in r.sources:
                            st.markdown(f"- {s['doc_name']} — Page {s['page_number']}")

# --- Tab 3: About / explanation ------------------------------------------------
with tab_about:
    st.subheader("How the RAG pipeline works")
    st.markdown(
        """
1. **Upload** — PDFs are uploaded via the sidebar.
2. **Extraction** — Text is pulled out page-by-page (`pdf_processor.py`).
3. **Chunking** — Each page's text is split into overlapping ~1000-character
   chunks so the LLM gets focused, relevant context instead of whole pages.
4. **Embedding** — Each chunk is converted into a vector using a local
   Sentence-Transformers model (`embeddings.py`) — no API call needed.
5. **Storage** — Chunk text, embeddings, and metadata (document name, page
   number) are stored in **ChromaDB** (`vector_store.py`).
6. **Retrieval** — When you ask a question, it's embedded the same way and
   compared against all stored chunks to find the most semantically similar
   ones (`retrieval.py`).
7. **RAG generation** — The retrieved chunks are inserted into a prompt that
   instructs the LLM (via the Groq API) to answer *only* from that context,
   and to say clearly when the answer isn't in the documents (`rag_pipeline.py`).
8. **Citations** — The document name and page number of each chunk used are
   shown alongside the answer.
        """
    )

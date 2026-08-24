# 📄 AI PDF Knowledge Assistant
```
Workflow : 

PDF Upload → Text Extraction → Chunking → Embeddings → ChromaDB
  → Semantic Search → Relevant Chunks → RAG Prompt → LLM → Answer + Citations
```

---

## 1. Project Structure

```
ai_pdf_knowledge_assistant/
├── app.py              # Streamlit UI (upload, chat, testing tabs)
├── config.py            # Central configuration (env vars, defaults)
├── pdf_processor.py      # PDF text extraction + chunking
├── embeddings.py         # Local embedding generation (Sentence-Transformers)
├── vector_store.py       # ChromaDB wrapper (store + semantic search)
├── retrieval.py          # Question embedding + relevance filtering
├── rag_pipeline.py       # Prompt construction + Groq LLM call + citations
├── testing.py            # Batch test runner for the RAG pipeline
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

Each file has a single responsibility, matching the RAG pipeline stages above.

---

## 2. Installation

Requires Python 3.10+.

```bash 
cd ai_pdf_knowledge_assistant
python -m venv venv
venv\Scripts\activate           

pip install -r requirements.txt
```

> The first run will download the local embedding model (`all-MiniLM-L6-v2`, ~80MB) automatically via `sentence-transformers`.

---

## 3. Environment / API Key Setup

The LLM step uses **Groq** (fast, free-tier available).

1. Get a free API key: https://console.groq.com/keys
2. Copy the example env file:
   ```bash
   cp .env.example .env
   ```
3. Make a `.env`  using `.env.example` and paste your key:
   ```
   GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
   ```

**No API key is needed for embeddings or the vector store** — those run entirely locally.

---

## 4. Running the App

```bash
streamlit run app.py
```

This opens the app in your browser (default: http://localhost:8501).

**Basic flow:**
1. In the sidebar, upload one or more PDFs.
2. Click **"Process documents"** — this extracts text, chunks it, generates embeddings, and stores everything in ChromaDB.
3. Go to the **"Ask Questions"** tab and start chatting. Each answer shows its source document(s) and page number(s).
4. Use the **"Test the System"** tab to run a batch of questions and inspect retrieval/grounding diagnostics.

---

## 5. How the RAG Pipeline Works

1. **Upload** — PDFs come in via `st.file_uploader` in `app.py`.
2. **Extraction** (`pdf_processor.py`) — `pypdf` reads text **page by page**, so every piece of text keeps its page number.
3. **Chunking** (`pdf_processor.py`) — Each page's text is split into ~1000-character chunks with ~150-character overlap (a simple sliding window), preserving `doc_name`, `page_number`, and `chunk_index` as metadata.
4. **Embedding** (`embeddings.py`) — Each chunk is converted into a vector locally using `sentence-transformers` (`all-MiniLM-L6-v2`). No network call.
5. **Storage** (`vector_store.py`) — Chunk text + embedding + metadata go into a **ChromaDB** persistent collection, so multiple documents can coexist and be queried together.
6. **Retrieval** (`retrieval.py`) — The user's question is embedded the same way, ChromaDB returns the top-K nearest chunks by cosine distance, and chunks that are too dissimilar are filtered out (`MAX_RELEVANT_DISTANCE`).
7. **RAG generation** (`rag_pipeline.py`) — The retrieved chunks are formatted into a numbered context block and inserted into a strict system prompt instructing the LLM (Groq, `llama-3.1-8b-instant` by default) to answer **only** from that context and to explicitly say when information isn't present.
8. **Citations** — Each answer is paired with the deduplicated `(document, page)` pairs of the chunks actually used, shown in an expandable "Sources" section.

If no chunk clears the relevance threshold, the app **skips the LLM call entirely** and returns a "not found" message directly — saving an unnecessary API call and avoiding hallucination.

---

## 6. Example Test Questions

Once you've uploaded a document (e.g. a research paper), try:

- "What is the main topic of this document?"
- "Summarize the key contributions in a few sentences."
- "What methodology or approach is used?"
- "What are the limitations mentioned by the authors?"
- "What datasets or experiments are discussed?"

With **two or more documents** uploaded:
- "Compare the methodologies used in these two papers."
- "Which document discusses [topic], and what does it say?"

To test grounding/refusal behavior, ask something unrelated to your documents, e.g.:
- "What is the capital of France?" → should trigger the "not found in the uploaded documents" response.

The **Test the System** tab includes these as defaults, plus one deliberately unrelated question to verify the "not found" behavior works.

---

## 7. Hands-On Concept → Implementation Mapping

| Concept                     | Where it's implemented                                  |
|------------------------------|-----------------------------------------------------------|
| PDF file uploading            | `app.py` (`st.file_uploader`)                             |
| PDF text extraction           | `pdf_processor.py` → `extract_pages()`                    |
| Document chunking             | `pdf_processor.py` → `chunk_text()`                       |
| Text embeddings                | `embeddings.py` → `embed_texts()` / `embed_query()`        |
| ChromaDB                      | `vector_store.py` → `VectorStore` class                   |
| Semantic search                | `retrieval.py` → `retrieve_relevant_chunks()`              |
| RAG                            | `rag_pipeline.py` → `generate_answer()`                    |
| Question answering             | `app.py` "Ask Questions" tab + `rag_pipeline.py`            |
| Citation/source support        | Chunk metadata (`doc_name`, `page_number`) surfaced in `RAGResult.sources` |
| Multiple document support      | `VectorStore` holds chunks from all processed PDFs together |
| Response testing               | `testing.py` → `run_test_suite()`, "Test the System" tab   |

---

## 8. Design Decisions Worth Knowing

- **Chunking is character-based and manual** (no LangChain text splitter) — keeps dependencies minimal and the logic fully readable for a mini-project.
- **Embeddings are always local** — avoids API cost/latency and keeps the app usable offline except for the final answer step.
- **The LLM is only called when there's something to answer from** — if retrieval finds nothing relevant enough, the app short-circuits with a "not found" message instead of calling Groq.
- **A relevance distance threshold (`MAX_RELEVANT_DISTANCE`)** is used as a lightweight guard against the LLM answering from unrelated chunks — this is not a full reranking pipeline, just a simple cutoff, in keeping with "avoid unnecessary complexity."

---

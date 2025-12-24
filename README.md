# RAG Tutor

Overview

RAG Tutor is a Retrieval-Augmented-Generation (RAG) demo for ingesting PDFs, storing chunk embeddings in a Qdrant vector store, and answering questions using context retrieved from those PDFs.

Key components
- Ingestion pipeline: load and chunk PDFs, create embeddings, and upsert into Qdrant (`main.py`, `data_loader.py`).
- Query pipeline: embed the user question, retrieve top-k contexts from Qdrant, and call an LLM (via an OpenAI-compatible client) to answer using the retrieved context (`main.py`).
- UI: Streamlit app to upload PDFs and ask questions (`streamlit_app.py`).

---

## Features
- PDF ingestion (chunking and embeddings)
- Vector storage and similarity search with Qdrant
- Inngest event-driven workflow for ingestion and query handling
- Streamlit UI for manual testing

---

## Prerequisites
- Python 3.12 or later
- Qdrant (local or hosted) reachable at `http://localhost:6333` (default)
- An OpenAI-compatible embeddings and chat API (configured via environment variables)

You can run Qdrant locally with Docker:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## Setup & Installation 

1. Clone the repo and create a virtual environment:


2. Install the project dependencies:


3. Create a `.env` file in the project root with the following keys:

```text
OPENAI_API_KEY=sk-...
OPENAI_API_BASE_URL=https://api.openai.com/v1  # or your provider's base URL
INNGEST_API_BASE=http://127.0.0.1:8288/v1
QDRANT_URL=http://localhost:6333
```


## Running locally

1. Start Qdrant (see Prerequisites).
2. Start the FastAPI server (Inngest endpoints and functions):

```bash
uvicorn main:app --reload --port 8288
```

3. Start the Streamlit UI:

```bash
streamlit run streamlit_app.py
```

4. Use the Streamlit UI to upload PDFs and ask questions. The app triggers ingestion events and polls the local Inngest API for function run outputs.

---

## Implementation details
- Embeddings: `text-embedding-3-small` with dimension 1536
- Chunking: `SentenceSplitter(chunk_size=1000, chunk_overlap=200)` (from LlamaIndex)
- Vector store: Qdrant; collection defaults to `docs`
- Ingest flow: `rag/ingest_pdf` (loads, chunks, embeds, upserts)
- Query flow: `rag/query_pdf_ai` (embeds question, searches, calls LLM adapter)



# Multi-Loader RAG Assistant 🚀

A multimodal Retrieval-Augmented Generation (RAG) system built with **Streamlit**, **FastAPI**, **LangChain**, and **Hugging Face**.

This system ingests data across diverse document formats, OCR images, transcribed audio files, and SQL databases, indexes them into an in-memory vector store, and generates grounded answers using either a local LLM or OpenAI.

---

## 🌟 Key Features

- **Streamlit Web Application (`app.py` & `streamlit_app.py`)**:
  - Interactive chat interface with conversation history and source attribution.
  - Multi-file drag-and-drop uploader with 1-click sample document loading.
  - Direct SQL query runner and database table indexer.
  - Sidebar tuning: similarity threshold, top-K retrieved chunks, and LLM selection.
- **FastAPI Backend (`api.py`)**:
  - Full REST API with Swagger UI documentation at `/docs`.
  - Supports `/upload`, `/sql`, and `/query` endpoints.
- **Multi-Format Ingestion**:
  - **Documents**: PDF (`.pdf`), Microsoft Word (`.docx`), Plain Text (`.txt`), CSV (`.csv`)
  - **Images (OCR)**: Extracts text from images (`.png`, `.jpg`, `.jpeg`) using **Tesseract OCR**
  - **Audio (Speech-to-Text)**: Transcribes audio files (`.wav`, `.mp3`, `.m4a`) using **OpenAI Whisper** (with pure-Python WAV decoding via `scipy`)
  - **SQL Databases**: Connects via **SQLAlchemy** to ingest rows from SQLite, MySQL, PostgreSQL, or SQL queries into the knowledge base
- **Dual Inference Engine**:
  - **OpenAI**: Fast, zero local RAM usage via `gpt-4o-mini` (cloud-optimized for Streamlit Cloud).
  - **Local Model**: `Qwen/Qwen2.5-0.5B-Instruct` or extractive fallback running offline.
  - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (free, runs locally).
- **High Precision & Hallucination Prevention**:
  - Semantic similarity search with cosine score calculation.
  - Strict relevance filtering (similarity threshold `>= 0.25`) to prevent cross-document contamination.
  - Fallback message when facts are not grounded in uploaded context.

---

## 📁 Project Structure

```
├── app.py              # Streamlit Web Application (UI)
├── streamlit_app.py    # Streamlit Cloud entrypoint alias
├── api.py              # FastAPI REST API & Swagger UI
├── loaders.py          # Unified document loader (PDF, DOCX, TXT, CSV, OCR images, audio)
├── audio_loader.py     # Whisper audio transcription loader
├── sql_loader.py       # SQLAlchemy database loader
├── vectorstore.py      # Text splitting, MiniLM embeddings & in-memory vector store
├── rag.py              # Qwen local LLM generation pipeline & relevance scoring
├── packages.txt        # System packages for Streamlit Cloud (tesseract-ocr, ffmpeg)
├── requirements.txt    # Project Python dependencies (CPU-optimized PyTorch)
├── test_files/         # Sample test documents, images, and data
├── test_loader.py      # Script to verify loaders independently
├── test_rag.py         # End-to-end RAG pipeline test script
├── .gitignore          # Git ignore rules
└── .env.example        # Environment configuration template
```

---

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/AnaghaBorkar17/Mulltiloader_RAG_API.git
cd Mulltiloader_RAG_API
```

### 2. Create and Activate Virtual Environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the Application

### Option A: Launch the Streamlit Web App (Recommended)

```bash
streamlit run app.py
```
Opens in your browser at `http://localhost:8501`.

### Option B: Launch the FastAPI REST Backend

```bash
uvicorn api:app --reload
```
The API documentation is accessible at `http://127.0.0.1:8000/docs`.

---

## ☁️ Deploying on Streamlit Community Cloud

1. Push your repository to GitHub.
2. In [Streamlit Community Cloud](https://share.streamlit.io):
   - **Repository**: `AnaghaBorkar17/Mulltiloader_RAG_API`
   - **Branch**: `main`
   - **Main file path**: `app.py`
3. (Optional) In **App Settings** > **Secrets**, add your OpenAI API Key for cloud inference:
   ```toml
   OPENAI_API_KEY = "sk-proj-..."
   ```

---

## 🛡️ License

This project is open-source and available under the MIT License.

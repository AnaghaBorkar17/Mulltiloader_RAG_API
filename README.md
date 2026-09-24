# Multi-Loader RAG API 🚀

A multimodal Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **LangChain**, and **Hugging Face**.

This system ingests data across diverse document types, OCR images, transcribed audio files, and SQL databases, indexes them into an in-memory vector store, and generates grounded answers using a local, lightweight LLM running completely offline on CPU.

---

## 🌟 Key Features

- **Multi-Format Ingestion**:
  - **Documents**: PDF (`.pdf`), Microsoft Word (`.docx`), Plain Text (`.txt`), CSV (`.csv`)
  - **Images (OCR)**: Extracts text from images (`.png`, `.jpg`, `.jpeg`) using **Tesseract OCR**
  - **Audio (Speech-to-Text)**: Transcribes audio files (`.wav`, `.mp3`, `.m4a`) using **OpenAI Whisper** (with pure-Python WAV decoding via `scipy`)
  - **SQL Databases**: Connects via **SQLAlchemy** to ingest rows from SQLite, MySQL, PostgreSQL, or SQL queries into the knowledge base
- **Local & Offline Inference**:
  - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` (free, runs locally)
  - **LLM**: `Qwen/Qwen2.5-0.5B-Instruct` via Hugging Face `transformers` (optimized for local CPU inference without requiring OpenAI API keys or credits)
- **High Precision & Hallucination Prevention**:
  - Semantic similarity search with cosine score calculation
  - Strict relevance filtering (similarity threshold `>= 0.25`) to prevent cross-document contamination
  - Fallback message when facts are not grounded in uploaded context
  - Source tracking and deduplication returning exact file names and types
- **Interactive Swagger UI**:
  - Custom OpenAPI schema enabling direct file drag-and-drop in `/docs`

---

## 📁 Project Structure

```
├── app.py              # FastAPI application & API endpoints
├── loaders.py          # Unified document loader (PDF, DOCX, TXT, CSV, OCR images, audio)
├── audio_loader.py     # Whisper audio transcription loader
├── sql_loader.py       # SQLAlchemy database loader
├── vectorstore.py      # Text splitting, MiniLM embeddings & in-memory vector store
├── rag.py              # Qwen local LLM generation pipeline & relevance scoring
├── requirements.txt    # Project Python dependencies
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

### 4. (Optional) External Prerequisites

- **Tesseract OCR** (for image OCR):
  - Windows: Install [Tesseract-OCR](https://github.com/UB-Mannheim/tesseract/wiki) to `C:\Program Files\Tesseract-OCR\tesseract.exe`
  - Linux: `sudo apt-get install tesseract-ocr`
  - macOS: `brew install tesseract`

---

## 🚀 Running the Application

Start the FastAPI development server:

```bash
uvicorn app:app --reload
```

The API will be available at:
- **Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
- **ReDoc Documentation**: `http://127.0.0.1:8000/redoc`

---

## 📡 API Endpoints

### 1. Health Check (`GET /`)
Returns server status and supported file formats.

### 2. Upload Files (`POST /upload`)
Upload one or multiple files simultaneously (PDF, DOCX, TXT, CSV, PNG, JPG, MP3, WAV, etc.).

**Response:**
```json
{
  "message": "Files processed successfully",
  "files": [
    "sample.pdf",
    "product_spec_image.png"
  ],
  "total_chunks": 42
}
```

### 3. Ingest SQL Data (`POST /sql`)
Ingest query results from any SQL database.

**Request Body:**
```json
{
  "database_url": "sqlite:///sample.db",
  "query": "SELECT * FROM employees;"
}
```

### 4. Query RAG (`POST /query`)
Query the knowledge base using natural language.

**Request Body:**
```json
{
  "question": "What is the warranty policy for laptop purchases?"
}
```

**Response:**
```json
{
  "answer": "All laptops include a 1-year limited warranty covering manufacturer defects.",
  "sources": [
    {
      "source": "uploads/sample.pdf",
      "file_type": "pdf"
    }
  ]
}
```

---

## 🧪 Testing

Run test scripts locally:

```bash
# Test file loading across supported formats
python test_loader.py

# Test end-to-end RAG pipeline
python test_rag.py
```

---

## 🛡️ License

This project is open-source and available under the MIT License.

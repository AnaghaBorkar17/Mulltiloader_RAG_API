import os
import shutil
import tempfile
from pathlib import Path
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# Import project modules
from loaders import load_document
from vectorstore import (
    add_documents,
    reset_vector_store,
    get_total_chunks,
    search_documents_with_score
)
from rag import answer_question
from sql_loader import load_sql_data

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Multi-Loader RAG Assistant",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Styling (Glassmorphism & Modern Gradients)
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e1e2f 0%, #2d1b4e 50%, #1e1e2f 100%);
        padding: 1.8rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        color: white;
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        font-size: 0.8rem;
        font-weight: 600;
        border-radius: 9999px;
        background: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        border: 1px solid rgba(56, 189, 248, 0.3);
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        text-align: center;
    }
    
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
    }
    
    .source-box {
        background: rgba(255, 255, 255, 0.02);
        border-left: 3px solid #818cf8;
        padding: 0.8rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# State Initialization
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []

if "total_chunks_indexed" not in st.session_state:
    st.session_state.total_chunks_indexed = get_total_chunks()

# ---------------------------------------------------------
# Sidebar Settings
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Engine Settings")

    default_key = os.getenv("OPENAI_API_KEY", "")
    try:
        if not default_key and "OPENAI_API_KEY" in st.secrets:
            default_key = st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    llm_provider = st.selectbox(
        "LLM Inference Engine",
        options=["OpenAI (Fast & Cloud-Optimized)", "Local Model (Qwen-2.5-0.5B / Extractive)"],
        index=0 if default_key else 1,
        help="OpenAI runs on cloud API without local RAM usage. Local model runs offline."
    )

    openai_key_input = ""
    if "OpenAI" in llm_provider:
        openai_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            value=default_key,
            placeholder="sk-proj-...",
            help="Your key is kept only in session memory."
        )

    st.markdown("---")
    st.markdown("### 🔍 Retrieval Tuning")

    similarity_thresh = st.slider(
        "Similarity Threshold (Cosine)",
        min_value=0.10,
        max_value=0.60,
        value=0.25,
        step=0.05,
        help="Rejects passages below this relevance score to prevent hallucinations."
    )

    top_k = st.slider(
        "Top-K Chunks to Retrieve",
        min_value=1,
        max_value=8,
        value=4,
        step=1
    )

    st.markdown("---")
    st.markdown("### 📊 Status & Controls")
    st.write(f"**Indexed Chunks:** `{get_total_chunks()}`")
    st.write(f"**Loaded Files:** `{len(st.session_state.uploaded_file_names)}`")

    if st.button("🗑️ Reset Knowledge Base", use_container_width=True):
        reset_vector_store()
        st.session_state.uploaded_file_names = []
        st.session_state.total_chunks_indexed = 0
        st.session_state.messages = []
        st.success("Knowledge base cleared!")
        st.rerun()

    st.markdown("---")
    st.caption("Developed with ❤️ | Multi-Loader RAG v2.0")

# ---------------------------------------------------------
# Hero Banner
# ---------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>Multi-Loader RAG Assistant 🚀</h1>
    <p style="margin: 0.4rem 0 1rem 0; color: #cbd5e1; font-size: 1.05rem;">
        Multimodal Retrieval-Augmented Generation across Documents, OCR Images, Audio Transcripts & SQL Databases.
    </p>
    <div>
        <span class="badge-pill">📄 PDF & DOCX</span>
        <span class="badge-pill">📊 CSV & TXT</span>
        <span class="badge-pill">🖼️ OCR Images (PNG/JPG)</span>
        <span class="badge-pill">🎙️ Audio Whisper (WAV/MP3)</span>
        <span class="badge-pill">🗄️ SQL Tables</span>
        <span class="badge-pill">🧠 MiniLM-L6-v2 Embeddings</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Tabs Navigation
# ---------------------------------------------------------
tab_chat, tab_ingest, tab_sql, tab_info = st.tabs([
    "💬 RAG Chat & Query",
    "📁 Document & Media Ingest",
    "🗄️ SQL Ingest",
    "ℹ️ System Overview"
])

# ---------------------------------------------------------
# TAB 1: Chat & Query
# ---------------------------------------------------------
with tab_chat:
    total_active = get_total_chunks()
    if total_active == 0:
        st.info("💡 **Your knowledge base is currently empty.** Head to the **'Document & Media Ingest'** tab to upload files or click the sample data button to get started!")

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📚 Source Attribution"):
                    for src in msg["sources"]:
                        st.markdown(f"- **{src.get('file_type', 'unknown').upper()}**: `{src.get('source', 'Unknown')}`")

    # Chat input
    if prompt := st.chat_input("Ask a question about your indexed files or database..."):
        # Append user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Assistant thinking
        with st.chat_message("assistant"):
            with st.spinner("Searching semantic index & synthesizing answer..."):
                try:
                    result = answer_question(
                        question=prompt,
                        k=top_k,
                        similarity_threshold=similarity_thresh,
                        openai_api_key=openai_key_input
                    )
                    answer = result.get("answer", "No answer generated.")
                    sources = result.get("sources", [])

                    st.markdown(answer)
                    if sources:
                        with st.expander("📚 Source Attribution"):
                            for src in sources:
                                st.markdown(f"- **{src.get('file_type', 'unknown').upper()}**: `{src.get('source', 'Unknown')}`")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                except Exception as e:
                    err_msg = f"⚠️ Query error: {str(e)}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                        "sources": []
                    })

# ---------------------------------------------------------
# TAB 2: Document & Media Ingest
# ---------------------------------------------------------
with tab_ingest:
    col_upload, col_samples = st.columns([2, 1])

    with col_upload:
        st.subheader("Upload Documents & Media")
        uploaded_files = st.file_uploader(
            "Choose files to index",
            type=["pdf", "docx", "txt", "csv", "png", "jpg", "jpeg", "mp3", "wav", "m4a"],
            accept_multiple_files=True,
            help="Files are processed, chunked, embedded, and added to the in-memory vector index."
        )

        if st.button("📥 Process & Index Uploaded Files", type="primary", disabled=not uploaded_files):
            with st.spinner("Processing documents, running OCR / speech recognition, and building embeddings..."):
                temp_dir = tempfile.mkdtemp()
                success_count = 0
                total_new_chunks = 0
                errors = []

                for uploaded in uploaded_files:
                    target_file = os.path.join(temp_dir, uploaded.name)
                    with open(target_file, "wb") as f:
                        f.write(uploaded.getbuffer())

                    try:
                        docs = load_document(target_file)
                        chunks = add_documents(docs)
                        total_new_chunks += chunks
                        success_count += 1
                        if uploaded.name not in st.session_state.uploaded_file_names:
                            st.session_state.uploaded_file_names.append(uploaded.name)
                    except Exception as e:
                        errors.append(f"{uploaded.name}: {str(e)}")

                shutil.rmtree(temp_dir, ignore_errors=True)

                if success_count > 0:
                    st.success(f"✅ Successfully processed {success_count} file(s)! Created {total_new_chunks} chunks.")
                if errors:
                    for err in errors:
                        st.warning(f"⚠️ {err}")
                st.rerun()

    with col_samples:
        st.subheader("⚡ Quick Start")
        st.markdown("Test the RAG engine immediately using pre-packaged test files:")

        test_dir = Path("test_files")
        if test_dir.exists():
            sample_files = list(test_dir.glob("*.*"))
            for sf in sample_files:
                st.markdown(f"- `{sf.name}` ({sf.suffix.upper()[1:]})")

            if st.button("⚡ Load All Sample Documents"):
                with st.spinner("Indexing pre-packaged test files..."):
                    total_new_chunks = 0
                    for sf in sample_files:
                        try:
                            docs = load_document(str(sf))
                            chunks = add_documents(docs)
                            total_new_chunks += chunks
                            if sf.name not in st.session_state.uploaded_file_names:
                                st.session_state.uploaded_file_names.append(sf.name)
                        except Exception as e:
                            st.error(f"Error loading {sf.name}: {e}")
                    st.success(f"Indexed {total_new_chunks} chunks from test files!")
                    st.rerun()
        else:
            st.caption("No sample files found in test_files directory.")

    st.markdown("---")
    st.subheader("📚 Currently Indexed Files")
    if st.session_state.uploaded_file_names:
        for fname in st.session_state.uploaded_file_names:
            st.markdown(f"- 📄 `{fname}`")
    else:
        st.caption("No files indexed yet.")

# ---------------------------------------------------------
# TAB 3: SQL Ingestion
# ---------------------------------------------------------
with tab_sql:
    st.subheader("🗄️ Ingest SQL Database Records")
    st.markdown("Connect to any SQLAlchemy-supported database (SQLite, MySQL, PostgreSQL) and index query rows as searchable context.")

    col1, col2 = st.columns(2)
    with col1:
        db_url = st.text_input(
            "Database Connection URL or SQLite File Path",
            value="sqlite:///sample.db",
            help="Examples: sqlite:///C:/path/db.sqlite or mysql+pymysql://user:pass@host/dbname"
        )
    with col2:
        sql_query = st.text_input(
            "SQL Query",
            value="SELECT * FROM employees LIMIT 50;",
            help="SQL query to execute and convert into document chunks."
        )

    if st.button("🚀 Ingest SQL Data", type="primary"):
        with st.spinner("Connecting to database, executing query, and embedding rows..."):
            try:
                documents = load_sql_data(db_url, sql_query)
                chunks = add_documents(documents)
                st.success(f"✅ Ingested {len(documents)} database rows ({chunks} vector chunks)!")
                sql_source_name = f"SQL: {sql_query[:30]}..."
                if sql_source_name not in st.session_state.uploaded_file_names:
                    st.session_state.uploaded_file_names.append(sql_source_name)
                st.rerun()
            except Exception as e:
                st.error(f"❌ SQL Ingestion Failed: {str(e)}")

# ---------------------------------------------------------
# TAB 4: System Overview
# ---------------------------------------------------------
with tab_info:
    st.subheader("Architecture & Technology Stack")
    st.markdown("""
    - **FastAPI Backend**: Run standalone via `python api.py` or `uvicorn api:app --reload` with Swagger UI at `/docs`.
    - **Streamlit Frontend**: Interactive web interface for uploading files, asking questions, and inspecting retrieved source attribution.
    - **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` generating 384-dimensional dense semantic vectors.
    - **Vector Store**: `InMemoryVectorStore` from LangChain providing instant cosine similarity scoring without requiring external DB instances.
    - **LLM Pipeline**:
      - **OpenAI (Optional / Recommended for Cloud)**: Uses `gpt-4o-mini` with zero cloud RAM overhead.
      - **Local HuggingFace Model**: `Qwen/Qwen2.5-0.5B-Instruct` running purely on CPU offline.
    - **Multimodal Document Loaders**:
      - **PDF**: `PyPDFLoader` (`pypdf`)
      - **Word**: `Docx2txtLoader` (`docx2txt`)
      - **CSV / TXT**: LangChain structured loaders
      - **Images**: OCR text extraction via `pytesseract` & `Pillow`
      - **Audio**: Speech-to-text transcription via `openai-whisper` & `scipy`
      - **SQL**: Database table ingestion via `SQLAlchemy`
    """)

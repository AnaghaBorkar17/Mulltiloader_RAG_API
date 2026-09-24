from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore


# Free, local Hugging Face embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# In-memory vector store
vectorstore = InMemoryVectorStore(
    embeddings
)

# Text splitter: 400 chars with 50 overlap preserves tabular rows & document sections
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)


def add_documents(documents):
    """
    Split incoming documents into clean chunks and index them into the vector store.
    """
    chunks = text_splitter.split_documents(documents)
    if chunks:
        vectorstore.add_documents(chunks)
    return len(chunks)


def search_documents_with_score(query: str, k: int = 4):
    """
    Perform semantic similarity search and return document chunks along with cosine similarity scores.
    """
    results = vectorstore.similarity_search_with_score(
        query,
        k=k
    )
    return results


def search_documents(query: str, k: int = 4):
    """
    Backward-compatible search returning documents only.
    """
    results = vectorstore.similarity_search(
        query,
        k=k
    )
    return results


def create_vector_store(documents):
    """
    Helper function to create and populate a fresh vector store from a list of documents.
    Used by test scripts.
    """
    vs = InMemoryVectorStore(embeddings)
    chunks = text_splitter.split_documents(documents)
    if chunks:
        vs.add_documents(chunks)
    return vs

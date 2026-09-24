from loaders import load_document
from vectorstore import add_documents
from rag import answer_question

files = [
    "test_files/sample.pdf",
    "test_files/sample.docx",
    "test_files/sample.txt",
    "test_files/sample.csv"
]

print("=== 1. LOADING AND INDEXING DOCUMENTS ===")
total_chunks = 0
for file in files:
    try:
        print(f"Loading: {file}")
        docs = load_document(file)
        chunks = add_documents(docs)
        total_chunks += chunks
        print(f"  -> Loaded {len(docs)} documents ({chunks} chunks)")
    except Exception as e:
        print(f"  -> Error loading {file}: {e}")

print(f"\nTotal indexed chunks: {total_chunks}")

print("\n=== 2. TESTING QUESTION-SPECIFIC RAG QUERIES ===")
queries = [
    "What is Artificial Intelligence?",
    "What is Rahul's score?",
    "What are the components of a typical RAG pipeline?",
    "What ingredients are listed in the recipe?"
]

for q in queries:
    print(f"\n----------------------------------------")
    print(f"Question: {q}")
    result = answer_question(q)
    print(f"Answer: {result['answer']}")
    print(f"Sources: {result['sources']}")

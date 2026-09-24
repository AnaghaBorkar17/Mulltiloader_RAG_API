from loaders import load_document

files = [
    "test_files/sample.pdf",
    "test_files/sample.docx",
    "test_files/sample.txt",
    "test_files/sample.csv"
]

for file in files:

    print("\n==============================")
    print("Testing:", file)
    print("==============================")

    documents = load_document(file)

    print("Documents loaded:", len(documents))

    for document in documents:
        print(document.page_content[:200])
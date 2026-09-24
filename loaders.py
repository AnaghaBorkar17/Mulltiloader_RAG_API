# app/loaders.py

import os
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    Docx2txtLoader,
    CSVLoader
)
from langchain_core.documents import Document
from PIL import Image
import pytesseract

# Configure Tesseract OCR path for Windows
TESSERACT_EXE = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
if os.path.exists(TESSERACT_EXE):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE


def load_document(file_path: str):
    """
    Load document across multiple supported formats and ensure proper metadata
    (source and file_type) is attached to every document chunk.
    """
    extension = Path(file_path).suffix.lower()

    # PDF
    if extension == ".pdf":
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_type"] = "pdf"
            doc.metadata["source"] = file_path
        return docs

    # DOCX
    elif extension == ".docx":
        loader = Docx2txtLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_type"] = "docx"
            doc.metadata["source"] = file_path
        return docs

    # TXT
    elif extension == ".txt":
        loader = TextLoader(file_path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_type"] = "txt"
            doc.metadata["source"] = file_path
        return docs

    # CSV
    elif extension == ".csv":
        loader = CSVLoader(file_path, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["file_type"] = "csv"
            doc.metadata["source"] = file_path
        return docs

    # IMAGE (OCR via Tesseract)
    elif extension in [".png", ".jpg", ".jpeg"]:
        image = Image.open(file_path)
        text = pytesseract.image_to_string(image)

        if not text.strip():
            raise ValueError("No text could be extracted from image.")

        return [
            Document(
                page_content=text.strip(),
                metadata={
                    "source": file_path,
                    "file_type": "image"
                }
            )
        ]

    # AUDIO (Transcription via Whisper)
    elif extension in [".mp3", ".wav", ".m4a"]:
        from audio_loader import load_audio
        return load_audio(file_path)

    else:
        raise ValueError(f"Unsupported file type: {extension}")

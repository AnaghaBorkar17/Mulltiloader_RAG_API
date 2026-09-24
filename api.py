import os
import shutil
from typing import Annotated

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from loaders import load_document
from vectorstore import add_documents
from rag import answer_question
from sql_loader import load_sql_data


load_dotenv()


app = FastAPI(
    title="Multi-Loader RAG API",
    description="RAG API supporting documents, images, audio and SQL data",
    version="2.0"
)


UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# --------------------------------
# Fix Swagger file upload display
# --------------------------------

def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema

    from fastapi.openapi.utils import get_openapi

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    # Convert application/octet-stream
    # to binary so Swagger shows file picker
    for schema in openapi_schema.get("components", {}).get("schemas", {}).values():
        properties = schema.get("properties", {})
        for prop in properties.values():
            if prop.get("contentMediaType") == "application/octet-stream":
                prop.pop("contentMediaType", None)
                prop["format"] = "binary"

            items = prop.get("items", {})
            if items.get("contentMediaType") == "application/octet-stream":
                items.pop("contentMediaType", None)
                items["format"] = "binary"

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


# --------------------------------
# Request Models
# --------------------------------

class QueryRequest(BaseModel):
    question: str = Field(
        ...,
        description="The question to query against the indexed documents",
        json_schema_extra={
            "example": "How many laptops were sold?"
        }
    )
    similarity_threshold: float = Field(
        0.25,
        description="Cosine similarity threshold for relevance filtering",
        ge=0.0,
        le=1.0
    )


class SQLRequest(BaseModel):
    database_url: str = Field(
        ...,
        description="SQLAlchemy database URL or SQLite path",
        json_schema_extra={
            "example": "sqlite:///sample.db"
        }
    )
    query: str = Field(
        ...,
        description="SQL query to execute",
        json_schema_extra={
            "example": "SELECT * FROM employees;"
        }
    )


# --------------------------------
# Home
# --------------------------------

@app.get("/")
def home():
    return {
        "message": "Multi-Loader RAG API is running",
        "version": "2.0",
        "supported_files": [
            "PDF",
            "DOCX",
            "TXT",
            "CSV",
            "JPG",
            "JPEG",
            "PNG",
            "MP3",
            "WAV",
            "M4A"
        ]
    }


# --------------------------------
# Upload Files
# --------------------------------

@app.post("/upload")
async def upload_files(
    files: Annotated[
        list[UploadFile],
        File(description="Upload one or more files")
    ]
):
    total_chunks = 0
    uploaded_files = []
    errors = []

    for file in files:
        file_path = os.path.join(
            UPLOAD_DIR,
            file.filename
        )

        try:
            # Save file
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(
                    file.file,
                    buffer
                )

            # Load document
            documents = load_document(
                file_path
            )

            # Split + embed + store
            chunks = add_documents(
                documents
            )

            total_chunks += chunks
            uploaded_files.append(
                file.filename
            )

        except Exception as e:
            errors.append(f"{file.filename}: {str(e)}")

    if not uploaded_files and errors:
        raise HTTPException(
            status_code=400,
            detail="; ".join(errors)
        )

    response = {
        "message": "Files processed successfully" if not errors else "Files processed with warnings",
        "files": uploaded_files,
        "total_chunks": total_chunks
    }
    if errors:
        response["warnings"] = errors

    return response


# --------------------------------
# SQL
# --------------------------------

@app.post("/sql")
def upload_sql_data(
    request: SQLRequest
):
    try:
        documents = load_sql_data(
            request.database_url,
            request.query
        )

        chunks = add_documents(
            documents
        )

        return {
            "message": "SQL data processed successfully",
            "rows": len(documents),
            "chunks": chunks
        }

    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"SQL processing error: {str(e)}"
        )


# --------------------------------
# Query
# --------------------------------

@app.post("/query")
def query(
    request: QueryRequest
):
    try:
        result = answer_question(
            request.question,
            similarity_threshold=request.similarity_threshold
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG query error: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

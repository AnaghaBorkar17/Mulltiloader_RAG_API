# app/sql_loader.py

from sqlalchemy import create_engine, text
from langchain_core.documents import Document


def load_sql_data(
    database_url: str,
    query: str
):
    """
    Connect to a SQL database and load rows as LangChain Documents.
    Automatically handles raw SQLite file paths (e.g. C:/path/db.sqlite or C:\path\db.sqlite)
    and converts them to SQLAlchemy-compatible URLs (sqlite:///C:/path/db.sqlite).
    """
    url = database_url.strip()

    # Automatically normalize SQLite file paths if dialect prefix is missing
    known_dialects = ("sqlite:", "mysql:", "postgresql:", "mssql:", "oracle:")
    if not any(url.lower().startswith(d) for d in known_dialects):
        clean_path = url.replace("\\", "/")
        url = f"sqlite:///{clean_path}"

    engine = create_engine(url)

    documents = []

    with engine.connect() as connection:
        result = connection.execute(
            text(query)
        )

        rows = result.fetchall()
        columns = result.keys()

        for row in rows:
            row_data = dict(
                zip(columns, row)
            )

            content = "\n".join(
                f"{key}: {value}"
                for key, value in row_data.items()
            )

            documents.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": "SQL Database",
                        "file_type": "sql"
                    }
                )
            )

    return documents

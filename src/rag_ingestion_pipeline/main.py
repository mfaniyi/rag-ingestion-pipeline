from uuid import uuid4
from pathlib import Path
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from rag_ingestion_pipeline.chunking import fixed_size_chunk
from rag_ingestion_pipeline.database import create_tables, save_chunks
from rag_ingestion_pipeline.embeddings import generate_embedding
from rag_ingestion_pipeline.ingestion import extract_document, validate_upload


app = FastAPI(title="RAG Ingestion Pipeline")


# Create the database tables when the application starts.
create_tables()


@app.get("/")
def root():
    return {"message": "RAG ingestion pipeline is running"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    # Read the uploaded document into memory.
    content = await file.read()

    try:
        # Validate the uploaded file before processing it.
        validate_upload(file, len(content))

        # Generate a unique ID for the uploaded document.
        document_id = str(uuid4())

        # Extract text while preserving page numbers.
        pages = extract_document(content, file.filename or "")

        # Create fixed-size chunks with 50-character overlap.
        chunks = fixed_size_chunk(
            document_id=document_id,
            pages=pages,
            chunk_size=500,
            overlap=50,
        )

        # Generate an embedding for every chunk.
        embeddings = [
            generate_embedding(chunk.text)
            for chunk in chunks
        ]

        # Store the chunks, metadata, and embeddings in PostgreSQL.
        save_chunks(chunks, embeddings)

    except ValueError as error:
        # Return validation and processing errors as a client error.
        raise HTTPException(status_code=400, detail=str(error)) from error

    return {
        "filename": file.filename,
        "document_id": document_id,
        "chunks_created": len(chunks),
        "embeddings_generated": len(embeddings),
        "stored": len(chunks),
        "message": "Document successfully ingested",
    }

@app.get("/ui", response_class=HTMLResponse)
def upload_ui():
    # Locate the HTML upload page in the project templates directory.
    template_path = Path("templates/upload.html")

    # Return the upload page to the browser.
    return template_path.read_text(encoding="utf-8")

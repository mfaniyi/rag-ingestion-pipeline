from uuid import uuid4
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from rag_ingestion_pipeline.chunking import fixed_size_chunk
from rag_ingestion_pipeline.database import create_tables, engine, save_chunks
from rag_ingestion_pipeline.embeddings import generate_embedding
from rag_ingestion_pipeline.ingestion import extract_document, validate_upload
from rag_ingestion_pipeline.retrieval import hybrid_search
from rag_ingestion_pipeline.reranking import rerank_chunks
from rag_ingestion_pipeline.generation import generate_grounded_answer


app = FastAPI(title="RAG Ingestion Pipeline")


class AskRequest(BaseModel):
    """Represent a question about a specific uploaded document."""

    # Store the user's question.
    question: str

    # Store the document that the question should search.
    document_id: str


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

        # Reject documents that contain no usable text.
        if not chunks:
            raise ValueError(
                "The uploaded document is empty or contains no usable text."
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
        raise HTTPException(
            status_code=400,
            detail=str(error),
        ) from error

    return {
        "filename": file.filename,
        "document_id": document_id,
        "chunks_created": len(chunks),
        "embeddings_generated": len(embeddings),
        "stored": len(chunks),
        "message": "Document successfully ingested",
    }


@app.post("/ask")
def ask_question(request: AskRequest):
    """Retrieve evidence and generate a grounded answer."""

    # Generate an embedding for the user's question.
    query_embedding = generate_embedding(request.question)

    # Connect to PostgreSQL for retrieval.
    with Session(engine) as session:
        # Retrieve a larger candidate pool using hybrid search.
        hybrid_results = hybrid_search(
            session,
            request.question,
            query_embedding,
            top_k=10,
            document_id=request.document_id,
        )

    # Extract the retrieved chunks from the hybrid results.
    candidate_chunks = [
        chunk
        for chunk, _ in hybrid_results
    ]

    # Rerank the candidate chunks using the cross-encoder.
    reranked_results = rerank_chunks(
        request.question,
        candidate_chunks,
        top_k=5,
    )

    # Refuse to answer when no evidence was retrieved.
    if not reranked_results:
        return {
            "question": request.question,
            "answer": (
                "I could not find enough information in the uploaded "
                "documents to answer this question."
            ),
            "sources": [],
        }

    # Build citation-labelled context for the language model.
    context_parts = []

    for chunk, _ in reranked_results:
        # Build a page-aware citation for documents that have page numbers.
        if chunk.page_number is not None:
            citation = (
                f"[Page {chunk.page_number}, "
                f"Chunk {chunk.chunk_index}]"
            )
        else:
            # Use only the chunk number for documents without pages, such as TXT.
            citation = f"[Chunk {chunk.chunk_index}]"

        # Add the citation before the chunk so the model can reference it.
        context_parts.append(
            f"{citation}\n"
            f"{chunk.text}"
        )

    # Combine all retrieved chunks into one context.
    context = "\n\n".join(context_parts)

    # Generate an answer grounded only in the retrieved evidence.
    answer = generate_grounded_answer(
        question=request.question,
        context=context,
    )

    # Return the grounded answer and supporting sources.
    return {
        "question": request.question,
        "answer": answer,
        "sources": [
            {
                "document_id": chunk.document_id,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "text": chunk.text,
            }
            for chunk, _ in reranked_results
        ],
    }


@app.get("/ui", response_class=HTMLResponse)
def upload_ui():
    # Locate the HTML upload page in the project templates directory.
    template_path = Path("templates/upload.html")

    # Return the upload page to the browser.
    return template_path.read_text(encoding="utf-8")

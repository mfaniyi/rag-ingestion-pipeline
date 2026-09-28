# RAG Ingestion Pipeline

A document ingestion pipeline that prepares PDF and TXT documents for Retrieval-Augmented Generation (RAG).

The pipeline accepts a document, validates it, extracts its text, splits it into chunks, generates vector embeddings, and stores the chunks and embeddings in PostgreSQL with pgvector.

## Pipeline

```text
Upload
   ↓
Validate
   ↓
Extract
   ↓
Chunk
   ↓
Generate Embeddings
   ↓
Store in PostgreSQL + pgvector
```

## Project Objective

This project implements the **ingestion half of a RAG system**.

The goal is to build a working:

> **Upload → Chunk → Embed → Store**

pipeline while understanding the design decisions behind document chunking, embeddings, metadata, vector storage, and safe document ingestion.

## Learning Objectives

* Implement a defensible and explainable document chunking strategy.
* Compare different chunking strategies.
* Generate embeddings for document chunks.
* Store embeddings with useful retrieval metadata.
* Select and use an appropriate vector store.
* Implement safe document upload with file-type and size restrictions.
* Understand the trade-offs involved in document chunking.

## Technologies

* **Python**
* **FastAPI** — API and web interface
* **Pydantic / FastAPI validation**
* **pypdf** — PDF text extraction
* **Sentence Transformers** — embedding generation
* **all-MiniLM-L6-v2** — embedding model
* **SQLAlchemy** — database ORM
* **PostgreSQL** — relational database
* **pgvector** — vector storage
* **JupyterLab** — experimentation and analysis
* **Docker** — PostgreSQL + pgvector environment
* **uv** — Python project and dependency management

## Project Structure

```text
rag-ingestion-pipeline/
│
├── docs/
│   └── chunking-strategy-comparison.md
│
├── notebooks/
│   └── 01_model.ipynb
│
├── templates/
│   └── upload.html
│
├── src/
│   └── rag_ingestion_pipeline/
│       ├── __init__.py
│       ├── models.py
│       ├── ingestion.py
│       ├── chunking.py
│       ├── embeddings.py
│       ├── database.py
│       └── main.py
│
├── .gitignore
├── pyproject.toml
├── README.md
└── uv.lock
```

## Core Components

### 1. Document Upload

The FastAPI application provides an upload endpoint:

```text
POST /upload
```

Supported file types:

* `.pdf`
* `.txt`

Maximum file size:

```text
10 MB
```

The application does not fetch user-supplied URLs, preventing server-side URL fetching/SSRF behavior in the ingestion workflow.

### 2. Text Extraction

PDF documents are processed using `pypdf`.

Page numbers are preserved so that chunks can retain their source-page metadata.

TXT files are decoded as UTF-8 text.

### 3. Chunking

Two chunking strategies were implemented and compared.

#### Fixed-size chunking

Configuration:

```text
Chunk size: 500 characters
Overlap: 50 characters
```

This approach creates relatively consistent-sized chunks while the overlap helps preserve context across chunk boundaries.

#### Paragraph-based chunking

The document is divided using paragraph boundaries.

This preserves larger semantic sections but can produce substantially larger chunks.

### Comparison

The comparison was performed using the same document, `MLT.pdf`.

| Metric             | Fixed-size | Paragraph |
| ------------------ | ---------: | --------: |
| Number of chunks   |        101 |        23 |
| Minimum characters |         20 |     1,019 |
| Maximum characters |        500 |     2,762 |
| Average characters |     438.75 |  1,755.65 |
| Overlap            |         50 |      None |

The detailed comparison is available in:

```text
docs/chunking-strategy-comparison.md
```

### Selected Strategy

The project uses:

```text
Fixed-size chunking
500 characters
50-character overlap
```

as the primary ingestion strategy.

The strategy was selected because it provides predictable chunk sizes, creates smaller retrieval units, and uses overlap to preserve some context between adjacent chunks.

A limitation is that fixed character boundaries can split words or sentences.

## Embeddings

The project uses the Sentence Transformers model:

```text
all-MiniLM-L6-v2
```

Each chunk is converted into a numerical vector.

The resulting embedding size is:

```text
384 dimensions
```

For example:

```text
Document chunk
      ↓
all-MiniLM-L6-v2
      ↓
384-dimensional vector
```

## Vector Storage

PostgreSQL with the **pgvector** extension is used as the vector store.

The database stores:

* `chunk_id`
* `document_id`
* `text`
* `chunk_index`
* `page_number`
* `strategy`
* `embedding`

The embedding column is:

```text
VECTOR(384)
```

### Why PostgreSQL + pgvector?

PostgreSQL was selected because it allows the project to keep document metadata and vector embeddings in the same database.

This provides a straightforward foundation for later RAG retrieval without introducing a separate vector database.

## Running PostgreSQL + pgvector

The project uses Docker to run PostgreSQL with pgvector.

```bash
docker run -d \
  --name rag-postgres \
  -e POSTGRES_USER=raguser \
  -e POSTGRES_PASSWORD=ragpassword \
  -e POSTGRES_DB=ragdb \
  -p 5433:5432 \
  pgvector/pgvector:pg18
```

Enable the pgvector extension:

```bash
docker exec -it rag-postgres psql -U raguser -d ragdb
```

Then:

```sql
CREATE EXTENSION vector;
```

The application connects using:

```text
localhost:5433
```

## Installation

Clone the repository and enter the project directory:

```bash
git clone https://github.com/mfaniyi/rag-ingestion-pipeline.git
cd rag-ingestion-pipeline
```

Create the project environment:

```bash
uv venv
```

Activate the environment in Git Bash:

```bash
source .venv/Scripts/activate
```

Install the project dependencies:

```bash
uv sync
```

## Running the Application

Start the FastAPI application:

```bash
uv run uvicorn rag_ingestion_pipeline.main:app --reload
```

The application will be available at:

```text
http://127.0.0.1:8000
```

## Web Upload Interface

A simple browser-based upload interface is available at:

```text
http://127.0.0.1:8000/ui
```

The interface allows users to select a PDF or TXT document and trigger the complete ingestion pipeline.

```text
Upload Document
      ↓
Validate
      ↓
Extract
      ↓
Chunk
      ↓
Embed
      ↓
Store
```

After processing, the interface displays:

* Filename
* Number of chunks created
* Number of embeddings generated
* Number of chunks stored

## API Documentation

FastAPI automatically provides interactive Swagger documentation at:

```text
http://127.0.0.1:8000/docs
```

The main endpoint is:

```text
POST /upload
```

Uploading a document through Swagger triggers the complete ingestion pipeline.

Example response:

```json
{
  "filename": "What is Machine Learning.txt",
  "document_id": "a99cc6bc-0cd5-4137-b0e8-34dcff325730",
  "chunks_created": 5,
  "embeddings_generated": 5,
  "stored": 5,
  "message": "Document successfully ingested"
}
```

## Jupyter Notebook

The notebook:

```text
notebooks/01_model.ipynb
```

was used for experimentation and analysis.

It includes:

* Document extraction experiments
* Chunking experiments
* Chunk-size analysis
* Comparison of chunking strategies
* Embedding generation
* Embedding dimension checks
* Pipeline testing
* Data visualization

Start JupyterLab with:

```bash
uv run jupyter lab
```

## Database Verification

The stored chunks can be inspected using PostgreSQL:

```bash
docker exec -it rag-postgres psql -U raguser -d ragdb
```

For example:

```sql
SELECT
    chunk_id,
    document_id,
    chunk_index,
    strategy
FROM document_chunks
ORDER BY document_id, chunk_index;
```

The database can also be used to verify the stored embedding dimensions.

## Assignment Deliverables

### Learning objectives

* [x] Defensible document chunking strategy
* [x] Embedding generation
* [x] Embedding storage with metadata
* [x] Vector store selection and justification
* [x] Safe document upload
* [x] File type restrictions
* [x] File size limits
* [x] No server-side fetching of user URLs

### Assignment

Implement and compare two chunking strategies on the same document set.

* [x] Fixed-size chunking
* [x] Paragraph-based chunking
* [x] Compare both strategies
* [x] Document the results
* [x] Explain the trade-offs
* [x] Select a primary strategy

### Milestone / Exit Check

* [x] Working upload pipeline
* [x] Document extraction
* [x] Chunking
* [x] Embedding generation
* [x] Vector storage
* [x] Useful metadata
* [x] Written chunking comparison
* [x] Populated PostgreSQL + pgvector database
* [x] End-to-end upload → chunk → embed → store workflow

## Current Scope

This project currently covers the **ingestion half of RAG**:

```text
Upload → Chunk → Embed → Store
```

Retrieval, similarity search, prompt construction, and LLM generation are outside the scope of this ingestion assignment.

## Conclusion

The completed pipeline demonstrates how documents can be safely uploaded, transformed into meaningful chunks, converted into vector embeddings, and stored with metadata in a PostgreSQL database using pgvector.

The project provides the foundation required for a future retrieval stage of a RAG application.

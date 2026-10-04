# Week 1: Build the ingestion foundation.

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

## Verification

The ingestion pipeline was verified end-to-end using the web upload interface.

A PDF document was uploaded through `/ui` and successfully processed:

- File: `dangote-petroleum-refinery-faqs-CGG-sPAx.pdf`
- Chunks created: 62
- Embeddings generated: 62
- Chunks stored: 62

PostgreSQL verification confirmed:

- Database rows: 62
- Rows with embeddings: 62

Automated testing was also added:

- Local pytest result: 8 passed
- GitHub Actions CI: passed

The verification confirms the complete ingestion flow:

Upload → Validation → Extraction → Chunking → Embedding → PostgreSQL/pgvector Storage

## Conclusion

The completed pipeline demonstrates how documents can be safely uploaded, transformed into meaningful chunks, converted into vector embeddings, and stored with metadata in a PostgreSQL database using pgvector.

The project provides the foundation required for a future retrieval stage of a RAG application.


# Week 2 - RAG Retrieval, Grounded Answers & Evaluation

This section documents the continuation of the **Retrieval-Augmented Generation (RAG)** project developed in the previous assignment.

The first assignment focused on building the document ingestion pipeline:

```text
Upload → Validate → Extract → Chunk → Embed → Store
```

This week's assignment extends that foundation into a complete RAG workflow:

```text
Ask → Retrieve → Rerank → Generate → Cite
```

---

## Week 2 Project Objective

The objective of this phase is to complete the RAG workflow by implementing:

- Vector retrieval
- Keyword retrieval
- Hybrid retrieval
- Reciprocal Rank Fusion (RRF)
- Cross-encoder reranking
- Grounded answer generation
- Source citations
- Retrieval evaluation

The completed system provides a `/ask` endpoint that retrieves relevant evidence from uploaded documents and uses that evidence to generate a grounded answer with source information.

---

## Extended RAG Architecture

```text
                    DOCUMENT INGESTION
                           │
                           ▼
                    Upload Document
                           │
                           ▼
                       Validate
                           │
                           ▼
                    Extract Text
                           │
                           ▼
                         Chunk
                           │
                           ▼
                 Generate Embeddings
                           │
                           ▼
                 PostgreSQL + pgvector
                           │
                    ───────┴───────
                           │
                           ▼
                       USER QUERY
                           │
                           ▼
                Generate Query Embedding
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       Vector Search               Keyword Search
             │                           │
             └─────────────┬─────────────┘
                           ▼
                  Hybrid Retrieval
                           │
                           ▼
                   RRF Score Fusion
                           │
                           ▼
                  Candidate Chunks
                           │
                           ▼
              Cross-Encoder Reranking
                           │
                           ▼
                     Top 5 Chunks
                           │
                           ▼
                 Grounded Generation
                           │
                           ▼
                   Answer + Citations
```

---

## 1. Vector Search

Vector search was implemented using **PostgreSQL** and **pgvector**.

The user's question is converted into a **384-dimensional embedding** using:

```text
all-MiniLM-L6-v2
```

The system then performs cosine-distance similarity search against the stored document embeddings.

The retrieval stage initially selects the **top 10 vector candidates**.

---

## 2. Keyword Search

PostgreSQL full-text search was implemented alongside vector search.

The implementation uses:

```text
to_tsvector()
websearch_to_tsquery()
ts_rank_cd()
```

Keyword search provides a complementary retrieval mechanism for exact terms and concepts that may not always be represented optimally through semantic similarity.

This allows the system to benefit from both semantic understanding and traditional lexical matching.

---

## 3. Hybrid Retrieval

The vector and keyword search results are combined using **Reciprocal Rank Fusion (RRF)**.

The implementation uses an RRF constant of:

```text
60
```

The purpose of hybrid retrieval is to combine the strengths of both retrieval approaches:

```text
Semantic Similarity
        +
Keyword Relevance
        │
        ▼
Better Candidate Set
```

The `/ask` endpoint retrieves up to **10 hybrid candidates** before reranking.

---

## 4. Cross-Encoder Reranking

The hybrid candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

The cross-encoder evaluates the relationship between the user's question and each candidate chunk.

After reranking, the **top 5 candidates** are passed to the generation stage.

The complete retrieval pipeline is:

```text
Vector Search
      +
Keyword Search
      │
      ▼
Hybrid Search
      │
      ▼
RRF
      │
      ▼
Cross-Encoder
      │
      ▼
Top 5 Evidence Chunks
```

---

## 5. Grounded Answer Generation

The `/ask` endpoint uses the retrieved chunks as context for answer generation.

The generation process is designed to keep responses **grounded in the retrieved document evidence** rather than relying solely on the language model's general knowledge.

When sufficient evidence cannot be retrieved, the system returns a fallback response indicating that there is not enough information in the uploaded documents to answer the question.

This helps reduce unsupported or ungrounded responses.

---

## 6. Source Citations

The `/ask` endpoint returns source information alongside the generated answer.

A successful response follows the general structure:

```json
{
  "question": "What is query folding?",
  "answer": "...",
  "sources": [
    {
      "chunk_id": "...",
      "document_id": "...",
      "page_number": 1
    }
  ]
}
```

Each source identifies the supporting chunk and its originating document.

This makes it possible to trace generated answers back to the retrieved document evidence.

---

## 7. `/ask` Endpoint

The main RAG question-answering endpoint is:

```http
POST /ask
```

### Example Request

```json
{
  "question": "What is query folding and what are its benefits?",
  "document_id": "mlt-001"
}
```

### Processing Flow

The endpoint performs the following operations:

```text
Question
   │
   ▼
Query Embedding
   │
   ▼
Hybrid Retrieval
   │
   ▼
RRF
   │
   ▼
Cross-Encoder Reranking
   │
   ▼
Context Construction
   │
   ▼
Grounded Generation
   │
   ▼
Answer + Sources
```

---

## 8. Retrieval Evaluation

A labelled retrieval evaluation dataset was created to measure retrieval quality.

The evaluation dataset contains:

```text
10 questions
```

Each question contains one or more manually identified relevant chunk IDs.

The dataset is stored in:

```text
evaluation/labelled_questions.json
```

The evaluation can be reproduced by running:

```bash
uv run python evaluation/evaluate_retrieval.py
```

---

## 9. Evaluation Metric

The primary retrieval evaluation metric is:

```text
Precision@5
```

**Precision@5** measures the proportion of the five retrieved chunks that are relevant to the question.

```text
              Relevant Chunks in Top 5
Precision@5 = ─────────────────────────
                          5
```

The final baseline score is calculated as the **mean Precision@5 across all 10 evaluation questions**.

---

## 10. Evaluation Results

The current retrieval pipeline achieved:

```text
Mean Precision@5: 0.32
```

### Individual Results

| Question | Precision@5 |
|----------|------------:|
| Q1 | 0.40 |
| Q2 | 0.40 |
| Q3 | 0.20 |
| Q4 | 0.20 |
| Q5 | 0.40 |
| Q6 | 0.40 |
| Q7 | 0.20 |
| Q8 | 0.40 |
| Q9 | 0.40 |
| Q10 | 0.20 |
| **Mean** | **0.32** |

This result represents a baseline measured against the current labelled evaluation dataset.

It should not be interpreted as a general performance measurement across every document or question type because the evaluation currently contains only **10 questions** from the `mlt-001` document.

---

## 11. Retrieval Failure Analysis

The evaluation revealed an important limitation in the retrieval pipeline.

A relevant chunk may be successfully identified during vector retrieval but still fail to enter the final hybrid candidate set.

If a relevant chunk is removed during candidate selection, the cross-encoder cannot recover it because reranking only operates on the candidates provided to it.

This highlights an important distinction between:

```text
Candidate Retrieval
        │
        ▼
Candidate Reranking
```

**Candidate retrieval** determines which chunks are available for further processing.

**Candidate reranking** only changes the ordering of those candidates.

Therefore:

> Reranking can improve the ordering of retrieved evidence, but it cannot recover relevant evidence that was excluded during the candidate retrieval stage.

This provides an important area for future retrieval optimization.

---

## 12. Evaluation Files

The evaluation implementation is organized as follows:

```text
evaluation/
├── evaluate_retrieval.py
├── labelled_questions.json
└── retrieval_evaluation.md
```

### `evaluate_retrieval.py`

Runs the complete retrieval evaluation against the PostgreSQL database.

### `labelled_questions.json`

Contains the manually labelled questions and their relevant chunk IDs.

### `retrieval_evaluation.md`

Contains the detailed evaluation methodology, results, observations, and limitations.

---

## 13. Testing

The project includes automated tests covering:

- Document ingestion
- File uploads
- Generation
- Retrieval
- API functionality
- Related RAG components

The full test suite currently passes:

```text
19 passed
```

Run the complete test suite with:

```bash
uv run pytest -q
```

---

## 14. CI/CD

**GitHub Actions** is configured to automatically:

1. Start PostgreSQL with pgvector.
2. Install project dependencies.
3. Enable the pgvector extension.
4. Run the automated test suite.

The CI workflow validates that the project continues to work correctly in a clean environment whenever the configured workflow is triggered.

---

## 15. Week 2 Technologies Added

In addition to the technologies used in the original assignment, the RAG retrieval and generation phase introduced:

- PostgreSQL Full-Text Search
- Reciprocal Rank Fusion (RRF)
- Cross-Encoder Reranking
- Grounded LLM Generation
- Retrieval Evaluation
- Precision@5
- GitHub Actions CI

### Retrieval Models

| Purpose | Model |
|---------|-------|
| Embedding | `all-MiniLM-L6-v2` |
| Cross-Encoder Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

---

## 16. Week 2 Learning Outcomes

This phase extended the original ingestion pipeline into a working end-to-end RAG system.

The implementation demonstrates the ability to:

1. Perform vector similarity search.
2. Perform keyword-based retrieval.
3. Combine retrieval methods using hybrid search.
4. Apply Reciprocal Rank Fusion.
5. Rerank retrieved candidates using a cross-encoder.
6. Construct grounded context for an LLM.
7. Return answers with source information.
8. Handle questions where sufficient evidence is unavailable.
9. Create a labelled retrieval evaluation dataset.
10. Measure retrieval quality using Precision@5.
11. Analyse retrieval failure cases.
12. Automate testing through CI/CD.

---

## 17. Complete Project Flow

The project now covers both document ingestion and retrieval.

```text
                    RAG INGESTION
                         │
                         ▼
                       Upload
                         │
                         ▼
                      Validate
                         │
                         ▼
                       Extract
                         │
                         ▼
                        Chunk
                         │
                         ▼
                        Embed
                         │
                         ▼
                PostgreSQL + pgvector
                         │
                  ───────┴───────
                         │
                         ▼
                    User Question
                         │
                         ▼
                Vector + Keyword
                         │
                         ▼
                  Hybrid Search
                         │
                         ▼
                        RRF
                         │
                         ▼
                   Cross-Encoder
                     Reranking
                         │
                         ▼
                   Top 5 Chunks
                         │
                         ▼
                Grounded Generation
                         │
                         ▼
                  Answer + Sources
                         │
                         ▼
               Retrieval Evaluation
                         │
                         ▼
                 Precision@5 = 0.32
```

---

## Week 2 Assignment Status

The Week 2 objectives have been implemented and evaluated.

| Requirement | Status |
|-------------|:------:|
| Vector Search | ✅ |
| Keyword Search | ✅ |
| Hybrid Retrieval | ✅ |
| Reciprocal Rank Fusion (RRF) | ✅ |
| Cross-Encoder Reranking | ✅ |
| Grounded Generation | ✅ |
| Source Citations | ✅ |
| `/ask` Endpoint | ✅ |
| Retrieval Evaluation | ✅ |
| Precision@5 Baseline | ✅ |
| Automated Tests | ✅ |
| CI/CD | ✅ |

---

## Week 2 Summary

The second phase of the project transforms the original document ingestion pipeline into a complete **Retrieval-Augmented Generation system**.

The final workflow combines **vector similarity search, PostgreSQL full-text search, Reciprocal Rank Fusion, cross-encoder reranking, grounded generation, and source attribution** to answer questions based on uploaded documents.

The current retrieval evaluation establishes a baseline:

```text
Mean Precision@5: 0.32
```

While the system successfully implements the complete RAG workflow, the evaluation also demonstrates that **candidate retrieval remains an important area for optimization**. Future improvements should focus on increasing retrieval recall, expanding the evaluation dataset, tuning hybrid retrieval parameters, and evaluating the system across a broader range of documents and question types.

---
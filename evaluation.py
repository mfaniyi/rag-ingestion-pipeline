from sqlalchemy.orm import Session

from src.rag_ingestion_pipeline.database import engine
from src.rag_ingestion_pipeline.embeddings import generate_embedding
from src.rag_ingestion_pipeline.retrieval import keyword_search, vector_search
from src.rag_ingestion_pipeline.retrieval import (hybrid_search,keyword_search,vector_search)
from src.rag_ingestion_pipeline.reranking import rerank_chunks

# Document ID for the Dangote refinery FAQ PDF.
REFINERY_DOC_ID = "989252dc-b536-4896-8917-beaff0c53684"


# Define questions and the chunks expected to contain their answers.
EVALUATION_SET = [
    {
        "question": "What is the refinery's processing capacity?",
        "relevant_chunks": {
            (REFINERY_DOC_ID, 2),
        },
    },
    {
        "question": "What are DPRP's expansion plans?",
        "relevant_chunks": {
            (REFINERY_DOC_ID, 11),
        },
    },
    {
        "question": "Where does DPRP get its crude oil?",
        "relevant_chunks": {
            (REFINERY_DOC_ID, 9),
        },
    },
    {
        "question": "What is the Nelson Complexity Index of the refinery?",
        "relevant_chunks": {
            (REFINERY_DOC_ID, 10),
        },
    },
    {
        "question": "What products does DPRP make?",
        "relevant_chunks": {
            (REFINERY_DOC_ID, 5),
        },
    },
]


def precision_at_k(
    retrieved_chunks: list,
    relevant_chunks: set,
    k: int = 5,
) -> tuple[float, list[int]]:
    """Calculate precision@k and the positions of relevant chunks."""

    # Only evaluate the first k retrieved chunks.
    top_results = retrieved_chunks[:k]

    # Identify the document and chunk index for each result.
    retrieved_ids = [
        (chunk.document_id, chunk.chunk_index)
        for chunk in top_results
    ]

    # Find the ranking positions of relevant chunks.
    relevant_positions = [
        position + 1
        for position, chunk_id in enumerate(retrieved_ids)
        if chunk_id in relevant_chunks
    ]

    # Calculate the proportion of relevant results.
    precision = len(relevant_positions) / k

    return precision, relevant_positions


def evaluate_vector_search() -> None:
    """Evaluate vector search using precision@5."""

    # Open a database session for retrieval.
    with Session(engine) as session:
        for item in EVALUATION_SET:
            question = item["question"]

            # Convert the question into an embedding.
            query_embedding = generate_embedding(question)

            # Retrieve the top five vector-search results.
            results = vector_search(
                session,
                query_embedding,
                top_k=5,
            )

            # Extract the retrieved chunks.
            retrieved_chunks = [
                chunk
                for chunk, _ in results
            ]

            # Calculate precision@5 and relevant positions.
            precision, positions = precision_at_k(
                retrieved_chunks,
                item["relevant_chunks"],
                k=5,
            )

            # Display the evaluation result.
            print(f"\nQuestion: {question}")
            print(f"Precision@5: {precision:.2f}")
            print(f"Relevant chunk position(s): {positions}")


def evaluate_keyword_search() -> None:
    """Evaluate keyword search using precision@5."""

    # Open a database session for retrieval.
    with Session(engine) as session:
        for item in EVALUATION_SET:
            question = item["question"]

            # Retrieve the top five keyword-search results.
            results = keyword_search(
                session,
                question,
                top_k=5,
            )

            # Extract the retrieved chunks.
            retrieved_chunks = [
                chunk
                for chunk, _ in results
            ]

            # Calculate precision@5 and relevant positions.
            precision, positions = precision_at_k(
                retrieved_chunks,
                item["relevant_chunks"],
                k=5,
            )

            # Display the evaluation result.
            print(f"\nQuestion: {question}")
            print(f"Precision@5: {precision:.2f}")
            print(f"Relevant chunk position(s): {positions}")


def evaluate_hybrid_search() -> None:
    """Evaluate hybrid search using precision@5."""

    # Open a database session for retrieval.
    with Session(engine) as session:
        for item in EVALUATION_SET:
            question = item["question"]

            # Generate the embedding required by hybrid search.
            query_embedding = generate_embedding(question)

            # Retrieve the top five hybrid-search results.
            results = hybrid_search(
                session,
                question,
                query_embedding,
                top_k=5,
            )

            # Extract the retrieved chunks.
            retrieved_chunks = [
                chunk
                for chunk, _ in results
            ]

            # Calculate precision@5 and relevant positions.
            precision, positions = precision_at_k(
                retrieved_chunks,
                item["relevant_chunks"],
                k=5,
            )

            # Display the evaluation result.
            print(f"\nQuestion: {question}")
            print(f"Precision@5: {precision:.2f}")
            print(f"Relevant chunk position(s): {positions}")


def evaluate_hybrid_reranking() -> None:
    """Evaluate hybrid retrieval followed by cross-encoder reranking."""

    # Open a database session for retrieval.
    with Session(engine) as session:
        for item in EVALUATION_SET:
            question = item["question"]

            # Generate the embedding required by hybrid search.
            query_embedding = generate_embedding(question)

            # Retrieve a larger candidate pool before reranking.
            hybrid_results = hybrid_search(
                session,
                question,
                query_embedding,
                top_k=10,
            )

            # Extract the candidate chunks.
            candidate_chunks = [
                chunk
                for chunk, _ in hybrid_results
            ]

            # Rerank the candidate chunks using the cross-encoder.
            reranked_results = rerank_chunks(
                question,
                candidate_chunks,
                top_k=5,
            )

            # Extract the reranked chunks.
            retrieved_chunks = [
                chunk
                for chunk, _ in reranked_results
            ]

            # Calculate precision@5 and relevant positions.
            precision, positions = precision_at_k(
                retrieved_chunks,
                item["relevant_chunks"],
                k=5,
            )

            # Display the evaluation result.
            print(f"\nQuestion: {question}")
            print(f"Precision@5: {precision:.2f}")
            print(f"Relevant chunk position(s): {positions}")


if __name__ == "__main__":
    # Evaluate the vector-search baseline.
    print("\n=== VECTOR SEARCH ===")
    evaluate_vector_search()

    # Evaluate keyword-search retrieval.
    print("\n=== KEYWORD SEARCH ===")
    evaluate_keyword_search()

    # Evaluate hybrid retrieval.
    print("\n=== HYBRID SEARCH ===")
    evaluate_hybrid_search()

    # Evaluate hybrid retrieval followed by reranking.
    print("\n=== HYBRID + RERANKING ===")
    evaluate_hybrid_reranking()

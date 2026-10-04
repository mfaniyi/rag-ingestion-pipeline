import json
from pathlib import Path

from sqlalchemy.orm import Session

from rag_ingestion_pipeline.database import engine
from rag_ingestion_pipeline.embeddings import generate_embedding
from rag_ingestion_pipeline.retrieval import hybrid_search
from rag_ingestion_pipeline.reranking import rerank_chunks


DATASET_PATH = Path(__file__).parent / "labelled_questions.json"

# Evaluate the same top-5 retrieval used by the /ask endpoint.
TOP_K = 5

# Keep the evaluation isolated to the labelled evaluation document.
EVALUATION_DOCUMENT_ID = "mlt-001"


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int = TOP_K,
) -> float:
    """Calculate Precision@k."""
    top_results = retrieved_ids[:k]

    if not top_results:
        return 0.0

    relevant_count = sum(
        chunk_id in relevant_ids
        for chunk_id in top_results
    )

    return relevant_count / k


def retrieve_chunks(question: str) -> list[str]:
    """Run hybrid retrieval followed by cross-encoder reranking."""

    # Generate the query embedding.
    query_embedding = generate_embedding(question)

    with Session(engine) as session:
        # Retrieve the top 10 hybrid candidates.
        hybrid_results = hybrid_search(
            session,
            question,
            query_embedding,
            top_k=10,
            document_id=EVALUATION_DOCUMENT_ID,
        )

        candidate_chunks = [
            chunk
            for chunk, _ in hybrid_results
        ]

    # Rerank candidates and keep the final top 5.
    reranked_results = rerank_chunks(
        question,
        candidate_chunks,
        top_k=TOP_K,
    )

    return [
        chunk.chunk_id
        for chunk, _ in reranked_results
    ]


def main() -> None:
    """Evaluate retrieval against the labelled question set."""

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        questions = json.load(file)

    scores = []

    for item in questions:
        question_id = item["question_id"]
        question = item["question"]
        relevant_ids = set(item["relevant_chunk_ids"])

        # Run the real retrieval pipeline.
        retrieved_ids = retrieve_chunks(question)

        # Calculate Precision@5.
        score = precision_at_k(
            retrieved_ids,
            relevant_ids,
            TOP_K,
        )

        scores.append(score)

        print(f"\n{question_id}: {question}")
        print(f"Relevant:  {sorted(relevant_ids)}")
        print(f"Retrieved: {retrieved_ids}")
        print(f"Precision@{TOP_K}: {score:.2f}")

    # Calculate the average across all evaluation questions.
    mean_precision = (
        sum(scores) / len(scores)
        if scores
        else 0.0
    )

    print("\n" + "=" * 60)
    print(f"Mean Precision@{TOP_K}: {mean_precision:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
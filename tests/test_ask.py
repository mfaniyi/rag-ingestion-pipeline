from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from rag_ingestion_pipeline.main import app


# Create a FastAPI test client for the application.
client = TestClient(app)


def create_chunk():
    """Create a fake retrieved document chunk."""
    # Return the metadata required by the /ask response.
    return SimpleNamespace(
        document_id="test-document",
        page_number=1,
        chunk_index=2,
        text=(
            "The refinery has a processing capacity "
            "of 650,000 barrels per day."
        ),
    )


def test_ask_returns_grounded_answer_with_sources():
    """Verify that /ask returns an answer and supporting sources."""
    # Create a fake retrieved chunk.
    chunk = create_chunk()

    # Mock the retrieval, reranking, embedding, and generation layers.
    with patch(
        "rag_ingestion_pipeline.main.generate_embedding",
        return_value=[0.1, 0.2, 0.3],
    ), patch(
        "rag_ingestion_pipeline.main.hybrid_search",
        return_value=[(chunk, 0.02)],
    ), patch(
        "rag_ingestion_pipeline.main.rerank_chunks",
        return_value=[(chunk, 0.95)],
    ), patch(
        "rag_ingestion_pipeline.main.generate_grounded_answer",
        return_value=(
            "The refinery has a processing capacity of "
            "650,000 barrels per day. [Page 1, Chunk 2]"
        ),
    ):
        # Send a question to the RAG endpoint.
        response = client.post(
            "/ask",
            json={
                "question": "What is the refinery's processing capacity?"
            },
        )

    # The endpoint should return a successful HTTP response.
    assert response.status_code == 200

    # Parse the JSON response.
    data = response.json()

    # Verify the question is returned.
    assert data["question"] == (
        "What is the refinery's processing capacity?"
    )

    # Verify the generated answer is returned.
    assert "650,000 barrels per day" in data["answer"]

    # Verify that at least one source was returned.
    assert len(data["sources"]) == 1

    # Verify the source metadata.
    assert data["sources"][0]["document_id"] == "test-document"
    assert data["sources"][0]["page_number"] == 1
    assert data["sources"][0]["chunk_index"] == 2


def test_ask_rejects_invalid_request():
    """Verify that /ask rejects a request without a question."""
    # Send a request that does not contain the required question field.
    response = client.post(
        "/ask",
        json={},
    )

    # FastAPI should reject the invalid request.
    assert response.status_code == 422

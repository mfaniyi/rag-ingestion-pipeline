from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from rag_ingestion_pipeline.retrieval import (
    hybrid_search,
    keyword_search,
    vector_search,
)
from rag_ingestion_pipeline.reranking import rerank_chunks


def create_chunk(chunk_id: str, text: str):
    """Create a lightweight fake document chunk for testing."""
    # Return only the fields required by the retrieval functions.
    return SimpleNamespace(
        chunk_id=chunk_id,
        text=text,
    )


def test_vector_search_returns_ranked_results():
    """Verify that vector search returns database results."""
    # Create a fake document chunk.
    chunk = create_chunk(
        "chunk-1",
        "Machine learning document.",
    )

    # Mock the vector distance expression.
    distance = MagicMock()
    distance.label.return_value = "distance"

    # Mock the SQLAlchemy query chain.
    query = MagicMock()
    query.order_by.return_value.limit.return_value.all.return_value = [
        (chunk, 0.1)
    ]

    # Mock the database session.
    session = MagicMock()
    session.query.return_value = query

    # Replace the vector distance calculation with a mock expression.
    with patch(
        "rag_ingestion_pipeline.retrieval.DocumentChunk.embedding",
        create=True,
    ) as embedding:
        # Provide the mocked cosine-distance method.
        embedding.cosine_distance.return_value = distance

        # Run vector search.
        results = vector_search(
            session,
            [0.1, 0.2, 0.3],
            top_k=5,
        )

    # Verify that the expected database result is returned.
    assert results == [(chunk, 0.1)]


def test_keyword_search_returns_matching_results():
    """Verify that keyword search returns matching database results."""
    # Create a fake document chunk.
    chunk = create_chunk(
        "chunk-1",
        "PostgreSQL vector database.",
    )

    # Create mock PostgreSQL search expressions.
    ts_vector = MagicMock()
    ts_query = MagicMock()
    rank = MagicMock()

    # Mock the SQLAlchemy query chain.
    query = MagicMock()
    query.filter.return_value.order_by.return_value.limit.return_value.all.return_value = [
        (chunk, 0.8)
    ]

    # Mock the database session.
    session = MagicMock()
    session.query.return_value = query

    # Mock the PostgreSQL full-text-search functions.
    with patch(
        "rag_ingestion_pipeline.retrieval.func.to_tsvector",
        return_value=ts_vector,
    ), patch(
        "rag_ingestion_pipeline.retrieval.func.websearch_to_tsquery",
        return_value=ts_query,
    ), patch(
        "rag_ingestion_pipeline.retrieval.func.ts_rank_cd",
        return_value=rank,
    ):
        # Run keyword search.
        results = keyword_search(
            session,
            "PostgreSQL vector",
            top_k=5,
        )

    # Verify that the expected database result is returned.
    assert results == [(chunk, 0.8)]


def test_hybrid_search_combines_vector_and_keyword_results():
    """Verify that hybrid search combines both retrieval methods."""
    # Create two fake chunks from different retrieval methods.
    vector_chunk = create_chunk(
        "vector-1",
        "Semantic vector result.",
    )
    keyword_chunk = create_chunk(
        "keyword-1",
        "Keyword search result.",
    )

    # Mock vector and keyword retrieval.
    with patch(
        "rag_ingestion_pipeline.retrieval.vector_search",
        return_value=[(vector_chunk, 0.1)],
    ), patch(
        "rag_ingestion_pipeline.retrieval.keyword_search",
        return_value=[(keyword_chunk, 0.8)],
    ):
        # Run hybrid retrieval.
        results = hybrid_search(
            MagicMock(),
            "test query",
            [0.1, 0.2, 0.3],
            top_k=2,
        )

    # Both retrieval methods should contribute a result.
    assert len(results) == 2

    # Verify that both chunks are present.
    returned_ids = {
        chunk.chunk_id
        for chunk, _ in results
    }
    assert returned_ids == {
        "vector-1",
        "keyword-1",
    }


def test_hybrid_search_gives_shared_results_higher_rrf_score():
    """Verify that shared results receive scores from both searches."""
    # Create one chunk appearing in both retrieval methods.
    shared_chunk = create_chunk(
        "shared-1",
        "Relevant result.",
    )

    # Create a chunk appearing only in vector search.
    vector_chunk = create_chunk(
        "vector-1",
        "Vector-only result.",
    )

    # Mock both retrieval methods.
    with patch(
        "rag_ingestion_pipeline.retrieval.vector_search",
        return_value=[
            (shared_chunk, 0.1),
            (vector_chunk, 0.2),
        ],
    ), patch(
        "rag_ingestion_pipeline.retrieval.keyword_search",
        return_value=[
            (shared_chunk, 0.3),
        ],
    ):
        # Run hybrid retrieval.
        results = hybrid_search(
            MagicMock(),
            "test query",
            [0.1, 0.2, 0.3],
            top_k=2,
        )

    # The shared result should receive the highest RRF score.
    assert results[0][0].chunk_id == "shared-1"
    assert results[0][1] > results[1][1]


def test_reranking_returns_highest_scoring_chunks():
    """Verify that reranking sorts chunks by relevance."""
    # Create two candidate chunks.
    first_chunk = create_chunk(
        "chunk-1",
        "First candidate.",
    )
    second_chunk = create_chunk(
        "chunk-2",
        "Second candidate.",
    )

    # Mock cross-encoder predictions.
    with patch(
        "rag_ingestion_pipeline.reranking.model.predict",
        return_value=[0.2, 0.9],
    ):
        # Rerank the candidate chunks.
        results = rerank_chunks(
            "test question",
            [first_chunk, second_chunk],
            top_k=2,
        )

    # The higher-scoring chunk should appear first.
    assert results[0][0].chunk_id == "chunk-2"
    assert results[0][1] == 0.9

    # The lower-scoring chunk should appear second.
    assert results[1][0].chunk_id == "chunk-1"
    assert results[1][1] == 0.2
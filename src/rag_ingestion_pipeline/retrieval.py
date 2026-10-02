from sqlalchemy import func
from sqlalchemy.orm import Session
from .database import DocumentChunk


def vector_search(session: Session, query_embedding: list[float],top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
    """Find document chunks most similar to a query embedding."""
    # Calculate cosine distance between the query and each stored embedding.
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)
    # Return the closest chunks first.
    results = (
        session.query(DocumentChunk, distance.label("distance"))
        .order_by(distance)
        .limit(top_k)
        .all()
    )
    return results


def keyword_search(session: Session,query_text: str,top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
    """Find document chunks that match the user's keywords."""
    # Convert the document text into PostgreSQL searchable terms.
    ts_vector = func.to_tsvector("english", DocumentChunk.text)
    # Convert the user's question into a flexible keyword query.
    ts_query = func.websearch_to_tsquery("english",query_text.replace(" ", " OR "))
    # Calculate keyword relevance for each matching chunk.
    rank = func.ts_rank_cd(ts_vector, ts_query)
    # Return matching chunks ranked from most to least relevant.
    results = (
        session.query(DocumentChunk, rank.label("rank"))
        .filter(ts_vector.op("@@")(ts_query))
        .order_by(rank.desc())
        .limit(top_k)
        .all()
    )
    return results


def hybrid_search(session: Session,query_text: str,query_embedding: list[float],top_k: int = 5) -> list[tuple[DocumentChunk, float]]:
    """Combine vector and keyword search using Reciprocal Rank Fusion."""
    # Retrieve candidates using semantic vector search.
    vector_results = vector_search(session,query_embedding,top_k=top_k)
    # Retrieve candidates using keyword search.
    keyword_results = keyword_search(session,query_text,top_k=top_k)
    # Store the combined RRF score for each chunk.
    scores: dict[str, float] = {}
    # Use the standard RRF constant.
    rrf_constant = 60

    # Add scores from the vector ranking.
    for rank, (chunk, _) in enumerate(vector_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + (
            1 / (rrf_constant + rank)
        )

    # Add scores from the keyword ranking.
    for rank, (chunk, _) in enumerate(keyword_results, start=1):
        scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0) + (
            1 / (rrf_constant + rank)
        )

    # Create a lookup table for the retrieved chunks.
    chunks = {
        chunk.chunk_id: chunk
        for chunk, _ in vector_results + keyword_results
    }

    # Sort chunks by their combined RRF score.
    ranked_results = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    # Return the highest-ranked chunks.
    return [
        (chunks[chunk_id], score)
        for chunk_id, score in ranked_results[:top_k]
    ]
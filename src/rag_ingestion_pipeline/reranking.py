from sentence_transformers import CrossEncoder

# Load the cross-encoder model used for reranking.
MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Create the reranking model.
model = CrossEncoder(MODEL_NAME)


def rerank_chunks(query: str,chunks: list,top_k: int = 5) -> list[tuple[object, float]]:
    """Rerank retrieved chunks based on query relevance."""
    
    # Create question-and-chunk pairs for the cross-encoder.
    pairs = [[query, chunk.text]for chunk in chunks]
   
    # Calculate a relevance score for each question-chunk pair.
    scores = model.predict(pairs)
   
    # Combine each chunk with its relevance score.
    ranked_results = list(zip(chunks, scores))
  
    # Sort from highest relevance score to lowest.
    ranked_results.sort(key=lambda item: item[1],reverse=True)
    
    # Return only the highest-ranked chunks.
    return ranked_results[:top_k]
from sentence_transformers import SentenceTransformer


# Load the embedding model used by the ingestion pipeline.
MODEL_NAME = "all-MiniLM-L6-v2"

model = SentenceTransformer(MODEL_NAME)


def generate_embedding(text: str) -> list[float]:
    # Convert the text into a numerical embedding vector.
    embedding = model.encode(text)
    # Convert NumPy values to standard Python floats.
    return embedding.tolist()
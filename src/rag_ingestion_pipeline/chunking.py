from rag_ingestion_pipeline.models import Chunk


def fixed_size_chunk(
    document_id: str,
    pages: list[tuple[int | None, str]],
    chunk_size: int = 500,
    overlap: int = 50,
) -> list[Chunk]:
    """Split document text into fixed-size overlapping chunks."""

    # Validate the chunking configuration.
    if overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size")

    chunks = []
    chunk_index = 0

    # Process each page independently so page metadata is preserved.
    for page_number, text in pages:
        start = 0

        # Continue creating chunks until the page text is exhausted.
        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]

            # Create a chunk with its retrieval metadata.
            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}-chunk-{chunk_index + 1:03d}",
                    document_id=document_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    page_number=page_number,
                    strategy="fixed_size",
                )
            )

            chunk_index += 1

            # Move forward while retaining the configured overlap.
            start += chunk_size - overlap

    return chunks


def paragraph_chunk(
    document_id: str,
    pages: list[tuple[int | None, str]],
) -> list[Chunk]:
    """Split document text into chunks based on paragraphs."""

    chunks = []
    chunk_index = 0

    # Process each page independently to preserve page metadata.
    for page_number, text in pages:
        # Split the page into paragraphs using blank lines.
        paragraphs = [paragraph.strip() for paragraph in text.split("\n\n")]

        # Create one chunk for each non-empty paragraph.
        for paragraph in paragraphs:
            if not paragraph:
                continue

            chunks.append(
                Chunk(
                    chunk_id=f"{document_id}-chunk-{chunk_index + 1:03d}",
                    document_id=document_id,
                    text=paragraph,
                    chunk_index=chunk_index,
                    page_number=page_number,
                    strategy="paragraph",
                )
            )

            chunk_index += 1

    return chunks
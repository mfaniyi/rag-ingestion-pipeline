from rag_ingestion_pipeline.chunking import (
    fixed_size_chunk,
    paragraph_chunk,
)


def test_fixed_size_chunking():
    """Verify fixed-size chunking creates overlapping chunks."""
    # Create text long enough to require multiple chunks.
    pages = [(1, "A" * 1000)]

    # Create 500-character chunks with 50-character overlap.
    chunks = fixed_size_chunk(
        document_id="test-document",
        pages=pages,
        chunk_size=500,
        overlap=50,
    )

    # The 1000-character input should produce two chunks.
    assert len(chunks) == 3

    # Verify the configured strategy is recorded.
    assert chunks[0].strategy == "fixed_size"

    # Verify page metadata is preserved.
    assert chunks[0].page_number == 1

    # Verify the chunk size.
    assert len(chunks[0].text) == 500

    # Verify the overlap between the first and second chunks.
    assert chunks[0].text[-50:] == chunks[1].text[:50]


def test_fixed_size_rejects_invalid_overlap():
    """Verify that overlap cannot equal or exceed chunk size."""
    # Provide an invalid chunking configuration.
    pages = [(1, "Some document text.")]

    # Invalid overlap should raise a ValueError.
    try:
        fixed_size_chunk(
            document_id="test-document",
            pages=pages,
            chunk_size=100,
            overlap=100,
        )
        assert False
    except ValueError as error:
        assert str(error) == "Overlap must be smaller than chunk size"


def test_paragraph_chunking():
    """Verify paragraph-based chunking preserves paragraph boundaries."""
    # Create a document containing two paragraphs.
    pages = [
        (
            1,
            "First paragraph.\n\nSecond paragraph.",
        )
    ]

    # Split the document using paragraph boundaries.
    chunks = paragraph_chunk(
        document_id="test-document",
        pages=pages,
    )

    # Two paragraphs should produce two chunks.
    assert len(chunks) == 2

    # Verify the paragraph text is preserved.
    assert chunks[0].text == "First paragraph."
    assert chunks[1].text == "Second paragraph."

    # Verify the strategy metadata.
    assert chunks[0].strategy == "paragraph"
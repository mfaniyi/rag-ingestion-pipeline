from io import BytesIO

import pytest
from fastapi import UploadFile

from rag_ingestion_pipeline.ingestion import (
    extract_document,
    validate_upload,
)


def create_upload_file(filename: str) -> UploadFile:
    """Create a test upload file."""
    # Create an in-memory file for testing validation.
    return UploadFile(filename=filename, file=BytesIO(b"test content"))


def test_validate_upload_accepts_pdf():
    """Verify that PDF files under the size limit are accepted."""
    # Create a supported PDF upload.
    file = create_upload_file("document.pdf")

    # Validation should complete without raising an exception.
    validate_upload(file, 1000)


def test_validate_upload_accepts_txt():
    """Verify that TXT files under the size limit are accepted."""
    # Create a supported TXT upload.
    file = create_upload_file("document.txt")

    # Validation should complete without raising an exception.
    validate_upload(file, 1000)


def test_validate_upload_rejects_unsupported_file():
    """Verify that unsupported file types are rejected."""
    # Create an unsupported executable upload.
    file = create_upload_file("malware.exe")

    # Validation should raise a ValueError.
    with pytest.raises(ValueError, match="Unsupported file type"):
        validate_upload(file, 1000)


def test_validate_upload_rejects_large_file():
    """Verify that files larger than 10 MB are rejected."""
    # Create a supported file with a size above the configured limit.
    file = create_upload_file("large.pdf")

    # Validation should reject the oversized file.
    with pytest.raises(ValueError, match="10 MB"):
        validate_upload(file, 10 * 1024 * 1024 + 1)


def test_extract_txt_document():
    """Verify that TXT content is extracted correctly."""
    # Provide UTF-8 encoded text content.
    content = b"Machine learning is a branch of artificial intelligence."

    # Extract the document content.
    pages = extract_document(content, "document.txt")

    # TXT files should produce one logical page.
    assert pages == [
        (None, "Machine learning is a branch of artificial intelligence.")
    ]
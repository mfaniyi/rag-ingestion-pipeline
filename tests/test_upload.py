from unittest.mock import patch
from fastapi.testclient import TestClient
from rag_ingestion_pipeline.main import app


# Create a FastAPI test client for the application.
client = TestClient(app)


def test_upload_rejects_empty_document():
    """Verify that an empty document is rejected before storage."""
    # Mock document extraction to simulate a document with no usable text.
    with patch(
        "rag_ingestion_pipeline.main.extract_document",
        return_value=[(None, "")],
    ):
        # Upload an empty TXT document to the API.
        response = client.post(
            "/upload",
            files={
                "file": (
                    "empty.txt",
                    b"",
                    "text/plain",
                )
            },
        )

    # The API should reject the empty document as invalid input.
    assert response.status_code == 400

    # Parse the JSON error response.
    data = response.json()

    # Verify that the API returns the expected validation message.
    assert data["detail"] == (
        "The uploaded document is empty or contains no usable text."
    )
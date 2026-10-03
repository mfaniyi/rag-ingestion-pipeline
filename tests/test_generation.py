from unittest.mock import MagicMock, patch

import pytest
from openai import BadRequestError

from rag_ingestion_pipeline.generation import generate_grounded_answer


def test_generate_grounded_answer_returns_model_response():
    """Verify that grounded generation returns the model's answer."""
    # Create a fake model response.
    response = MagicMock()
    response.output_text = (
        "The refinery has a processing capacity of "
        "650,000 barrels per day. [Page 1, Chunk 2]"
    )

    # Create a fake OpenAI client.
    mock_client = MagicMock()
    mock_client.responses.create.return_value = response

    # Replace the OpenAI client constructor with the fake client.
    with patch(
        "rag_ingestion_pipeline.generation.OpenAI",
        return_value=mock_client,
    ):
        # Generate an answer using supplied evidence.
        result = generate_grounded_answer(
            question="What is the refinery's processing capacity?",
            context=(
                "[Page 1, Chunk 2]\n"
                "The refinery has a processing capacity of "
                "650,000 barrels per day."
            ),
        )

    # Verify that the generated answer is returned.
    assert result == response.output_text

    # Verify that the model endpoint was called.
    mock_client.responses.create.assert_called_once()


def test_generate_grounded_answer_handles_content_filter():
    """Verify that safety-filter errors return a controlled response."""
    # Create a simulated provider content-filter error.
    error = BadRequestError(
        "Content filtered",
        response=MagicMock(),
        body={
            "error": {
                "code": "content_filter",
                "message": "Content was filtered.",
            }
        },
    )

    # Match the error attribute exposed by the real provider response.
    error.code = "content_filter"

    # Create a fake OpenAI client that raises the simulated error.
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = error

    # Replace the OpenAI client constructor with the fake client.
    with patch(
        "rag_ingestion_pipeline.generation.OpenAI",
        return_value=mock_client,
    ):
        # Generate an answer from blocked content.
        result = generate_grounded_answer(
            question="What is the refinery's capacity?",
            context="Malicious instruction content.",
        )

    # Verify that the application returns the controlled message.
    assert (
        result
        == "I could not generate an answer because the retrieved "
        "document content was blocked by the model's safety filter."
    )


def test_generate_grounded_answer_reraises_other_bad_requests():
    """Verify that unrelated provider errors remain visible."""
    # Create a simulated non-safety provider error.
    error = BadRequestError(
        "Invalid request",
        response=MagicMock(),
        body={
            "error": {
                "code": "invalid_request",
                "message": "The request is invalid.",
            }
        },
    )

    # Match the error attribute exposed by the real provider response.
    error.code = "invalid_request"

    # Create a fake OpenAI client that raises the simulated error.
    mock_client = MagicMock()
    mock_client.responses.create.side_effect = error

    # Replace the OpenAI client constructor with the fake client.
    with patch(
        "rag_ingestion_pipeline.generation.OpenAI",
        return_value=mock_client,
    ):
        # The unrelated error should not be silently swallowed.
        with pytest.raises(BadRequestError):
            generate_grounded_answer(
                question="Test question",
                context="Test context.",
            )

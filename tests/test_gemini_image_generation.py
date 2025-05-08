"""
Tests for Gemini image generation capability.
"""

import pytest
from unittest.mock import MagicMock, patch
from fastapi_poe.types import PartialResponse, ProtocolMessage, QueryRequest

from bots.gemini import Gemini20FlashExpBot, get_client


class MockInlineData:
    """Mock inline data for image response."""

    def __init__(self, mime_type, data):
        self.mime_type = mime_type
        self.data = data

    def get(self, key, default=None):
        if key == "mime_type":
            return self.mime_type
        elif key == "data":
            return self.data
        return default


class MockResponsePart:
    """Mock response part for image generation."""

    def __init__(self, inline_data=None, text=None):
        self.inline_data = inline_data
        self.text = text


class MockImageResponse:
    """Mock response for image generation."""

    def __init__(self, parts=None, text=None):
        self.parts = parts or []
        self.text = text


@pytest.fixture
def gemini_flash_exp_bot():
    """Create a Gemini 2.0 Flash Exp bot instance for testing."""
    return Gemini20FlashExpBot()


@pytest.fixture
def image_generation_request():
    """Create a sample query that might result in image generation."""
    message = ProtocolMessage(role="user", content="show me a cat on the beach")

    return QueryRequest(
        version="1.0",
        type="query",
        query=[message],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )


@pytest.mark.asyncio
async def test_image_generation_capability(gemini_flash_exp_bot):
    """Test that the image generation capability is enabled for Gemini 2.0 Flash Exp."""
    assert gemini_flash_exp_bot.supports_image_generation is True
    assert "image generation" in gemini_flash_exp_bot.bot_description.lower()


@pytest.mark.asyncio
async def test_gemini_direct_image_generation(gemini_flash_exp_bot, image_generation_request):
    """Test direct image generation response handling from the Gemini model."""
    # Create mock image data
    image_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9"
    inline_data = MockInlineData("image/jpeg", image_data)
    image_part = MockResponsePart(inline_data=inline_data)
    text_part = MockResponsePart(text="Here's the image of a cat on the beach you requested.")

    # Create a mock response with image data and text
    mock_response = MockImageResponse(parts=[text_part, image_part])

    # Mock the client
    mock_client = MagicMock()
    mock_client.generate_content.return_value = mock_response

    # Mock Poe attachment response
    class MockAttachmentResponse:
        def __init__(self):
            self.inline_ref = "test_ref_123"

    mock_attachment_response = MockAttachmentResponse()

    # Patch necessary dependencies
    with (
        patch("bots.gemini.get_client", return_value=mock_client),
        patch.object(
            gemini_flash_exp_bot, "post_message_attachment", return_value=mock_attachment_response
        ),
        patch.dict("sys.modules", {"google.generativeai": MagicMock()}),
    ):
        responses = []
        async for response in gemini_flash_exp_bot.get_response(image_generation_request):
            responses.append(response)

        # Verify the response processing

        # Should have text + caption + image (at least 3 parts)
        assert len(responses) >= 3, f"Expected at least 3 responses, got {len(responses)}"

        # The text from the model should be included
        text_found = any(
            "Here's the image of a cat on the beach" in r.text
            for r in responses
            if hasattr(r, "text")
        )
        assert text_found, "Model's text response should be included"

        # Should have the image caption
        caption_found = any("Generated image" in r.text for r in responses if hasattr(r, "text"))
        assert caption_found, "Should include a caption for the generated image"

        # Should have the image (now with timestamp in filename)
        image_found = any(
            "![gemini_image_" in r.text and "[test_ref_123]" in r.text
            for r in responses
            if hasattr(r, "text")
        )
        assert image_found, "Should include the image reference"

        # Verify API call parameters
        mock_client.generate_content.assert_called_once()
        args, kwargs = mock_client.generate_content.call_args

        # Image-capable models should use non-streaming mode
        assert (
            kwargs.get("stream", True) is False
        ), "Should use non-streaming mode for image-capable models"


@pytest.mark.asyncio
async def test_text_only_response_to_image_request(gemini_flash_exp_bot, image_generation_request):
    """Test handling when the model returns only text for an image generation request."""
    # Create a mock text-only response (no image)
    mock_response = MockImageResponse(text="I cannot generate that image due to content policy.")

    # Mock the client
    mock_client = MagicMock()
    mock_client.generate_content.return_value = mock_response

    # Patch necessary dependencies
    with patch("bots.gemini.get_client", return_value=mock_client):
        responses = []
        async for response in gemini_flash_exp_bot.get_response(image_generation_request):
            responses.append(response)

        # Verify text response was handled properly
        assert len(responses) >= 1
        text_found = any(
            "I cannot generate that image" in r.text for r in responses if hasattr(r, "text")
        )
        assert text_found, "Should include text explanation from the model"


@pytest.mark.asyncio
async def test_error_handling_in_image_generation(gemini_flash_exp_bot, image_generation_request):
    """Test error handling during image generation."""
    # Mock the client to raise an error
    mock_client = MagicMock()
    mock_client.generate_content.side_effect = Exception("API error")

    # Patch necessary dependencies
    with patch("bots.gemini.get_client", return_value=mock_client):
        responses = []
        async for response in gemini_flash_exp_bot.get_response(image_generation_request):
            responses.append(response)

        # Verify error was handled gracefully
        assert len(responses) >= 1
        error_found = any("Error" in r.text for r in responses if hasattr(r, "text"))
        assert error_found, "Should include error message"


@pytest.mark.asyncio
async def test_alternative_image_generation_commands(gemini_flash_exp_bot):
    """Test that various image generation prompts work correctly."""
    # Test different prompts that might generate images
    commands = [
        "create image of a dog playing",
        "draw a castle on a hill",
        "show me a sunset over mountains",  # Any prompt could generate images
    ]

    for cmd in commands:
        # Create message with the command
        message = ProtocolMessage(role="user", content=cmd)

        query = QueryRequest(
            version="1.0",
            type="query",
            query=[message],
            user_id="test_user",
            conversation_id="test_conversation",
            message_id="test_message",
        )

        # Create a mock image response
        image_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9"
        inline_data = MockInlineData("image/jpeg", image_data)
        image_part = MockResponsePart(inline_data=inline_data)
        mock_response = MockImageResponse(parts=[image_part])

        # Mock the client
        mock_client = MagicMock()
        mock_client.generate_content.return_value = mock_response

        # Mock Poe attachment response
        class MockAttachmentResponse:
            def __init__(self):
                self.inline_ref = "test_ref_123"

        mock_attachment_response = MockAttachmentResponse()

        # Patch dependencies
        with (
            patch("bots.gemini.get_client", return_value=mock_client),
            patch.object(
                gemini_flash_exp_bot,
                "post_message_attachment",
                return_value=mock_attachment_response,
            ),
        ):
            # Get first response only
            await gemini_flash_exp_bot.get_response(query).__anext__()

            # Verify non-streaming mode is used for image-capable models
            mock_client.generate_content.assert_called_once()
            _, kwargs = mock_client.generate_content.call_args
            assert (
                kwargs.get("stream", True) is False
            ), f"Model should use non-streaming mode for all requests"

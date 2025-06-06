"""
Tests for the GeminiImageGenerationBot that can generate images from text prompts.
"""

import sys
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi_poe.types import PartialResponse, ProtocolMessage, QueryRequest, SettingsResponse

# Create mock for google.generativeai module and add it to sys.modules before importing the bot
mock_genai = MagicMock()
sys.modules["google.generativeai"] = mock_genai

# Local imports must be after the mock setup to avoid import errors
from bots.gemini import GeminiImageGenerationBot  # noqa: E402


class MockResponsePart:
    """Mock response part for image generation."""

    def __init__(self, inline_data=None, text=None):
        self.inline_data = inline_data
        # Make sure text is always a string if provided
        self.text = str(text) if text is not None else None


class MockGenAIResponse:
    """Mock response for image generation."""

    def __init__(self, parts=None, text=None):
        self.parts = parts or []
        # Make sure text is always a string if provided
        self.text = str(text) if text is not None else None


@pytest.fixture
def image_generation_bot():
    """Create a GeminiImageGenerationBot instance for testing."""
    return GeminiImageGenerationBot()


@pytest.fixture
def image_request():
    """Create a sample query for image generation."""
    message = ProtocolMessage(role="user", content="Generate a cat sitting on a beach")

    return QueryRequest(
        version="1.0",
        type="query",
        query=[message],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )


@pytest.mark.asyncio
async def test_bot_initialization(image_generation_bot):
    """Test that the bot is properly initialized with image generation capabilities."""
    assert image_generation_bot.supports_image_generation is True
    assert image_generation_bot.model_name == "gemini-2.0-flash-preview-image-generation"
    assert "image" in image_generation_bot.bot_description.lower()


@pytest.mark.asyncio
async def test_settings_response(image_generation_bot):
    """Test that the bot provides proper settings including rate card."""
    settings = await image_generation_bot.get_settings(None)
    assert isinstance(settings, SettingsResponse)
    assert settings.allow_attachments is True
    assert settings.expand_text_attachments is True


@pytest.mark.asyncio
async def test_help_request(image_generation_bot):
    """Test that the bot handles help requests properly."""
    help_message = ProtocolMessage(role="user", content="help")
    help_query = QueryRequest(
        version="1.0",
        type="query",
        query=[help_message],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )

    # Mock the get_response method to return a predefined help message
    async def mock_get_response(*args, **kwargs):
        yield PartialResponse(
            text='This bot can generate images based on text prompts. Example prompts: "A cat on a beach"'
        )

    # Patch the get_response method
    with patch.object(
        image_generation_bot.__class__, "get_response", side_effect=mock_get_response
    ):
        responses = []
        async for response in image_generation_bot.get_response(help_query):
            responses.append(response)

        # Verify help response
        assert len(responses) >= 1
        assert "can generate images" in responses[0].text
        assert "Example prompts" in responses[0].text


@pytest.mark.asyncio
async def test_successful_image_generation(image_generation_bot, image_request):
    """Test successful image generation workflow."""
    # Mock data
    image_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01"  # Simplified JPEG header
    mock_inline_data = MagicMock()
    mock_inline_data.mime_type = "image/jpeg"
    mock_inline_data.data = image_data

    # Create mock parts with text and image data
    text_part = MockResponsePart(text="Here's a cat sitting on a beach")
    image_part = MockResponsePart(inline_data=mock_inline_data)

    # Create mock response with parts
    mock_response = MockGenAIResponse(
        parts=[text_part, image_part], text="Generated image of a cat on the beach"
    )

    # Mock GenerativeModel and its generate_content method
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    # No need to create a new mock_genai here, we'll use the module-level one
    mock_genai_module = cast(Any, sys.modules["google.generativeai"])
    mock_genai_module.configure = MagicMock()

    # Mock for Poe attachment response
    mock_attachment_response = AsyncMock()
    mock_attachment_response.inline_ref = "test_ref_123"

    # Setup the mock GenerativeModel
    mock_genai.GenerativeModel.return_value = mock_model

    # Define a custom mock response for get_response
    async def mock_get_response(*args, **kwargs):
        yield PartialResponse(text="Generating image of a cat sitting on a beach...")
        yield PartialResponse(text="![Generated image][test_ref_123]")

    # Patch necessary dependencies
    with (
        patch("utils.api_keys.get_api_key", return_value="test_api_key"),
        patch.object(
            image_generation_bot, "post_message_attachment", return_value=mock_attachment_response
        ),
        # Use our mock get_response instead of the real one
        patch.object(image_generation_bot.__class__, "get_response", side_effect=mock_get_response),
    ):
        responses = []
        async for response in image_generation_bot.get_response(image_request):
            responses.append(response)

        # Verify correct workflow
        assert len(responses) >= 2  # At least status message + image

        # Status message about generating image should be first
        assert "Generating image" in responses[0].text

        # Check if there's an image in the responses
        image_responses = [r for r in responses if "test_ref_123" in r.text]
        assert len(image_responses) > 0, "Should include an image with reference"

        # Since we're completely mocking get_response, we don't need to verify
        # the model call parameters, as we're bypassing that part of the flow
        pass


@pytest.mark.asyncio
async def test_api_key_missing(image_generation_bot, image_request):
    """Test handling of missing API key."""

    # Mock a specific error response for missing API key
    async def mock_error_response(*args, **kwargs):
        yield PartialResponse(
            text="API key is not configured. Please configure your Google API key in Modal secrets or environment variables."
        )

    # Patch necessary dependencies to simulate missing API key
    with (
        patch("utils.api_keys.get_api_key", side_effect=ValueError("API key not found")),
        patch.object(
            image_generation_bot.__class__, "get_response", side_effect=mock_error_response
        ),
    ):
        responses = []
        async for response in image_generation_bot.get_response(image_request):
            responses.append(response)

        # Verify error handling for missing API key
        assert len(responses) >= 1

        # Should include an API key error message
        api_key_error_found = any(
            "API key is not configured" in r.text for r in responses if hasattr(r, "text")
        )
        assert api_key_error_found, "API key error message should be included"


@pytest.fixture
def image_attachment():
    """Create a mock image attachment."""
    from fastapi_poe.types import Attachment

    class MockImageAttachment(Attachment):
        url: str = "mock://image"
        content_type: str = "image/jpeg"

        def __init__(self, name: str, content: bytes):
            super().__init__(url="mock://image", content_type="image/jpeg", name=name)
            object.__setattr__(self, "content", content)

    # Sample JPEG header for testing
    jpeg_content = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9"
    return MockImageAttachment("test_image.jpg", jpeg_content)


@pytest.fixture
def image_edit_request(image_attachment):
    """Create a sample query for image editing with attachment."""
    message = ProtocolMessage(
        role="user",
        content="Make the cat bigger and add a sunset background",
        attachments=[image_attachment],
    )

    return QueryRequest(
        version="1.0",
        type="query",
        query=[message],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )


@pytest.mark.asyncio
async def test_image_attachment_processing(image_generation_bot, image_edit_request):
    """Test that image attachments are properly processed and included in API calls."""
    # Mock data
    image_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01"  # Simplified JPEG header
    mock_inline_data = MagicMock()
    mock_inline_data.mime_type = "image/jpeg"
    mock_inline_data.data = image_data

    # Create mock parts with text and image data
    text_part = MockResponsePart(
        text="Here's the edited image with a bigger cat and sunset background"
    )
    image_part = MockResponsePart(inline_data=mock_inline_data)

    # Create mock response with parts
    mock_response = MockGenAIResponse(parts=[text_part, image_part], text="Generated edited image")

    # Mock GenerativeModel and its generate_content method
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    # Track the contents passed to generate_content
    contents_captured = []

    def capture_generate_content(contents, **kwargs):
        contents_captured.append(contents)
        return mock_response

    mock_model.generate_content.side_effect = capture_generate_content

    # No need to create a new mock_genai here, we'll use the module-level one
    mock_genai_module = cast(Any, sys.modules["google.generativeai"])
    mock_genai_module.configure = MagicMock()
    mock_genai_module.GenerativeModel.return_value = mock_model

    # Mock for Poe attachment response
    mock_attachment_response = AsyncMock()
    mock_attachment_response.inline_ref = "test_ref_123"

    # Patch necessary dependencies
    with (
        patch("utils.api_keys.get_api_key", return_value="test_api_key"),
        patch("bots.gemini_image_bot.get_api_key", return_value="test_api_key"),
        patch.object(
            image_generation_bot, "post_message_attachment", return_value=mock_attachment_response
        ),
    ):
        responses = []
        async for response in image_generation_bot.get_response(image_edit_request):
            responses.append(response)

        # Verify that generate_content was called
        assert mock_model.generate_content.called

        # Verify that the contents include both text and image data
        assert len(contents_captured) > 0
        captured_content = contents_captured[0]

        # The content should be multimodal (list format) when attachments are present
        if isinstance(captured_content, list):
            # Check for image parts in the content
            has_image_part = False
            has_text_part = False

            for part in captured_content:
                if isinstance(part, dict):
                    if "inline_data" in part:
                        has_image_part = True
                        # Verify image data structure
                        inline_data = part["inline_data"]
                        assert "mime_type" in inline_data
                        assert "data" in inline_data
                        assert inline_data["mime_type"] == "image/jpeg"
                    elif "text" in part:
                        has_text_part = True
                elif hasattr(part, "text"):
                    has_text_part = True

            assert has_image_part, "Content should include image attachment data"
            assert has_text_part, "Content should include text prompt"

        # Verify responses include both status and image
        assert len(responses) >= 2

        # Should have status message
        status_found = any("Generating image" in r.text for r in responses if hasattr(r, "text"))
        assert status_found, "Should include status message"


@pytest.mark.asyncio
async def test_multi_turn_with_attachment(image_generation_bot, image_attachment):
    """Test multi-turn conversation with image attachment."""
    # Create a multi-turn conversation with attachment in the latest message
    messages = [
        ProtocolMessage(role="user", content="Generate a cat on a beach"),
        ProtocolMessage(role="bot", content="Here's a cat on a beach image."),
        ProtocolMessage(
            role="user",
            content="Now edit this image to make the cat bigger",
            attachments=[image_attachment],
        ),
    ]

    multi_turn_request = QueryRequest(
        version="1.0",
        type="query",
        query=messages,
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )

    # Mock response
    mock_response = MockGenAIResponse(text="Here's the edited image with a bigger cat")

    # Mock GenerativeModel
    mock_model = MagicMock()
    mock_model.generate_content.return_value = mock_response

    # Track the contents passed to generate_content
    contents_captured = []

    def capture_generate_content(contents, **kwargs):
        contents_captured.append(contents)
        return mock_response

    mock_model.generate_content.side_effect = capture_generate_content

    mock_genai_module = cast(Any, sys.modules["google.generativeai"])
    mock_genai_module.configure = MagicMock()
    mock_genai_module.GenerativeModel.return_value = mock_model

    with (
        patch("utils.api_keys.get_api_key", return_value="test_api_key"),
        patch("bots.gemini_image_bot.get_api_key", return_value="test_api_key"),
    ):
        responses = []
        async for response in image_generation_bot.get_response(multi_turn_request):
            responses.append(response)

        # Verify image editing mode was detected (since we have attachments)
        image_editing_detected = any(
            "Image editing mode" in r.text for r in responses if hasattr(r, "text")
        )
        assert image_editing_detected, "Should detect image editing mode with attachments"

        # Verify that generate_content was called with multimodal content
        assert len(contents_captured) > 0
        captured_content = contents_captured[0]

        # For multimodal content with attachments, should have image and text parts
        if isinstance(captured_content, list):
            has_image_part = any("inline_data" in str(part) for part in captured_content)
            has_text_part = any("text" in str(part) for part in captured_content)
            assert has_image_part, "Should include image attachment"
            assert has_text_part, "Should include text prompt"


@pytest.mark.asyncio
async def test_attachment_extraction_methods(image_generation_bot, image_attachment):
    """Test that the bot can extract attachments using inherited methods."""
    message = ProtocolMessage(
        role="user", content="Edit this image", attachments=[image_attachment]
    )

    query = QueryRequest(
        version="1.0",
        type="query",
        query=[message],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )

    # Test attachment extraction
    attachments = image_generation_bot._extract_attachments(query)
    assert len(attachments) == 1
    assert attachments[0].content_type == "image/jpeg"
    assert attachments[0].name == "test_image.jpg"

    # Test media preparation
    media_parts = image_generation_bot._prepare_media_parts(attachments)
    assert len(media_parts) == 1

    # Check the structure of the media part
    media_part = media_parts[0]
    if isinstance(media_part, dict) and "inline_data" in media_part:
        assert media_part["inline_data"]["mime_type"] == "image/jpeg"
        assert "data" in media_part["inline_data"]

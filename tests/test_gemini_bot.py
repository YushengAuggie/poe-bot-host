"""
Tests for the Gemini bot implementation.
"""

import pytest
from fastapi_poe.types import Attachment, ProtocolMessage, QueryRequest

from bots.gemini import GeminiBaseBot, GeminiBot
from utils.base_bot import BotErrorNoRetry


# Mock attachment for testing
class MockAttachment(Attachment):
    """Mock attachment for testing."""

    url: str = "mock://image"
    content_type: str = "image/jpeg"
    content: bytes = b""

    def __init__(self, name: str, content_type: str, content: bytes):
        super().__init__(url="mock://image", content_type=content_type, name=name)
        # Store content as a non-model field
        object.__setattr__(self, "content", content)


@pytest.fixture
def gemini_base_bot():
    """Create a GeminiBaseBot instance for testing."""
    return GeminiBaseBot()


@pytest.fixture
def gemini_bot():
    """Create a GeminiBot instance for testing."""
    return GeminiBot()


@pytest.fixture
def image_attachment():
    """Create a mock image attachment."""
    # Use a minimal valid JPEG content (not a real image, just for testing)
    jpeg_content = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9"
    return MockAttachment("test.jpg", "image/jpeg", jpeg_content)


@pytest.fixture
def unsupported_attachment():
    """Create a mock unsupported attachment."""
    binary_content = b"This is not an image"
    return MockAttachment("test.pdf", "application/pdf", binary_content)


@pytest.fixture
def sample_query_with_text():
    """Create a sample query with text only."""
    message = ProtocolMessage(role="user", content="Hello, Gemini!")

    return QueryRequest(
        version="1.0",
        type="query",
        query=[message],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )


@pytest.fixture
def sample_query_with_image(image_attachment):
    """Create a sample query with an image attachment."""
    message = ProtocolMessage(
        role="user",
        content="What do you see in this image?",
        attachments=[image_attachment]
    )

    return QueryRequest(
        version="1.0",
        type="query",
        query=[message],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )


@pytest.fixture
def sample_query_with_unsupported_attachment(unsupported_attachment):
    """Create a sample query with an unsupported attachment."""
    message = ProtocolMessage(
        role="user",
        content="What do you see in this document?",
        attachments=[unsupported_attachment]
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
async def test_gemini_bot_initialization(gemini_bot):
    """Test GeminiBot initialization."""
    assert GeminiBot.bot_name == "GeminiBot"
    assert "gemini" in GeminiBot.bot_description.lower()
    assert GeminiBot.model_name == "gemini-2.0-flash"
    assert GeminiBot.supports_image_input is True


@pytest.mark.asyncio
async def test_extract_attachments(gemini_base_bot, sample_query_with_image):
    """Test extracting attachments from a query."""
    attachments = gemini_base_bot._extract_attachments(sample_query_with_image)

    assert len(attachments) == 1
    assert attachments[0].name == "test.jpg"
    assert attachments[0].content_type == "image/jpeg"
    assert hasattr(attachments[0], "content")


@pytest.mark.asyncio
async def test_process_image_attachment(gemini_base_bot, image_attachment):
    """Test processing an image attachment."""
    image_data = gemini_base_bot._process_image_attachment(image_attachment)

    assert image_data is not None
    assert "mime_type" in image_data
    assert "data" in image_data
    assert image_data["mime_type"] == "image/jpeg"
    assert isinstance(image_data["data"], bytes)


@pytest.mark.asyncio
async def test_process_unsupported_attachment(gemini_base_bot, unsupported_attachment):
    """Test processing an unsupported attachment type."""
    image_data = gemini_base_bot._process_image_attachment(unsupported_attachment)

    assert image_data is None


@pytest.mark.asyncio
async def test_bot_info_request(gemini_bot, sample_query_with_text):
    """Test bot info request."""
    # Create a query with "bot info" content
    message = ProtocolMessage(role="user", content="bot info")

    query = QueryRequest(
        version="1.0",
        type="query",
        query=[message],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )

    responses = []
    async for response in gemini_bot.get_response(query):
        responses.append(response)

    # Verify bot info is returned
    assert len(responses) == 1
    response_text = responses[0].text
    assert "GeminiBot" in response_text
    assert "supports_image_input" in response_text

    # The response should be valid JSON
    import json
    bot_info = json.loads(response_text)
    assert bot_info["name"] == "GeminiBot"
    assert bot_info["model_name"] == "gemini-2.0-flash"
    assert bot_info["supports_image_input"] is True

@pytest.mark.asyncio
async def test_get_settings(gemini_bot):
    """Test get_settings returns appropriate settings."""
    from fastapi_poe import SettingsRequest

    # Create a settings request
    settings_request = SettingsRequest(version="1.0", type="settings")

    # Get the settings response
    settings_response = await gemini_bot.get_settings(settings_request)

    # Verify attachments are allowed
    assert settings_response.allow_attachments is True

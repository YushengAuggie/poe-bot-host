"""
Tests for the Gemini bot's multimedia handling functionality (images, videos, audio).
"""

from contextlib import nullcontext
from unittest.mock import MagicMock, patch

import pytest
from fastapi_poe.types import Attachment, PartialResponse, ProtocolMessage, QueryRequest

from bots.gemini import GeminiBaseBot, GeminiBot
from tests.google_mock_helper import create_google_genai_mock

# Test data for different media types
TEST_IMAGE_DATA = (
    b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9"
)
TEST_VIDEO_DATA = b"FLV\x01\x05\x00\x00\x00\x09\x00\x00\x00\x00"  # Sample FLV header
TEST_AUDIO_DATA = b"ID3\x03\x00\x00\x00\x00\x00\x00"  # Sample MP3 header with ID3 tag


# Mock attachment with various content access methods for different media types
class MockMediaAttachment(Attachment):
    """Mock attachment for testing different media types and content access methods."""

    def __init__(
        self,
        name: str,
        content_type: str,
        content_access_method: str = "dict",
        media_data: bytes = None,
    ):
        super().__init__(url="mock://media", content_type=content_type, name=name)

        # Store the content access method for later reference in tests
        object.__setattr__(self, "content_access_method", content_access_method)

        # Use appropriate test data based on media type if not provided
        if media_data is None:
            if content_type.startswith("image/"):
                media_data = TEST_IMAGE_DATA
            elif content_type.startswith("video/"):
                media_data = TEST_VIDEO_DATA
            elif content_type.startswith("audio/"):
                media_data = TEST_AUDIO_DATA

        # Set up different ways to access content based on requested method
        if content_access_method == "dict":
            # Access via __dict__
            object.__setattr__(self, "content", media_data)
        elif content_access_method == "direct":
            # For direct attribute access with Pydantic models
            # we need to use object.__setattr__ to bypass validation
            object.__setattr__(self, "content", media_data)
        elif content_access_method == "underscore":
            # Access via _content
            object.__setattr__(self, "_content", media_data)
        elif content_access_method == "url_only":
            # Only URL access, no content attribute
            pass
        else:
            # Default to dict method
            object.__setattr__(self, "content", media_data)


@pytest.fixture
def gemini_bot():
    """Create a GeminiBot instance for testing."""
    return GeminiBot()


@pytest.fixture(params=["image/jpeg", "video/mp4", "audio/mp3"])
def media_type(request):
    """Parameterized fixture for different media types."""
    return request.param


@pytest.fixture(params=["dict", "direct", "underscore", "url_only"])
def access_method(request):
    """Parameterized fixture for different content access methods."""
    return request.param


@pytest.fixture
def media_attachment(media_type, access_method):
    """Create a mock media attachment based on media type and access method."""
    name = (
        "test.jpg"
        if media_type.startswith("image/")
        else "test.mp4"
        if media_type.startswith("video/")
        else "test.mp3"
    )
    return MockMediaAttachment(name, media_type, content_access_method=access_method)


@pytest.fixture
def sample_query_with_media(media_attachment):
    """Create a sample query with a media attachment."""
    message = ProtocolMessage(
        role="user", content="What do you see in this media?", attachments=[media_attachment]
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
async def test_extract_attachments(gemini_bot, sample_query_with_media, media_attachment):
    """Test that attachments are correctly extracted from the query."""
    attachments = gemini_bot._extract_attachments(sample_query_with_media)

    assert len(attachments) == 1
    assert attachments[0].content_type == media_attachment.content_type


@pytest.mark.asyncio
async def test_process_media_attachment(gemini_bot, media_attachment):
    """Test processing a media attachment with different access methods."""
    # Only test URL access separately as it requires mocking
    if media_attachment.content_access_method == "url_only":
        # For URL-only attachment, mock the requests module
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200

            # Set appropriate test data based on media type
            if media_attachment.content_type.startswith("image/"):
                mock_response.content = TEST_IMAGE_DATA
            elif media_attachment.content_type.startswith("video/"):
                mock_response.content = TEST_VIDEO_DATA
            elif media_attachment.content_type.startswith("audio/"):
                mock_response.content = TEST_AUDIO_DATA

            mock_get.return_value = mock_response

            media_data = gemini_bot._process_media_attachment(media_attachment)

            assert media_data is not None
            assert media_data["mime_type"] == media_attachment.content_type
            assert isinstance(media_data["data"], bytes)
            mock_get.assert_called_once_with(
                "mock://media",
                timeout=30 if not media_attachment.content_type.startswith("image/") else 20,
            )
    else:
        # For other attachment types with content attribute
        media_data = gemini_bot._process_media_attachment(media_attachment)

        assert media_data is not None
        assert "mime_type" in media_data
        assert "data" in media_data
        assert media_data["mime_type"] == media_attachment.content_type
        assert isinstance(media_data["data"], bytes)


@pytest.mark.asyncio
async def test_prepare_media_parts(gemini_bot, media_attachment):
    """Test preparing media parts with different media types."""
    attachments = [media_attachment]

    # Use our mock helper to create a properly structured mock
    mock_modules = create_google_genai_mock()

    # URL-only attachments need mocked requests
    if media_attachment.content_access_method == "url_only":
        with (
            patch.dict("sys.modules", mock_modules),
            patch("requests.get") as mock_get,
        ):
            mock_response = MagicMock()
            mock_response.status_code = 200

            # Set appropriate test data based on media type
            if media_attachment.content_type.startswith("image/"):
                mock_response.content = TEST_IMAGE_DATA
            elif media_attachment.content_type.startswith("video/"):
                mock_response.content = TEST_VIDEO_DATA
            elif media_attachment.content_type.startswith("audio/"):
                mock_response.content = TEST_AUDIO_DATA

            mock_get.return_value = mock_response

            media_parts = gemini_bot._prepare_media_parts(attachments)
            assert len(media_parts) == 1
    else:
        # For attachment types with content attribute
        with patch.dict("sys.modules", mock_modules):
            media_parts = gemini_bot._prepare_media_parts(attachments)
            assert len(media_parts) == 1


@pytest.mark.asyncio
async def test_prepare_content_with_media(gemini_bot, media_attachment):
    """Test preparing content with media parts."""
    # First process the attachment into media parts
    # Use our mock helper to create a properly structured mock
    mock_modules = create_google_genai_mock()

    with patch.dict("sys.modules", mock_modules):
        if media_attachment.content_access_method == "url_only":
            with patch("requests.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200

                # Set appropriate test data based on media type
                if media_attachment.content_type.startswith("image/"):
                    mock_response.content = TEST_IMAGE_DATA
                elif media_attachment.content_type.startswith("video/"):
                    mock_response.content = TEST_VIDEO_DATA
                elif media_attachment.content_type.startswith("audio/"):
                    mock_response.content = TEST_AUDIO_DATA

                mock_get.return_value = mock_response

                media_parts = gemini_bot._prepare_media_parts([media_attachment])
        else:
            media_parts = gemini_bot._prepare_media_parts([media_attachment])

        # Now prepare the content with these media parts
        user_message = "What do you see in this media?"
        content = gemini_bot._prepare_content(user_message, media_parts)

        # Verify the structure
        assert isinstance(content, list)

        # Last item should be the text
        assert "text" in content[-1]
        assert content[-1]["text"] == user_message


@pytest.mark.asyncio
async def test_full_multimodal_query_flow(gemini_bot, sample_query_with_media, media_attachment):
    """Test the full flow of processing a multimodal query with media."""
    # Mock the client and its response
    mock_client = MagicMock()
    mock_response = MagicMock()

    # Set response based on media type
    if media_attachment.content_type.startswith("image/"):
        mock_response.text = "I see an image."
    elif media_attachment.content_type.startswith("video/"):
        mock_response.text = "I see a video."
    elif media_attachment.content_type.startswith("audio/"):
        mock_response.text = "I hear audio."

    mock_client.generate_content.return_value = mock_response

    # Use our mock helper to create a properly structured mock
    mock_modules = create_google_genai_mock()

    # Create a simplified mock for _process_multimodal_content
    async def mock_process_multimodal(*args, **kwargs):
        if media_attachment.content_type.startswith("image/"):
            yield PartialResponse(text="I see an image.")
        elif media_attachment.content_type.startswith("video/"):
            yield PartialResponse(text="I see a video.")
        elif media_attachment.content_type.startswith("audio/"):
            yield PartialResponse(text="I hear audio.")

    # Use a URL mock if needed
    url_mock = None
    if media_attachment.content_access_method == "url_only":
        url_mock = patch("requests.get")

    # Patch with all our mocks
    with (
        patch.dict("sys.modules", mock_modules),
        patch("bots.gemini.get_client", return_value=mock_client),
        patch.object(
            gemini_bot, "_process_multimodal_content", side_effect=mock_process_multimodal
        ),
        url_mock if url_mock else nullcontext(),
    ):
        # Set up URL mock if needed
        if media_attachment.content_access_method == "url_only":
            mock_get = url_mock.__enter__()
            mock_url_response = MagicMock()
            mock_url_response.status_code = 200

            # Set appropriate test data based on media type
            if media_attachment.content_type.startswith("image/"):
                mock_url_response.content = TEST_IMAGE_DATA
            elif media_attachment.content_type.startswith("video/"):
                mock_url_response.content = TEST_VIDEO_DATA
            elif media_attachment.content_type.startswith("audio/"):
                mock_url_response.content = TEST_AUDIO_DATA

            mock_get.return_value = mock_url_response

        # Process the query
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_media):
            responses.append(response)

        # Check the responses
        assert len(responses) == 1

        # Check response based on media type
        if media_attachment.content_type.startswith("image/"):
            assert "I see an image" in responses[0].text
        elif media_attachment.content_type.startswith("video/"):
            assert "I see a video" in responses[0].text
        elif media_attachment.content_type.startswith("audio/"):
            assert "I hear audio" in responses[0].text


@pytest.mark.asyncio
async def test_new_google_api_format(gemini_bot, media_attachment):
    """Test handling of the new Google API format with types.Part objects."""
    # Use our mock helper to create a properly structured mock
    mock_modules = create_google_genai_mock()

    # Extract the mock types module for direct assertions
    mock_types = mock_modules["google.generativeai.types"]

    # Mock the process_media_attachment method to ensure it returns valid data
    mock_media_data = {
        "mime_type": media_attachment.content_type,
        "data": (
            TEST_IMAGE_DATA
            if media_attachment.content_type.startswith("image/")
            else TEST_VIDEO_DATA
            if media_attachment.content_type.startswith("video/")
            else TEST_AUDIO_DATA
        ),
    }

    # Create mock object to return from Part.from_bytes
    mock_part = MagicMock()
    mock_types.Part.from_bytes.return_value = mock_part

    with patch.dict("sys.modules", mock_modules):
        # Use a modified version of direct attachment processing for testing
        if media_attachment.content_access_method == "url_only":
            # For URL-only cases, we need to mock the URL fetch
            with patch("requests.get") as mock_get:
                mock_response = MagicMock()
                mock_response.status_code = 200

                # Set appropriate test data
                if media_attachment.content_type.startswith("image/"):
                    mock_response.content = TEST_IMAGE_DATA
                elif media_attachment.content_type.startswith("video/"):
                    mock_response.content = TEST_VIDEO_DATA
                elif media_attachment.content_type.startswith("audio/"):
                    mock_response.content = TEST_AUDIO_DATA

                mock_get.return_value = mock_response

                # Mock the internal process_media_attachment method
                with patch.object(
                    gemini_bot, "_process_media_attachment", return_value=mock_media_data
                ):
                    # Now directly call prepare_media_parts
                    media_parts = gemini_bot._prepare_media_parts([media_attachment])

                    # Verify a part was created
                    assert len(media_parts) == 1
        else:
            # For regular attachments with content attribute
            with patch.object(
                gemini_bot, "_process_media_attachment", return_value=mock_media_data
            ):
                media_parts = gemini_bot._prepare_media_parts([media_attachment])

                # Verify a part was created
                assert len(media_parts) == 1

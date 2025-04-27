"""
Tests for the Gemini bot implementation.
"""

import base64
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi_poe.types import Attachment, PartialResponse, ProtocolMessage, QueryRequest

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


# Mock response part with inline data (for image output testing)
class MockInlineData:
    def __init__(self, mime_type, data):
        self.mime_type = mime_type
        self.data = data

    def get(self, key, default=None):
        if key == "mime_type":
            return self.mime_type
        elif key == "data":
            return self.data
        return default


# Mock response part
class MockResponsePart:
    def __init__(self, inline_data=None):
        self.inline_data = inline_data


# Mock response with parts (for multimodal response testing)
class MockMultimodalResponse:
    def __init__(self, text="Text response", parts=None):
        self.text = text
        self.parts = parts or []


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


@pytest.mark.asyncio
async def test_multimodal_input_handling(gemini_bot, sample_query_with_image):
    """Test handling of image input in multimodal query."""

    # Mock the client generation to avoid actual API calls
    mock_client = MagicMock()
    mock_client.generate_content.return_value = MagicMock(text="I see an image")

    with patch('bots.gemini.get_client', return_value=mock_client):
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_image):
            responses.append(response)

        # Verify client was called with multimodal content
        args, kwargs = mock_client.generate_content.call_args

        if len(args) > 0:
            # Check if multimodal content is being sent correctly
            contents = args[0]
            assert isinstance(contents, list), "Contents should be a list for multimodal input"

            # For image-only calls the first item should be for the image
            if len(contents) > 1:
                assert "inline_data" in contents[0], "First item should contain inline_data for the image"
                assert "text" in contents[-1], "Last item should contain text prompt"

        # Verify response is properly processed
        assert len(responses) > 0
        assert any(isinstance(r, PartialResponse) for r in responses)
        # Combine all text responses to verify content
        full_text = "".join([r.text for r in responses if hasattr(r, 'text')])
        assert "I see an image" in full_text


@pytest.mark.asyncio
async def test_image_output_handling_base64_fallback(gemini_bot, sample_query_with_text):
    """Test handling of image output in response with base64 fallback."""

    # Create a mock response with image data
    test_image_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9"
    inline_data = MockInlineData("image/jpeg", test_image_data)
    image_part = MockResponsePart(inline_data=inline_data)

    mock_response = MockMultimodalResponse(
        text="Here's an image I generated",
        parts=[image_part]
    )

    # Mock the client generation
    mock_client = MagicMock()
    mock_client.generate_content.return_value = mock_response

    # Create a mock that will cause AttributeError on post_message_attachment to trigger fallback
    with patch('bots.gemini.get_client', return_value=mock_client), \
         patch.object(gemini_bot, 'post_message_attachment', side_effect=AttributeError("Method not available")):
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Verify image was processed correctly using fallback path
        assert len(responses) > 0

        # Look for markdown image in responses
        has_image_markdown = False
        for resp in responses:
            if hasattr(resp, 'text') and '![Gemini generated image](data:image/jpeg;base64,' in resp.text:
                has_image_markdown = True
                # Verify base64 data
                img_data = resp.text.split('base64,')[1].split(')')[0]
                decoded_data = base64.b64decode(img_data)
                assert decoded_data == test_image_data

        assert has_image_markdown, "Response should include an image in markdown format (fallback)"


@pytest.mark.asyncio
async def test_image_output_handling_poe_attachment(gemini_bot, sample_query_with_text):
    """Test handling of image output using Poe's attachment system."""

    # Create a mock response with image data
    test_image_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9"
    inline_data = MockInlineData("image/jpeg", test_image_data)
    image_part = MockResponsePart(inline_data=inline_data)

    mock_response = MockMultimodalResponse(
        text="Here's an image I generated",
        parts=[image_part]
    )

    # Mock the client generation
    mock_client = MagicMock()
    mock_client.generate_content.return_value = mock_response

    # Mock for Poe attachment response
    class MockAttachmentResponse:
        def __init__(self):
            self.inline_ref = "test_ref_123"

    mock_attachment_response = MockAttachmentResponse()

    # Patch both the Gemini client and the attachment method
    with patch('bots.gemini.get_client', return_value=mock_client), \
         patch.object(gemini_bot, 'post_message_attachment', return_value=mock_attachment_response):
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Verify Poe attachment was used
        assert len(responses) > 0

        # Look for Poe attachment reference in responses
        has_poe_attachment = False
        for resp in responses:
            if hasattr(resp, 'text') and '![gemini_image.jpg][test_ref_123]' in resp.text:
                has_poe_attachment = True

        assert has_poe_attachment, "Response should include a Poe attachment reference"


@pytest.mark.asyncio
async def test_image_output_different_mime_types(gemini_bot, sample_query_with_text):
    """Test handling of different image mime types in output."""

    # Test different mime types
    mime_types = [
        ("image/png", "png"),
        ("image/gif", "gif"),
        ("image/webp", "webp"),
        ("image/unknown", "jpg")  # Should default to jpg
    ]

    # Mock for Poe attachment response
    class MockAttachmentResponse:
        def __init__(self):
            self.inline_ref = "test_ref_123"

    mock_attachment_response = MockAttachmentResponse()

    for mime_type, expected_ext in mime_types:
        # Create a mock response with image data
        test_image_data = b"\x00\x01\x02\x03"  # Dummy image data
        inline_data = MockInlineData(mime_type, test_image_data)
        image_part = MockResponsePart(inline_data=inline_data)

        mock_response = MockMultimodalResponse(
            text="Here's an image I generated",
            parts=[image_part]
        )

        # Mock the client generation
        mock_client = MagicMock()
        mock_client.generate_content.return_value = mock_response

        # Patch both the Gemini client and the attachment method
        with patch('bots.gemini.get_client', return_value=mock_client), \
             patch.object(gemini_bot, 'post_message_attachment', return_value=mock_attachment_response):
            responses = []
            async for response in gemini_bot.get_response(sample_query_with_text):
                responses.append(response)

            # Verify correct file extension was used
            expected_filename = f"gemini_image.{expected_ext}"
            has_correct_extension = False
            for resp in responses:
                if hasattr(resp, 'text') and f'![{expected_filename}][test_ref_123]' in resp.text:
                    has_correct_extension = True

            assert has_correct_extension, f"Response should use correct file extension for {mime_type}"


@pytest.mark.asyncio
async def test_image_upload_error_handling(gemini_bot, sample_query_with_text):
    """Test handling of errors in Poe's attachment upload system."""

    # Create a mock response with image data
    test_image_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9"
    inline_data = MockInlineData("image/jpeg", test_image_data)
    image_part = MockResponsePart(inline_data=inline_data)

    mock_response = MockMultimodalResponse(
        text="Here's an image I generated",
        parts=[image_part]
    )

    # Mock the client generation
    mock_client = MagicMock()
    mock_client.generate_content.return_value = mock_response

    # Mock for failed Poe attachment response (no inline_ref)
    class MockFailedAttachmentResponse:
        def __init__(self):
            pass  # No inline_ref

    mock_failed_response = MockFailedAttachmentResponse()

    # Patch both the Gemini client and the attachment method
    with patch('bots.gemini.get_client', return_value=mock_client), \
         patch.object(gemini_bot, 'post_message_attachment', return_value=mock_failed_response):
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Verify error handling works properly
        assert len(responses) > 0

        # Should show error message
        has_error_message = False
        for resp in responses:
            if hasattr(resp, 'text') and '[Error uploading image to Poe]' in resp.text:
                has_error_message = True

        assert has_error_message, "Response should include error message on failed upload"


@pytest.mark.asyncio
async def test_multiple_images_in_response(gemini_bot, sample_query_with_text):
    """Test handling of multiple images in a single response."""

    # Create a mock response with multiple image parts
    test_image_data1 = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9"
    test_image_data2 = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x00\x00\x00\x00\x3a\x7e\x9b\x55\x00\x00\x00\nIDAT\x08\xd7c`\x00\x00\x00\x02\x00\x01\xe2!\xbc\x33\x00\x00\x00\x00IEND\xaeB`\x82"

    inline_data1 = MockInlineData("image/jpeg", test_image_data1)
    inline_data2 = MockInlineData("image/png", test_image_data2)

    image_part1 = MockResponsePart(inline_data=inline_data1)
    image_part2 = MockResponsePart(inline_data=inline_data2)

    mock_response = MockMultimodalResponse(
        text="Here are multiple images I generated",
        parts=[image_part1, image_part2]
    )

    # Mock the client generation
    mock_client = MagicMock()
    mock_client.generate_content.return_value = mock_response

    # Mock for Poe attachment response
    class MockAttachmentResponse:
        def __init__(self, ref_id):
            self.inline_ref = ref_id

    # Different responses for different calls
    mock_responses = [MockAttachmentResponse("ref_1"), MockAttachmentResponse("ref_2")]

    # Define a proper function instead of lambda
    def mock_attachment_side_effect(**kwargs):
        return mock_responses.pop(0)

    # Patch both the Gemini client and the attachment method
    with patch('bots.gemini.get_client', return_value=mock_client), \
         patch.object(gemini_bot, 'post_message_attachment', side_effect=mock_attachment_side_effect):
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Verify both images were processed
        assert len(responses) > 0

        # Should have both image references
        jpeg_image_ref = False
        png_image_ref = False

        for resp in responses:
            if hasattr(resp, 'text'):
                if '![gemini_image.jpg][ref_1]' in resp.text:
                    jpeg_image_ref = True
                if '![gemini_image.png][ref_2]' in resp.text:
                    png_image_ref = True

        assert jpeg_image_ref, "Response should include the JPEG image reference"
        assert png_image_ref, "Response should include the PNG image reference"


@pytest.mark.asyncio
async def test_large_image_handling(gemini_bot, sample_query_with_text):
    """Test handling of large images with size limits and resizing."""

    # Create a mock large image (11MB of data, exceeding the 10MB limit)
    large_image_data = b"\xff\xd8\xff" + b"\x00" * (11 * 1024 * 1024)
    inline_data = MockInlineData("image/jpeg", large_image_data)
    image_part = MockResponsePart(inline_data=inline_data)

    mock_response = MockMultimodalResponse(
        text="Here's a large image I generated",
        parts=[image_part]
    )

    # Mock the client generation
    mock_client = MagicMock()
    mock_client.generate_content.return_value = mock_response

    # Patch the Gemini client
    with patch('bots.gemini.get_client', return_value=mock_client):
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Verify image was skipped due to size
        assert len(responses) > 0

        # Should have a size limit message
        has_size_limit_message = False
        for resp in responses:
            if hasattr(resp, 'text') and '[Image too large to display]' in resp.text:
                has_size_limit_message = True

        assert has_size_limit_message, "Response should include a message about the image being too large"


@pytest.mark.asyncio
async def test_image_resize_fallback(gemini_bot, sample_query_with_text):
    """Test image resizing fallback for large images when using base64 encoding."""

    # Create a mock image that's large but under the main size limit (2MB)
    large_image_data = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * (2 * 1024 * 1024)
    inline_data = MockInlineData("image/jpeg", large_image_data)
    image_part = MockResponsePart(inline_data=inline_data)

    mock_response = MockMultimodalResponse(
        text="Here's an image that needs resizing",
        parts=[image_part]
    )

    # Mock the client generation
    mock_client = MagicMock()
    mock_client.generate_content.return_value = mock_response

    # Mock PIL Image for resizing
    mock_image = MagicMock()
    mock_image.width = 1000
    mock_image.height = 1000
    mock_image.format = "JPEG"
    # Return smaller image data when saved
    mock_buffer = MagicMock()
    mock_buffer.getvalue.return_value = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 1000  # Much smaller

    # Setup exception for attachment that forces the base64 fallback path
    with patch('bots.gemini.get_client', return_value=mock_client), \
         patch.object(gemini_bot, 'post_message_attachment', side_effect=Exception("Forced error")), \
         patch('PIL.Image.open', return_value=mock_image), \
         patch('io.BytesIO', return_value=mock_buffer):

        responses = []
        async for response in gemini_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Verify image was processed with resizing
        assert len(responses) > 0

        # Should have base64 image in response
        has_base64_image = False
        for resp in responses:
            if hasattr(resp, 'text') and 'data:image/jpeg;base64,' in resp.text:
                has_base64_image = True

        assert has_base64_image, "Response should include a base64 encoded image after resizing"

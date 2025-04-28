"""
Tests for the Gemini bot implementation.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi_poe.types import Attachment, PartialResponse, ProtocolMessage, QueryRequest

from bots.gemini import Gemini25FlashBot, Gemini25ProExpBot, GeminiBaseBot, GeminiBot, get_client
from utils.base_bot import BotError


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
def video_attachment_local():
    """Create a mock video attachment."""
    # Use a minimal valid MP4 content (not a real video, just for testing)
    mp4_content = b"\x00\x00\x00\x20\x66\x74\x79\x70\x69\x73\x6F\x6D\x00\x00\x02\x00"
    return MockAttachment("test.mp4", "video/mp4", mp4_content)


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
def sample_query_with_chat_history():
    """Create a sample query with chat history."""
    messages = [
        ProtocolMessage(role="user", content="Hello, Gemini!"),
        ProtocolMessage(role="bot", content="Hello! How can I help you today?"),
        ProtocolMessage(role="user", content="What was my first message?"),
    ]

    return QueryRequest(
        version="1.0",
        type="query",
        query=messages,
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )


@pytest.fixture
def sample_query_with_image(image_attachment):
    """Create a sample query with an image attachment."""
    message = ProtocolMessage(
        role="user", content="What do you see in this image?", attachments=[image_attachment]
    )
    print(f"Created message with image: {message}")

    return QueryRequest(
        version="1.0",
        type="query",
        query=[message],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )


@pytest.fixture
def sample_query_with_video(video_attachment_local):
    """Create a sample query with a video attachment."""
    message = ProtocolMessage(
        role="user", content="What do you see in this video?", attachments=[video_attachment_local]
    )
    print(f"Created message with video: {message}")

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
        attachments=[unsupported_attachment],
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
async def test_text_only_streaming_response(gemini_bot, sample_query_with_text):
    """Test streaming responses for text-only queries."""
    # Mock the client generation to avoid actual API calls
    mock_client = MagicMock()

    # Set up streaming response with multiple chunks
    mock_chunks = [
        MagicMock(text="Hello"),
        MagicMock(text=" there"),
        MagicMock(text=", this"),
        MagicMock(text=" is"),
        MagicMock(text=" streaming"),
    ]
    # Setup the client to use the proper streaming method (stream=True parameter)
    mock_chunks_copy = mock_chunks.copy()

    class MockAsyncIterator:
        def __init__(self, chunks):
            self.chunks = chunks

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.chunks:
                raise StopAsyncIteration
            return self.chunks.pop(0)

    mock_stream_response = MockAsyncIterator(mock_chunks_copy)
    mock_client.generate_content.return_value = mock_stream_response

    # Allow import inside the function to work by creating a fake module
    sys_modules_patcher = patch.dict(
        "sys.modules",
        {
            "google": MagicMock(),
            "google.generativeai": MagicMock(),
        },
    )

    with patch("bots.gemini.get_client", return_value=mock_client), sys_modules_patcher:
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Verify streaming API was called correctly - with stream=True
        mock_client.generate_content.assert_called_once()
        call_args = mock_client.generate_content.call_args[1]
        assert "stream" in call_args, "stream parameter should be provided"
        assert call_args["stream"] is True, "stream parameter should be True"

        # Verify all chunks are returned as separate responses
        assert len(responses) == len(
            mock_chunks
        ), "Each chunk should be returned as a separate response"

        # Verify the combined text is correct
        full_text = "".join([r.text for r in responses if hasattr(r, "text")])
        assert full_text == "Hello there, this is streaming"


@pytest.mark.asyncio
async def test_extract_attachments(gemini_base_bot, sample_query_with_image):
    """Test extracting attachments from a query."""
    attachments = gemini_base_bot._extract_attachments(sample_query_with_image)

    assert len(attachments) == 1
    assert attachments[0].name == "test.jpg"
    assert attachments[0].content_type == "image/jpeg"
    assert hasattr(attachments[0], "content")


@pytest.mark.asyncio
async def test_format_chat_history(gemini_base_bot, sample_query_with_chat_history):
    """Test formatting chat history from a query."""
    chat_history = gemini_base_bot._format_chat_history(sample_query_with_chat_history)

    # Should have 3 messages (user, bot, user)
    assert len(chat_history) == 3

    # Check first message
    assert chat_history[0]["role"] == "user"
    assert chat_history[0]["parts"][0]["text"] == "Hello, Gemini!"

    # Check second message
    assert chat_history[1]["role"] == "model"  # bot role mapped to model for Gemini
    assert chat_history[1]["parts"][0]["text"] == "Hello! How can I help you today?"

    # Check third message
    assert chat_history[2]["role"] == "user"
    assert chat_history[2]["parts"][0]["text"] == "What was my first message?"


@pytest.mark.asyncio
async def test_process_image_attachment(gemini_base_bot, image_attachment):
    """Test processing an image attachment."""
    media_data = gemini_base_bot._process_media_attachment(image_attachment)

    assert media_data is not None
    assert "mime_type" in media_data
    assert "data" in media_data
    assert media_data["mime_type"] == "image/jpeg"
    assert isinstance(media_data["data"], bytes)


@pytest.mark.asyncio
async def test_process_video_attachment(gemini_base_bot, video_attachment_local):
    """Test processing a video attachment."""
    media_data = gemini_base_bot._process_media_attachment(video_attachment_local)

    assert media_data is not None
    assert "mime_type" in media_data
    assert "data" in media_data
    assert media_data["mime_type"] == "video/mp4"
    assert isinstance(media_data["data"], bytes)


@pytest.mark.asyncio
async def test_process_unsupported_attachment(gemini_base_bot, unsupported_attachment):
    """Test processing an unsupported attachment type."""
    media_data = gemini_base_bot._process_media_attachment(unsupported_attachment)

    assert media_data is None


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
    # Bot name might be null in the response if not explicitly set
    assert "description" in response_text
    assert "supports_image_input" in response_text

    # The response should be valid JSON
    bot_info = json.loads(response_text)
    # Bot name comes from the fixture which might be null
    assert "model_name" in bot_info
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

    # For multimodal content, we use non-streaming mode
    mock_client.generate_content.return_value = MagicMock(text="I see an image")

    with patch("bots.gemini.get_client", return_value=mock_client):
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_image):
            responses.append(response)

        # Verify client was called with multimodal content
        assert mock_client.generate_content.called, "API should be called"
        args, kwargs = mock_client.generate_content.call_args

        if len(args) > 0:
            # Check if multimodal content is being sent correctly
            contents = args[0]
            assert isinstance(contents, list), "Contents should be a list for multimodal input"

            # For image-only calls the first item should be for the image
            if len(contents) > 1:
                assert (
                    "inline_data" in contents[0]
                ), "First item should contain inline_data for the image"
                assert "text" in contents[-1], "Last item should contain text prompt"

        # Verify response is properly processed
        assert len(responses) > 0
        assert any(isinstance(r, PartialResponse) for r in responses)
        # Combine all text responses to verify content
        full_text = "".join([r.text for r in responses if hasattr(r, "text")])
        assert "I see an image" in full_text


@pytest.mark.asyncio
async def test_image_output_handling_base64_fallback(gemini_bot, sample_query_with_text):
    """Test handling of image output in response with base64 fallback."""

    # Create a mock response with image data
    test_image_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9"
    inline_data = MockInlineData("image/jpeg", test_image_data)
    image_part = MockResponsePart(inline_data=inline_data)

    mock_response = MockMultimodalResponse(text="Here's an image I generated", parts=[image_part])

    # Mock the client generation
    mock_client = MagicMock()

    # First, mock the stream to return a chunk that has image parts
    # This will trigger the code path to get the full response
    image_chunk = MagicMock()
    image_chunk.parts = [
        MockResponsePart(inline_data=None)
    ]  # At least one part to trigger image check
    image_chunk.text = "Here's an image"

    mock_client.generate_content_stream.return_value = [image_chunk]
    mock_client.generate_content.return_value = mock_response

    # Create a proper image attachment that will pass the _process_image_attachment check
    mock_attachment = MockAttachment(
        "test.jpg",
        "image/jpeg",
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9",
    )
    gemini_bot._extract_attachments = MagicMock(return_value=[mock_attachment])

    # Create a mock that will cause AttributeError on post_message_attachment to trigger fallback
    with (
        patch("bots.gemini.get_client", return_value=mock_client),
        patch.object(
            gemini_bot,
            "post_message_attachment",
            side_effect=AttributeError("Method not available"),
        ),
        patch("PIL.Image.open"),
        patch("io.BytesIO"),
        patch("base64.b64encode", return_value=b"fake_base64_data"),
    ):
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Verify image was processed correctly using fallback path
        assert len(responses) > 0

        # Verify the full content API was called to process images
        assert mock_client.generate_content.called

        # Look for markdown image in responses
        has_image_markdown = False
        for resp in responses:
            if (
                hasattr(resp, "text")
                and "![Gemini generated image](data:image/jpeg;base64," in resp.text
            ):
                has_image_markdown = True
                # For the fake base64 data we don't need to verify decoded data
                # since we're mocking the actual encoding process

        assert has_image_markdown, "Response should include an image in markdown format (fallback)"


@pytest.mark.asyncio
async def test_image_output_handling_poe_attachment(gemini_bot, sample_query_with_text):
    """Test handling of image output using Poe's attachment system."""

    # Create a mock response with image data
    test_image_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9"
    inline_data = MockInlineData("image/jpeg", test_image_data)
    image_part = MockResponsePart(inline_data=inline_data)

    mock_response = MockMultimodalResponse(text="Here's an image I generated", parts=[image_part])

    # Mock the client generation
    mock_client = MagicMock()

    # First, mock the stream to return a chunk that has image parts
    # This will trigger the code path to get the full response
    image_chunk = MagicMock()
    image_chunk.parts = [MockResponsePart(inline_data=None)]
    image_chunk.text = "Here's an image"

    mock_client.generate_content_stream.return_value = [image_chunk]
    mock_client.generate_content.return_value = mock_response

    # Mock for Poe attachment response
    class MockAttachmentResponse:
        def __init__(self):
            self.inline_ref = "test_ref_123"

    mock_attachment_response = MockAttachmentResponse()

    # Create a proper image attachment that will pass the _process_image_attachment check
    mock_attachment = MockAttachment(
        "test.jpg",
        "image/jpeg",
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9",
    )
    gemini_bot._extract_attachments = MagicMock(return_value=[mock_attachment])

    # Patch both the Gemini client and the attachment method
    with (
        patch("bots.gemini.get_client", return_value=mock_client),
        patch.object(gemini_bot, "post_message_attachment", return_value=mock_attachment_response),
    ):
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Verify Poe attachment was used
        assert len(responses) > 0

        # Verify the full content API was called to process images
        assert mock_client.generate_content.called

        # Look for Poe attachment reference in responses
        has_poe_attachment = False
        for resp in responses:
            if hasattr(resp, "text") and "![gemini_image.jpg][test_ref_123]" in resp.text:
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
        ("image/unknown", "jpg"),  # Should default to jpg
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
            text="Here's an image I generated", parts=[image_part]
        )

        # Mock the client generation
        mock_client = MagicMock()

        # First, mock the stream to return a chunk that has image parts
        # This will trigger the code path to get the full response
        image_chunk = MagicMock()
        image_chunk.parts = [MockResponsePart(inline_data=None)]
        image_chunk.text = "Here's an image"

        mock_client.generate_content_stream.return_value = [image_chunk]
        mock_client.generate_content.return_value = mock_response

        # Create a proper image attachment that will pass the _process_image_attachment check
        mock_attachment = MockAttachment(
            "test.jpg",
            "image/jpeg",
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9",
        )
        gemini_bot._extract_attachments = MagicMock(return_value=[mock_attachment])

        # Patch both the Gemini client and the attachment method
        with (
            patch("bots.gemini.get_client", return_value=mock_client),
            patch.object(
                gemini_bot, "post_message_attachment", return_value=mock_attachment_response
            ),
        ):
            responses = []
            async for response in gemini_bot.get_response(sample_query_with_text):
                responses.append(response)

            # Verify correct file extension was used
            expected_filename = f"gemini_image.{expected_ext}"
            has_correct_extension = False
            for resp in responses:
                if hasattr(resp, "text") and f"![{expected_filename}][test_ref_123]" in resp.text:
                    has_correct_extension = True

            assert (
                has_correct_extension
            ), f"Response should use correct file extension for {mime_type}"


@pytest.mark.asyncio
async def test_image_upload_error_handling(gemini_bot, sample_query_with_text):
    """Test handling of errors in Poe's attachment upload system."""

    # Create a mock response with image data
    test_image_data = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9"
    inline_data = MockInlineData("image/jpeg", test_image_data)
    image_part = MockResponsePart(inline_data=inline_data)

    mock_response = MockMultimodalResponse(text="Here's an image I generated", parts=[image_part])

    # Mock the client generation
    mock_client = MagicMock()

    # First, mock the stream to return a chunk that has image parts
    # This will trigger the code path to get the full response
    image_chunk = MagicMock()
    image_chunk.parts = [MockResponsePart(inline_data=None)]
    image_chunk.text = "Here's an image"

    mock_client.generate_content_stream.return_value = [image_chunk]
    mock_client.generate_content.return_value = mock_response

    # Mock for failed Poe attachment response (no inline_ref)
    class MockFailedAttachmentResponse:
        def __init__(self):
            pass  # No inline_ref

    mock_failed_response = MockFailedAttachmentResponse()

    # Create a proper image attachment that will pass the _process_image_attachment check
    mock_attachment = MockAttachment(
        "test.jpg",
        "image/jpeg",
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9",
    )
    gemini_bot._extract_attachments = MagicMock(return_value=[mock_attachment])

    # Patch both the Gemini client and the attachment method
    with (
        patch("bots.gemini.get_client", return_value=mock_client),
        patch.object(gemini_bot, "post_message_attachment", return_value=mock_failed_response),
    ):
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Verify error handling works properly
        assert len(responses) > 0

        # Should show error message
        has_error_message = False
        for resp in responses:
            if hasattr(resp, "text") and "[Error uploading image to Poe]" in resp.text:
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
        text="Here are multiple images I generated", parts=[image_part1, image_part2]
    )

    # Mock the client generation
    mock_client = MagicMock()

    # First, mock the stream to return a chunk that has image parts
    # This will trigger the code path to get the full response
    image_chunk = MagicMock()
    image_chunk.parts = [MockResponsePart(inline_data=None)]
    image_chunk.text = "Here are multiple images"

    mock_client.generate_content_stream.return_value = [image_chunk]
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

    # Create a proper image attachment that will pass the _process_image_attachment check
    mock_attachment = MockAttachment(
        "test.jpg",
        "image/jpeg",
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9",
    )
    gemini_bot._extract_attachments = MagicMock(return_value=[mock_attachment])

    # Patch both the Gemini client and the attachment method
    with (
        patch("bots.gemini.get_client", return_value=mock_client),
        patch.object(
            gemini_bot, "post_message_attachment", side_effect=mock_attachment_side_effect
        ),
    ):
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Verify both images were processed
        assert len(responses) > 0

        # Should have both image references
        jpeg_image_ref = False
        png_image_ref = False

        for resp in responses:
            if hasattr(resp, "text"):
                if "![gemini_image.jpg][ref_1]" in resp.text:
                    jpeg_image_ref = True
                if "![gemini_image.png][ref_2]" in resp.text:
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
        text="Here's a large image I generated", parts=[image_part]
    )

    # Mock the client generation
    mock_client = MagicMock()

    # First, mock the stream to return a chunk that has image parts
    # This will trigger the code path to get the full response
    image_chunk = MagicMock()
    image_chunk.parts = [MockResponsePart(inline_data=None)]
    image_chunk.text = "Here's a large image"

    mock_client.generate_content_stream.return_value = [image_chunk]
    mock_client.generate_content.return_value = mock_response

    # Create a proper image attachment that will pass the _process_image_attachment check
    mock_attachment = MockAttachment(
        "test.jpg",
        "image/jpeg",
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9",
    )
    gemini_bot._extract_attachments = MagicMock(return_value=[mock_attachment])

    # Patch the Gemini client
    with patch("bots.gemini.get_client", return_value=mock_client):
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Verify image was skipped due to size
        assert len(responses) > 0

        # Should have a size limit message
        has_size_limit_message = False
        for resp in responses:
            if hasattr(resp, "text") and "[Image too large to display]" in resp.text:
                has_size_limit_message = True

        assert (
            has_size_limit_message
        ), "Response should include a message about the image being too large"


@pytest.mark.asyncio
async def test_process_streaming_helper_method(gemini_base_bot):
    """Test the _process_streaming_response helper method directly."""
    # Create mock stream with multiple chunks to test the helper function
    mock_chunks = [
        MagicMock(text="Hello"),
        MagicMock(text=" world"),
        # Test a chunk with parts instead of direct text
        MagicMock(text=None, parts=[MagicMock(text=" with"), MagicMock(text=" parts")]),
    ]

    # Create a proper async iterable stream
    class MockStream:
        def __init__(self, chunks):
            self.chunks = chunks

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.chunks:
                raise StopAsyncIteration
            return self.chunks.pop(0)

    # Call the helper method directly
    mock_stream = MockStream(mock_chunks.copy())  # Use copy to keep original chunks
    responses = []
    async for response in gemini_base_bot._process_streaming_response(mock_stream):
        responses.append(response)

    # Verify the method handled all chunk types correctly
    assert len(responses) == 4  # 2 direct text chunks + 2 from parts
    full_text = "".join([r.text for r in responses if hasattr(r, "text")])
    assert full_text == "Hello world with parts"

    # Test error handling in the streaming method
    error_stream = MagicMock()
    error_stream.__aiter__ = MagicMock(side_effect=Exception("Test streaming error"))

    error_responses = []
    async for response in gemini_base_bot._process_streaming_response(error_stream):
        error_responses.append(response)

    assert len(error_responses) == 1
    assert "Error streaming from Gemini" in error_responses[0].text


@pytest.mark.asyncio
async def test_process_streaming_with_non_async_iterable(gemini_base_bot):
    """Test that the _process_streaming_response method handles non-async iterables correctly."""
    # Create a mock stream that is iterable but not async iterable
    mock_chunks = [
        MagicMock(text="Hello"),
        MagicMock(text=" from"),
        MagicMock(text=" synchronous"),
        MagicMock(text=" iterator"),
    ]

    # Standard iterator (not async)
    class MockSyncStream:
        def __init__(self, chunks):
            self.chunks = chunks

        def __iter__(self):
            return self

        def __next__(self):
            if not self.chunks:
                raise StopIteration
            return self.chunks.pop(0)

    # Test with sync iterator
    mock_sync_stream = MockSyncStream(mock_chunks.copy())
    responses = []
    async for response in gemini_base_bot._process_streaming_response(mock_sync_stream):
        responses.append(response)

    # Verify sync iteration works
    assert len(responses) == 4
    full_text = "".join([r.text for r in responses if hasattr(r, "text")])
    assert full_text == "Hello from synchronous iterator"


@pytest.mark.asyncio
async def test_handling_missing_aiter_error(gemini_base_bot):
    """Test that our fix correctly handles the original error case."""

    # Create a mock object that lacks __aiter__ but would be used with async for
    class MockResponseWithoutAiter:
        def __init__(self):
            self.text = "Response without __aiter__"

        # Only implement __iter__ but not __aiter__
        def __iter__(self):
            yield MagicMock(text="This would fail with async for directly")

    # This would raise "'async for' requires an object with __aiter__ method"
    # if used directly with async for
    mock_response = MockResponseWithoutAiter()

    # Verify our implementation correctly handles this case
    responses = []
    async for response in gemini_base_bot._process_streaming_response(mock_response):
        responses.append(response)

    # Should fall back to synchronous iteration
    assert len(responses) == 1
    assert "This would fail with async for directly" in responses[0].text


@pytest.mark.asyncio
async def test_direct_response_object_handling(gemini_base_bot):
    """Test handling of direct response objects that are not iterables."""
    # Create a mock direct response (has text but is not iterable)
    mock_direct_response = MagicMock()
    mock_direct_response.text = "Direct non-iterable response text"
    # Need to properly handle our special case by deleting the attributes
    # that would make the object be considered iterable
    delattr(mock_direct_response, "__iter__")
    delattr(mock_direct_response, "__aiter__")

    # Verify our implementation handles direct response objects
    responses = []
    async for response in gemini_base_bot._process_streaming_response(mock_direct_response):
        responses.append(response)

    # Should extract text directly from the response
    assert len(responses) == 1
    assert responses[0].text == "Direct non-iterable response text"


@pytest.mark.skip(reason="Test needs refinement")
@pytest.mark.asyncio
async def test_fallback_to_non_streaming(gemini_base_bot, sample_query_with_text):
    """Test fallback from streaming to non-streaming if streaming fails."""
    # Mock client where streaming throws an error but non-streaming works
    mock_client = MagicMock()

    # Create a mock response with text
    mock_response = MagicMock()
    mock_response.text = "Non-streaming response"

    # Mock the generate_content method to fail on first call and succeed on second call
    def mock_generate_content(*args, **kwargs):
        if kwargs.get("stream") is True:
            raise Exception("Streaming failed")
        else:
            return mock_response

    mock_client.generate_content.side_effect = mock_generate_content

    # Test with a direct patch to isolate the function from other aspects
    with (
        patch.object(gemini_base_bot, "_extract_attachments", return_value=[]),
        patch.object(gemini_base_bot, "_prepare_image_parts", return_value=[]),
    ):
        responses = []
        async for response in gemini_base_bot._process_user_query(
            mock_client, "Hello", sample_query_with_text
        ):
            responses.append(response)

        # Should have generated a successful response through fallback
        assert len(responses) == 1
        assert responses[0].text == "Non-streaming response"

        # Verify both streaming and non-streaming were attempted
        assert mock_client.generate_content.call_count == 2
        assert mock_client.generate_content.call_args_list[0][1].get("stream") is True
        assert "stream" in mock_client.generate_content.call_args_list[1][1]
        assert mock_client.generate_content.call_args_list[1][1].get("stream") is False


@pytest.mark.asyncio
async def test_process_user_query_helper(
    gemini_base_bot, sample_query_with_text, sample_query_with_image
):
    """Test the _process_user_query helper method to ensure it handles different content types correctly."""
    # Mock client
    mock_client = MagicMock()

    # Test 1: Text-only content should use streaming
    class MockAsyncIterator:
        def __init__(self, text):
            self.yielded = False
            self.text = text

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.yielded:
                raise StopAsyncIteration
            self.yielded = True
            mock_chunk = MagicMock()
            mock_chunk.text = self.text
            return mock_chunk

    mock_stream = MockAsyncIterator("Streaming text response")
    mock_client.generate_content.return_value = mock_stream

    # Allow import inside the function to work by creating a fake module
    sys_modules_patcher = patch.dict(
        "sys.modules",
        {
            "google": MagicMock(),
            "google.generativeai": MagicMock(),
        },
    )

    # For text-only queries
    responses = []
    with sys_modules_patcher:
        async for response in gemini_base_bot._process_user_query(
            mock_client, "Hello", sample_query_with_text
        ):
            responses.append(response)

    # Verify streaming was used
    mock_client.generate_content.assert_called_once()
    assert mock_client.generate_content.call_args[1].get("stream") is True
    assert len(responses) == 1
    assert "Streaming text response" in responses[0].text

    # Test 2: Image content should use non-streaming
    # Create a new mock client for the second test to avoid reusing state
    mock_client = MagicMock()

    # Set up a mock image response
    mock_image_response = MagicMock(text="Image response")
    mock_client.generate_content.return_value = mock_image_response

    # Create a generator for multimodal content that will be yielded
    async def mock_multimodal_generator(*args, **kwargs):
        yield PartialResponse(text="Image response")

    # Mock image processing
    with (
        patch.object(gemini_base_bot, "_extract_attachments", return_value=["mock_image"]),
        patch.object(
            gemini_base_bot,
            "_prepare_media_parts",
            return_value=[{"mime_type": "image/jpeg", "data": b"fake-image"}],
        ),
        patch.object(
            gemini_base_bot,
            "_process_multimodal_content",
            side_effect=mock_multimodal_generator,
        ),
        # Reset the sys.modules patch
        sys_modules_patcher,
    ):
        # For image queries
        responses = []
        async for response in gemini_base_bot._process_user_query(
            mock_client, "What's in this image?", sample_query_with_image
        ):
            responses.append(response)

        # For multimodal, should not use streaming
        assert len(responses) == 1
        assert "Image response" in responses[0].text


# Helper for async generator
async def async_generator(value):
    yield value


@pytest.mark.asyncio
async def test_client_stub_compatibility(gemini_base_bot):
    """Test that the GeminiClientStub works correctly with the new streaming API."""
    # Import the stub class
    from bots.gemini import GeminiClientStub

    # Create an instance
    stub = GeminiClientStub(model_name="test-model")

    # Test non-streaming mode
    response = stub.generate_content("Hello")
    assert "not available" in response.text

    # Test streaming mode
    stream_response = stub.generate_content("Hello", stream=True)

    # Should be iterable for normal for loop
    chunks = list(stream_response)
    assert len(chunks) == 1
    assert "not available" in chunks[0].text

    # Should be async iterable
    chunks = []
    async for chunk in stream_response:
        chunks.append(chunk)

    assert len(chunks) == 1
    assert "not available" in chunks[0].text


@pytest.mark.asyncio
async def test_image_resize_fallback(gemini_bot, sample_query_with_text):
    """Test image resizing fallback for large images when using base64 encoding."""

    # Create a mock image that's large but under the main size limit (2MB)
    large_image_data = b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * (2 * 1024 * 1024)
    inline_data = MockInlineData("image/jpeg", large_image_data)
    image_part = MockResponsePart(inline_data=inline_data)

    mock_response = MockMultimodalResponse(
        text="Here's an image that needs resizing", parts=[image_part]
    )

    # Mock the client generation
    mock_client = MagicMock()

    # First, mock the stream to return a chunk that has image parts
    # This will trigger the code path to get the full response
    image_chunk = MagicMock()
    image_chunk.parts = [MockResponsePart(inline_data=None)]
    image_chunk.text = "Here's an image that needs resizing"

    mock_client.generate_content_stream.return_value = [image_chunk]
    mock_client.generate_content.return_value = mock_response

    # Mock PIL Image for resizing
    mock_image = MagicMock()
    mock_image.width = 1000
    mock_image.height = 1000
    mock_image.format = "JPEG"
    # Return smaller image data when saved
    mock_buffer = MagicMock()
    mock_buffer.getvalue.return_value = (
        b"\xff\xd8\xff\xe0\x00\x10JFIF" + b"\x00" * 1000
    )  # Much smaller

    # Create a proper image attachment that will pass the _process_image_attachment check
    mock_attachment = MockAttachment(
        "test.jpg",
        "image/jpeg",
        b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00\x43\x00\xff\xd9",
    )
    gemini_bot._extract_attachments = MagicMock(return_value=[mock_attachment])

    # Setup exception for attachment that forces the base64 fallback path
    with (
        patch("bots.gemini.get_client", return_value=mock_client),
        patch.object(gemini_bot, "post_message_attachment", side_effect=Exception("Forced error")),
        patch("PIL.Image.open", return_value=mock_image),
        patch("io.BytesIO", return_value=mock_buffer),
    ):
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Verify image was processed with resizing
        assert len(responses) > 0

        # Should have base64 image in response
        has_base64_image = False
        for resp in responses:
            if hasattr(resp, "text") and "data:image/jpeg;base64," in resp.text:
                has_base64_image = True

        assert has_base64_image, "Response should include a base64 encoded image after resizing"


@pytest.mark.asyncio
async def test_prepare_content(gemini_base_bot):
    """Test the _prepare_content helper method."""
    # Test with text only (no chat history)
    text_only_content = gemini_base_bot._prepare_content("Hello, Gemini!", [])
    assert isinstance(text_only_content, list)
    assert len(text_only_content) == 1
    assert text_only_content[0]["text"] == "Hello, Gemini!"

    # Test with images
    image_parts = [
        {"mime_type": "image/jpeg", "data": b"fake-jpeg-data"},
        {"mime_type": "image/png", "data": b"fake-png-data"},
    ]

    multimodal_content = gemini_base_bot._prepare_content("Describe these images", image_parts)
    assert isinstance(multimodal_content, list)
    assert len(multimodal_content) == 3  # 2 images + 1 text prompt

    # First two items should be images
    assert "inline_data" in multimodal_content[0]
    assert "inline_data" in multimodal_content[1]
    assert multimodal_content[0]["inline_data"] == image_parts[0]
    assert multimodal_content[1]["inline_data"] == image_parts[1]

    # Last item should be the text prompt
    assert multimodal_content[2]["text"] == "Describe these images"

    # Test with chat history
    chat_history = [
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Hi there"}]},
    ]

    # New user message should be added to history
    content_with_history = gemini_base_bot._prepare_content("How are you?", [], chat_history)
    assert isinstance(content_with_history, list)
    assert len(content_with_history) == 3  # 3 messages in history

    # First two messages should be unchanged
    assert content_with_history[0]["role"] == "user"
    assert content_with_history[0]["parts"][0]["text"] == "Hello"
    assert content_with_history[1]["role"] == "model"
    assert content_with_history[1]["parts"][0]["text"] == "Hi there"

    # Third message should be the new user message
    assert content_with_history[2]["role"] == "user"
    assert content_with_history[2]["parts"][0]["text"] == "How are you?"

    # Test with chat history where the last message is from the user
    chat_history_user_last = [
        {"role": "user", "parts": [{"text": "Hello"}]},
        {"role": "model", "parts": [{"text": "Hi there"}]},
        {"role": "user", "parts": [{"text": "Old message"}]},
    ]

    # Should update the last user message instead of adding a new one
    content_with_history_updated = gemini_base_bot._prepare_content(
        "New message", [], chat_history_user_last
    )
    assert isinstance(content_with_history_updated, list)
    assert len(content_with_history_updated) == 3  # Still 3 messages

    # First two messages should be unchanged
    assert content_with_history_updated[0]["role"] == "user"
    assert content_with_history_updated[0]["parts"][0]["text"] == "Hello"
    assert content_with_history_updated[1]["role"] == "model"
    assert content_with_history_updated[1]["parts"][0]["text"] == "Hi there"

    # Third message should be updated
    assert content_with_history_updated[2]["role"] == "user"
    assert content_with_history_updated[2]["parts"][0]["text"] == "New message"  # Updated


@pytest.mark.asyncio
async def test_multiturn_conversation(gemini_bot, sample_query_with_chat_history):
    """Test handling of a multi-turn conversation."""
    # Mock the client generation to avoid actual API calls
    mock_client = MagicMock()

    # Set up streaming response with a simple response
    mock_response = MagicMock(text="Your first message was 'Hello, Gemini!'")
    mock_chunks = [mock_response]

    # Setup async iterator for streaming
    class MockAsyncIterator:
        def __init__(self, chunks):
            self.chunks = chunks

        def __aiter__(self):
            return self

        async def __anext__(self):
            if not self.chunks:
                raise StopAsyncIteration
            return self.chunks.pop(0)

    mock_stream_response = MockAsyncIterator(mock_chunks.copy())
    mock_client.generate_content.return_value = mock_stream_response

    # Allow import inside the function to work by creating a fake module
    sys_modules_patcher = patch.dict(
        "sys.modules",
        {
            "google": MagicMock(),
            "google.generativeai": MagicMock(),
        },
    )

    with patch("bots.gemini.get_client", return_value=mock_client), sys_modules_patcher:
        responses = []
        async for response in gemini_bot.get_response(sample_query_with_chat_history):
            responses.append(response)

        # Verify chat history was used in the API call
        mock_client.generate_content.assert_called_once()
        args, kwargs = mock_client.generate_content.call_args

        # Extract the contents argument
        contents = args[0]

        # Check that the content includes multiple messages (chat history)
        assert isinstance(contents, list)
        assert len(contents) >= 3  # At least 3 messages from our chat history

        # Check that streaming was enabled
        assert kwargs.get("stream") is True

        # Check response
        assert len(responses) == 1
        assert "Your first message was 'Hello, Gemini!'" in responses[0].text


@pytest.mark.asyncio
async def test_process_user_query(gemini_bot, sample_query_with_text, sample_query_with_image):
    """Test the _process_user_query helper method."""
    # Mock client
    mock_client = MagicMock()

    # Test text-only query with streaming
    class MockAsyncIterator:
        def __init__(self, text):
            self.yielded = False
            self.text = text

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.yielded:
                raise StopAsyncIteration
            self.yielded = True
            mock_chunk = MagicMock()
            mock_chunk.text = self.text
            return mock_chunk

    mock_stream = MockAsyncIterator("Hello from Gemini")
    mock_client.generate_content.return_value = mock_stream

    # Allow import inside the function to work by creating a fake module
    sys_modules_patcher = patch.dict(
        "sys.modules",
        {
            "google": MagicMock(),
            "google.generativeai": MagicMock(),
        },
    )

    # Process text-only query
    responses = []
    with sys_modules_patcher:
        async for response in gemini_bot._process_user_query(
            mock_client, "Hello", sample_query_with_text
        ):
            responses.append(response)

    # Verify text-only processing
    assert len(responses) == 1
    assert responses[0].text == "Hello from Gemini"
    assert mock_client.generate_content.called
    assert mock_client.generate_content.call_args[1].get("stream") is True

    # Create a new mock client for the image test
    mock_client2 = MagicMock()

    # Test multimodal query with image
    # Mock for image handling
    mock_response = MagicMock(text="I see an image")
    mock_client2.generate_content.return_value = mock_response

    # Create mock async generator for multimodal content
    async def mock_multimodal_generator(*args, **kwargs):
        yield PartialResponse(text="I see an image")

    # Mock the attachments processing to return an image part
    with (
        patch.object(gemini_bot, "_extract_attachments", return_value=["mock_image"]),
        patch.object(
            gemini_bot,
            "_prepare_media_parts",
            return_value=[{"mime_type": "image/jpeg", "data": b"fake-image-data"}],
        ),
        patch.object(
            gemini_bot,
            "_process_multimodal_content",
            side_effect=mock_multimodal_generator,
        ),
        sys_modules_patcher,
    ):
        responses = []
        async for response in gemini_bot._process_user_query(
            mock_client2, "What's in this image?", sample_query_with_image
        ):
            responses.append(response)

        # Verify multimodal processing
        assert len(responses) == 1
        assert responses[0].text == "I see an image"
        # For multimodal content, it should call generate_content instead of streaming
        assert mock_client.generate_content.called


@pytest.mark.asyncio
async def test_process_user_query_for_all_model_versions(sample_query_with_text):
    """Test the _process_user_query method across different Gemini model versions."""
    # Create instances of different model versions
    gemini_20_bot = GeminiBot()  # 2.0 model
    gemini_25_flash_bot = Gemini25FlashBot()  # 2.5 Flash model
    gemini_25_pro_bot = Gemini25ProExpBot()  # 2.5 Pro model

    bots = [gemini_20_bot, gemini_25_flash_bot, gemini_25_pro_bot]

    # Create two different mock response classes for different model behaviors

    # Mock for newer models with resolve() method
    class MockResolveResponse:
        """Mock a streaming response for newer Gemini models that use resolve()."""

        def __init__(self, text):
            self.text = text

        def __iter__(self):
            yield MagicMock(text=self.text)

        def resolve(self):
            return MagicMock(text=self.text)

    # Mock for models that use async iteration
    class MockAsyncResponse:
        """Mock a streaming response that uses async iteration."""

        def __init__(self, text):
            self.text = text

        def __aiter__(self):
            return self

        async def __anext__(self):
            if hasattr(self, "_yielded") and self._yielded:
                raise StopAsyncIteration
            self._yielded = True
            return MagicMock(text=self.text)

    # Create a mock client that works for all scenarios
    mock_client = MagicMock()

    # Set up sys.modules patch to allow imports
    sys_modules_patcher = patch.dict(
        "sys.modules",
        {
            "google": MagicMock(),
            "google.generativeai": MagicMock(),
        },
    )

    # Test all bot instances with different streaming responses
    with sys_modules_patcher:
        for bot in bots:
            # Test 1: With resolve() method (newer models)
            mock_resolve_response = MockResolveResponse("Hello from Gemini with resolve")
            mock_client.generate_content.return_value = mock_resolve_response

            responses = []
            async for response in bot._process_user_query(
                mock_client, "Hello", sample_query_with_text
            ):
                responses.append(response)

            # All bots should produce a response with the resolve path
            assert len(responses) > 0
            assert "Hello from Gemini" in responses[0].text

            # Test 2: With async iteration (older models)
            mock_client.reset_mock()
            mock_async_response = MockAsyncResponse("Hello from Gemini async")
            mock_client.generate_content.return_value = mock_async_response

            responses = []
            async for response in bot._process_user_query(
                mock_client, "Hello async", sample_query_with_text
            ):
                responses.append(response)

            # Should still produce a response with the async iteration path
            assert len(responses) > 0
            assert "Hello from Gemini async" in responses[0].text


def test_get_client_with_valid_key():
    """Test the get_client function with a valid API key."""
    with (
        patch("bots.gemini.get_api_key", return_value="fake-api-key"),
        patch("google.generativeai.GenerativeModel") as mock_model,
        patch("google.generativeai.configure") as mock_configure,
    ):
        client = get_client("gemini-2.0-pro")

        # Verify API key was configured
        mock_configure.assert_called_once_with(api_key="fake-api-key")

        # Verify model was created with correct name
        mock_model.assert_called_once_with(model_name="gemini-2.0-pro")

        # Should return a client
        assert client is not None


def test_get_client_with_missing_key():
    """Test the get_client function with a missing API key."""
    with patch("bots.gemini.get_api_key", return_value=None):
        client = get_client("gemini-2.0-pro")
        assert client is None


def test_get_client_with_import_error():
    """Test the get_client function with an import error."""
    # Create a mock ImportError that will be raised when attempting to import google.generativeai
    mock_import_error = ImportError("No module named 'google.generativeai'")

    # Use patch.dict to modify sys.modules to trigger ImportError for google.generativeai
    with (
        patch.dict("sys.modules", {"google.generativeai": None, "google": None}),
        patch("bots.gemini.get_api_key", return_value="fake-api-key"),
        patch("bots.gemini.GeminiClientStub") as mock_stub,
    ):
        # Set model name for verification
        mock_stub.return_value.model_name = "gemini-2.0-pro"

        client = get_client("gemini-2.0-pro")

        # Should return a stub client
        assert client is not None
        assert mock_stub.called
        mock_stub.assert_called_once_with(model_name="gemini-2.0-pro")

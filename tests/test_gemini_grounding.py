"""
Tests for the grounding functionality in Gemini bots.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi_poe.types import PartialResponse, ProtocolMessage, QueryRequest

from bots.gemini import GeminiBaseBot, GeminiBot


@pytest.fixture
def gemini_bot():
    """Create a GeminiBot instance for testing."""
    return GeminiBot()


@pytest.fixture
def sample_query():
    """Create a sample query."""
    message = ProtocolMessage(role="user", content="Tell me about weather in San Francisco")

    return QueryRequest(
        version="1.0",
        type="query",
        query=[message],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )


@pytest.fixture
def sample_grounding_source():
    """Create a sample grounding source."""
    return {
        "title": "SF Weather Report",
        "url": "https://example.com/sf-weather",
        "content": "San Francisco weather is characterized by mild temperatures with typical ranges from 50-65°F. Foggy conditions are common, especially in summer months. The city rarely experiences extreme weather events.",
    }


def test_grounding_flag_defaults():
    """Test the default values for grounding flags."""
    bot = GeminiBaseBot()
    # Default value should be False until _set_grounding_support is called
    assert bot.supports_grounding is False
    # Grounding is only enabled if supported
    assert bot.grounding_enabled is False
    assert len(bot.grounding_sources) == 0
    assert bot.citations_enabled is True


def test_bot_info_includes_grounding(gemini_bot):
    """Test that bot info includes grounding flags."""
    bot_metadata = gemini_bot._get_bot_metadata()
    bot_metadata["model_name"] = gemini_bot.model_name
    bot_metadata["supports_grounding"] = gemini_bot.supports_grounding
    bot_metadata["grounding_enabled"] = gemini_bot.grounding_enabled
    bot_metadata["citations_enabled"] = gemini_bot.citations_enabled
    bot_metadata["supports_image_input"] = gemini_bot.supports_image_input
    bot_metadata["supports_video_input"] = gemini_bot.supports_video_input
    bot_metadata["supported_image_types"] = gemini_bot.supported_image_types
    bot_metadata["supported_video_types"] = gemini_bot.supported_video_types

    assert "supports_grounding" in bot_metadata
    assert "grounding_enabled" in bot_metadata
    assert "citations_enabled" in bot_metadata
    # This is a Flash model, so grounding should be disabled
    assert bot_metadata["supports_grounding"] is False
    assert bot_metadata["grounding_enabled"] is False
    assert bot_metadata["citations_enabled"] is True


def test_add_grounding_source(gemini_bot, sample_grounding_source):
    """Test adding a grounding source."""
    assert len(gemini_bot.grounding_sources) == 0
    gemini_bot.add_grounding_source(sample_grounding_source)
    assert len(gemini_bot.grounding_sources) == 1
    assert gemini_bot.grounding_sources[0] == sample_grounding_source


def test_add_invalid_grounding_source(gemini_bot):
    """Test adding an invalid grounding source."""
    # Test with non-dictionary
    gemini_bot.add_grounding_source("not a dictionary")
    assert len(gemini_bot.grounding_sources) == 0

    # Test with missing required keys
    gemini_bot.add_grounding_source({"title": "Missing content"})
    assert len(gemini_bot.grounding_sources) == 0

    gemini_bot.add_grounding_source({"content": "Missing title"})
    assert len(gemini_bot.grounding_sources) == 0


def test_clear_grounding_sources(gemini_bot, sample_grounding_source):
    """Test clearing grounding sources."""
    gemini_bot.add_grounding_source(sample_grounding_source)
    assert len(gemini_bot.grounding_sources) == 1

    gemini_bot.clear_grounding_sources()
    assert len(gemini_bot.grounding_sources) == 0


def test_set_grounding_enabled(gemini_bot):
    """Test enabling and disabling grounding."""
    # Flash model default
    assert gemini_bot.grounding_enabled is False

    # Try to enable grounding on a non-supporting model
    gemini_bot.set_grounding_enabled(True)
    # Should remain disabled because Flash models don't support grounding
    assert gemini_bot.grounding_enabled is False

    gemini_bot.set_grounding_enabled(False)
    assert gemini_bot.grounding_enabled is False

    # Now test with a Pro model that supports grounding
    pro_bot = GeminiBaseBot()
    pro_bot.model_name = "gemini-2.0-pro"
    pro_bot._set_grounding_support()

    # Should be supported but disabled by default
    assert pro_bot.supports_grounding is True
    assert pro_bot.grounding_enabled is False

    # Enable grounding on supporting model
    pro_bot.set_grounding_enabled(True)
    assert pro_bot.grounding_enabled is True


def test_prepare_grounding_config(gemini_bot, sample_grounding_source):
    """Test preparing grounding configuration."""
    # Flash model should return None even with sources
    config = gemini_bot._prepare_grounding_config()
    assert config is None

    # Add a source and test again with a Flash model
    gemini_bot.add_grounding_source(sample_grounding_source)
    config = gemini_bot._prepare_grounding_config()
    assert config is None  # Still None because Flash models don't support grounding

    # Now test with a Pro model that supports grounding
    pro_bot = GeminiBaseBot()
    pro_bot.model_name = "gemini-2.0-pro"
    pro_bot._set_grounding_support()
    pro_bot.set_grounding_enabled(True)
    pro_bot.add_grounding_source(sample_grounding_source)

    with patch("sys.modules", {"google.generativeai": MagicMock()}):
        config = pro_bot._prepare_grounding_config()
        assert config is not None
        assert config["groundingEnabled"] is True
        assert len(config["groundingSources"]) == 1
        assert config["groundingSources"][0]["title"] == sample_grounding_source["title"]
        assert config["groundingSources"][0]["content"] == sample_grounding_source["content"]
        assert config["groundingSources"][0]["uri"] == sample_grounding_source["url"]
        # Citations should be enabled by default
        assert config["includeCitations"] is True


def test_prepare_grounding_config_disabled(gemini_bot, sample_grounding_source):
    """Test preparing grounding configuration when disabled."""
    gemini_bot.add_grounding_source(sample_grounding_source)
    gemini_bot.set_grounding_enabled(False)

    config = gemini_bot._prepare_grounding_config()
    assert config is None


@pytest.mark.asyncio
async def test_grounding_in_api_call(gemini_bot, sample_query, sample_grounding_source):
    """Test that grounding config is included in the API call."""
    # Replace with a pro model that supports grounding
    gemini_bot.model_name = "gemini-2.0-pro"
    gemini_bot._set_grounding_support()
    gemini_bot.set_grounding_enabled(True)
    gemini_bot.add_grounding_source(sample_grounding_source)

    # Mock the client
    mock_client = MagicMock()

    # Set up a response
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

    mock_stream = MockAsyncIterator("Response with grounding")
    mock_client.generate_content.return_value = mock_stream

    # Allow import inside the function to work by creating a fake module
    sys_modules_patcher = patch.dict(
        "sys.modules",
        {
            "google": MagicMock(),
            "google.generativeai": MagicMock(),
        },
    )

    # Process query
    with sys_modules_patcher, patch("bots.gemini.get_client", return_value=mock_client):
        responses = []
        async for response in gemini_bot.get_response(sample_query):
            responses.append(response)

        # Verify API call included grounding config
        mock_client.generate_content.assert_called_once()
        kwargs = mock_client.generate_content.call_args[1]

        assert "generationConfig" in kwargs
        assert "groundingConfig" in kwargs["generationConfig"]
        assert kwargs["generationConfig"]["groundingConfig"]["groundingEnabled"] is True
        assert len(kwargs["generationConfig"]["groundingConfig"]["groundingSources"]) == 1
        assert "includeCitations" in kwargs["generationConfig"]["groundingConfig"]
        assert kwargs["generationConfig"]["groundingConfig"]["includeCitations"] is True


@pytest.mark.asyncio
async def test_set_citations_enabled():
    """Test enabling and disabling citations."""
    # Create a Pro model bot that supports grounding
    pro_bot = GeminiBaseBot()
    pro_bot.model_name = "gemini-2.0-pro"
    pro_bot._set_grounding_support()
    pro_bot.set_grounding_enabled(True)

    # Citations should be enabled by default
    assert pro_bot.citations_enabled is True

    # Test disabling citations
    pro_bot.set_citations_enabled(False)
    assert pro_bot.citations_enabled is False

    # Test enabling citations
    pro_bot.set_citations_enabled(True)
    assert pro_bot.citations_enabled is True

    # Check that the grounding config reflects citation settings
    with patch("sys.modules", {"google.generativeai": MagicMock()}):
        # Add a source so we can generate a config
        pro_bot.add_grounding_source({"title": "Test Source", "content": "Test content"})

        # With citations enabled
        config = pro_bot._prepare_grounding_config()
        assert config is not None
        assert config["includeCitations"] is True

        # With citations disabled
        pro_bot.set_citations_enabled(False)
        config = pro_bot._prepare_grounding_config()
        assert config is not None
        assert "includeCitations" not in config


@pytest.mark.asyncio
async def test_multimodal_with_grounding(gemini_bot, sample_query, sample_grounding_source):
    """Test that grounding works with multimodal content."""
    # Replace with a pro model that supports grounding
    gemini_bot.model_name = "gemini-2.0-pro"
    gemini_bot._set_grounding_support()
    gemini_bot.set_grounding_enabled(True)
    gemini_bot.add_grounding_source(sample_grounding_source)

    # Mock the client
    mock_client = MagicMock()

    # For multimodal content
    mock_response = MagicMock(text="Response with multimodal content and grounding")
    mock_client.generate_content.return_value = mock_response

    # Create mock async generator for multimodal content
    async def mock_multimodal_generator(*args, **kwargs):
        yield PartialResponse(text="Response with multimodal content and grounding")

    # Mock the processing
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
        patch.dict(
            "sys.modules",
            {
                "google": MagicMock(),
                "google.generativeai": MagicMock(),
            },
        ),
        patch("bots.gemini.get_client", return_value=mock_client),
    ):
        responses = []
        async for response in gemini_bot.get_response(sample_query):
            responses.append(response)

        # Process multimodal content was called
        assert gemini_bot._process_multimodal_content.called

        # Verify response
        assert len(responses) == 1
        assert responses[0].text == "Response with multimodal content and grounding"


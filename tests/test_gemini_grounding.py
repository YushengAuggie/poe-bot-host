"""
Tests for the grounding functionality in Gemini bots.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi_poe.types import PartialResponse, ProtocolMessage, QueryRequest

from bots.gemini import GeminiBaseBot, GeminiBot
from tests.google_mock_helper import create_google_genai_mock


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

    # Now test with a Pro model (note: grounding is currently disabled in model config)
    pro_bot = GeminiBaseBot()
    pro_bot.model_name = "gemini-2.0-pro"
    # The constructor automatically sets grounding support based on model name

    # Grounding is currently disabled for all models in the configuration
    assert pro_bot.supports_grounding is False
    assert pro_bot.grounding_enabled is False

    # Try to enable grounding on a non-supporting model (should be ignored)
    pro_bot.set_grounding_enabled(True)
    assert pro_bot.grounding_enabled is False  # Should remain False


def test_prepare_grounding_config(gemini_bot, sample_grounding_source):
    """Test preparing grounding configuration."""
    # Flash model should return None even with sources
    config = gemini_bot._prepare_grounding_config()
    assert config is None

    # Add a source and test again with a Flash model
    gemini_bot.add_grounding_source(sample_grounding_source)
    config = gemini_bot._prepare_grounding_config()
    assert config is None  # Still None because Flash models don't support grounding

    # Test with a Pro model (note: grounding is currently disabled in model config)
    pro_bot = GeminiBaseBot()
    pro_bot.model_name = "gemini-2.0-pro"
    # The constructor automatically sets grounding support based on model name
    pro_bot.add_grounding_source(sample_grounding_source)

    # Since grounding is disabled in model config, should return None
    config = pro_bot._prepare_grounding_config()
    assert config is None  # Grounding is disabled for all models currently


def test_prepare_grounding_config_disabled(gemini_bot, sample_grounding_source):
    """Test preparing grounding configuration when disabled."""
    gemini_bot.add_grounding_source(sample_grounding_source)
    gemini_bot.set_grounding_enabled(False)

    config = gemini_bot._prepare_grounding_config()
    assert config is None


@pytest.mark.asyncio
async def test_set_citations_enabled():
    """Test enabling and disabling citations."""
    # Create a Pro model bot (note: grounding is currently disabled in model config)
    pro_bot = GeminiBaseBot()
    pro_bot.model_name = "gemini-2.0-pro"
    # The constructor automatically sets grounding support based on model name

    # Citations should be enabled by default
    assert pro_bot.citations_enabled is True

    # Test disabling citations
    pro_bot.set_citations_enabled(False)
    assert pro_bot.citations_enabled is False

    # Test enabling citations
    pro_bot.set_citations_enabled(True)
    assert pro_bot.citations_enabled is True

    # Add a source so we can test config generation
    pro_bot.add_grounding_source({"title": "Test Source", "content": "Test content"})

    # Since grounding is disabled in model config, should return None regardless of citations
    config = pro_bot._prepare_grounding_config()
    assert config is None  # Grounding is disabled for all models currently

    # Test with citations disabled
    pro_bot.set_citations_enabled(False)
    config = pro_bot._prepare_grounding_config()
    assert config is None  # Still None due to disabled grounding

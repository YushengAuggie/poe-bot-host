"""
Tests for the ChatGPT bot implementation.
"""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi_poe.types import PartialResponse, ProtocolMessage, QueryRequest

from bots.chatgpt import ChatgptBot, get_client


@pytest.fixture
def chatgpt_bot():
    """Create a ChatgptBot instance for testing."""
    return ChatgptBot()


@pytest.fixture
def sample_query_with_text():
    """Create a sample query with text only."""
    message = ProtocolMessage(role="user", content="Hello, ChatGPT!")

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
        ProtocolMessage(role="user", content="Hello, ChatGPT!"),
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


@pytest.mark.asyncio
async def test_chatgpt_bot_initialization(chatgpt_bot):
    """Test ChatgptBot initialization."""
    # The bot_name class attribute is defined in the ChatgptBot class
    assert ChatgptBot.bot_name == "ChatgptBot"


@pytest.mark.asyncio
async def test_format_chat_history(chatgpt_bot, sample_query_with_chat_history):
    """Test formatting chat history from a query."""
    chat_history = chatgpt_bot._format_chat_history(sample_query_with_chat_history)

    # Should have 3 messages (user, assistant, user)
    assert len(chat_history) == 3

    # Check first message
    assert chat_history[0]["role"] == "user"
    assert chat_history[0]["content"] == "Hello, ChatGPT!"

    # Check second message - note the role mapping
    assert chat_history[1]["role"] == "assistant"  # bot role mapped to assistant for OpenAI
    assert chat_history[1]["content"] == "Hello! How can I help you today?"

    # Check third message
    assert chat_history[2]["role"] == "user"
    assert chat_history[2]["content"] == "What was my first message?"


@pytest.mark.asyncio
async def test_single_turn_response(chatgpt_bot, sample_query_with_text):
    """Test single-turn response generation."""
    # Create a mock OpenAI client response
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(delta=MagicMock(content="Hello from ChatGPT!"))]

    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [mock_response]

    # Patch the global client variable
    with (
        patch("bots.chatgpt.client", mock_client),
        patch("bots.chatgpt.get_client", return_value=mock_client),
    ):

        responses = []
        async for response in chatgpt_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Check that the API was called correctly
        mock_client.chat.completions.create.assert_called_once()

        # Extract call arguments
        call_args = mock_client.chat.completions.create.call_args[1]

        # Check that the right model was used
        assert call_args["model"] == "gpt-4.1-nano-2025-04-14"

        # Check that streaming was enabled
        assert call_args["stream"] is True

        # Check that the messages were formatted correctly
        messages = call_args["messages"]
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, ChatGPT!"

        # Check the response
        assert len(responses) == 1
        assert responses[0].text == "Hello from ChatGPT!"


@pytest.mark.asyncio
async def test_multiturn_conversation(chatgpt_bot, sample_query_with_chat_history):
    """Test handling of a multi-turn conversation."""
    # Create a mock OpenAI client response
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(delta=MagicMock(content="Your first message was 'Hello, ChatGPT!'"))
    ]

    # Mock the OpenAI client
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = [mock_response]

    # Patch the global client variable
    with (
        patch("bots.chatgpt.client", mock_client),
        patch("bots.chatgpt.get_client", return_value=mock_client),
    ):

        responses = []
        async for response in chatgpt_bot.get_response(sample_query_with_chat_history):
            responses.append(response)

        # Check that the API was called correctly
        mock_client.chat.completions.create.assert_called_once()

        # Extract call arguments
        call_args = mock_client.chat.completions.create.call_args[1]

        # Check that the messages were formatted correctly (should include chat history)
        messages = call_args["messages"]
        assert len(messages) == 3  # Full chat history

        # First message should be the user's initial message
        assert messages[0]["role"] == "user"
        assert messages[0]["content"] == "Hello, ChatGPT!"

        # Second message should be the bot's response
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Hello! How can I help you today?"

        # Third message should be the user's follow-up
        assert messages[2]["role"] == "user"
        assert messages[2]["content"] == "What was my first message?"

        # Check the response
        assert len(responses) == 1
        assert responses[0].text == "Your first message was 'Hello, ChatGPT!'"


@pytest.mark.asyncio
async def test_bot_info_request(chatgpt_bot):
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

    # Mock the client to ensure we don't make actual API calls
    with patch("bots.chatgpt.client", None), patch("bots.chatgpt.get_client"):

        responses = []
        async for response in chatgpt_bot.get_response(query):
            responses.append(response)

        # Verify bot info is returned
        assert len(responses) == 1
        response_text = responses[0].text
        assert "ChatgptBot" in response_text

        # The response should be valid JSON
        bot_info = json.loads(response_text)
        assert bot_info["name"] == "ChatgptBot"


@pytest.mark.asyncio
async def test_api_key_not_configured(chatgpt_bot, sample_query_with_text):
    """Test handling when API key is not configured."""
    # Mock get_client to return None (indicating no API key)
    with patch("bots.chatgpt.client", None), patch("bots.chatgpt.get_client", return_value=None):

        responses = []
        async for response in chatgpt_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Check the error message
        assert len(responses) == 1
        assert "Error: OpenAI API key is not configured" in responses[0].text


@pytest.mark.asyncio
async def test_api_error_handling(chatgpt_bot, sample_query_with_text):
    """Test handling of API errors."""
    # Mock client that raises an exception
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API Error")

    # Patch the global client variable
    with (
        patch("bots.chatgpt.client", mock_client),
        patch("bots.chatgpt.get_client", return_value=mock_client),
    ):

        responses = []
        async for response in chatgpt_bot.get_response(sample_query_with_text):
            responses.append(response)

        # Check the error message
        assert len(responses) == 1
        assert "Error: Could not get response from OpenAI" in responses[0].text
        assert "API Error" in responses[0].text

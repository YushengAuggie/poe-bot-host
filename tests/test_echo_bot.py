"""
Tests for the EchoBot implementation.
"""

from typing import List

import pytest
from fastapi_poe.types import PartialResponse, QueryRequest

from bots.echo_bot import EchoBot


@pytest.fixture
def echo_bot():
    """Create an EchoBot instance for testing."""
    return EchoBot()

@pytest.fixture
def sample_query():
    """Create a sample query for testing."""
    return QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": "Hello, Echo Bot!"}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message"
    )

@pytest.mark.asyncio
async def test_echo_bot_initialization(echo_bot):
    """Test EchoBot initialization."""
    assert echo_bot.bot_name == "EchoBot"
    assert echo_bot.path == "/echobot"

@pytest.mark.asyncio
async def test_echo_bot_response(echo_bot, sample_query):
    """Test EchoBot response."""
    # Collect responses
    responses: List[PartialResponse] = []
    async for response in echo_bot.get_response(sample_query):
        responses.append(response)

    # Check response
    assert len(responses) == 1
    assert responses[0].text == "Hello, Echo Bot!"

@pytest.mark.asyncio
async def test_echo_bot_empty_message(echo_bot):
    """Test EchoBot with an empty message."""
    # Create query with empty message
    empty_query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": ""}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message"
    )

    # Collect responses
    responses: List[PartialResponse] = []
    async for response in echo_bot.get_response(empty_query):
        responses.append(response)

    # Check response (should echo empty string)
    assert len(responses) == 1
    assert responses[0].text == ""

@pytest.mark.asyncio
async def test_echo_bot_special_characters(echo_bot):
    """Test EchoBot with special characters."""
    # Create query with special characters
    special_query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": "!@#$%^&*()_+<>?:\"{}|~`-=[]\\;',./"}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message"
    )

    # Collect responses
    responses: List[PartialResponse] = []
    async for response in echo_bot.get_response(special_query):
        responses.append(response)

    # Check response (should echo special characters exactly)
    assert len(responses) == 1
    assert responses[0].text == "!@#$%^&*()_+<>?:\"{}|~`-=[]\\;',./"

@pytest.mark.asyncio
async def test_echo_bot_long_message(echo_bot):
    """Test EchoBot with a long message."""
    # Create a long message
    long_message = "A" * 1000

    # Create query with long message
    long_query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": long_message}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message"
    )

    # Collect responses
    responses: List[PartialResponse] = []
    async for response in echo_bot.get_response(long_query):
        responses.append(response)

    # Check response (should echo the long message exactly)
    assert len(responses) == 1
    assert len(responses[0].text) == 1000
    assert responses[0].text == long_message

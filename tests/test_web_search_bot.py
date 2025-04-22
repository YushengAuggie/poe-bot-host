"""
Tests for the WebSearchBot implementation.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from typing import List, Dict, Any
from fastapi_poe.types import QueryRequest, PartialResponse
from bots.web_search_bot import WebSearchBot
from utils.base_bot import BotError, BotErrorNoRetry

@pytest.fixture
def web_search_bot():
    """Create a WebSearchBot instance for testing."""
    return WebSearchBot()

@pytest.fixture
def mock_search_results():
    """Create mock search results for testing."""
    return {
        "search_metadata": {
            "status": "Success",
            "created_at": "2023-04-13 12:34:56 UTC",
        },
        "search_parameters": {
            "q": "test query",
            "engine": "google",
        },
        "organic_results": [
            {
                "position": 1,
                "title": "Test Result 1",
                "link": "https://example.com/1",
                "snippet": "This is the first test result."
            },
            {
                "position": 2,
                "title": "Test Result 2",
                "link": "https://example.com/2",
                "snippet": "This is the second test result."
            },
            {
                "position": 3,
                "title": "Test Result 3",
                "link": "https://example.com/3",
                "snippet": "This is the third test result."
            }
        ]
    }

@pytest.mark.asyncio
async def test_web_search_bot_initialization(web_search_bot):
    """Test WebSearchBot initialization."""
    # Check the class attributes rather than instance attributes
    assert WebSearchBot.bot_name == "WebSearchBot"
    assert "search" in WebSearchBot.bot_description.lower()

@pytest.mark.asyncio
async def test_web_search_bot_help_command(web_search_bot):
    """Test web search bot help command."""
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": "help"}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message"
    )
    
    responses = []
    async for response in web_search_bot._process_message("help", query):
        responses.append(response)
    
    # Verify help content is returned
    assert len(responses) == 1
    assert "Web Search Bot" in responses[0].text
    assert "search query" in responses[0].text.lower()

@pytest.mark.asyncio
async def test_web_search_bot_empty_query(web_search_bot):
    """Test web search bot with empty query."""
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": ""}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message"
    )
    
    responses = []
    async for response in web_search_bot._process_message("", query):
        responses.append(response)
    
    # Verify prompt for search query is returned
    assert len(responses) == 1
    assert "Please enter a search query" in responses[0].text

@pytest.mark.asyncio
async def test_web_search_bot_no_api_key(web_search_bot):
    """Test web search bot with no API key (should use mock data)."""
    search_query = "test query"
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": search_query}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message"
    )
    
    # Make sure the api_key is empty (it should be by default, but just to be sure)
    web_search_bot.api_key = ""
    
    responses = []
    async for response in web_search_bot._process_message(search_query, query):
        responses.append(response)
    
    # Verify response contains mock data notice
    assert len(responses) == 2  # "Searching..." and the results
    assert "Searching" in responses[0].text
    assert "mock results" in responses[1].text.lower() or "using mock" in responses[1].text.lower()

@pytest.mark.asyncio
async def test_web_search_with_api_key(web_search_bot, mock_search_results):
    """Test web search bot with API key."""
    search_query = "test query"
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": search_query}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message"
    )
    
    # Set a fake API key
    web_search_bot.api_key = "fake_api_key"
    
    # Mock the httpx.AsyncClient.get method
    with patch('httpx.AsyncClient.get') as mock_get:
        # Create a mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = mock_search_results
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response
        
        responses = []
        async for response in web_search_bot._process_message(search_query, query):
            responses.append(response)
        
        # Verify httpx.get was called with the right parameters
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs['params']['q'] == search_query
        assert kwargs['params']['api_key'] == "fake_api_key"
        
        # Verify response contains search results
        assert len(responses) == 2  # "Searching..." and the results
        assert "Searching" in responses[0].text
        assert "Test Result 1" in responses[1].text
        assert "Test Result 2" in responses[1].text
        assert "Test Result 3" in responses[1].text

@pytest.mark.asyncio
async def test_format_search_results(web_search_bot, mock_search_results):
    """Test formatting of search results."""
    query = "test query"
    formatted_results = web_search_bot._format_search_results(mock_search_results, query)
    
    # Check that formatting contains key elements
    assert f"Search Results for '{query}'" in formatted_results
    assert "Test Result 1" in formatted_results
    assert "https://example.com/1" in formatted_results
    assert "This is the first test result." in formatted_results
    assert "Test Result 2" in formatted_results
    assert "Test Result 3" in formatted_results

@pytest.mark.asyncio
async def test_search_error_handling(web_search_bot):
    """Test error handling during search."""
    search_query = "test query"
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": search_query}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message"
    )
    
    # Set a fake API key
    web_search_bot.api_key = "fake_api_key"
    
    # Mock the _search_web method to raise an error
    with patch.object(web_search_bot, '_search_web', new_callable=AsyncMock) as mock_search:
        mock_search.side_effect = BotError("Search API error")
        
        responses = []
        async for response in web_search_bot._process_message(search_query, query):
            responses.append(response)
        
        # Verify error handling
        assert len(responses) == 2  # "Searching..." and the error
        assert "Searching" in responses[0].text
        assert "Search error" in responses[1].text
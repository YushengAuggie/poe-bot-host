"""
Tests for the WeatherBot implementation.
"""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from typing import List, Dict, Any
from fastapi_poe.types import QueryRequest, PartialResponse
from bots.weather_bot import WeatherBot

@pytest.fixture
def weather_bot():
    """Create a WeatherBot instance for testing."""
    return WeatherBot()

@pytest.fixture
def mock_weather_data():
    """Create mock weather data for testing."""
    return {
        "name": "Test City",
        "main": {
            "temp": 22.5,
            "feels_like": 23.0,
            "temp_min": 20.0,
            "temp_max": 25.0,
            "pressure": 1012,
            "humidity": 65
        },
        "weather": [
            {
                "id": 800,
                "main": "Clear",
                "description": "clear sky",
                "icon": "01d"
            }
        ],
        "wind": {
            "speed": 3.6,
            "deg": 160
        },
        "sys": {
            "country": "TS",
            "sunrise": 1618884000,
            "sunset": 1618930800
        },
        "dt": 1618910400,
        "timezone": 0,
        "mock_data": True
    }

@pytest.mark.asyncio
async def test_weather_bot_initialization(weather_bot):
    """Test WeatherBot initialization."""
    # Check the class attributes rather than instance attributes
    assert WeatherBot.bot_name == "WeatherBot"
    assert "weather" in WeatherBot.bot_description.lower()

@pytest.mark.asyncio
async def test_weather_bot_help_command(weather_bot):
    """Test weather bot help command."""
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": "help"}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message"
    )
    
    responses = []
    async for response in weather_bot._process_message("help", query):
        responses.append(response)
    
    # Verify help content is returned
    assert len(responses) == 1
    assert "Weather Bot" in responses[0].text
    assert "location" in responses[0].text.lower()

@pytest.mark.asyncio
async def test_weather_bot_empty_query(weather_bot):
    """Test weather bot with empty query."""
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": ""}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message"
    )
    
    responses = []
    async for response in weather_bot._process_message("", query):
        responses.append(response)
    
    # Verify prompt for location is returned
    assert len(responses) == 1
    assert "Please enter a location" in responses[0].text

@pytest.mark.asyncio
async def test_weather_bot_generic_location(weather_bot):
    """Test weather bot with generic location terms."""
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": "my location"}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message"
    )
    
    responses = []
    async for response in weather_bot._process_message("my location", query):
        responses.append(response)
    
    # Verify prompt for specific location is returned
    assert len(responses) == 1
    assert "Please specify a location" in responses[0].text

@pytest.mark.asyncio
async def test_weather_bot_get_weather(weather_bot, mock_weather_data):
    """Test weather bot getting weather data."""
    location = "Test City"
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": location}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message"
    )
    
    # Mock the _get_weather method to return our test data
    with patch.object(weather_bot, '_get_weather', new_callable=AsyncMock) as mock_get_weather:
        mock_get_weather.return_value = mock_weather_data
        
        responses = []
        async for response in weather_bot._process_message(location, query):
            responses.append(response)
        
        # First response should be "Getting weather..."
        assert len(responses) == 2
        assert "Getting weather" in responses[0].text
        
        # Second response should contain the formatted weather data
        assert "Weather for Test City" in responses[1].text
        assert "Clear" in responses[1].text
        assert "22.5°C" in responses[1].text

@pytest.mark.asyncio
async def test_weather_bot_location_not_found(weather_bot):
    """Test weather bot with non-existent location."""
    location = "NonExistentCity"
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": location}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message"
    )
    
    # Mock the _get_weather method to raise a BotErrorNoRetry
    with patch.object(weather_bot, '_get_weather', new_callable=AsyncMock) as mock_get_weather:
        from utils.base_bot import BotErrorNoRetry
        mock_get_weather.side_effect = BotErrorNoRetry(f"Location '{location}' not found.")
        
        responses = []
        async for response in weather_bot._process_message(location, query):
            responses.append(response)
        
        # Should have two responses: "Getting weather..." and the error
        assert len(responses) == 2
        assert "Getting weather" in responses[0].text
        assert "Error" in responses[1].text
        assert "not found" in responses[1].text

@pytest.mark.asyncio
async def test_format_weather_data(weather_bot, mock_weather_data):
    """Test formatting of weather data."""
    formatted_data = weather_bot._format_weather_data(mock_weather_data)
    
    # Check that formatting contains key elements
    assert "Weather for Test City" in formatted_data
    assert "Current Conditions: Clear" in formatted_data
    assert "22.5°C" in formatted_data  # Current temp
    assert "23.0°C" in formatted_data  # Feels like
    assert "65%" in formatted_data     # Humidity
    assert "3.6 m/s" in formatted_data # Wind speed
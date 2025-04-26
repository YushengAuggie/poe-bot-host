"""
Tests for the WeatherBot implementation.
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi_poe.types import QueryRequest

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

    # Since we're testing a newer version of the bot, we should use get_response 
    # instead of _process_message which is deprecated
    responses = []
    async for response in weather_bot.get_response(query):
        responses.append(response)

    # Verify help content is returned
    assert len(responses) > 0
    help_text = " ".join([r.text for r in responses])
    assert "Weather Bot" in help_text
    assert "location" in help_text.lower()

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
    async for response in weather_bot.get_response(query):
        responses.append(response)

    # Verify prompt for location is returned
    assert len(responses) > 0
    response_text = " ".join([r.text for r in responses])
    assert "Please enter a location" in response_text

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
    async for response in weather_bot.get_response(query):
        responses.append(response)

    # Verify prompt for specific location is returned
    assert len(responses) > 0
    response_text = " ".join([r.text for r in responses])
    assert "Please specify a location" in response_text

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
        async for response in weather_bot.get_response(query):
            responses.append(response)

        # Check for the expected response content
        assert len(responses) > 0
        response_text = " ".join([r.text for r in responses])
        assert "Weather for Test City" in response_text
        assert "Clear" in response_text
        assert "22.5°C" in response_text

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
        async for response in weather_bot.get_response(query):
            responses.append(response)

        # Check if we have responses and at least one contains the error
        assert len(responses) > 0
        response_text = " ".join([r.text for r in responses])
        assert "Error" in response_text or "error" in response_text.lower()
        assert "not found" in response_text

@pytest.mark.asyncio
async def test_format_weather_data(weather_bot, mock_weather_data):
    """Test formatting of weather data."""
    formatted_data = weather_bot._format_weather_data(mock_weather_data)

    # Check that formatting contains key elements
    assert "Weather for Test City" in formatted_data
    assert "Clear" in formatted_data  # Weather condition
    assert "22.5°C" in formatted_data  # Current temp
    assert "23.0°C" in formatted_data  # Feels like
    assert "65%" in formatted_data     # Humidity
    assert "3.6 m/s" in formatted_data # Wind speed

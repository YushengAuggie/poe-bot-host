"""
Tests for the CalculatorBot implementation.
"""

import pytest
from fastapi_poe.types import QueryRequest

from bots.calculator_bot import CalculatorBot


@pytest.fixture
def calculator_bot():
    """Create a CalculatorBot instance for testing."""
    return CalculatorBot()


@pytest.fixture
def sample_query():
    """Create a sample query with a mathematical expression."""

    def _create_query(expression):
        return QueryRequest(
            version="1.0",
            type="query",
            query=[{"role": "user", "content": expression}],
            user_id="test_user",
            conversation_id="test_conversation",
            message_id="test_message",
        )

    return _create_query


@pytest.mark.asyncio
async def test_calculator_initialization(calculator_bot):
    """Test CalculatorBot initialization."""
    # Check the class attributes rather than instance attributes
    assert CalculatorBot.bot_name == "CalculatorBot"
    assert "calculator" in CalculatorBot.bot_description.lower()


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("2 + 2", "4"),
        ("10 - 5", "5"),
        ("3 * 4", "12"),
        ("20 / 4", "5"),
        ("2^3", "8"),
        ("sqrt(16)", "4"),
    ],
)
@pytest.mark.asyncio
async def test_calculator_basic_operations(calculator_bot, expression, expected):
    """Test basic calculator operations."""
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": expression}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )

    responses = []
    async for response in calculator_bot.get_response(query):
        responses.append(response)

    # Verify response contains the expected calculation result
    assert len(responses) == 1
    assert expected in responses[0].text


@pytest.mark.parametrize(
    "expression",
    [
        "help",
        "?",
        "/help",
    ],
)
@pytest.mark.asyncio
async def test_calculator_help_command(calculator_bot, expression):
    """Test calculator help command."""
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": expression}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )

    responses = []
    async for response in calculator_bot.get_response(query):
        responses.append(response)

    # Verify help content is returned
    assert len(responses) == 1
    assert "Calculator Bot" in responses[0].text
    assert "Basic Arithmetic" in responses[0].text


@pytest.mark.parametrize(
    "conversion_request,expected_result",
    [
        ("convert 32 f to c", "0"),
        ("convert 100 c to f", "212"),
        ("convert 10 km to miles", "6.21"),
        ("convert 5 miles to km", "8.04"),  # Updated to match more precisely
    ],
)
@pytest.mark.asyncio
async def test_calculator_unit_conversions(calculator_bot, conversion_request, expected_result):
    """Test calculator unit conversions."""
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": conversion_request}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )

    responses = []
    async for response in calculator_bot.get_response(query):
        responses.append(response)

    # Verify conversion result is correct
    assert len(responses) == 1
    assert expected_result in responses[0].text


@pytest.mark.parametrize(
    "invalid_expression",
    [
        "1/0",  # Division by zero
        "print('hello')",  # Attempt to execute code
        "os.system('ls')",  # Attempt to access os module
        "__import__('os')",  # Another attempt to import modules
    ],
)
@pytest.mark.asyncio
async def test_calculator_handles_invalid_expressions(calculator_bot, invalid_expression):
    """Test calculator handles invalid expressions safely."""
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": invalid_expression}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
    )

    responses = []
    async for response in calculator_bot.get_response(query):
        responses.append(response)

    # Verify error message is returned
    assert len(responses) == 1
    assert "Error" in responses[0].text or "Invalid" in responses[0].text

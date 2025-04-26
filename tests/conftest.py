"""
Pytest configuration for the Poe Bots Framework.
"""

import os
import sys

import pytest

# Add the parent directory to the path for imports to work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Set testing environment variables
os.environ["POE_ALLOW_WITHOUT_KEY"] = "true"
os.environ["DEBUG"] = "true"

@pytest.fixture
def sample_query():
    """Sample query for testing."""
    from fastapi_poe.types import QueryRequest

    return QueryRequest(
        query=[{"role": "user", "content": "Hello, world!"}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
        protocol="poe"
    )

@pytest.fixture
def simple_query():
    """Simple string query for testing."""
    from fastapi_poe.types import QueryRequest

    return QueryRequest(
        query="Hello, world!",
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
        protocol="poe"
    )

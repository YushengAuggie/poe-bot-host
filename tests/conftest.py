"""
Pytest configuration for the Poe Bots Framework.
"""

import os
import sys
from unittest.mock import patch

import pytest

# Add the parent directory to the path for imports to work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Set testing environment variables
os.environ["POE_ALLOW_WITHOUT_KEY"] = "true"
os.environ["DEBUG"] = "true"


@pytest.fixture(autouse=False)
def mock_get_api_key():
    """Mock get_api_key to return a test API key.

    This is NOT auto-used to avoid interfering with tests that need specific API key values.
    Apply this fixture explicitly to tests that need a generic API key.
    """
    with patch("utils.api_keys.get_api_key", return_value="test_api_key"):
        yield


class NoRaiseMock:
    """A helper to avoid ValueError being raised by get_api_key."""

    def __init__(self, return_value="test_api_key"):
        self.return_value = return_value

    def __call__(self, key_name):
        return self.return_value


@pytest.fixture(autouse=True)
def allow_api_key_access(request):
    """Ensure API key checks don't fail in tests by default."""
    # Skip this fixture for test_modal_integration.py
    if request.module.__name__.endswith("test_modal_integration"):
        yield
        return

    # For most tests, just make sure get_api_key doesn't raise errors
    with patch("utils.api_keys.get_api_key", NoRaiseMock()):
        yield


@pytest.fixture
def sample_query():
    """Sample query for testing."""
    from fastapi_poe.types import QueryRequest

    return QueryRequest(
        query=[{"role": "user", "content": "Hello, world!"}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message",
        protocol="poe",
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
        protocol="poe",
    )


@pytest.fixture
def video_attachment():
    """Create a mock video attachment."""
    from fastapi_poe.types import Attachment

    # Create a minimal class that extends Attachment with content field
    class MockVideoAttachment(Attachment):
        url: str = "mock://video"
        content_type: str = "video/mp4"
        content: bytes = b""

        def __init__(self, name: str, content: bytes):
            super().__init__(url="mock://video", content_type="video/mp4", name=name)
            # Store content as a non-model field
            object.__setattr__(self, "content", content)

    # Sample MP4 header (not a real video, just for testing)
    mp4_content = b"\x00\x00\x00\x20\x66\x74\x79\x70\x69\x73\x6f\x6d\x00\x00\x02\x00"

    return MockVideoAttachment("test_video.mp4", mp4_content)

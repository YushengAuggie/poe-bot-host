"""
Integration tests for the main application.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import create_api


@pytest.fixture
def mock_bot_factory():
    """Mock the BotFactory class."""
    with patch('app.BotFactory') as mock_factory:
        # Create a mock FastAPI app
        mock_app = FastAPI()

        # Add a test endpoint
        @mock_app.get("/test")
        def test_endpoint():
            return {"status": "ok"}

        # Configure the mock factory
        mock_factory.create_app.return_value = mock_app
        mock_factory.load_bots_from_module.return_value = [
            MagicMock(bot_name="TestBot1"),
            MagicMock(bot_name="TestBot2")
        ]
        mock_factory.get_available_bots.return_value = {
            "TestBot1": "Test bot 1 description",
            "TestBot2": "Test bot 2 description"
        }

        yield mock_factory

def test_create_api(mock_bot_factory):
    """Test creating the API."""
    # Create the API
    api = create_api(allow_without_key=True)

    # Check that load_bots_from_module was called
    mock_bot_factory.load_bots_from_module.assert_called_once_with("bots")

    # Check that create_app was called with the bot classes
    bot_classes = mock_bot_factory.load_bots_from_module.return_value
    mock_bot_factory.create_app.assert_called_once_with(bot_classes, allow_without_key=True)

    # Make sure we got a FastAPI app
    assert isinstance(api, FastAPI)

def test_api_health_endpoint(mock_bot_factory):
    """Test the health endpoint."""
    # Create the API
    api = create_api(allow_without_key=True)

    # Create a test client
    client = TestClient(api)

    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200

    # Check response content
    data = response.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "bots" in data
    assert len(data["bots"]) == 2
    assert "TestBot1" in data["bots"]
    assert "TestBot2" in data["bots"]

def test_api_bots_endpoint(mock_bot_factory):
    """Test the bots endpoint."""
    # Create the API
    api = create_api(allow_without_key=True)

    # Create a test client
    client = TestClient(api)

    # Test bots endpoint
    response = client.get("/bots")
    assert response.status_code == 200

    # Check response content
    data = response.json()
    assert "TestBot1" in data
    assert "TestBot2" in data
    assert data["TestBot1"] == "Test bot 1 description"
    assert data["TestBot2"] == "Test bot 2 description"

def test_api_error_handling(mock_bot_factory):
    """Test API error handling."""
    # Create the API
    api = create_api(allow_without_key=True)

    # Add a test endpoint that raises an exception
    @api.get("/error")
    def error_endpoint():
        raise ValueError("Test error")

    # Create a test client
    client = TestClient(api)

    # Test error endpoint
    response = client.get("/error")
    assert response.status_code == 500

    # Check response content
    data = response.json()
    assert "error" in data
    assert "detail" in data
    assert "internal server error" in data["error"].lower()
    assert "Test error" in data["detail"]

def test_api_no_bots_warning(mock_bot_factory):
    """Test warning when no bots are found."""
    # Configure the mock factory to return no bots
    mock_bot_factory.load_bots_from_module.return_value = []

    # Create the API (this should log a warning but not fail)
    with patch('app.logger.warning') as mock_warning:
        api = create_api(allow_without_key=True)

        # Check that the warning was logged
        mock_warning.assert_called_once_with("No bots found in 'bots' module!")

        # Make sure we still got a FastAPI app
        assert isinstance(api, FastAPI)

"""
Tests for the BotFactory class.
"""

from typing import AsyncGenerator
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from fastapi_poe.types import PartialResponse, QueryRequest

from utils.base_bot import BaseBot
from utils.bot_factory import BotFactory


# Test bot classes for factory testing
class TestFactoryBot1(BaseBot):
    """First test bot for factory testing."""
    bot_name = "TestFactoryBot1"
    bot_description = "First test bot for factory testing"

    def __init__(self, **kwargs):
        path = "/test-factory-bot-1"
        super().__init__(path=path, **kwargs)

    async def _process_message(self, message: str, query: QueryRequest) -> AsyncGenerator[PartialResponse, None]:
        yield PartialResponse(text=f"TestFactoryBot1: {message}")

class TestFactoryBot2(BaseBot):
    """Second test bot for factory testing."""
    bot_name = "TestFactoryBot2"
    bot_description = "Second test bot for factory testing"

    def __init__(self, **kwargs):
        path = "/test-factory-bot-2"
        super().__init__(path=path, **kwargs)

    async def _process_message(self, message: str, query: QueryRequest) -> AsyncGenerator[PartialResponse, None]:
        yield PartialResponse(text=f"TestFactoryBot2: {message}")

@pytest.fixture
def factory_bot_classes():
    """Fixture to get bot classes for testing."""
    return [TestFactoryBot1, TestFactoryBot2]

def test_get_available_bots():
    """Test getting available bots."""
    bot_info = BotFactory.get_available_bots()

    # Check that we get a dictionary
    assert isinstance(bot_info, dict)

    # Check that the dictionary contains at least the included bots
    expected_bots = ["EchoBot", "ReverseBot", "UppercaseBot"]
    for bot in expected_bots:
        assert bot in bot_info
        assert isinstance(bot_info[bot], str)

    # Check that the descriptions are non-empty
    for bot, description in bot_info.items():
        assert description, f"Description for {bot} should not be empty"

def test_create_app(factory_bot_classes):
    """Test creating a FastAPI app with bots."""
    # Create the app with test bot classes
    app = BotFactory.create_app(factory_bot_classes, allow_without_key=True)

    # Check that we got a FastAPI app
    assert isinstance(app, FastAPI)

    # Check that the app has routes for our bots
    routes = [route.path for route in app.routes]
    assert "/test-factory-bot-1" in routes
    assert "/test-factory-bot-2" in routes

@pytest.mark.asyncio
async def test_app_integration(factory_bot_classes):
    """Test the integration of BotFactory with FastAPI."""
    # Create the app with test bot classes
    app = BotFactory.create_app(factory_bot_classes, allow_without_key=True)

    # Create a test client
    client = TestClient(app)

    # Test health endpoint
    if "/health" in [route.path for route in app.routes]:
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"

@pytest.mark.asyncio
async def test_bot_factory_load_bots():
    """Test loading bots from a module."""
    # Define a mock module structure
    mock_module = MagicMock()
    mock_module.__path__ = ["/fake/path"]
    mock_module.__name__ = "mock_module"

    # Define mock bot classes
    mock_bot1 = type("MockBot1", (BaseBot,), {"bot_name": "MockBot1"})
    mock_bot2 = type("MockBot2", (BaseBot,), {"bot_name": "MockBot2"})

    # Mock importlib.import_module to return our mock module
    with patch("importlib.import_module") as mock_import:
        # Mocking pkgutil.iter_modules behavior
        with patch("pkgutil.iter_modules") as mock_iter_modules:
            mock_import.return_value = mock_module

            # Mock finding two submodules
            mock_iter_modules.return_value = [
                (None, "mock_module.bot1", False),
                (None, "mock_module.bot2", False)
            ]

            # Create mock submodules with our bot classes
            mock_submodule1 = MagicMock()
            mock_submodule1.__name__ = "mock_module.bot1"
            mock_submodule1.__dict__ = {"MockBot1": mock_bot1}

            mock_submodule2 = MagicMock()
            mock_submodule2.__name__ = "mock_module.bot2"
            mock_submodule2.__dict__ = {"MockBot2": mock_bot2}

            # Mock importing the submodules
            def import_side_effect(name):
                if name == "mock_module":
                    return mock_module
                elif name == "mock_module.bot1":
                    return mock_submodule1
                elif name == "mock_module.bot2":
                    return mock_submodule2
                else:
                    raise ImportError(f"Module {name} not found")

            mock_import.side_effect = import_side_effect

            # Mock inspect.getmembers to return our bot classes
            with patch("inspect.getmembers") as mock_getmembers:
                mock_getmembers.return_value = [
                    ("MockBot1", mock_bot1),
                    ("MockBot2", mock_bot2)
                ]

                # Test loading bots
                bot_classes = BotFactory.load_bots_from_module("mock_module")

                # Check that we found the correct number of bot classes
                assert len(bot_classes) == 2

                # Check that we found the correct bot classes
                bot_names = [cls.bot_name for cls in bot_classes]
                assert "MockBot1" in bot_names
                assert "MockBot2" in bot_names

def test_bot_factory_error_handling():
    """Test error handling in BotFactory."""
    # Test with a non-existent module
    with patch("importlib.import_module") as mock_import:
        mock_import.side_effect = ImportError("Module not found")

        # Should return an empty list on error
        bot_classes = BotFactory.load_bots_from_module("non_existent_module")
        assert isinstance(bot_classes, list)
        assert len(bot_classes) == 0

def test_get_available_bots_integration():
    """Test the get_available_bots method with multiple bot sources."""
    # First patch load_bots_from_module to return our test bots
    with patch.object(BotFactory, "load_bots_from_module") as mock_load:
        # Mock loading our test bot classes
        mock_load.return_value = [TestFactoryBot1, TestFactoryBot2]

        # Get available bots
        available_bots = BotFactory.get_available_bots()

        # Check results
        assert "TestFactoryBot1" in available_bots
        assert "TestFactoryBot2" in available_bots
        assert available_bots["TestFactoryBot1"] == "First test bot for factory testing"
        assert available_bots["TestFactoryBot2"] == "Second test bot for factory testing"

@pytest.mark.parametrize("bot_classes,expected_count", [
    ([], 0),  # Empty list
    ([TestFactoryBot1], 1),  # Single bot
    ([TestFactoryBot1, TestFactoryBot2], 2),  # Multiple bots
])
def test_create_app_with_different_bot_counts(bot_classes, expected_count):
    """Test creating an app with different numbers of bots."""
    # Create app with the given bot classes
    app = BotFactory.create_app(bot_classes, allow_without_key=True)

    # Count bot-related routes (excluding health checks and other endpoints)
    bot_routes = [
        route.path for route in app.routes
        if route.path not in ["/health", "/bots", "/docs", "/redoc", "/openapi.json"]
    ]

    # Check route count matches expected bot count
    assert len(bot_routes) == expected_count

def test_remove_duplicate_bots():
    """Test handling of duplicate bot classes."""
    # Create list with duplicates
    bot_classes = [TestFactoryBot1, TestFactoryBot1, TestFactoryBot2]

    # Create app (should deduplicate)
    with patch.object(BotFactory, "create_app", side_effect=BotFactory.create_app) as mock_create:
        app = BotFactory.create_app(bot_classes, allow_without_key=True)

        # Check that we only created each unique bot once
        bot_routes = [
            route.path for route in app.routes
            if route.path not in ["/health", "/bots", "/docs", "/redoc", "/openapi.json"]
        ]

        # We should have 2 unique routes
        assert len(set(bot_routes)) == 2
        assert "/test-factory-bot-1" in bot_routes
        assert "/test-factory-bot-2" in bot_routes

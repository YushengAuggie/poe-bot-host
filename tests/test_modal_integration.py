"""
Tests for Modal integration with API keys.
"""

import os
from unittest.mock import MagicMock, patch

import modal
import pytest


class TestModalIntegration:
    """Tests for Modal integration."""

    @patch("modal.Secret.from_name")
    def test_api_key_retrieval(self, mock_from_name):
        """Test retrieving API keys using Secret.from_name."""
        # Set up the mock
        mock_secret = MagicMock()
        mock_secret.get.return_value = "mock-api-key-value"
        mock_from_name.return_value = mock_secret

        # Import after mocking to ensure the mock is applied
        from utils.api_keys import get_api_key

        # Set up a test environment with no env vars
        with patch.dict(os.environ, {}, clear=True):
            # Completely replace the get_api_key mock with the actual function
            with patch("utils.api_keys.get_api_key", side_effect=get_api_key):
                # Mock Modal environment
                with patch("modal.is_local", return_value=False):
                    # Test retrieving a key
                    result = get_api_key("TEST_KEY")

                    # Verify Secret.from_name was called with the right key
                    mock_from_name.assert_called_with("TEST_KEY")
                    assert result == "mock-api-key-value"

    @pytest.mark.parametrize(
        "key_name,env_value",
        [
            ("OPENAI_API_KEY", "test-openai-key"),
            ("GOOGLE_API_KEY", "test-google-key"),
            ("CUSTOM_API_KEY", "test-custom-key"),
        ],
    )
    @patch("modal.is_local")
    def test_api_key_environment_priority(self, mock_is_local, key_name, env_value):
        """Test that environment variables take priority over Modal secrets."""
        mock_is_local.return_value = False

        # Import the API key function
        from utils.api_keys import get_api_key

        # Set up environment with the key
        with patch.dict(os.environ, {key_name: env_value}, clear=True):
            # Completely replace the get_api_key mock with the actual function
            with patch("utils.api_keys.get_api_key", side_effect=get_api_key):
                # Set up a mock that would be called if env var wasn't used
                with patch("modal.Secret.from_name") as mock_from_name:
                    # The env var should be used, so Secret.from_name shouldn't be called
                    assert get_api_key(key_name) == env_value
                    mock_from_name.assert_not_called()

    @pytest.mark.parametrize("secret_available", [True, False])
    @patch("modal.is_local")
    @patch.dict(os.environ, {}, clear=True)
    def test_modal_secret_fallback(self, mock_is_local, secret_available):
        """Test Modal secret fallback when environment variable is not set."""
        from utils.api_keys import get_api_key

        # Mock Modal environment
        mock_is_local.return_value = False

        # Mock Secret.from_name behavior based on whether secret is available
        mock_secret = MagicMock()
        mock_secret.get.return_value = "modal-secret-key"

        # Completely replace the get_api_key mock with the actual function
        with patch("utils.api_keys.get_api_key", side_effect=get_api_key):
            with patch("modal.Secret.from_name") as mock_from_name:
                if secret_available:
                    mock_from_name.return_value = mock_secret
                    assert get_api_key("OPENAI_API_KEY") == "modal-secret-key"
                    mock_from_name.assert_called_with("OPENAI_API_KEY")
                else:
                    mock_from_name.side_effect = ValueError("Secret not found")
                    with pytest.raises(ValueError):
                        get_api_key("OPENAI_API_KEY")

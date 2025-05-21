import os
from unittest.mock import MagicMock, patch

import pytest

from utils.api_keys import get_api_key


class TestApiKeys:
    @patch.dict(os.environ, {"TEST_API_KEY": "test-key"}, clear=True)
    def test_get_api_key_from_env(self):
        assert get_api_key("TEST_API_KEY") == "test-key"

    def test_get_api_key_missing(self):
        # Create a clean environment without the key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                get_api_key("TEST_API_KEY")

    @patch("modal.is_local", return_value=False)
    @patch("modal.Secret.from_name")
    def test_get_api_key_from_modal(self, mock_from_name, mock_is_local):
        # Create a mock secret that returns a value when get() is called
        mock_secret = MagicMock()
        mock_secret.get.return_value = "modal-secret-key"
        mock_from_name.return_value = mock_secret

        # Create a clean environment without the key
        with patch.dict(os.environ, {}, clear=True):
            assert get_api_key("TEST_API_KEY") == "modal-secret-key"
            mock_from_name.assert_called_once_with("TEST_API_KEY")

    @patch("modal.is_local", return_value=False)
    @patch("modal.Secret.from_name")
    def test_get_api_key_env_priority(self, mock_from_name, mock_is_local):
        # Test that environment variables take priority over Modal secrets
        mock_secret = MagicMock()
        mock_secret.get.return_value = "modal-secret-key"
        mock_from_name.return_value = mock_secret

        with patch.dict(os.environ, {"TEST_API_KEY": "env-key"}, clear=True):
            assert get_api_key("TEST_API_KEY") == "env-key"
            # Should not try to get the Modal secret
            mock_from_name.assert_not_called()

    @patch("modal.is_local", return_value=False)
    @patch("modal.Secret.from_name")
    def test_get_api_key_modal_error(self, mock_from_name, mock_is_local):
        # Test that we handle errors when trying to get Modal secrets
        mock_from_name.side_effect = ValueError("Secret not found")

        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                get_api_key("TEST_API_KEY")

    def test_get_api_key_examples(self):
        # These tests are primarily to demonstrate example usages
        with patch.dict(os.environ, {"OPENAI_API_KEY": "openai-key"}, clear=True):
            assert get_api_key("OPENAI_API_KEY") == "openai-key"

        with patch.dict(os.environ, {"GOOGLE_API_KEY": "google-key"}, clear=True):
            assert get_api_key("GOOGLE_API_KEY") == "google-key"

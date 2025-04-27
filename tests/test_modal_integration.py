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
    def test_key_secret_application(self, mock_from_name):
        """Test using the KEY secret in a Modal app function."""
        # Since we can't easily test the Modal function object directly,
        # we'll patch the Secret.from_name and verify it's called correctly
        mock_secret = MagicMock()
        mock_from_name.return_value = mock_secret

        # Import the module to trigger the function definition
        from examples.modal_api_key_example import check_key_secret

        # Verify the Secret.from_name was called with KEY
        mock_from_name.assert_any_call("KEY")

    def test_service_secrets_application(self):
        """Test using service-specific secrets in a Modal app function."""
        # Test a different way - inspect the actual code
        # Import the module to access the functions
        import inspect

        from examples.modal_api_key_example import app
        with open("/Users/yding/workspace_quora/poe_bots/examples/modal_api_key_example.py", "r") as f:
            content = f.read()

        # Verify the content contains the expected patterns
        assert "secrets=get_function_secrets([\"openai\", \"google\"])" in content.replace("'", "\"")
        assert "@app.function(secrets=get_function_secrets" in content

    @pytest.mark.parametrize("secret_available", [True, False])
    @patch("modal.is_local")
    @patch.dict(os.environ, {}, clear=True)
    def test_openai_key_retrieval_in_modal(self, mock_is_local, secret_available):
        """Test retrieving OpenAI key in Modal environment."""
        from utils.api_keys import get_openai_api_key

        # Mock Modal environment
        mock_is_local.return_value = False

        # Set up environment as needed
        if secret_available:
            os.environ["OPENAI_API_KEY"] = "test-modal-key"
            assert get_openai_api_key() == "test-modal-key"
        else:
            with pytest.raises(ValueError):
                get_openai_api_key()

    @pytest.mark.parametrize("secret_available", [True, False])
    @patch("modal.is_local")
    @patch.dict(os.environ, {}, clear=True)
    def test_google_key_retrieval_in_modal(self, mock_is_local, secret_available):
        """Test retrieving Google key in Modal environment."""
        from utils.api_keys import get_google_api_key

        # Mock Modal environment
        mock_is_local.return_value = False

        # Set up environment as needed
        if secret_available:
            os.environ["GOOGLE_API_KEY"] = "test-modal-google-key"
            assert get_google_api_key() == "test-modal-google-key"
        else:
            with pytest.raises(ValueError):
                get_google_api_key()

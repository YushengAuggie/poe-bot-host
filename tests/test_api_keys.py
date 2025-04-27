import json
import os
import tempfile
from unittest.mock import MagicMock, patch

import modal
import pytest

from utils.api_keys import (
    get_function_secrets,
    get_google_api_key,
    get_google_credentials,
    get_openai_api_key,
)


class TestApiKeys:
    @patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True)
    def test_get_openai_api_key(self):
        assert get_openai_api_key() == "test-key"

    def test_get_openai_api_key_missing(self):
        # Create a clean environment without the key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                get_openai_api_key()

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-google-key"}, clear=True)
    def test_get_google_api_key(self):
        assert get_google_api_key() == "test-google-key"

    def test_get_google_api_key_missing(self):
        # Create a clean environment without the key
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError):
                get_google_api_key()

    @patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS_JSON": '{"key": "value"}'}, clear=True)
    def test_get_google_credentials_json_string(self):
        assert get_google_credentials() == {"key": "value"}

    def test_get_google_credentials_file_path(self):
        # Create a temporary credentials file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json') as temp_file:
            temp_file.write('{"key": "file-value"}')
            temp_file.flush()

            # Set the environment to use this file
            with patch.dict(os.environ, {"GOOGLE_APPLICATION_CREDENTIALS": temp_file.name}, clear=True):
                assert get_google_credentials() == {"key": "file-value"}

    @patch.dict(os.environ, {"GOOGLE_API_KEY": "test-api-key"}, clear=True)
    def test_get_google_credentials_modal(self):
        assert get_google_credentials() == {"api_key": "test-api-key"}

    @patch.dict(os.environ, {}, clear=True)
    def test_get_google_credentials_missing(self):
        with pytest.raises(ValueError):
            get_google_credentials()

    @patch("modal.is_local")
    @patch.dict(os.environ, {"OPENAI_API_KEY": "modal-openai-key"}, clear=True)
    def test_get_openai_api_key_in_modal(self, mock_is_local):
        mock_is_local.return_value = False
        assert get_openai_api_key() == "modal-openai-key"

    @patch("modal.is_local")
    @patch.dict(os.environ, {"GOOGLE_API_KEY": "modal-google-key"}, clear=True)
    def test_get_google_api_key_in_modal(self, mock_is_local):
        mock_is_local.return_value = False
        assert get_google_api_key() == "modal-google-key"

    @patch("modal.Secret.from_name")
    def test_get_function_secrets_key_secret(self, mock_from_name):
        mock_secret = MagicMock()
        mock_from_name.return_value = mock_secret

        secrets = get_function_secrets(["openai"])

        # Should try to get KEY secret
        mock_from_name.assert_any_call("KEY")
        assert mock_secret in secrets

    @patch("modal.Secret.from_name")
    def test_get_function_secrets_fallback(self, mock_from_name):
        # Mock the from_name method to raise ValueError for the KEY secret
        # but return a valid secret for the service-specific secret
        def side_effect(name):
            if name == "KEY":
                raise ValueError("Secret not found")
            return MagicMock()

        mock_from_name.side_effect = side_effect

        secrets = get_function_secrets(["openai"])

        # Should try to get OPENAI_API_KEY secret
        mock_from_name.assert_any_call("OPENAI_API_KEY")
        assert len(secrets) == 1  # Should have one secret (service-specific)

    @patch("modal.Secret.from_name")
    def test_get_function_secrets_multiple_services(self, mock_from_name):
        mock_secrets = {
            "KEY": MagicMock(name="key_secret"),
            "OPENAI_API_KEY": MagicMock(name="openai_secret"),
            "GOOGLE_API_KEY": MagicMock(name="google_secret")
        }

        def side_effect(name):
            if name in mock_secrets:
                return mock_secrets[name]
            raise ValueError(f"Secret {name} not found")

        mock_from_name.side_effect = side_effect

        secrets = get_function_secrets(["openai", "google"])

        # Should contain the KEY secret and both service secrets
        assert mock_secrets["KEY"] in secrets
        assert mock_secrets["OPENAI_API_KEY"] in secrets
        assert mock_secrets["GOOGLE_API_KEY"] in secrets
        assert len(secrets) == 3

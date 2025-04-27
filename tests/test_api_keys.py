import json
import os
import tempfile
from unittest.mock import patch

import pytest

from utils.api_keys import get_google_api_key, get_google_credentials, get_openai_api_key


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

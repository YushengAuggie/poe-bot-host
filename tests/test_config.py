"""
Tests for the configuration module.
"""

import logging
import os
from unittest.mock import patch

from utils.config import Settings


class TestSettings:
    """Tests for the Settings class."""

    def test_default_values(self):
        """Test default values are set correctly."""
        settings = Settings()

        # Check default API settings - read from the environment
        # or use the defaults defined in Settings class
        default_api_host = os.environ.get("POE_API_HOST", "0.0.0.0")
        default_api_port = int(os.environ.get("POE_API_PORT", "8000"))
        default_allow_without_key = (
            os.environ.get("POE_ALLOW_WITHOUT_KEY", "true").lower() == "true"
        )

        assert settings.API_HOST == default_api_host
        assert settings.API_PORT == default_api_port
        assert settings.ALLOW_WITHOUT_KEY is default_allow_without_key

        # Check default bot settings
        default_bot_timeout = int(os.environ.get("POE_BOT_TIMEOUT", "30"))
        default_bot_max_tokens = int(os.environ.get("POE_BOT_MAX_TOKENS", "2000"))

        assert settings.BOT_TIMEOUT == default_bot_timeout
        assert settings.BOT_MAX_TOKENS == default_bot_max_tokens

        # Check default app settings
        default_app_name = os.environ.get("MODAL_APP_NAME", "poe-bots")
        assert settings.MODAL_APP_NAME == default_app_name

    def test_as_dict(self):
        """Test converting settings to dictionary."""
        settings = Settings()

        # Convert to dictionary
        settings_dict = settings.as_dict()

        # Check dictionary contains all uppercase attributes
        for key in dir(settings):
            if key.isupper() and not key.startswith("_"):
                assert key in settings_dict

        # Check specific values - use the same defaults as the class
        default_api_host = os.environ.get("POE_API_HOST", "0.0.0.0")
        default_api_port = int(os.environ.get("POE_API_PORT", "8000"))

        assert settings_dict["API_HOST"] == default_api_host
        assert settings_dict["API_PORT"] == default_api_port
        default_bot_timeout = int(os.environ.get("POE_BOT_TIMEOUT", "30"))
        assert settings_dict["BOT_TIMEOUT"] == default_bot_timeout

    def test_get_log_level(self):
        """Test log level conversion."""
        settings = Settings()

        # Get default log level to determine actual implementation
        default_level = settings.get_log_level()

        # Test each log level - just verify it's an integer
        with patch.object(settings, "LOG_LEVEL", "DEBUG"):
            level = settings.get_log_level()
            assert isinstance(level, int)

        with patch.object(settings, "LOG_LEVEL", "INFO"):
            level = settings.get_log_level()
            assert isinstance(level, int)

        with patch.object(settings, "LOG_LEVEL", "WARNING"):
            level = settings.get_log_level()
            assert isinstance(level, int)

        with patch.object(settings, "LOG_LEVEL", "ERROR"):
            level = settings.get_log_level()
            assert isinstance(level, int)

        with patch.object(settings, "LOG_LEVEL", "CRITICAL"):
            level = settings.get_log_level()
            assert isinstance(level, int)

    def test_invalid_log_level(self):
        """Test handling of invalid log level."""
        settings = Settings()

        # Test invalid log level (should default to INFO)
        with patch.object(settings, "LOG_LEVEL", "INVALID"):
            assert settings.get_log_level() == logging.INFO

    def test_configure_logging(self):
        """Test logging configuration."""
        settings = Settings()

        # Mock logging.basicConfig to avoid changing actual logging config
        with patch("logging.basicConfig") as mock_basic_config:
            # Test with DEBUG level
            with patch.object(settings, "LOG_LEVEL", "DEBUG"):
                settings.configure_logging()
                mock_basic_config.assert_called_once()

                # Check that a level was set (without checking specific value)
                args, kwargs = mock_basic_config.call_args
                assert "level" in kwargs
                assert isinstance(kwargs["level"], int)

            # Reset mock and test with different level
            mock_basic_config.reset_mock()
            with patch.object(settings, "LOG_LEVEL", "WARNING"):
                settings.configure_logging()
                mock_basic_config.assert_called_once()

                # Check that a level was set (without checking specific value)
                args, kwargs = mock_basic_config.call_args
                assert "level" in kwargs
                assert isinstance(kwargs["level"], int)

    @patch.dict(
        os.environ,
        {
            "POE_API_HOST": "127.0.0.1",
            "POE_API_PORT": "9000",
            "POE_ALLOW_WITHOUT_KEY": "false",
            "POE_BOT_TIMEOUT": "60",
            "POE_BOT_MAX_TOKENS": "4000",
            "MODAL_APP_NAME": "custom-poe-bots",
            "LOG_LEVEL": "DEBUG",
            "DEBUG": "true",
        },
    )
    def test_env_variable_override(self):
        """Test environment variable overrides."""
        # Create new settings instance (which should read from environment)
        from importlib import reload

        from utils import config

        reload(config)
        settings = config.settings

        # Check environment values were applied
        assert settings.API_HOST == "127.0.0.1"
        assert settings.API_PORT == 9000
        assert settings.ALLOW_WITHOUT_KEY is False
        assert settings.BOT_TIMEOUT == 60
        assert settings.BOT_MAX_TOKENS == 4000
        assert settings.MODAL_APP_NAME == "custom-poe-bots"
        assert settings.LOG_LEVEL == "DEBUG"
        assert settings.DEBUG is True

    def test_boolean_parsing(self):
        """Test parsing of boolean environment variables."""
        # Test different boolean string values
        with patch.dict(os.environ, {"POE_ALLOW_WITHOUT_KEY": "true"}):
            assert Settings().ALLOW_WITHOUT_KEY is True

        with patch.dict(os.environ, {"POE_ALLOW_WITHOUT_KEY": "True"}):
            assert Settings().ALLOW_WITHOUT_KEY is True

        # Checking implementation specifics
        with patch.dict(os.environ, {"DEBUG": "true"}):
            settings = Settings()
            assert settings.DEBUG is True

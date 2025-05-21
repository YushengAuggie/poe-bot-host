"""
Configuration module for the Poe Bot Host.

This module provides a centralized configuration system for the framework,
loading settings from environment variables and configuration files.
"""

import logging
import os
from typing import Any, ClassVar, Dict, Mapping, TypedDict, cast

from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG", "").lower() == "true" else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("poe_bots.config")


class SettingsDict(TypedDict):
    """Type definition for settings dictionary."""

    API_HOST: str
    API_PORT: int
    ALLOW_WITHOUT_KEY: bool
    BOT_TIMEOUT: int
    BOT_MAX_TOKENS: int
    LOG_LEVEL: str
    DEBUG: bool
    MODAL_APP_NAME: str


class Settings:
    """Configuration settings for the Poe Bot Host."""

    # API Settings
    API_HOST: ClassVar[str] = os.environ.get("POE_API_HOST", "0.0.0.0")
    API_PORT: ClassVar[int] = int(os.environ.get("POE_API_PORT", "8000"))
    ALLOW_WITHOUT_KEY: ClassVar[bool] = (
        os.environ.get("POE_ALLOW_WITHOUT_KEY", "true").lower() == "true"
    )

    # Bot Settings
    BOT_TIMEOUT: ClassVar[int] = int(os.environ.get("POE_BOT_TIMEOUT", "30"))
    BOT_MAX_TOKENS: ClassVar[int] = int(os.environ.get("POE_BOT_MAX_TOKENS", "2000"))

    # Logging Settings
    LOG_LEVEL: ClassVar[str] = os.environ.get("LOG_LEVEL", "INFO")
    DEBUG: ClassVar[bool] = os.environ.get("DEBUG", "").lower() == "true"

    # Modal Settings
    MODAL_APP_NAME: ClassVar[str] = os.environ.get("MODAL_APP_NAME", "poe-bots")

    @classmethod
    def get_log_level(cls) -> int:
        """Get the log level as an integer value for logging module."""
        levels: Dict[str, int] = {
            "CRITICAL": logging.CRITICAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG,
        }
        return levels.get(cls.LOG_LEVEL.upper(), logging.INFO)

    @classmethod
    def as_dict(cls) -> SettingsDict:
        """Return all settings as a dictionary."""
        result = {
            key: value
            for key, value in cls.__dict__.items()
            if not key.startswith("_") and key.isupper()
        }
        return cast(SettingsDict, result)

    @classmethod
    def configure_logging(cls) -> None:
        """Configure logging based on settings."""
        log_level: int = cls.get_log_level()
        logging.basicConfig(
            level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger.debug(f"Logging configured with level: {cls.LOG_LEVEL}")


# Initialize settings
settings = Settings()

# Configure logging if not already configured
if os.environ.get("CONFIGURE_LOGGING", "true").lower() == "true":
    settings.configure_logging()

logger.debug(f"Loaded configuration: {settings.as_dict()}")

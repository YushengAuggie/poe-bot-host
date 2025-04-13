"""
Configuration module for the Poe Bot Host.

This module provides a centralized configuration system for the framework,
loading settings from environment variables and configuration files.
"""

import os
import logging
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Load environment variables from .env file if present
load_dotenv()

# Configure root logger
logging.basicConfig(
    level=logging.DEBUG if os.environ.get("DEBUG", "").lower() == "true" else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger("poe_bots.config")

class Settings:
    """Configuration settings for the Poe Bot Host."""
    
    # API Settings
    API_HOST: str = os.environ.get("POE_API_HOST", "0.0.0.0")
    API_PORT: int = int(os.environ.get("POE_API_PORT", "8000"))
    ALLOW_WITHOUT_KEY: bool = os.environ.get("POE_ALLOW_WITHOUT_KEY", "true").lower() == "true"
    
    # Bot Settings
    BOT_TIMEOUT: int = int(os.environ.get("POE_BOT_TIMEOUT", "30"))
    BOT_MAX_TOKENS: int = int(os.environ.get("POE_BOT_MAX_TOKENS", "2000"))
    
    # Logging Settings
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")
    DEBUG: bool = os.environ.get("DEBUG", "").lower() == "true"
    
    # Modal Settings
    MODAL_APP_NAME: str = os.environ.get("MODAL_APP_NAME", "poe-bots")
    
    @classmethod
    def get_log_level(cls) -> int:
        """Get the log level as an integer value for logging module."""
        levels = {
            "CRITICAL": logging.CRITICAL,
            "ERROR": logging.ERROR,
            "WARNING": logging.WARNING,
            "INFO": logging.INFO,
            "DEBUG": logging.DEBUG
        }
        return levels.get(cls.LOG_LEVEL.upper(), logging.INFO)
    
    @classmethod
    def as_dict(cls) -> Dict[str, Any]:
        """Return all settings as a dictionary."""
        return {
            key: value for key, value in cls.__dict__.items() 
            if not key.startswith("_") and key.isupper()
        }
    
    @classmethod
    def configure_logging(cls) -> None:
        """Configure logging based on settings."""
        log_level = cls.get_log_level()
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger.debug(f"Logging configured with level: {cls.LOG_LEVEL}")
        
# Initialize settings
settings = Settings()

# Configure logging if not already configured
if os.environ.get("CONFIGURE_LOGGING", "true").lower() == "true":
    settings.configure_logging()
    
logger.debug(f"Loaded configuration: {settings.as_dict()}")
"""
Utility modules for the Poe Bots Framework.

This package contains utilities for creating, managing, and running
bots on the Poe platform.
"""

from utils.base_bot import BaseBot, BotError, BotErrorNoRetry
from utils.bot_factory import BotFactory

# Optional imports (may not be available in all environments)
try:
    from utils.config import settings

    __all__ = ["BaseBot", "BotError", "BotErrorNoRetry", "BotFactory", "settings"]
except ImportError:
    settings = None
    __all__ = ["BaseBot", "BotError", "BotErrorNoRetry", "BotFactory"]

"""
Authentication and API key management utilities.

This module provides clean, testable API key resolution strategies
using configuration-driven patterns instead of hardcoded logic.
"""

import logging
import os
from typing import List, Optional

from .bot_config import get_access_key_patterns

logger = logging.getLogger(__name__)


class APIKeyResolver:
    """Resolves API keys for bots using configurable strategies."""

    def __init__(self, bot_name: str):
        """
        Initialize the resolver for a specific bot.

        Args:
            bot_name: The name of the bot needing API key resolution
        """
        self.bot_name = bot_name
        self.bot_id = bot_name.strip().lower()

    def resolve(self) -> Optional[str]:
        """
        Resolve the API key using configuration-driven patterns.

        Returns:
            The API key if found, None otherwise
        """
        try:
            # Get patterns from configuration
            patterns = get_access_key_patterns(self.bot_name)
            logger.debug(f"Trying {len(patterns)} patterns for bot {self.bot_name}")

            # Try each pattern in order
            for pattern in patterns:
                key = os.environ.get(pattern)
                if key:
                    logger.debug(f"Found API key using pattern: {pattern}")
                    return key
                logger.debug(f"Pattern {pattern} not found in environment")

            # Fallback: fuzzy matching
            return self._fuzzy_match_strategy()

        except Exception as e:
            logger.error(f"Error resolving API key for {self.bot_name}: {e}")
            return None

    def _fuzzy_match_strategy(self) -> Optional[str]:
        """Try fuzzy matching against available environment variables."""
        try:
            # Get all environment variables with ACCESS_KEY
            all_access_keys = [k for k in os.environ.keys() if "ACCESS_KEY" in k]

            if not all_access_keys:
                logger.debug("No ACCESS_KEY environment variables found")
                return None

            # Clean bot name for comparison
            bot_clean = self.bot_id.replace("-", "").replace("_", "").replace(".", "")

            for env_var in all_access_keys:
                env_clean = (
                    env_var.lower().replace("_", "").replace("access", "").replace("key", "")
                )

                # Check for substantial overlap
                if bot_clean in env_clean or env_clean in bot_clean:
                    if len(bot_clean) > 2 and len(env_clean) > 2:  # Avoid false positives
                        logger.debug(f"Fuzzy match found: {env_var} for bot {self.bot_name}")
                        return os.environ.get(env_var)

            logger.debug(f"No fuzzy matches found for bot {self.bot_name}")
            return None

        except Exception as e:
            logger.debug(f"Fuzzy matching failed: {e}")
            return None

    def get_available_keys(self) -> List[str]:
        """
        Get list of available API key environment variables.

        Returns:
            List of environment variable names containing API keys
        """
        return [k for k in os.environ.keys() if "ACCESS_KEY" in k or "API_KEY" in k]

    def get_tried_patterns(self) -> List[str]:
        """
        Get list of patterns that would be tried for this bot.

        Returns:
            List of environment variable patterns
        """
        return get_access_key_patterns(self.bot_name)


def get_bot_access_key(bot_name: str) -> Optional[str]:
    """
    Convenience function to get access key for a bot.

    Args:
        bot_name: The name of the bot

    Returns:
        The access key if found, None otherwise
    """
    resolver = APIKeyResolver(bot_name)
    return resolver.resolve()


def debug_access_key_resolution(bot_name: str) -> dict:
    """
    Debug function to show access key resolution process.

    Args:
        bot_name: The name of the bot

    Returns:
        Dictionary with debug information
    """
    resolver = APIKeyResolver(bot_name)

    patterns = resolver.get_tried_patterns()
    available_keys = resolver.get_available_keys()
    found_key = resolver.resolve()

    return {
        "bot_name": bot_name,
        "patterns_tried": patterns,
        "available_env_keys": available_keys,
        "resolved_key": found_key[:10] + "..." if found_key else None,
        "resolution_successful": found_key is not None,
    }

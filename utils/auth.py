"""
Authentication and API key management utilities.

This module provides clean, testable API key resolution strategies
to replace the complex logic in BaseBot.
"""

import logging
import os
from typing import List, Optional

logger = logging.getLogger(__name__)


class APIKeyResolver:
    """Resolves API keys for bots using multiple strategies."""

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
        Resolve the API key using multiple strategies.

        Returns:
            The API key if found, None otherwise
        """
        strategies = [
            self._exact_match_strategy,
            self._normalized_match_strategy,
            self._fuzzy_match_strategy,
            self._special_case_strategy,
        ]

        for strategy in strategies:
            try:
                key = strategy()
                if key:
                    logger.debug(f"Found API key using {strategy.__name__}")
                    return key
            except Exception as e:
                logger.debug(f"Strategy {strategy.__name__} failed: {e}")

        logger.debug(f"No API key found for bot {self.bot_name}")
        return None

    def _exact_match_strategy(self) -> Optional[str]:
        """Try exact matches for the bot name."""
        candidates = [f"{self.bot_name}_ACCESS_KEY", f"{self.bot_name.upper()}_ACCESS_KEY"]

        return self._try_env_vars(candidates)

    def _normalized_match_strategy(self) -> Optional[str]:
        """Try normalized variations of the bot name."""
        normalized_name = self.bot_name.upper().replace("-", "_")
        candidates = [
            f"{normalized_name}_ACCESS_KEY",
            f"{normalized_name}",
            f"{self.bot_name.replace('-', '_')}_ACCESS_KEY",
            f"{self.bot_name.replace('-', '_').upper()}_ACCESS_KEY",
        ]

        return self._try_env_vars(candidates)

    def _special_case_strategy(self) -> Optional[str]:
        """Handle special naming conventions for specific bots."""
        candidates = []

        # Echo bot variations
        if "echo" in self.bot_id:
            parts = self.bot_id.split("-")
            if len(parts) > 0:
                candidates.append(f"{parts[0].upper()}_ECHOBOT_ACCESS_KEY")

        # Gemini bot variations
        if "gemini" in self.bot_id:
            candidates.extend(self._get_gemini_variants())

        return self._try_env_vars(candidates)

    def _get_gemini_variants(self) -> List[str]:
        """Get Gemini-specific API key variants."""
        candidates = []

        # Extract version information
        version_match = None
        if "2.0" in self.bot_id or "20" in self.bot_id:
            version_match = "2_0"
        elif "2.5" in self.bot_id or "25" in self.bot_id:
            version_match = "2_5"

        if version_match:
            variants = ["FLASH", "PRO", "PRO_EXP"]
            for variant in variants:
                variant_clean = variant.lower().replace("_", "")
                if variant_clean in self.bot_id.replace("-", "").replace("_", ""):
                    candidates.extend(
                        [
                            f"GEMINI_{version_match}_{variant}_ACCESS_KEY",
                            f"GEMINI_{version_match}_{variant}_JY_ACCESS_KEY",
                        ]
                    )

        return candidates

    def _fuzzy_match_strategy(self) -> Optional[str]:
        """Try fuzzy matching against available environment variables."""
        # Get all environment variables with ACCESS_KEY
        all_access_keys = [k for k in os.environ.keys() if "ACCESS_KEY" in k]

        # Clean bot name for comparison
        bot_clean = self.bot_id.replace("-", "").replace("_", "").replace(".", "")

        for env_var in all_access_keys:
            env_clean = env_var.lower().replace("_", "").replace("access", "").replace("key", "")

            # Check for substantial overlap
            if bot_clean in env_clean or env_clean in bot_clean:
                if len(bot_clean) > 2 and len(env_clean) > 2:  # Avoid false positives
                    return os.environ.get(env_var)

        return None

    def _try_env_vars(self, candidates: List[str]) -> Optional[str]:
        """
        Try a list of environment variable candidates.

        Args:
            candidates: List of environment variable names to try

        Returns:
            First matching environment variable value or None
        """
        for candidate in candidates:
            value = os.environ.get(candidate)
            if value:
                return value
        return None

    def get_available_keys(self) -> List[str]:
        """
        Get list of available API key environment variables.

        Returns:
            List of environment variable names containing API keys
        """
        return [k for k in os.environ.keys() if "ACCESS_KEY" in k or "API_KEY" in k]


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

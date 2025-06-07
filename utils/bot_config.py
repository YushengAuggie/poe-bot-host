"""
Bot configuration management for access keys and other settings.

This module provides a flexible, configuration-driven approach to bot settings
instead of hardcoded patterns.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class BotConfig:
    """Configuration manager for bot settings including access key patterns."""

    def __init__(self):
        """Initialize with default configurations."""
        self._config = self._load_default_config()
        self._load_custom_config()

    def _load_default_config(self) -> Dict[str, Any]:
        """Load default bot configuration patterns."""
        return {
            "access_key_patterns": {
                # Global fallbacks
                "global": ["POE_ACCESS_KEY", "GLOBAL_ACCESS_KEY"],
                # Bot-specific patterns
                "GeminiImageGenerationBot": [
                    "GEMINIIMAGEGENERATION_ACCESS_KEY",
                    "GEMINI_IMAGE_GENERATION_ACCESS_KEY",
                    "GEMINI_IMAGE_GEN_ACCESS_KEY",
                    "GEMINI_IMG_GEN_ACCESS_KEY",
                ],
                # Pattern templates for bot families
                "gemini_patterns": [
                    "GEMINI_{variant}_ACCESS_KEY",
                    "GEMINI_{version}_{variant}_ACCESS_KEY",
                    "GEMINI_{version}_{variant}_JY_ACCESS_KEY",
                ],
                # Standard patterns that work for any bot
                "standard_patterns": [
                    "{bot_name}_ACCESS_KEY",
                    "{bot_name_upper}_ACCESS_KEY",
                    "{bot_name_no_bot}_ACCESS_KEY",
                    "{bot_name_no_bot_upper}_ACCESS_KEY",
                    "{bot_name_normalized}_ACCESS_KEY",
                ],
            },
            "bot_metadata": {
                "GeminiImageGenerationBot": {
                    "family": "gemini",
                    "version": "2.0",
                    "variant": "flash",
                    "capabilities": ["image_generation"],
                    "requires_access_key": True,
                }
            },
            "name_transformations": {
                "normalize": {
                    "remove_suffixes": ["Bot", "bot"],
                    "replace_chars": {"-": "_", ".": "_"},
                    "case_variants": ["upper", "lower", "original"],
                }
            },
        }

    def _load_custom_config(self):
        """Load custom configuration from file if it exists."""
        config_file = os.path.join(os.path.dirname(__file__), "bot_config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    custom_config = json.load(f)
                    self._merge_config(custom_config)
                    logger.info(f"Loaded custom bot configuration from {config_file}")
            except Exception as e:
                logger.warning(f"Failed to load custom config: {e}")

    def _merge_config(self, custom_config: Dict[str, Any]):
        """Merge custom configuration with defaults."""
        for key, value in custom_config.items():
            if (
                key in self._config
                and isinstance(value, dict)
                and isinstance(self._config[key], dict)
            ):
                self._config[key].update(value)
            else:
                self._config[key] = value

    def get_access_key_patterns(self, bot_name: str) -> List[str]:
        """Get access key patterns for a specific bot."""
        patterns = []

        # 1. Bot-specific patterns first
        if bot_name in self._config["access_key_patterns"]:
            patterns.extend(self._config["access_key_patterns"][bot_name])

        # 2. Family-specific patterns
        bot_metadata = self._config["bot_metadata"].get(bot_name, {})
        if bot_metadata.get("family") == "gemini":
            patterns.extend(self._expand_gemini_patterns(bot_name, bot_metadata))

        # 3. Standard patterns with name transformations
        patterns.extend(self._expand_standard_patterns(bot_name))

        # 4. Global fallbacks
        patterns.extend(self._config["access_key_patterns"]["global"])

        # Remove duplicates while preserving order
        return list(dict.fromkeys(patterns))

    def _expand_gemini_patterns(self, bot_name: str, metadata: Dict[str, Any]) -> List[str]:
        """Expand Gemini-specific patterns with metadata."""
        patterns = []
        gemini_patterns = self._config["access_key_patterns"]["gemini_patterns"]

        version = metadata.get("version", "").replace(".", "_")
        variant = metadata.get("variant", "").upper()

        for pattern in gemini_patterns:
            try:
                expanded = pattern.format(version=version, variant=variant, bot_name=bot_name)
                patterns.append(expanded)
            except KeyError:
                # Skip patterns that can't be expanded
                continue

        return patterns

    def _expand_standard_patterns(self, bot_name: str) -> List[str]:
        """Expand standard patterns with name transformations."""
        patterns = []
        standard_patterns = self._config["access_key_patterns"]["standard_patterns"]

        # Generate name variants
        name_variants = self._generate_name_variants(bot_name)

        for pattern in standard_patterns:
            for variant_key, variant_value in name_variants.items():
                try:
                    expanded = pattern.format(**{variant_key: variant_value})
                    patterns.append(expanded)
                except KeyError:
                    continue

        return patterns

    def _generate_name_variants(self, bot_name: str) -> Dict[str, str]:
        """Generate name variants for pattern expansion."""
        transforms = self._config["name_transformations"]["normalize"]

        variants = {
            "bot_name": bot_name,
            "bot_name_upper": bot_name.upper(),
            "bot_name_lower": bot_name.lower(),
        }

        # Remove suffixes
        name_no_bot = bot_name
        for suffix in transforms["remove_suffixes"]:
            if name_no_bot.endswith(suffix):
                name_no_bot = name_no_bot[: -len(suffix)]

        variants.update(
            {
                "bot_name_no_bot": name_no_bot,
                "bot_name_no_bot_upper": name_no_bot.upper(),
                "bot_name_no_bot_lower": name_no_bot.lower(),
            }
        )

        # Apply character replacements
        normalized = bot_name
        for old_char, new_char in transforms["replace_chars"].items():
            normalized = normalized.replace(old_char, new_char)

        variants.update(
            {
                "bot_name_normalized": normalized,
                "bot_name_normalized_upper": normalized.upper(),
                "bot_name_normalized_lower": normalized.lower(),
            }
        )

        return variants

    def get_bot_metadata(self, bot_name: str) -> Dict[str, Any]:
        """Get metadata for a specific bot."""
        return self._config["bot_metadata"].get(bot_name, {})

    def requires_access_key(self, bot_name: str) -> bool:
        """Check if a bot requires an access key."""
        metadata = self.get_bot_metadata(bot_name)
        return metadata.get("requires_access_key", False)

    def add_bot_config(self, bot_name: str, config: Dict[str, Any]):
        """Add or update configuration for a specific bot."""
        if "access_key_patterns" in config:
            self._config["access_key_patterns"][bot_name] = config["access_key_patterns"]

        if "metadata" in config:
            self._config["bot_metadata"][bot_name] = config["metadata"]

    def save_custom_config(self, config_file: Optional[str] = None):
        """Save current configuration to a file."""
        if config_file is None:
            config_file = os.path.join(os.path.dirname(__file__), "bot_config.json")

        try:
            with open(config_file, "w") as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Saved bot configuration to {config_file}")
        except Exception as e:
            logger.error(f"Failed to save config: {e}")


# Global instance
_bot_config = None


def get_bot_config() -> BotConfig:
    """Get the global bot configuration instance."""
    global _bot_config
    if _bot_config is None:
        _bot_config = BotConfig()
    return _bot_config


def get_access_key_patterns(bot_name: str) -> List[str]:
    """Convenience function to get access key patterns for a bot."""
    return get_bot_config().get_access_key_patterns(bot_name)


def get_bot_metadata(bot_name: str) -> Dict[str, Any]:
    """Convenience function to get bot metadata."""
    return get_bot_config().get_bot_metadata(bot_name)


def requires_access_key(bot_name: str) -> bool:
    """Convenience function to check if bot requires access key."""
    return get_bot_config().requires_access_key(bot_name)

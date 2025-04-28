#!/usr/bin/env python
"""
Script to manually sync bot settings with Poe.

This script synchronizes the bot settings (such as attachment support) with the Poe platform.
Use this when changing settings and you need to update Poe without redeploying.

Authentication:
    This script uses per-bot access keys which can be obtained from your bot's settings page.

    Set the environment variable following one of these formats:
    - BOT_NAME_ACCESS_KEY
    - BOTNAME_ACCESS_KEY
    - BOT_NAME with underscores instead of hyphens

    Example: For a bot named "my-cool-bot"
    - MY_COOL_BOT_ACCESS_KEY=your_access_key
    - MYCOOLBOT_ACCESS_KEY=your_access_key

    Special formats for specific bots:
    - For JY-EchoBot: JY_ECHOBOT_ACCESS_KEY
    - For Gemini bots: GEMINI_X_Y_Z_ACCESS_KEY where X_Y_Z is the version
      (e.g., GEMINI_2_5_FLASH_JY_ACCESS_KEY for Gemini-2.5-Flash-JY bot)

Usage:
    python sync_bot_settings.py --bot BOT_NAME
    python sync_bot_settings.py --all
    python sync_bot_settings.py --bot BOT_NAME -v  # Verbose mode
"""

import argparse
import asyncio
import logging
import os
from typing import Any, Dict, Optional

import httpx
from fastapi_poe import sync_bot_settings as fp_sync_bot_settings
from fastapi_poe.types import SettingsRequest

from utils.base_bot import BaseBot
from utils.bot_factory import BotFactory

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def get_bot_settings(bot: BaseBot) -> Dict[str, Any]:
    """Get bot settings from the bot's get_settings method."""
    try:
        settings_request = SettingsRequest(version="1", type="settings")
        settings_response = await bot.get_settings(settings_request)

        # Convert to dict for easier inspection
        if hasattr(settings_response, "dict"):
            return settings_response.dict()
        else:
            return vars(settings_response)
    except Exception as e:
        # Get bot_name safely (may be on class or instance)
        bot_name = getattr(bot, "bot_name", None)
        if bot_name is None:
            bot_name = getattr(bot.__class__, "bot_name", bot.__class__.__name__)

        logger.error(f"Error getting settings for {bot_name}: {e}")
        return {}


def get_bot_access_key(bot_name: str) -> Optional[str]:
    """
    Get the access key for a bot from various possible environment variables.

    Args:
        bot_name: The name of the bot

    Returns:
        The access key if found, None otherwise
    """
    # Log all available access keys in environment for debugging
    all_access_keys = [k for k in os.environ.keys() if "ACCESS_KEY" in k]
    logger.debug(f"All environment variables with ACCESS_KEY: {all_access_keys}")

    # Try standardized bot naming schemes in order of preference
    potential_env_vars = []

    # Bot ID format (provided in --bot parameter)
    bot_id = bot_name.strip().lower()

    # 1. Exact case match with dashes (matches original bot name)
    potential_env_vars.append(f"{bot_name}_ACCESS_KEY")

    # 2. Uppercase with dashes (common env var style)
    potential_env_vars.append(f"{bot_name.upper()}_ACCESS_KEY")

    # 3. Uppercase with underscores instead of dashes (most common env var style)
    potential_env_vars.append(f"{bot_name.upper().replace('-', '_')}_ACCESS_KEY")

    # 4. Original name but with underscores
    potential_env_vars.append(f"{bot_name.replace('-', '_')}_ACCESS_KEY")

    # 5. Original name uppercased with underscores
    potential_env_vars.append(f"{bot_name.replace('-', '_').upper()}_ACCESS_KEY")

    # 6. Just the original name uppercased (without _ACCESS_KEY suffix)
    potential_env_vars.append(f"{bot_name.upper()}")

    # 7. No dashes or underscores, everything concatenated
    no_special_chars = "".join(c for c in bot_name if c.isalnum()).upper()
    potential_env_vars.append(f"{no_special_chars}_ACCESS_KEY")

    # 8. Special case for common bot naming conventions
    if "echobot" in bot_id:
        parts = bot_id.split("-")
        if len(parts) > 0:
            potential_env_vars.append(f"{parts[0].upper()}_ECHOBOT_ACCESS_KEY")

    if "gemini" in bot_id:
        # Try GEMINI_X_Y_FORMAT
        version_match = None
        if "2.0" in bot_id or "20" in bot_id:
            version_match = "2_0"
        elif "2.5" in bot_id or "25" in bot_id:
            version_match = "2_5"

        if version_match:
            for variant in ["FLASH", "PRO", "PRO_EXP"]:
                if variant.lower().replace("_", "") in bot_id.replace("-", "").replace("_", ""):
                    potential_env_vars.append(f"GEMINI_{version_match}_{variant}_ACCESS_KEY")
                    potential_env_vars.append(f"GEMINI_{version_match}_{variant}_JY_ACCESS_KEY")

    # Try to find fuzzy matches by removing special chars and comparing
    bot_clean = bot_id.replace("-", "").replace("_", "").replace(".", "")
    for env_var in all_access_keys:
        env_clean = env_var.lower().replace("_", "").replace("access", "").replace("key", "")
        # If the bot name is a significant part of the env var or vice versa
        if bot_clean in env_clean or env_clean in bot_clean:
            potential_env_vars.append(env_var)

    # Remove duplicates while preserving order
    seen = set()
    unique_env_vars = []
    for env_var in potential_env_vars:
        if env_var not in seen:
            seen.add(env_var)
            unique_env_vars.append(env_var)

    # Try each possible environment variable format
    logger.debug(f"Checking these environment variables for {bot_name}: {unique_env_vars}")
    for env_var in unique_env_vars:
        if env_var in os.environ:
            logger.debug(f"Found access key using environment variable: {env_var}")
            return os.environ[env_var]

    # If no match found via specific environment variables,
    # check if any environment variable contains the bot name
    for env_var in all_access_keys:
        if bot_id in env_var.lower():
            logger.debug(f"Found potential match by bot name in environment variable: {env_var}")
            return os.environ[env_var]

    # No match found
    return None


async def sync_bot_via_fastapi_poe(bot_name: str) -> bool:
    """Sync a bot's settings using fastapi-poe's sync_bot_settings.

    Args:
        bot_name: Name of the bot to sync

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create a bot instance to use its get_access_key method if possible
        try:
            # Try to find the bot class matching the name
            bot_classes = BotFactory.load_bots_from_module("bots")
            bot_instance = None

            for bot_class in bot_classes:
                # Check class name or bot_name attribute
                class_bot_name = getattr(bot_class, "bot_name", bot_class.__name__)
                if class_bot_name.lower() == bot_name.lower():
                    bot_instance = bot_class()
                    break

            # If we found a matching bot instance, try to get the access key from it
            if bot_instance and hasattr(bot_instance, "get_access_key"):
                access_key = bot_instance.get_access_key()
                if access_key:
                    logger.debug(f"Found access key for {bot_name} from bot instance")
                else:
                    # Fall back to the standalone function
                    access_key = get_bot_access_key(bot_name)
            else:
                # Use the standalone function if we couldn't create a bot instance
                access_key = get_bot_access_key(bot_name)
        except Exception as e:
            logger.debug(
                f"Error attempting to create bot instance: {e}. Using standalone key lookup."
            )
            # Fall back to the standalone function
            access_key = get_bot_access_key(bot_name)

        if not access_key:
            logger.error(f"No access key found for bot {bot_name}")
            logger.error(
                f"Please set the {bot_name.upper()}_ACCESS_KEY environment variable (or appropriate format)."
            )
            return False

        # Check if the function exists
        key_preview = access_key[:4] + "..." if access_key else "None"
        logger.info(
            f"Syncing settings for {bot_name} using fastapi_poe.sync_bot_settings() with key: {key_preview}"
        )
        fp_sync_bot_settings(bot_name=bot_name, access_key=access_key)
        return True
    except Exception as e:
        logger.error(f"Error syncing bot {bot_name} via fastapi_poe: {e}")
        return False


async def sync_bot_via_http(bot_name: str) -> bool:
    """Sync a bot's settings by making HTTP requests to the Poe API.

    Args:
        bot_name: Name of the bot to sync

    Returns:
        True if successful, False otherwise
    """
    try:
        # Create a bot instance to use its get_access_key method if possible
        try:
            # Try to find the bot class matching the name
            bot_classes = BotFactory.load_bots_from_module("bots")
            bot_instance = None

            for bot_class in bot_classes:
                # Check class name or bot_name attribute
                class_bot_name = getattr(bot_class, "bot_name", bot_class.__name__)
                if class_bot_name.lower() == bot_name.lower():
                    bot_instance = bot_class()
                    break

            # If we found a matching bot instance, try to get the access key from it
            if bot_instance and hasattr(bot_instance, "get_access_key"):
                access_key = bot_instance.get_access_key()
                if access_key:
                    logger.debug(f"Found access key for {bot_name} from bot instance")
                else:
                    # Fall back to the standalone function
                    access_key = get_bot_access_key(bot_name)
            else:
                # Use the standalone function if we couldn't create a bot instance
                access_key = get_bot_access_key(bot_name)
        except Exception as e:
            logger.debug(
                f"Error attempting to create bot instance: {e}. Using standalone key lookup."
            )
            # Fall back to the standalone function
            access_key = get_bot_access_key(bot_name)

        if not access_key:
            logger.error(f"No access key found for bot {bot_name}")
            logger.error(
                f"Please set the {bot_name.upper()}_ACCESS_KEY environment variable (or appropriate format)."
            )
            return False

        # URL for the settings endpoint
        base_url = "https://api.poe.com"
        settings_url = f"{base_url}/bot/fetch_settings/{bot_name}/{access_key}"

        key_preview = access_key[:4] + "..." if access_key else "None"
        logger.info(f"Syncing settings for {bot_name} using HTTP request with key: {key_preview}")

        # Make request to the settings endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(
                settings_url,
                timeout=30,
            )

            if response.status_code == 200:
                logger.info(f"Successfully synced settings for {bot_name}")
                return True
            else:
                logger.error(
                    f"Failed to sync settings for {bot_name}: {response.status_code} {response.text}"
                )
                return False
    except Exception as e:
        logger.error(f"Error syncing bot {bot_name} via HTTP: {e}")
        return False


async def sync_single_bot(bot_name: str) -> bool:
    """Sync settings for a single bot using multiple methods.

    Args:
        bot_name: Name of the bot to sync

    Returns:
        True if any method succeeded, False otherwise
    """
    # First try using fastapi_poe.sync_bot_settings()
    success = await sync_bot_via_fastapi_poe(bot_name)

    # If that fails, try the HTTP method
    if not success:
        success = await sync_bot_via_http(bot_name)

    return success


async def sync_all_bots() -> Dict[str, bool]:
    """Sync settings for all available bots.

    Returns:
        Dictionary mapping bot names to success status
    """
    bot_classes = BotFactory.load_bots_from_module("bots")

    results = {}
    for bot_class in bot_classes:
        try:
            # Create instance and get bot name
            bot = bot_class()

            # Check for bot_name in both instance and class
            if hasattr(bot, "bot_name") and bot.bot_name and bot.bot_name != "BaseBot":
                bot_name = bot.bot_name
            elif (
                hasattr(bot_class, "bot_name")
                and bot_class.bot_name
                and bot_class.bot_name != "BaseBot"
            ):
                bot_name = bot_class.bot_name
            else:
                logger.warning(f"Skipping {bot_class.__name__}: No bot_name attribute")
                continue

            # Get current settings to log
            bot_settings = await get_bot_settings(bot)
            allows_attachments = bot_settings.get("allow_attachments", False)
            logger.info(f"Bot {bot_name} current settings: allow_attachments={allows_attachments}")

            # Attempt to get access key directly from bot instance
            # This uses the get_access_key method added to BaseBot
            access_key = bot.get_access_key() if hasattr(bot, "get_access_key") else None

            if access_key:
                logger.info(f"Found access key for {bot_name} from bot instance")
                # Sync with the access key from the bot instance
                key_preview = access_key[:4] + "..." if access_key else "None"
                logger.info(f"Syncing settings for {bot_name} with key: {key_preview}")
                try:
                    # Try using fastapi_poe with the direct access key
                    fp_sync_bot_settings(bot_name=bot_name, access_key=access_key)
                    results[bot_name] = True
                    continue
                except Exception as e:
                    logger.warning(f"Failed to sync with fastapi_poe, trying HTTP method: {e}")
                    try:
                        # URL for the settings endpoint
                        base_url = "https://api.poe.com"
                        settings_url = f"{base_url}/bot/fetch_settings/{bot_name}/{access_key}"

                        # Make request to the settings endpoint
                        async with httpx.AsyncClient() as client:
                            response = await client.post(
                                settings_url,
                                timeout=30,
                            )

                            if response.status_code == 200:
                                logger.info(f"Successfully synced settings for {bot_name}")
                                results[bot_name] = True
                                continue
                    except Exception as http_e:
                        logger.warning(f"HTTP method also failed: {http_e}")

            # Fallback to the old approach if direct key retrieval fails
            logger.debug(f"Falling back to old sync method for {bot_name}")
            success = await sync_single_bot(bot_name)
            results[bot_name] = success

        except Exception as e:
            logger.error(f"Error processing {bot_class.__name__}: {e}")
            results[bot_class.__name__] = False

    return results


async def main_async():
    """Async main function."""
    parser = argparse.ArgumentParser(description="Sync bot settings with Poe")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Sync all available bots")
    group.add_argument("--bot", type=str, help="Name of the bot to sync")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set log level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Sync bots
    if args.all:
        logger.info("Syncing all bots...")
        logger.info(
            "Note: Each bot requires its own specific access key set as an environment variable."
        )
        results = await sync_all_bots()

        # Report results
        success_count = sum(1 for success in results.values() if success)
        logger.info(f"Synced {success_count}/{len(results)} bots successfully")

        for bot_name, success in results.items():
            status = "✅ Success" if success else "❌ Failed"
            logger.info(f"{bot_name}: {status}")

    else:
        logger.info(f"Syncing bot {args.bot}...")
        success = await sync_single_bot(args.bot)
        status = "✅ Success" if success else "❌ Failed"
        logger.info(f"{args.bot}: {status}")

        if not success:
            logger.error(
                f"Make sure you have set the access key for {args.bot} as an environment variable."
            )
            logger.error(f"Example: export {args.bot.upper()}_ACCESS_KEY=your_access_key")


def main():
    """Main function to parse args and run async main."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

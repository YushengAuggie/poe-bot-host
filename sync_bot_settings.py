#!/usr/bin/env python
"""
Script to manually sync bot settings with Poe.

This script synchronizes the bot settings (such as attachment support) with the Poe platform.
Use this when changing settings and you need to update Poe without redeploying.

Requirements:
    - POE_ACCESS_KEY environment variable must be set. Get this from https://poe.com/api_key
    - POE_API_KEY environment variable must be set. Get this from https://poe.com/api_key

Usage:
    python sync_bot_settings.py --bot BOT_NAME
    python sync_bot_settings.py --all
"""

import argparse
import asyncio
import logging
import os
from typing import Any, Dict

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
        logger.error(f"Error getting settings for {bot.bot_name}: {e}")
        return {}


async def sync_bot_via_fastapi_poe(bot_name: str) -> bool:
    """Sync a bot's settings using fastapi-poe's sync_bot_settings.

    Args:
        bot_name: Name of the bot to sync

    Returns:
        True if successful, False otherwise
    """
    try:
        # We need an access key to sync settings
        access_key = os.environ.get("POE_ACCESS_KEY")
        if not access_key:
            # Try a bot-specific key
            access_key = os.environ.get(f"{bot_name.upper()}_ACCESS_KEY")

        if not access_key:
            logger.error(f"No access key found for bot {bot_name}")
            logger.error("Please set the POE_ACCESS_KEY environment variable. See API_KEY_MANAGEMENT.md for details.")
            return False

        # Check if the function exists
        logger.info(f"Syncing settings for {bot_name} using fastapi_poe.sync_bot_settings()")
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
        # We need an access key to sync settings
        access_key = os.environ.get("POE_ACCESS_KEY")
        if not access_key:
            # Try a bot-specific key
            access_key = os.environ.get(f"{bot_name.upper()}_ACCESS_KEY")

        if not access_key:
            logger.error(f"No access key found for bot {bot_name}")
            logger.error("Please set the POE_ACCESS_KEY environment variable. See API_KEY_MANAGEMENT.md for details.")
            return False

        # URL for the settings endpoint
        base_url = "https://api.poe.com"
        settings_url = f"{base_url}/bot/fetch_settings/{bot_name}/{access_key}"

        logger.info(f"Syncing settings for {bot_name} using HTTP request")

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
                logger.error(f"Failed to sync settings for {bot_name}: {response.status_code} {response.text}")
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
            bot_name = getattr(bot, "bot_name", None)

            if not bot_name:
                logger.warning(f"Skipping {bot_class.__name__}: No bot_name attribute")
                continue

            # Get current settings to log
            bot_settings = await get_bot_settings(bot)
            allows_attachments = bot_settings.get("allow_attachments", False)
            logger.info(f"Bot {bot_name} current settings: allow_attachments={allows_attachments}")

            # Sync settings
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

    # Check if POE_ACCESS_KEY is set
    access_key = os.environ.get("POE_ACCESS_KEY")
    if not access_key:
        logger.error("POE_ACCESS_KEY environment variable is not set.")
        logger.error("Please set the POE_ACCESS_KEY environment variable. See API_KEY_MANAGEMENT.md for details.")
        return

    # Sync bots
    if args.all:
        logger.info("Syncing all bots...")
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


def main():
    """Main function to parse args and run async main."""
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

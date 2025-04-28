#!/usr/bin/env python
"""
Script to manually sync bot settings with Poe.

This script synchronizes the bot settings (such as attachment support) with the Poe platform.
Use this when changing settings and you need to update Poe without redeploying.

Usage:
    python sync_bot_settings.py --bot-name BOT_NAME --access-key ACCESS_KEY

Or sync all bots:
    python sync_bot_settings.py --all
"""

import argparse
import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

# Import local bot modules
from utils.base_bot import BaseBot
from utils.bot_factory import BotFactory

import fastapi_poe as fp
from fastapi_poe.types import SettingsRequest

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


async def get_bot_settings(bot: BaseBot) -> Dict[str, Any]:
    """Get bot settings from the bot's get_settings method."""
    try:
        settings_response = await bot.get_settings(
            SettingsRequest(version=fp.client.PROTOCOL_VERSION, type="settings")
        )
        return settings_response.dict()
    except Exception as e:
        logger.error(f"Error getting settings for {bot.bot_name}: {e}")
        return {}


def sync_single_bot(bot_name: str, access_key: str) -> bool:
    """Sync settings for a single bot."""
    try:
        # Try the direct sync method first
        fp.sync_bot_settings(bot_name=bot_name, access_key=access_key)
        logger.info(f"✅ Successfully synced settings for {bot_name}")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to sync settings for {bot_name}: {e}")
        # Try alternative curl method
        try:
            import requests
            response = requests.post(f"https://api.poe.com/bot/fetch_settings/{bot_name}/{access_key}")
            if response.status_code == 200:
                logger.info(f"✅ Successfully synced settings for {bot_name} using alternative method")
                return True
            else:
                logger.error(f"❌ Alternative sync method failed with status {response.status_code}: {response.text}")
        except Exception as curl_e:
            logger.error(f"❌ Alternative sync method also failed: {curl_e}")
        return False


def sync_all_bots() -> List[str]:
    """Sync settings for all available bots that have access keys."""
    bot_factory = BotFactory()
    bot_classes = bot_factory.discover_bot_classes()

    # Create a bot instance for each class to get settings
    successful_bots = []
    for bot_class in bot_classes:
        try:
            # Create instance
            bot = bot_class()
            bot_name = bot.bot_name if hasattr(bot, "bot_name") else None

            # Try to get access key from environment
            access_key = os.environ.get(f"{bot_name.upper()}_ACCESS_KEY")
            if not access_key:
                # Try a general access key
                access_key = os.environ.get("POE_ACCESS_KEY")

            # Skip if we don't have both bot name and access key
            if not bot_name or not access_key:
                logger.warning(f"Skipping {bot_class.__name__}: Missing bot_name or access_key")
                continue

            # Sync settings
            if sync_single_bot(bot_name, access_key):
                successful_bots.append(bot_name)
        except Exception as e:
            logger.error(f"Error syncing {bot_class.__name__}: {e}")

    return successful_bots


def main():
    """Main function to parse arguments and sync bot settings."""
    parser = argparse.ArgumentParser(description="Sync bot settings with Poe")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Sync all available bots")
    group.add_argument("--bot-name", type=str, help="Name of the bot to sync")
    parser.add_argument("--access-key", type=str, help="Access key for the bot")

    args = parser.parse_args()

    if args.all:
        successful_bots = sync_all_bots()
        if successful_bots:
            logger.info(f"Successfully synced {len(successful_bots)} bots: {', '.join(successful_bots)}")
        else:
            logger.error("Failed to sync any bots")
    else:
        if not args.access_key:
            # Try to get from environment
            access_key = os.environ.get(f"{args.bot_name.upper()}_ACCESS_KEY")
            if not access_key:
                access_key = os.environ.get("POE_ACCESS_KEY")

            if not access_key:
                parser.error("access-key is required when using --bot-name")
        else:
            access_key = args.access_key

        sync_single_bot(args.bot_name, access_key)


if __name__ == "__main__":
    main()

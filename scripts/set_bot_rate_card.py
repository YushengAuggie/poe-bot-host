#!/usr/bin/env python3
"""
Script to set the rate card for Poe bots.

This script provides a reusable way to set rate cards for any bot in the project.
It uses the Poe API via fastapi-poe to update bot settings.

Usage:
    python set_bot_rate_card.py <bot_name> <points_per_message>

Example:
    python set_bot_rate_card.py YouTubeDownloaderBot 300
"""
import argparse
import logging
import os
import sys
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_bot_rate_card(bot_name: str, points: int, access_key: Optional[str] = None) -> bool:
    """Set the rate card for a Poe bot.

    Args:
        bot_name: Name of the bot to update
        points: Points per message for the rate card
        access_key: Bot's access key (if not provided, will look in environment variables)

    Returns:
        True if successful, False otherwise
    """
    try:
        # If no access key provided, try to get it from environment variables
        if not access_key:
            # Try different formats of environment variable names
            env_vars = [
                f"{bot_name.upper()}_ACCESS_KEY",
                f"{bot_name.upper().replace('-', '_')}_ACCESS_KEY",
                f"{bot_name.replace('-', '_').upper()}_ACCESS_KEY",
            ]

            for env_var in env_vars:
                if env_var in os.environ:
                    access_key = os.environ[env_var]
                    logger.debug(f"Found access key using environment variable: {env_var}")
                    break

        if not access_key:
            logger.error(f"No access key found for {bot_name}")
            logger.error(f"Please set the {bot_name.upper()}_ACCESS_KEY environment variable")
            return False

        logger.info(f"Setting rate card for {bot_name} to {points} points per message")

        # Import here to avoid issues if the package is not installed
        try:
            from fastapi_poe.client import set_bot_settings

            # Create settings dictionary
            settings_dict = {
                "rate_card": {"api_calling_cost": points, "api_pricing_type": "per_message"}
            }

            # Set bot settings using the Poe API
            set_bot_settings(bot_name=bot_name, access_key=access_key, settings=settings_dict)
            logger.info(f"Successfully set rate card for {bot_name}")
            return True

        except ImportError:
            # Fallback to the older method
            logger.warning("Could not import set_bot_settings, trying sync_bot_settings")
            from fastapi_poe.client import sync_bot_settings

            # Sync bot settings, but this may not properly set the rate card
            # This will be fixed in a future version of fastapi-poe
            sync_bot_settings(bot_name=bot_name, access_key=access_key)

            # Additionally, try to make a direct API call to ensure rate card is set
            import json

            import requests

            url = "https://api.poe.com/bot/settings"
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_key}"}

            data = {
                "bot_name": bot_name,
                "rate_card": {"api_calling_cost": points, "api_pricing_type": "per_message"},
            }

            response = requests.post(url, headers=headers, data=json.dumps(data))
            if response.status_code == 200:
                logger.info(f"Successfully set rate card for {bot_name} via direct API call")
                return True
            else:
                logger.warning(f"HTTP {response.status_code} from API: {response.text}")
                logger.warning("Rate card may not have been set correctly")
                return False

    except Exception as e:
        logger.error(f"Failed to set rate card: {str(e)}", exc_info=True)
        return False


def main():
    """Parse command line arguments and call the appropriate function."""
    parser = argparse.ArgumentParser(description="Set rate card for a Poe bot")
    parser.add_argument("bot_name", help="Name of the bot to update")
    parser.add_argument("points", type=int, help="Points per message for the rate card")
    parser.add_argument(
        "--access-key",
        help="Bot's access key (if not provided, will look in environment variables)",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    # Set log level based on verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG)

    success = set_bot_rate_card(args.bot_name, args.points, args.access_key)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main()

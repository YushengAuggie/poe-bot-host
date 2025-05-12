#!/usr/bin/env python3
"""
Script to update bot settings via Poe API.

This script provides a robust way to update various bot settings including
rate cards, attachment support, and other configuration options.

Usage:
    python update_bot_settings.py <bot_name> --rate-card <points> [--allow-attachments] [--access-key <key>]

Example:
    python update_bot_settings.py YouTubeDownloaderBot --rate-card 300 --no-attachments
"""
import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def update_bot_settings(
    bot_name: str, settings: Dict[str, Any], access_key: Optional[str] = None
) -> bool:
    """Update settings for a Poe bot via direct API calls.

    Args:
        bot_name: Name of the bot to update
        settings: Dictionary of settings to update
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
                bot_name.upper(),
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

        # Prepare the request
        url = "https://api.poe.com/bot/settings"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_key}"}

        # Add bot_name to settings
        full_settings = settings.copy()
        full_settings["bot_name"] = bot_name

        # Log what we're doing (without revealing the full access key)
        key_preview = access_key[:4] + "..." if access_key else "None"
        logger.info(f"Updating settings for {bot_name} with key: {key_preview}")
        logger.debug(f"Settings: {json.dumps(full_settings, indent=2)}")

        # Make the API request
        response = requests.post(url, headers=headers, data=json.dumps(full_settings))

        # Check response
        if response.status_code == 200:
            logger.info(f"Successfully updated settings for {bot_name}")
            return True
        else:
            logger.error(f"Failed to update settings: HTTP {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False

    except Exception as e:
        logger.error(f"Error updating bot settings: {str(e)}", exc_info=True)
        return False


def main():
    """Parse command line arguments and update bot settings."""
    parser = argparse.ArgumentParser(description="Update settings for a Poe bot")
    parser.add_argument("bot_name", help="Name of the bot to update")
    parser.add_argument("--rate-card", type=int, help="Points per message for the rate card")
    parser.add_argument(
        "--allow-attachments",
        action="store_true",
        help="Enable file attachments (default is whatever the bot already has)",
    )
    parser.add_argument("--no-attachments", action="store_true", help="Disable file attachments")
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

    # Build settings dictionary
    settings = {}

    # Add rate card if specified
    if args.rate_card is not None:
        settings["rate_card"] = {
            "api_calling_cost": args.rate_card,
            "api_pricing_type": "per_message",
        }
        logger.info(f"Setting rate card to {args.rate_card} points per message")

    # Add attachment settings if specified
    if args.allow_attachments and args.no_attachments:
        logger.error("Cannot specify both --allow-attachments and --no-attachments")
        sys.exit(1)
    elif args.allow_attachments:
        settings["allow_attachments"] = True
        logger.info("Enabling file attachments")
    elif args.no_attachments:
        settings["allow_attachments"] = False
        logger.info("Disabling file attachments")

    # Make sure we have at least one setting to update
    if not settings:
        logger.error(
            "No settings specified. Use --rate-card, --allow-attachments, or --no-attachments"
        )
        parser.print_help()
        sys.exit(1)

    # Update the settings
    success = update_bot_settings(args.bot_name, settings, args.access_key)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    main()

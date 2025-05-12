#!/usr/bin/env python3
"""
Script to update settings for the YouTube Downloader Bot.

This is a convenience wrapper around the update_bot_settings.py script
specifically configured for the YouTube Downloader Bot.

Usage:
    python update_youtube_bot_settings.py [--rate-card <points>]

Example:
    python update_youtube_bot_settings.py --rate-card 300
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent)
sys.path.append(parent_dir)
from update_bot_settings import update_bot_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def update_youtube_bot_settings(rate_card=300, access_key=None):
    """Update settings for the YouTube Downloader Bot."""
    # Bot configuration
    bot_name = "YouTubeDownloaderBot"

    # Get access key from environment if not provided
    if not access_key:
        access_key = os.environ.get("YOUTUBEDOWNLOADERBOT_ACCESS_KEY")

    # Build settings
    settings = {
        "allow_attachments": False,  # Always disable file uploads for this bot
        "rate_card": {"api_calling_cost": rate_card, "api_pricing_type": "per_message"},
    }

    # Update the settings
    return update_bot_settings(bot_name, settings, access_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Update YouTube Downloader Bot settings")
    parser.add_argument(
        "--rate-card",
        type=int,
        default=300,
        help="Points per message for the rate card (default: 300)",
    )
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

    success = update_youtube_bot_settings(args.rate_card, args.access_key)
    sys.exit(0 if success else 1)

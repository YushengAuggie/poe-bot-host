#!/usr/bin/env python3
"""
Script to set the rate card for YouTube Downloader Bot.
"""
import logging
import os
import sys

from fastapi_poe.client import sync_bot_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_youtube_bot_rate_card(points=300):
    """Set the rate card for the YouTube Downloader Bot."""
    # Bot configuration
    bot_name = "YouTubeDownloaderBot"

    # Get access key from environment
    access_key = os.environ.get("YOUTUBEDOWNLOADERBOT_ACCESS_KEY")
    if not access_key:
        logger.error("No access key found for YouTubeDownloaderBot")
        logger.error("Please set the YOUTUBEDOWNLOADERBOT_ACCESS_KEY environment variable")
        return False

    logger.info(f"Setting rate card for {bot_name} to {points} points per message")

    # Set bot settings including rate card
    try:
        settings_dict = {
            "bot_name": bot_name,
            "allow_attachments": False,  # Disable file upload for this bot
            "rate_card": {"api_calling_cost": points, "api_pricing_type": "per_message"},
        }

        # Send settings to Poe API
        sync_bot_settings(bot_name=bot_name, access_key=access_key)
        logger.info(f"Successfully set rate card for {bot_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to set rate card: {str(e)}")
        return False


if __name__ == "__main__":
    # Get points from command line argument if provided
    points = 300  # Default value
    if len(sys.argv) > 1:
        try:
            points = int(sys.argv[1])
        except ValueError:
            logger.error("Points must be an integer")
            sys.exit(1)

    success = set_youtube_bot_rate_card(points)
    sys.exit(0 if success else 1)

#!/usr/bin/env python3
"""
Script to set the rate card for the YouTube Downloader Bot.

This is a convenience wrapper around the generic set_bot_rate_card.py script.

Usage:
    python set_youtube_bot_rate_card.py [points_per_message]

Example:
    python set_youtube_bot_rate_card.py 300
"""
import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
parent_dir = str(Path(__file__).parent)
sys.path.append(parent_dir)
from set_bot_rate_card import set_bot_rate_card

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_youtube_bot_rate_card(points=300):
    """Set the rate card for the YouTube Downloader Bot."""
    # Bot configuration
    bot_name = "YouTubeDownloaderBot"

    # Get access key from environment
    access_key = os.environ.get("YOUTUBEDOWNLOADERBOT_ACCESS_KEY")

    # Call the generic function
    return set_bot_rate_card(bot_name, points, access_key)


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

#!/usr/bin/env python3
"""
Direct script to set the rate card for a Poe bot using the HTTP API.
Usage: python set_rate_card_direct.py <bot_name> <points_per_message> <access_key>
"""

import json
import logging
import sys

import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_rate_card_direct(bot_name: str, points: int, access_key: str) -> bool:
    """Set the rate card for a bot to charge points per message using direct API access."""
    try:
        logger.info(f"Setting rate card for {bot_name} to {points} points per message")

        url = "https://api.poe.com/bot/settings"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_key}"}

        data = {
            "bot_name": bot_name,
            "rate_card": {"api_calling_cost": points, "api_pricing_type": "per_message"},
        }

        response = requests.post(url, headers=headers, json=data)

        if response.status_code == 200:
            logger.info(f"Successfully set rate card for {bot_name}")
            return True
        else:
            logger.error(f"Failed to set rate card: HTTP {response.status_code} - {response.text}")
            return False

    except Exception as e:
        logger.error(f"Failed to set rate card: {str(e)}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python set_rate_card_direct.py <bot_name> <points_per_message> <access_key>")
        sys.exit(1)

    bot_name = sys.argv[1]
    try:
        points = int(sys.argv[2])
    except ValueError:
        logger.error("Points must be an integer")
        sys.exit(1)

    access_key = sys.argv[3]
    success = set_rate_card_direct(bot_name, points, access_key)
    sys.exit(0 if success else 1)

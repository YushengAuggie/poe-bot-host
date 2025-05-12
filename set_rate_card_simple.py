#!/usr/bin/env python3
"""
Simple script to set the rate card for a Poe bot.
"""
import json
import logging
import os
import sys
import urllib.request

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_rate_card(bot_name, points, access_key):
    """Set the rate card for a bot using only standard library."""
    try:
        logger.info(f"Setting rate card for {bot_name} to {points} points per message")

        url = "https://api.poe.com/bot/settings"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_key}"}

        data = {
            "bot_name": bot_name,
            "rate_card": {"api_calling_cost": points, "api_pricing_type": "per_message"},
        }

        # Convert data to JSON
        data_json = json.dumps(data).encode("utf-8")

        # Create request
        req = urllib.request.Request(url, data=data_json, headers=headers, method="POST")

        # Send request
        with urllib.request.urlopen(req) as response:
            response_data = response.read().decode("utf-8")
            status_code = response.getcode()

        if status_code == 200:
            logger.info(f"Successfully set rate card for {bot_name}")
            return True
        else:
            logger.error(f"Failed to set rate card: HTTP {status_code} - {response_data}")
            return False

    except Exception as e:
        logger.error(f"Failed to set rate card: {str(e)}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python set_rate_card_simple.py <bot_name> <points_per_message> <access_key>")
        sys.exit(1)

    bot_name = sys.argv[1]
    try:
        points = int(sys.argv[2])
    except ValueError:
        logger.error("Points must be an integer")
        sys.exit(1)

    access_key = sys.argv[3]
    success = set_rate_card(bot_name, points, access_key)
    sys.exit(0 if success else 1)

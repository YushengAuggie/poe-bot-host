#!/usr/bin/env python3
"""
Simple script to set the rate card for a Poe bot.
Usage: python set_rate_card.py <bot_name> <points_per_message> [<access_key>]

If access_key is not provided, it will look for an environment variable named
POE_ACCESS_KEY_{BOT_NAME} (all uppercase).
"""

import logging
import os
import sys
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def set_rate_card(bot_name: str, points: int, access_key: Optional[str] = None) -> bool:
    """Set the rate card for a bot to charge points per message."""
    try:
        # Get the bot access key from environment if not provided
        if not access_key:
            env_var_name = f"POE_ACCESS_KEY_{bot_name.upper()}"
            access_key = os.environ.get(env_var_name)

            if not access_key:
                logger.error(f"No access key found for {bot_name} in {env_var_name}")
                return False

        # Import here to avoid dependency issues
        from fastapi_poe.client import set_poe_bot_settings

        logger.info(f"Setting rate card for {bot_name} to {points} points per message")

        settings_dict = {
            "bot_name": bot_name,
            "rate_card": {"api_calling_cost": points, "api_pricing_type": "per_message"},
        }

        set_poe_bot_settings(bot_access_key=access_key, **settings_dict)
        logger.info(f"Successfully set rate card for {bot_name}")
        return True

    except Exception as e:
        logger.error(f"Failed to set rate card: {str(e)}")
        return False


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python set_rate_card.py <bot_name> <points_per_message> [<access_key>]")
        sys.exit(1)

    bot_name = sys.argv[1]
    try:
        points = int(sys.argv[2])
    except ValueError:
        logger.error("Points must be an integer")
        sys.exit(1)

    access_key = sys.argv[3] if len(sys.argv) > 3 else None
    success = set_rate_card(bot_name, points, access_key)
    sys.exit(0 if success else 1)

#!/usr/bin/env python
"""
Bot Testing CLI Tool

This script provides a command-line interface for testing bots running
in the Poe Bots Framework. It can:
  - Test specific bots with custom messages
  - Check API health and status
  - Show available bots
  - Display OpenAPI schema information
  - Test multiple bots at once

Usage:
    # Test the first available bot
    python scripts/test_bot_cli.py

    # Test a specific bot
    python scripts/test_bot_cli.py --bot EchoBot

    # Test with a custom message
    python scripts/test_bot_cli.py --bot ReverseBot --message "Reverse this text"

    # Check API health
    python scripts/test_bot_cli.py --health

    # Show API endpoints
    python scripts/test_bot_cli.py --schema
"""

import argparse
import base64
import json
import logging
import mimetypes
import os
import sys
from typing import Any, Dict, Optional

import requests

# Add the project root to path so utils module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ruff: noqa: E402
from utils.config import settings

# Set up logger
logger = logging.getLogger("poe_bots.test_bot")


def get_bot_api_key(bot_name: str) -> Optional[str]:
    """Auto-detect API key for known bots.

    Args:
        bot_name: Name of the bot

    Returns:
        API key if found, None otherwise
    """
    # Try different API key patterns for the bot
    bot_upper = bot_name.upper()
    bot_normalized = bot_upper.replace("-", "_")

    # List of possible environment variable names
    possible_keys = [
        f"{bot_normalized}_ACCESS_KEY",
        f"{bot_name.upper()}_ACCESS_KEY",
        f"{bot_name}_ACCESS_KEY",
        "POE_ACCESS_KEY",  # Fallback generic key
    ]

    for key_name in possible_keys:
        key_value = os.environ.get(key_name)
        if key_value:
            logger.debug(f"Found API key for {bot_name} using {key_name}")
            return key_value

    logger.debug(f"No API key found for {bot_name}")
    return None


def get_available_bots(base_url: str) -> Dict[str, str]:
    """Get list of available bots from the API.

    Args:
        base_url: Base URL of the API

    Returns:
        Dictionary of bot names and descriptions
    """
    try:
        logger.debug(f"Requesting bot list from: {base_url}/bots")
        response = requests.get(f"{base_url}/bots")
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to get bot list: {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error getting bot list: {str(e)}")
        return {}


def check_health(base_url: str) -> Dict[str, Any]:
    """Check API health status.

    Args:
        base_url: Base URL of the API

    Returns:
        Health check response
    """
    try:
        logger.debug(f"Checking health at: {base_url}/health")
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Health check failed: {response.status_code}")
            return {"status": "error", "code": response.status_code}
    except Exception as e:
        logger.error(f"Error checking health: {str(e)}")
        return {"status": "error", "message": str(e)}


def fetch_openapi_schema(base_url: str) -> Optional[Dict[str, Any]]:
    """Fetch the OpenAPI schema from the API.

    Args:
        base_url: Base URL of the API

    Returns:
        OpenAPI schema
    """
    try:
        response = requests.get(f"{base_url}/openapi.json")
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to fetch OpenAPI schema: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching OpenAPI schema: {str(e)}")
        return None


def test_bot_api(
    base_url: str = "http://localhost:8000",
    bot_name: str = "EchoBot",
    message: str = "Test message",
    api_key: Optional[str] = None,
    attachment_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Test a bot with a sample message and optional attachment.

    Args:
        base_url: Base URL of the API
        bot_name: Name of the bot to test
        message: Message to send to the bot
        api_key: Optional API key to use
        attachment_path: Optional path to image file to attach

    Returns:
        Bot response
    """
    url = f"{base_url}/{bot_name.lower()}"

    headers = {
        "Content-Type": "application/json",
    }

    # Add API key if provided
    if api_key:
        # Try both common header formats for Poe authentication
        headers["Authorization"] = f"Bearer {api_key}"
        headers["X-Poe-API-Key"] = api_key
    else:
        # Use dummy token for bots without authentication
        headers["Authorization"] = "Bearer dummytoken"  # Since allow_without_key=True in our bot

    try:
        # Log the exact request we're making
        logger.debug(f"Testing bot at URL: {url}")
        logger.debug(f"Headers: {headers}")

        # Prepare the request body
        query_message = {"role": "user", "content": message}

        # Add attachment if specified
        if attachment_path:
            try:
                # Get file type from extension
                content_type, _ = mimetypes.guess_type(attachment_path)
                if not content_type:
                    content_type = "application/octet-stream"

                # Read file contents
                with open(attachment_path, "rb") as f:
                    file_content = f.read()

                # Encode binary data as base64 for JSON transport
                file_content_b64 = base64.b64encode(file_content).decode("utf-8")

                # Get filename
                filename = os.path.basename(attachment_path)

                # Add attachment to message
                query_message["attachments"] = [
                    {
                        "url": f"file://{filename}",  # Dummy URL, not used by server
                        "name": filename,
                        "content_type": content_type,
                        "content": file_content_b64,
                    }
                ]

                # For debugging - print more details but not the whole base64 content
                content_preview = (
                    file_content_b64[:30] + "..."
                    if len(file_content_b64) > 30
                    else file_content_b64
                )
                logger.debug(
                    f"Added attachment: {filename} ({content_type}), size: {len(file_content)} bytes, content preview: {content_preview}"
                )
            except Exception as e:
                logger.error(f"Error adding attachment: {str(e)}")
                return {
                    "status_code": 0,
                    "error": f"Error adding attachment: {str(e)}",
                    "content": None,
                    "content_type": "error",
                }

        response = requests.post(
            url,
            headers=headers,
            json={
                "version": "1.0",
                "type": "query",
                "query": [query_message],
                "user_id": "test_user",
                "conversation_id": "test_convo_123",
                "message_id": "test_msg_123",
                "protocol": "poe",
            },
        )

        status_code = response.status_code
        headers = dict(response.headers)

        # Try to parse response as JSON
        try:
            content = response.json()
            content_type = "json"
        except ValueError:
            # For event stream responses
            content = response.text
            content_type = "text"

        return {
            "status_code": status_code,
            "headers": headers,
            "content": content,
            "content_type": content_type,
        }
    except Exception as e:
        logger.error(f"Error testing bot: {str(e)}")
        return {"status_code": 0, "error": str(e), "content": None, "content_type": "error"}


def print_banner():
    """Print a banner for the test script."""
    print("\n======================================================")
    print("                 POE BOTS TEST TOOL                   ")
    print("======================================================\n")


def check_key_configuration(base_url: str) -> Dict[str, Any]:
    """Check API key configuration in the deployment.

    Args:
        base_url: Base URL of the API

    Returns:
        API key configuration details
    """
    try:
        logger.debug(f"Checking key configuration at: {base_url}/keys_configured")
        response = requests.get(f"{base_url}/keys_configured")
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Key configuration check failed: {response.status_code}")
            return {"status": "error", "code": response.status_code}
    except Exception as e:
        logger.error(f"Error checking key configuration: {str(e)}")
        return {"status": "error", "message": str(e)}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Poe bots running locally",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=settings.API_HOST, help="Host where the API is running")
    parser.add_argument(
        "--port", type=int, default=settings.API_PORT, help="Port where the API is running"
    )
    parser.add_argument("--bot", help="Specific bot to test (omit for auto-select)")
    parser.add_argument("--message", default="Hello, world!", help="Message to send to the bot")
    parser.add_argument("--attachment", help="Path to image file to attach to the message")
    parser.add_argument(
        "--api-key", help="API key for authentication (auto-detected for known bots)"
    )
    parser.add_argument("--schema", action="store_true", help="Show OpenAPI schema")
    parser.add_argument("--health", action="store_true", help="Show health check")
    parser.add_argument("--all", action="store_true", help="Test all available bots")
    parser.add_argument("--keys", action="store_true", help="Check API key configuration")
    parser.add_argument(
        "--format", choices=["text", "json"], default="text", help="Output format (text or json)"
    )

    args = parser.parse_args()

    # Override environment variables with command line arguments if provided
    # Note: args.host should take precedence over environment variables
    host = args.host if args.host else os.environ.get("POE_API_HOST", "0.0.0.0")
    port = int(args.port if args.port else os.environ.get("POE_API_PORT", "8000"))

    # Check if this is a full URL (likely a Modal deployment)
    if host.startswith("http://") or host.startswith("https://"):
        base_url = host
    elif "modal.run" in host:  # Modal deployment
        base_url = f"https://{host}"
    else:
        base_url = f"http://{host}:{port}"

    # Debug output
    print(f"DEBUG: Using host={host}, port={port}, base_url={base_url}")

    # Print banner for non-JSON output
    if args.format == "text":
        print_banner()
        logger.info(f"Using API at {base_url}")

    # Check health if requested
    if args.health:
        health = check_health(base_url)
        if args.format == "json":
            print(json.dumps({"health": health}, indent=2))
        else:
            print("\nHealth Check:")
            print(json.dumps(health, indent=2))

    # Check API key configuration if requested
    if args.keys:
        keys_config = check_key_configuration(base_url)
        if args.format == "json":
            print(json.dumps({"keys_config": keys_config}, indent=2))
        else:
            print("\nAPI Key Configuration:")
            print(json.dumps(keys_config, indent=2))

    # Get list of available bots
    bots = get_available_bots(base_url)
    if bots:
        if args.format == "json":
            if not args.health and not args.schema:
                print(json.dumps({"bots": bots}, indent=2))
        else:
            print("\nAvailable Bots:")
            for bot, description in bots.items():
                print(f"  - {bot}: {description}")
    else:
        if args.format == "text":
            print("\nNo bots available!")
        else:
            print(json.dumps({"error": "No bots available"}, indent=2))
        return

    # Show OpenAPI schema if requested
    if args.schema:
        schema = fetch_openapi_schema(base_url)
        if schema:
            if args.format == "json":
                print(json.dumps({"schema": schema}, indent=2))
            else:
                print("\nAvailable Endpoints:")
                for path, methods in schema.get("paths", {}).items():
                    print(f"  {path}:")
                    for method in methods:
                        print(f"    - {method.upper()}")

    # Test a specific bot or all bots
    if args.all:
        # Test all bots
        results = {}
        for bot in bots.keys():
            # Auto-detect API key for each bot
            api_key = args.api_key or get_bot_api_key(bot)

            if args.format == "text":
                print(f"\nTesting bot: {bot}")
                print(f"Message: {args.message}")
                if api_key:
                    print(f"Using API key: {api_key[:10]}...")

            result = test_bot_api(
                base_url, bot, args.message, api_key=api_key, attachment_path=args.attachment
            )
            results[bot] = result

            if args.format == "text":
                print(f"Status Code: {result['status_code']}")

                if result["content_type"] == "json":
                    print("Response JSON:")
                    print(json.dumps(result["content"], indent=2))
                elif result["content_type"] == "text":
                    print("Response Content:")
                    for line in result["content"].split("\n"):
                        if line.strip():
                            print(f"  {line}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")

        if args.format == "json":
            print(json.dumps({"results": results}, indent=2))
    else:
        # Test a specific bot
        bot_to_test = args.bot
        if not bot_to_test and bots:
            bot_to_test = list(bots.keys())[0]
            if args.format == "text":
                print(f"\nNo bot specified, using first available bot: {bot_to_test}")

        if bot_to_test:
            # Auto-detect or use provided API key
            api_key = args.api_key or get_bot_api_key(bot_to_test)

            if args.format == "text":
                print(f"\nTesting bot: {bot_to_test}")
                print(f"Message: {args.message}")
                if api_key:
                    print(f"Using API key: {api_key[:10]}...")
                else:
                    print("No API key found - using dummy token")

            result = test_bot_api(
                base_url,
                bot_to_test,
                args.message,
                api_key=api_key,
                attachment_path=args.attachment,
            )

            if args.format == "json":
                print(json.dumps({"result": result}, indent=2))
            else:
                print(f"Status Code: {result['status_code']}")

                if result["content_type"] == "json":
                    print("Response JSON:")
                    print(json.dumps(result["content"], indent=2))
                elif result["content_type"] == "text":
                    print("Response Content:")
                    for line in result["content"].split("\n"):
                        if line.strip():
                            print(f"  {line}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Debug API call script for Gemini bot image handling.

This script adds extra debug logging to understand why the image
attachments aren't being properly processed through the API.
"""

import base64
import json
import logging
import os
import sys
from typing import Any, Dict, Optional

import requests

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("debug_api_call")

# Add the project root to path so modules can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def create_api_request(image_path, bot_name, prompt):
    """Create a request for the API with an attachment.

    Args:
        image_path: Path to the image file
        bot_name: Name of the bot to test
        prompt: Text prompt to send with the image

    Returns:
        Tuple of (url, headers, body)
    """
    # Check if image file exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        sys.exit(1)

    # Get file type from extension
    import mimetypes

    content_type, _ = mimetypes.guess_type(image_path)
    if not content_type:
        content_type = "application/octet-stream"

    # Read file contents
    with open(image_path, "rb") as f:
        file_content = f.read()

    # Encode binary data as base64 for JSON transport
    file_content_b64 = base64.b64encode(file_content).decode("utf-8")

    # Get filename
    filename = os.path.basename(image_path)

    url = f"http://localhost:8000/{bot_name.lower()}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer dummytoken",  # Since allow_without_key=True in our bot
    }

    # Prepare request body
    body = {
        "version": "1.0",
        "type": "query",
        "query": [
            {
                "role": "user",
                "content": prompt,
                "attachments": [
                    {
                        "url": f"file://{filename}",  # Dummy URL, not used by server
                        "name": filename,
                        "content_type": content_type,
                        "content": file_content_b64,
                    }
                ],
            }
        ],
        "user_id": "test_user",
        "conversation_id": "test_convo_123",
        "message_id": "test_msg_123",
        "protocol": "poe",
    }

    return url, headers, body


def send_request(url, headers, body):
    """Send the request to the API.

    Args:
        url: The API URL
        headers: The request headers
        body: The request body

    Returns:
        The response
    """
    logger.info(f"Sending request to: {url}")
    logger.info(f"Headers: {headers}")
    logger.debug(f"Body: {json.dumps(body)[:100]}...")

    # Send request
    response = requests.post(url, headers=headers, json=body)

    logger.info(f"Response status code: {response.status_code}")

    # Try to parse response
    try:
        content = response.json()
        logger.info(f"Response is JSON: {json.dumps(content)[:100]}...")
        return content
    except ValueError:
        # For event stream responses
        logger.info(f"Response is text: {response.text[:100]}...")
        return response.text


def main():
    """Main entry point."""
    # Get command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Debug API call")
    parser.add_argument(
        "image_path",
        nargs="?",
        default="assets/test_images/duck2.jpg",
        help="Path to image file (default: assets/test_images/duck2.jpg)",
    )
    parser.add_argument(
        "--bot", default="gemini20flashbot", help="Bot name to test (default: gemini20flashbot)"
    )
    parser.add_argument(
        "--prompt",
        default="Describe what you see in this image in detail.",
        help="Text prompt to send with the image",
    )
    args = parser.parse_args()

    # Check if Gemini API key is set
    gemini_api_key = os.environ.get("GOOGLE_API_KEY")
    if gemini_api_key:
        logger.info(f"GOOGLE_API_KEY is set: {gemini_api_key[:5]}...")
    else:
        logger.warning("GOOGLE_API_KEY environment variable is not set!")

    # Print all environment variables
    logger.debug("All environment variables that might be relevant:")
    for key in os.environ:
        if "API" in key or "KEY" in key or "SECRET" in key or "TOKEN" in key:
            value = os.environ[key]
            logger.debug(
                f"  {key}={value[:5]}..." if value and len(value) > 5 else f"  {key}={value}"
            )

    # Create request
    url, headers, body = create_api_request(args.image_path, args.bot, args.prompt)

    # Send request
    response = send_request(url, headers, body)

    # Print response details
    if isinstance(response, str):
        try:
            # Try to parse SSE format
            events = []
            for line in response.split("\n\n"):
                if line.startswith("event: text"):
                    # Find the data line
                    for data_line in line.split("\n"):
                        if data_line.startswith("data: "):
                            # Parse the JSON data
                            try:
                                data_json = json.loads(data_line[6:])
                                if "text" in data_json:
                                    events.append(data_json["text"])
                            except json.JSONDecodeError:
                                pass

            logger.info(f"Extracted {len(events)} events from SSE")
            for i, event in enumerate(events):
                logger.info(f"Event {i+1}: {event}")

            # Print full combined response
            if events:
                print("\nFull response:")
                print("".join(events))
            else:
                print("\nNo events found in SSE response")
                print(response)
        except Exception as e:
            logger.error(f"Error parsing SSE: {e}")
            print("\nRaw response:")
            print(response)
    else:
        # Handle JSON response
        if "messages" in response:
            messages = response["messages"]
            for i, message in enumerate(messages):
                logger.info(f"Message {i+1}: {message.get('content', '')[:100]}...")

            # Print full combined response
            if messages:
                print("\nFull response:")
                for message in messages:
                    print(message.get("content", ""))
            else:
                print("\nNo messages found in response")
                print(json.dumps(response, indent=2))
        else:
            print("\nUnknown response format:")
            print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()

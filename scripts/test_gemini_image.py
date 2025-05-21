#!/usr/bin/env python
"""
Test script for Gemini image handling.

This script allows testing Gemini bot's image processing capabilities
with either a built-in test image or a custom image file.

Usage:
    # Test with built-in 10x10 red image
    python scripts/test_gemini_image.py

    # Test with custom image
    python scripts/test_gemini_image.py --image duck.jpg

    # Specify a bot class
    python scripts/test_gemini_image.py --bot Gemini20FlashBot --image duck.jpg

    # Change the prompt
    python scripts/test_gemini_image.py --prompt "Describe this image in detail"
"""

import argparse
import asyncio
import base64
import json
import logging
import os
import sys
from typing import Any, AsyncGenerator, Dict, List, Optional

# Add the project root to path so utils module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import importlib

from fastapi_poe.types import PartialResponse, ProtocolMessage, QueryRequest

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("gemini_test")


class MockAttachment:
    """Mock attachment with base64 content."""

    def __init__(self, name: str, content_type: str, base64_content: str = None):
        """Initialize a mock attachment.

        Args:
            name: The filename
            content_type: MIME type
            base64_content: Base64-encoded content
        """
        self.name = name
        self.content_type = content_type
        self.content = base64_content if base64_content else "base64_data_here"
        self.url = f"file://{name}"


def get_bot_class(bot_name: str):
    """Import and return the specified bot class.

    Args:
        bot_name: Name of the bot class to import

    Returns:
        The bot class
    """
    try:
        # Import the bot module
        bot_module = importlib.import_module("bots.gemini")

        # Get the bot class
        if hasattr(bot_module, bot_name):
            return getattr(bot_module, bot_name)
        else:
            logger.error(f"Bot class {bot_name} not found in bots.gemini module")
            available_bots = [name for name in dir(bot_module) if "Bot" in name]
            logger.info(f"Available bot classes: {', '.join(available_bots)}")
            raise ValueError(f"Bot class {bot_name} not found")
    except Exception as e:
        logger.error(f"Error importing bot class: {str(e)}")
        raise


def load_image_from_file(file_path: str) -> tuple[bytes, str]:
    """Load image data from a file.

    Args:
        file_path: Path to the image file

    Returns:
        Tuple of (image_data, mime_type)
    """
    import imghdr
    import mimetypes
    import os

    from PIL import Image

    try:
        # Check if file exists
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"Image file not found: {file_path}")

        # Get mime type
        mime_type = mimetypes.guess_type(file_path)[0]
        if not mime_type or not mime_type.startswith("image/"):
            # Fallback to imghdr
            img_type = imghdr.what(file_path)
            if img_type:
                mime_type = f"image/{img_type}"
            else:
                mime_type = "image/jpeg"  # Default

        # Open and read the image
        with open(file_path, "rb") as f:
            image_data = f.read()

        # Validate it's a valid image
        try:
            Image.open(file_path)
        except Exception as e:
            raise ValueError(f"Not a valid image file: {str(e)}")

        return image_data, mime_type

    except Exception as e:
        logger.error(f"Error loading image from {file_path}: {str(e)}")
        raise


async def test_gemini_image_processing(
    bot_name: str = "Gemini20FlashBot",
    image_path: Optional[str] = None,
    prompt: str = "What is in this image?",
):
    """Test Gemini image processing.

    Args:
        bot_name: Name of the Gemini bot class to test
        image_path: Path to image file to test with (if None, uses a test image)
        prompt: Text prompt to send with the image
    """
    import io
    import os

    from PIL import Image

    # Import the bot class
    bot_class = get_bot_class(bot_name)
    print(f"Using bot class: {bot_class.__name__}")

    # Create or load the image
    if image_path:
        print(f"Loading image from file: {image_path}")
        try:
            image_data, mime_type = load_image_from_file(image_path)
            file_name = os.path.basename(image_path)
            print(f"Loaded image: {file_name} ({mime_type}, {len(image_data)} bytes)")
        except Exception as e:
            print(f"Error loading image: {str(e)}")
            return
    else:
        print("Creating test image (10x10 red square)")
        # Create a simple 10x10 red image
        img = Image.new("RGB", (100, 100), color="red")
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        image_data = buf.getvalue()
        mime_type = "image/jpeg"
        file_name = "test_red_square.jpg"

    # Encode the image
    base64_content = base64.b64encode(image_data).decode("utf-8")

    # Create an attachment with the image data
    attachment = MockAttachment(file_name, mime_type, base64_content)

    # Create a mock query
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[ProtocolMessage(role="user", content=prompt, attachments=[attachment])],
        user_id="test_user",
        conversation_id="test_convo",
        message_id="test_msg",
    )

    # Create a Gemini bot instance
    bot = bot_class()

    print("\n======================================")
    print(f"Testing {bot_name} with {prompt}")
    print("======================================")

    # Check if API key is set
    import os

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("\nWARNING: GOOGLE_API_KEY environment variable is not set!")
        print("The test will run with a mock/stub response.")
        print("To use the actual Google API, set the GOOGLE_API_KEY environment variable.")
        print("Example: export GOOGLE_API_KEY=your_api_key_here\n")
    else:
        print(f"\nFound GOOGLE_API_KEY environment variable (starts with: {api_key[:3]}...)\n")

    # Process the attachments
    print("\nStep 1: Extract attachments")
    attachments = bot._extract_attachments(query)
    print(f"Found {len(attachments)} attachments")

    print("\nStep 2: Process media attachments")
    for i, attachment in enumerate(attachments):
        print(f"Processing attachment {i+1}")
        media_data = bot._process_media_attachment(attachment)
        if media_data:
            print(f"Successfully processed attachment as {media_data['mime_type']}")
            print(f"Data size: {len(media_data['data'])} bytes")
        else:
            print("Failed to process attachment")

    print("\nStep 3: Prepare media parts for API")
    media_parts = bot._prepare_media_parts(attachments)
    print(f"Prepared {len(media_parts)} media parts")

    print("\nStep 4: Send query to bot")
    if not api_key:
        print("Since no API key is set, you'll get a simulated response")

    # Run the async function to get the response
    print("\nWaiting for bot response...")
    responses = []
    try:
        async for response in bot.get_response(query):
            if hasattr(response, "text"):
                responses.append(response.text)
    except Exception as e:
        print(f"Error getting response: {str(e)}")

    # Print the responses
    print("\nBot responses:")
    if responses:
        for i, response in enumerate(responses):
            print(f"Response chunk {i+1}:")
            print(f"{response}")
            print("---")
    else:
        print("No responses received")

    print("\nTest completed!")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Gemini bot image processing")
    parser.add_argument(
        "--bot",
        default="Gemini20FlashBot",
        help="Bot class name to test (default: Gemini20FlashBot)",
    )
    parser.add_argument(
        "--image", help="Path to image file to test with (if not provided, uses a test image)"
    )
    parser.add_argument(
        "--prompt",
        default="What is in this image?",
        help="Text prompt to send with the image (default: 'What is in this image?')",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Run the test
    asyncio.run(
        test_gemini_image_processing(bot_name=args.bot, image_path=args.image, prompt=args.prompt)
    )

#!/usr/bin/env python
"""
Simple test script for Gemini image handling.

This script tests the Gemini bot with an image attachment to see if the fix
for multimodal_model_name is working properly.

Usage:
    python scripts/test_simple_gemini.py
"""

import asyncio
import base64
import logging
import os
import sys
from typing import List, Optional

# Add the project root to path so bots module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi_poe.types import PartialResponse, ProtocolMessage, QueryRequest

from bots.gemini import Gemini20FlashBot

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("gemini_test")


class MockAttachment:
    """Mock attachment with base64 content."""

    def __init__(self, name: str, content_type: str, content: bytes):
        """Initialize a mock attachment.

        Args:
            name: The filename
            content_type: MIME type
            content: The raw binary data
        """
        self.name = name
        self.content_type = content_type
        # Base64 encode the content
        self.content = base64.b64encode(content).decode("utf-8")
        self.url = f"file://{name}"


async def run_test():
    """Run the Gemini test with a duck image."""
    # Load the duck image
    image_path = os.path.join(os.path.dirname(__file__), "duck.jpg")

    if not os.path.exists(image_path):
        print(f"Error: duck.jpg not found at {image_path}")
        return

    # Read the image file
    with open(image_path, "rb") as f:
        image_data = f.read()

    print(f"Loaded duck.jpg: {len(image_data)} bytes")

    # Create an attachment with the image
    attachment = MockAttachment("duck.jpg", "image/jpeg", image_data)

    # Create the message
    message = ProtocolMessage(
        role="user",
        content="What is in this image? Describe it in detail.",
        attachments=[attachment],
    )

    # Create a query request
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[message],
        user_id="test_user",
        conversation_id="test_convo",
        message_id="test_msg",
    )

    # Create the bot
    bot = Gemini20FlashBot()

    print("\n=== Testing Gemini20FlashBot with duck.jpg ===")
    print("Bot model: ", bot.model_name)
    print("Bot multimodal_model_name: ", getattr(bot, "multimodal_model_name", "Not set"))

    # Check for API key
    from utils.api_keys import get_api_key

    api_key = get_api_key("GOOGLE_API_KEY")
    if not api_key:
        print("\nWARNING: No Google API key found in environment variables.")
        print("To use the actual Google API, set the GOOGLE_API_KEY environment variable.")
        print("Example: export GOOGLE_API_KEY=your_api_key_here\n")
    else:
        print(f"Found Google API key (starts with: {api_key[:3]}...)")

    # Test the image processing steps
    print("\nStep 1: Extract attachments")
    attachments = bot._extract_attachments(query)
    print(f"Found {len(attachments)} attachments")

    if attachments:
        print("\nStep 2: Process media attachments")
        for i, att in enumerate(attachments):
            print(f"Processing attachment {i+1}")
            result = bot._process_media_attachment(att)
            if result:
                print(f"Successfully processed: {result['mime_type']}, {len(result['data'])} bytes")
            else:
                print("Failed to process attachment")

        print("\nStep 3: Prepare media parts")
        media_parts = bot._prepare_media_parts(attachments)
        print(f"Created {len(media_parts)} media parts")

    # Get response from bot
    print("\nStep 4: Getting response from bot")
    response_chunks = []
    try:
        async for response in bot.get_response(query):
            if hasattr(response, "text"):
                print(f"Response chunk: {response.text[:80]}...")
                response_chunks.append(response.text)
    except Exception as e:
        print(f"Error getting response: {str(e)}")

    # Print the full response
    if response_chunks:
        print("\nFull response:")
        print("\n".join(response_chunks))
    else:
        print("\nNo response received")


if __name__ == "__main__":
    asyncio.run(run_test())

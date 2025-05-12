"""
Test script for YouTube downloader bot.

This script allows testing the YouTube downloader bot locally.
"""

import asyncio
import json
import os
import sys
from typing import Any, Dict, List

# Then import our bot
from bots.youtube_downloader_bot import YouTubeDownloaderBot


# Create a simplified version of the Poe protocol types for testing
class ContentType:
    text = "text"


class ProtocolMessage:
    def __init__(self, role: str, content: str, content_type: str = "text"):
        self.role = role
        self.content = content
        self.content_type = content_type
        self.attachments = []


class QueryRequest:
    def __init__(self, query: List[ProtocolMessage]):
        self.query = query


# Add a sample YouTube URL to test with
TEST_YOUTUBE_URL = (
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Use a known small video for testing
)


async def test_youtube_bot():
    """Test the YouTube downloader bot with a sample URL."""
    print(f"Testing YouTube downloader bot with URL: {TEST_YOUTUBE_URL}")

    # Create the bot instance
    bot = YouTubeDownloaderBot()

    # Create a test message with a YouTube URL
    test_message = ProtocolMessage(
        role="user",
        content=TEST_YOUTUBE_URL,
        content_type=ContentType.text,
    )

    # Create a query request with the test message
    query = QueryRequest(query=[test_message])

    # Process the query through the bot
    print("Sending request to bot...")
    response_chunks = []

    try:
        async for response in bot.get_response(query):
            if hasattr(response, "text"):
                print(f"Response: {response.text}")
                response_chunks.append(response.text)

            # We can't actually test the attachment part locally since we're mocking the types
            # But we can check that the bot attempts to download

    except Exception as e:
        print(f"Error during testing: {str(e)}")
        import traceback

        print(traceback.format_exc())

    # Print results summary
    print("\nTest Results:")
    print(f"- Response received: {'Yes' if response_chunks else 'No'}")

    # Check if we got a download response
    success = any("Downloading video" in chunk for chunk in response_chunks)
    print(f"- Download attempt: {'Yes' if success else 'No'}")


if __name__ == "__main__":
    asyncio.run(test_youtube_bot())

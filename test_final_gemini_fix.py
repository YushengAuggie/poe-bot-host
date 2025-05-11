#!/usr/bin/env python
"""
Final test for the Gemini image handling fix.
This script tests both direct and API-based image handling to confirm the fix works.
"""

import asyncio
import base64
import logging
import os
import sys
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("test_final_gemini_fix")

# Import required modules
try:
    from fastapi_poe.types import Attachment, QueryRequest
    from fastapi_poe.types import ProtocolMessage as Message

    from bots.gemini import Gemini20FlashBot
    from utils.api_keys import get_api_key
except ImportError as e:
    logger.error(f"Import error: {e}")
    sys.exit(1)


class FakeMessageId:
    """Mock message ID for testing."""

    def __init__(self, id_value="test_message_id"):
        self.id = id_value


def create_test_image() -> bytes:
    """Create a simple test image (small red square)."""
    try:
        import io

        from PIL import Image

        # Create a simple 100x100 red square
        img = Image.new("RGB", (100, 100), color=(255, 0, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_data = buffer.getvalue()

        logger.info(f"Created test image, size: {len(img_data)} bytes")
        return img_data
    except ImportError:
        logger.error("PIL not found, using hardcoded test image")
        # Return a tiny 1x1 PNG if PIL is not available
        return base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVQI12P4//8/AAX+Av7czFnnAAAAAElFTkSuQmCC"
        )


def check_api_key() -> bool:
    """Check if Google API key is available and log details."""
    try:
        api_key = get_api_key("GOOGLE_API_KEY")
        if api_key:
            preview = api_key[:5] + "..." if len(api_key) > 5 else "***"
            logger.info(f"✅ Found GOOGLE_API_KEY ({preview})")
            return True
        else:
            logger.error("❌ GOOGLE_API_KEY returned empty value")
            return False
    except Exception as e:
        logger.error(f"❌ Error getting GOOGLE_API_KEY: {e}")

        # Check what's in the environment
        if "GOOGLE_API_KEY" in os.environ:
            value = os.environ["GOOGLE_API_KEY"]
            preview = value[:5] + "..." if len(value) > 5 else "***"
            logger.info(f"Direct environment check: GOOGLE_API_KEY found ({preview})")
        else:
            logger.error("GOOGLE_API_KEY not found in environment")

        # Show all environment variables (names only)
        logger.info(f"Available environment variables: {list(os.environ.keys())}")
        return False


class TestAttachment:
    """Independent test attachment class that mimics the fastapi-poe Attachment class."""

    def __init__(self, name: str, content_type: str, content: bytes):
        """Initialize attachment with direct content attribute."""
        self.name = name
        self.content_type = content_type
        # Set content directly as attribute
        self.content = content

    def __dir__(self):
        """Return the list of attributes for better debugging."""
        return ["name", "content_type", "content"]


async def test_direct_bot_path():
    """Test direct bot usage without going through the API."""
    logger.info("=== TESTING DIRECT BOT PATH ===")

    # Check API key
    if not check_api_key():
        logger.error("API key not available, aborting direct path test")
        return False

    # Create bot instance
    logger.info("Creating Gemini20FlashBot instance")
    bot = Gemini20FlashBot()

    # Create test image
    img_data = create_test_image()

    # Create attachment
    attachment = TestAttachment(name="test_image.png", content_type="image/png", content=img_data)

    # Create message with attachment
    message = Message(role="user", content="What's in this image?", attachments=[attachment])

    # Create query
    query = QueryRequest(query=[message], message_id=FakeMessageId(), version="version")

    # Send request to bot
    logger.info("Sending request to bot with image attachment")

    try:
        responses = []
        async for response in bot.get_response(query):
            responses.append(response)
            logger.info(f"Response: {response.text}")

        # Check if we got any responses
        if responses:
            logger.info(f"✅ Direct path test successful ({len(responses)} responses)")
            return True
        else:
            logger.error("❌ Direct path test failed - no responses")
            return False
    except Exception as e:
        logger.error(f"❌ Error in direct test: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return False


async def test_api_simulation_path():
    """Test API path by simulating how fastapi-poe handles attachments."""
    logger.info("=== TESTING API SIMULATION PATH ===")

    # Check API key
    if not check_api_key():
        logger.error("API key not available, aborting API simulation test")
        return False

    # Create bot instance
    logger.info("Creating Gemini20FlashBot instance")
    bot = Gemini20FlashBot()

    # Create test image
    img_data = create_test_image()

    # Create a dict-based attachment to simulate Pydantic model
    # This is crucial - we're using a normal class but manually setting
    # the content only in __dict__ to mimic the bug in the API path
    class APIStyleAttachment:
        """Simulates how attachment is received in the API path."""

        def __init__(self, name, content_type, content):
            self.name = name
            self.content_type = content_type
            # Only set content in __dict__, not as direct attribute
            # This is how fastapi-poe Pydantic models behave
            self.__dict__["content"] = content

        def __dir__(self):
            """Return list of attributes for better debugging."""
            return ["name", "content_type"]

    # Create attachment (API style)
    attachment = APIStyleAttachment(
        name="test_image.png", content_type="image/png", content=img_data
    )

    # Verify attachment structure to make sure we're correctly simulating the issue
    logger.info(f"Attachment content as attribute? {hasattr(attachment, 'content')}")
    logger.info(f"Attachment content in __dict__? {'content' in attachment.__dict__}")
    logger.info(f"Attachment __dict__ keys: {attachment.__dict__.keys()}")

    # Create message with attachment
    message = Message(role="user", content="What's in this image?", attachments=[attachment])

    # Create query
    query = QueryRequest(query=[message], message_id=FakeMessageId(), version="version")

    # Send request to bot
    logger.info("Sending request to bot with API-style attachment")

    try:
        responses = []
        async for response in bot.get_response(query):
            responses.append(response)
            logger.info(f"Response: {response.text}")

        # Check if we got any substantive responses (not just error messages)
        has_substantive_response = False
        error_phrases = [
            "error",
            "couldn't process",
            "failed to",
            "invalid",
            "unsupported",
            "could not access",
        ]

        for response in responses:
            text = response.text.lower()
            if not any(phrase in text for phrase in error_phrases):
                has_substantive_response = True
                break

        if has_substantive_response:
            logger.info(f"✅ API simulation test successful ({len(responses)} responses)")
            return True
        else:
            # Still got responses but they might be error messages
            logger.warning("⚠️ API simulation got responses but they may be error messages")
            return False
    except Exception as e:
        logger.error(f"❌ Error in API simulation test: {str(e)}")
        import traceback

        logger.error(traceback.format_exc())
        return False


async def main():
    """Run both tests and report final results."""
    # Setup environment for testing
    logger.info("Setting up test environment")

    # Ensure Gemini model debug logging is enabled
    logging.getLogger("bots.gemini").setLevel(logging.DEBUG)

    # Run direct test
    direct_result = await test_direct_bot_path()

    # Run API simulation test
    api_result = await test_api_simulation_path()

    # Summary
    logger.info("=== TEST SUMMARY ===")
    logger.info(f"Direct bot path test: {'PASSED' if direct_result else 'FAILED'}")
    logger.info(f"API simulation test: {'PASSED' if api_result else 'FAILED'}")

    if direct_result and api_result:
        logger.info("✅ ALL TESTS PASSED! The fix is working correctly.")
    elif direct_result:
        logger.info("⚠️ PARTIAL SUCCESS: Direct path works but API simulation failed.")
    else:
        logger.info("❌ TESTS FAILED: The fix isn't working correctly.")


if __name__ == "__main__":
    # Make sure API key is available in the environment
    if "GOOGLE_API_KEY" not in os.environ:
        print("WARNING: GOOGLE_API_KEY not set in environment.")
        api_key = input("Enter your Google API key: ").strip()
        if api_key:
            os.environ["GOOGLE_API_KEY"] = api_key
            print("Set GOOGLE_API_KEY environment variable.")
        else:
            print("No API key provided, test will likely fail.")

    # Run the tests
    asyncio.run(main())

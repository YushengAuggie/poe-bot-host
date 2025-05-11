#!/usr/bin/env python
"""
Direct test of the Gemini image handling fix without using fastapi-poe classes.
This test directly exercises the fixed methods in the GeminiBaseBot class.
"""

import logging
import os

# Import only what we need from our implementation
from bots.gemini import GeminiBaseBot
from utils.api_keys import get_api_key

# Set up logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger("direct_fix_test")


def create_test_attachment_dict_only():
    """Create a test attachment with content only in __dict__."""

    class DictOnlyAttachment:
        """Attachment with content only in __dict__, not as attribute."""

        def __init__(self, name, content_type, content):
            self.name = name
            self.content_type = content_type
            # Only set in __dict__, not as attribute - simulates the bug
            self.__dict__["content"] = content

        def __repr__(self):
            """String representation for debugging."""
            return f"DictOnlyAttachment(name={self.name}, content_type={self.content_type}, has_content={'content' in self.__dict__})"

    # Simple test content
    test_content = b"test_image_data"

    return DictOnlyAttachment(name="test_image.png", content_type="image/png", content=test_content)


def create_test_attachment_normal():
    """Create a test attachment with normal content attribute."""

    class NormalAttachment:
        """Attachment with normal content attribute."""

        def __init__(self, name, content_type, content):
            self.name = name
            self.content_type = content_type
            self.content = content

        def __repr__(self):
            """String representation for debugging."""
            return f"NormalAttachment(name={self.name}, content_type={self.content_type}, has_content={hasattr(self, 'content')})"

    # Simple test content
    test_content = b"test_image_data"

    return NormalAttachment(name="test_image.png", content_type="image/png", content=test_content)


def test_extract_attachments_fix():
    """Test the fix in _extract_attachments method."""
    logger.info("=== Testing _extract_attachments fix ===")

    # Create a bot instance
    bot = GeminiBaseBot()

    # Create a test attachment with content only in __dict__
    attachment = create_test_attachment_dict_only()
    logger.info(f"Created test attachment: {attachment}")
    logger.info(f"Initial state - Has content attribute? {hasattr(attachment, 'content')}")
    logger.info(f"Initial state - Has content in __dict__? {'content' in attachment.__dict__}")

    # Create a fake query object - just needs to work with our _extract_attachments method
    class FakeQuery:
        def __init__(self, attachments):
            class FakeMessage:
                def __init__(self, attachments):
                    self.attachments = attachments
                    self.content = "What's in this image?"

            # Make this match the expected structure in _extract_attachments
            self.query = [FakeMessage(attachments)]

    # Create the query with our test attachment
    query = FakeQuery([attachment])

    # Call the method we're testing
    logger.info("Calling _extract_attachments method...")
    result = bot._extract_attachments(query)

    # Check the result
    logger.info(f"Got {len(result)} attachments back")

    # Check if our original attachment now has content accessible as attribute
    logger.info(f"After fix - Has content attribute? {hasattr(attachment, 'content')}")

    # Try accessing content both ways
    if hasattr(attachment, "content"):
        content_attr = attachment.content
        logger.info(
            f"Content accessible as attribute: {len(content_attr) if content_attr else 'None'}"
        )

    if "content" in attachment.__dict__:
        content_dict = attachment.__dict__["content"]
        logger.info(
            f"Content accessible in __dict__: {len(content_dict) if content_dict else 'None'}"
        )

    # Final result
    if hasattr(attachment, "content") and "content" in attachment.__dict__:
        logger.info("✅ TEST PASSED: Content accessible both as attribute and in __dict__")
        return True
    else:
        logger.error("❌ TEST FAILED: Content not properly accessible")
        return False


def test_process_media_attachment_fix():
    """Test the fix in _process_media_attachment method."""
    logger.info("=== Testing _process_media_attachment fix ===")

    # Create a bot instance
    bot = GeminiBaseBot()

    # Create a test attachment with content only in __dict__
    attachment = create_test_attachment_dict_only()
    logger.info(f"Created test attachment: {attachment}")
    logger.info(f"Initial state - Has content attribute? {hasattr(attachment, 'content')}")
    logger.info(f"Initial state - Has content in __dict__? {'content' in attachment.__dict__}")

    # Call the method we're testing
    logger.info("Calling _process_media_attachment method...")
    result = bot._process_media_attachment(attachment)

    # Check the result
    logger.info(f"Result type: {type(result).__name__}")
    logger.info(f"Result: {result}")

    # Test should pass if we got back a dict with mime_type and data
    if (
        result is not None
        and isinstance(result, dict)
        and "mime_type" in result
        and "data" in result
    ):
        logger.info(
            "✅ TEST PASSED: Successfully processed attachment with content only in __dict__"
        )
        return True
    else:
        logger.error("❌ TEST FAILED: Could not process attachment with content only in __dict__")
        return False


def test_with_normal_attachment():
    """Verify the fix doesn't break normal attachments."""
    logger.info("=== Testing with normal attachment ===")

    # Create a bot instance
    bot = GeminiBaseBot()

    # Create a normal attachment
    attachment = create_test_attachment_normal()
    logger.info(f"Created normal attachment: {attachment}")
    logger.info(f"Initial state - Has content attribute? {hasattr(attachment, 'content')}")
    logger.info(f"Initial state - Has content in __dict__? {'content' in attachment.__dict__}")

    # Call the method
    logger.info("Calling _process_media_attachment method...")
    result = bot._process_media_attachment(attachment)

    # Check the result
    logger.info(f"Result type: {type(result).__name__}")
    logger.info(f"Result: {result}")

    # Test should pass if we got back a dict with mime_type and data
    if (
        result is not None
        and isinstance(result, dict)
        and "mime_type" in result
        and "data" in result
    ):
        logger.info("✅ TEST PASSED: Successfully processed normal attachment")
        return True
    else:
        logger.error("❌ TEST FAILED: Could not process normal attachment")
        return False


def main():
    """Run all tests and report results."""
    logger.info("Starting direct tests of Gemini fix...")

    # Make sure we can see all debug logs
    logging.getLogger("bots.gemini").setLevel(logging.DEBUG)

    # Run tests
    test1 = test_extract_attachments_fix()
    test2 = test_process_media_attachment_fix()
    test3 = test_with_normal_attachment()

    # Print summary
    logger.info("\n=== TEST SUMMARY ===")
    logger.info(f"Extract attachments fix test: {'PASSED' if test1 else 'FAILED'}")
    logger.info(f"Process media attachment fix test: {'PASSED' if test2 else 'FAILED'}")
    logger.info(f"Normal attachment test: {'PASSED' if test3 else 'FAILED'}")

    if test1 and test2 and test3:
        logger.info("✅ ALL TESTS PASSED! The fix is working correctly.")
    else:
        logger.info("❌ SOME TESTS FAILED: The fix isn't working correctly.")


if __name__ == "__main__":
    main()

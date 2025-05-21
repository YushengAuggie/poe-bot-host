#!/usr/bin/env python
"""
Test script for Gemini media processing functions.

This script directly tests the Gemini bot's media processing capabilities
by calling the _process_media_attachment and _prepare_media_parts functions.

Usage:
    python scripts/test_media_processing.py
"""

import base64
import logging
import os
import sys

# Add the project root to path so bots module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bots.gemini import Gemini20FlashBot, get_client

# Set up logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("media_test")


class SimpleAttachment:
    """Simple attachment class for testing."""

    def __init__(self, name, content_type, content):
        """Initialize the attachment.

        Args:
            name: Filename
            content_type: MIME type
            content: Binary content or base64 string
        """
        self.name = name
        self.content_type = content_type
        self.content = content


def test_image_processing():
    """Test image processing directly."""
    # Path to test image
    image_path = os.path.join(os.path.dirname(__file__), "duck.jpg")

    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: duck.jpg not found at {image_path}")
        return

    # Read the image
    with open(image_path, "rb") as f:
        image_data = f.read()

    print(f"Loaded duck.jpg: {len(image_data)} bytes")

    # Create attachment
    attachment = SimpleAttachment("duck.jpg", "image/jpeg", image_data)

    # Create bot instance
    bot = Gemini20FlashBot()
    print(f"Created {bot.__class__.__name__}")
    print(f"Model: {bot.model_name}")
    print(f"Multimodal model: {getattr(bot, 'multimodal_model_name', 'Not set')}")

    # Test image processing
    print("\n=== Testing _process_media_attachment ===")
    media_data = bot._process_media_attachment(attachment)

    if media_data:
        print("Successfully processed attachment")
        print(f"MIME type: {media_data['mime_type']}")
        print(f"Data size: {len(media_data['data'])} bytes")
        print(f"Data type: {type(media_data['data']).__name__}")

        # Test preparing media parts
        print("\n=== Testing _prepare_media_parts ===")
        media_parts = bot._prepare_media_parts([attachment])
        print(f"Created {len(media_parts)} media parts")

        for i, part in enumerate(media_parts):
            print(f"Part {i+1} type: {type(part).__name__}")
            if isinstance(part, dict):
                print(f"Part {i+1} keys: {list(part.keys())}")
                if "inline_data" in part:
                    inline_data = part["inline_data"]
                    print(f"  inline_data keys: {list(inline_data.keys())}")
                    mime_type = inline_data.get("mime_type")
                    data = inline_data.get("data")
                    print(f"  mime_type: {mime_type}")
                    print(f"  data type: {type(data).__name__}")
                    print(f"  data size: {len(data) if data else 0} bytes")

        # Test Gemini client creation
        print("\n=== Testing get_client ===")
        # Check for API key
        from utils.api_keys import get_api_key

        api_key = get_api_key("GOOGLE_API_KEY")
        if not api_key:
            print("WARNING: No Google API key found in environment variables")
            print("To use the actual Google API, set the GOOGLE_API_KEY environment variable")
        else:
            print(f"Found API key (starts with: {api_key[:3]}...)")

            # Try to create clients for both models
            standard_client = get_client(bot.model_name)
            print(f"Standard client created: {standard_client is not None}")

            if hasattr(bot, "multimodal_model_name"):
                multimodal_client = get_client(bot.multimodal_model_name)
                print(f"Multimodal client created: {multimodal_client is not None}")

                # Check Google Generative AI version
                try:
                    import google.generativeai as genai

                    print(
                        f"Google Generative AI version: {getattr(genai, '__version__', 'unknown')}"
                    )

                    # Check if types module exists
                    if hasattr(genai, "types"):
                        print("Types module exists")
                        if hasattr(genai.types, "Part") and hasattr(genai.types.Part, "from_bytes"):
                            print("Part.from_bytes method exists")
                        else:
                            print("Part.from_bytes method not found")
                    else:
                        print("Types module not found")
                except ImportError:
                    print("Failed to import google.generativeai")
                except Exception as e:
                    print(f"Error checking genai version: {str(e)}")

                # Test direct API call if we have a multimodal client
                if multimodal_client:
                    print("\n=== Testing direct API call with multimodal client ===")
                    try:
                        # Use dictionary format since it's more compatible
                        content = [
                            {
                                "inline_data": {
                                    "mime_type": media_data["mime_type"],
                                    "data": media_data["data"],
                                }
                            },
                            {"text": "What is in this image? Describe it in detail."},
                        ]

                        print("Calling generate_content with content structure:")
                        print(f"- Content has {len(content)} items")
                        print(f"- First item type: {type(content[0]).__name__}")
                        print(f"- First item has inline_data: {'inline_data' in content[0]}")

                        # Make the API call
                        response = multimodal_client.generate_content(content)

                        # Process response
                        print("\nAPI Response:")
                        print(f"Response type: {type(response).__name__}")
                        if hasattr(response, "text"):
                            print(f"Response text: {response.text[:200]}...")
                        else:
                            print("Response has no text attribute")

                        if hasattr(response, "parts"):
                            print(f"Response has {len(response.parts)} parts")
                            for i, part in enumerate(response.parts):
                                if hasattr(part, "text"):
                                    print(f"Part {i+1} text: {part.text[:100]}...")
                        else:
                            print("Response has no parts attribute")

                    except Exception as e:
                        print(f"Error calling API: {str(e)}")
                        import traceback

                        print(traceback.format_exc())
    else:
        print("Failed to process attachment")


if __name__ == "__main__":
    test_image_processing()

#!/usr/bin/env python
"""
Test script for Gemini bot image handling via API.

This script tests the Gemini bot's ability to process and analyze images
through the Poe Bots API, with several testing options.

Usage:
    python test_gemini_image_api.py [path/to/image.jpg]
    python test_gemini_image_api.py --auto  # Generates a test image
"""

import argparse
import base64
import io
import json
import os
import sys
from typing import Any, Dict, Optional

import requests

# Check if Google API key is set
if "GOOGLE_API_KEY" not in os.environ:
    print("WARNING: GOOGLE_API_KEY not found in environment.")
    print("Image processing may fail if the server doesn't have access to this key.")

# Set up argument parser
parser = argparse.ArgumentParser(description="Test Gemini bot image processing via API")
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
parser.add_argument(
    "--host", default="http://localhost:8000", help="Host URL (default: http://localhost:8000)"
)
parser.add_argument("--debug", action="store_true", help="Enable debug output")
parser.add_argument(
    "--auto", action="store_true", help="Auto-generate test image and run test automatically"
)
parser.add_argument(
    "--endpoint",
    default="bot",
    help="API endpoint (default: 'bot' for FastAPI direct, use 'api/chat' for Poe-style API)",
)
args = parser.parse_args()


def print_debug(message: str):
    """Print debug message if debug flag is set."""
    if args.debug:
        print(f"DEBUG: {message}")


def test_image_processing(
    image_path: str, bot_name: str, prompt: str, base_url: str
) -> Dict[str, Any]:
    """Test image processing with the Gemini bot via API.

    Args:
        image_path: Path to the image file
        bot_name: Name of the bot to test
        prompt: Text prompt to send with the image
        base_url: Base URL of the API

    Returns:
        Bot response
    """
    url = f"{base_url}/{bot_name.lower()}"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer dummytoken",  # Since allow_without_key=True in our bot
    }

    print_debug(f"Testing bot at URL: {url}")
    print_debug(f"Headers: {headers}")

    try:
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Error: Image file not found: {image_path}")
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

        # Prepare query message with attachment
        query_message = {
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

        print_debug(
            f"Added attachment: {filename} ({content_type}), size: {len(file_content)} bytes"
        )
        print_debug(f"Using prompt: {prompt}")

        # Send request to the bot
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
        print_debug(f"Response status code: {status_code}")

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
            "content": content,
            "content_type": content_type,
        }
    except Exception as e:
        print(f"Error testing bot: {str(e)}")
        return {"status_code": 0, "error": str(e), "content": None, "content_type": "error"}


def extract_text_from_sse(sse_content: str) -> str:
    """Extract text from Server-Sent Events response.

    Args:
        sse_content: The SSE response content

    Returns:
        Extracted text
    """
    combined_text = ""
    for line in sse_content.strip().split("\n"):
        if line.startswith("data: "):
            # Parse the JSON data
            try:
                data_json = json.loads(line[6:])  # Skip "data: "
                if "text" in data_json:
                    combined_text += data_json["text"]
            except json.JSONDecodeError:
                pass

    return combined_text


def create_test_image() -> tuple:
    """Create a test image for automated testing.

    Returns:
        Tuple of (file_content, filename, content_type)
    """
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create a colored square with text
        img = Image.new("RGB", (400, 400), color=(25, 25, 112))  # Midnight blue

        # Add some text
        draw = ImageDraw.Draw(img)

        # Try to use a font if available, otherwise use default
        try:
            font = ImageFont.truetype("Arial", 32)
        except IOError:
            font = None

        draw.text((100, 150), "Gemini Test Image", fill=(255, 255, 255), font=font)
        draw.text((100, 200), "This is a simple test", fill=(255, 255, 255), font=font)
        draw.text((100, 250), "with generated content", fill=(255, 255, 255), font=font)

        # Draw some shapes
        draw.rectangle([50, 50, 350, 350], outline=(255, 255, 255), width=2)
        draw.ellipse([150, 300, 250, 350], fill=(255, 0, 0))

        # Save to bytes
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)

        print_debug("Created test image with PIL")
        return buffer.getvalue(), "test_image.png", "image/png"

    except ImportError:
        print_debug("PIL not installed. Using hardcoded test image.")
        # Return a simple test image if PIL is not available
        return (
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAGQAAABkCAYAAABw4pVUAAAAnElEQVR4nO3RAQ0AMAgAoJviyWgFQT42+aKrwJQAAAAAAAAAAAAAAAAAAADgRd7ZdgX3DJqbZnAcR1F0Y9ybZnAcR1F0Y9ybZnAcR1F0Y9ybZnAcR1F0Y9ybZnAcR1F0Y9ybZnAcR1F0Y9ybZnAcR1F0Y9ybZnAcR1F0Y9ybZnAcR1F0Y9ybZnAcR1F0Y9ybZnAcR1F0Y9ybZnAcR1fZ4QUBAAAAAAAAAADAV3wCH3B7vS8AAAAASUVORK5CYII="
            ),
            "test_image.png",
            "image/png",
        )


def test_api_chat_endpoint(
    image_path: str, bot_name: str, prompt: str, base_url: str
) -> Dict[str, Any]:
    """Test image processing with the Gemini bot via API/chat endpoint.

    Args:
        image_path: Path to the image file or auto-generated content
        bot_name: Name of the bot to test
        prompt: Text prompt to send with the image
        base_url: Base URL of the API

    Returns:
        Bot response
    """
    url = f"{base_url}/api/chat"

    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer fake-api-key",  # Default key used by run_local.py
    }

    print_debug(f"Testing bot using Poe API at URL: {url}")
    print_debug(f"Headers: {headers}")

    try:
        # Handle auto-generated or file-based image
        if isinstance(image_path, tuple):
            # Auto-generated image
            file_content, filename, content_type = image_path
        else:
            # Check if image file exists
            if not os.path.exists(image_path):
                print(f"Error: Image file not found: {image_path}")
                sys.exit(1)

            # Get file type from extension
            import mimetypes

            content_type, _ = mimetypes.guess_type(image_path)
            if not content_type:
                content_type = "application/octet-stream"

            # Read file contents
            with open(image_path, "rb") as f:
                file_content = f.read()

            # Get filename
            filename = os.path.basename(image_path)

        # Encode binary data as base64 for JSON transport
        file_content_b64 = base64.b64encode(file_content).decode("utf-8")

        print_debug(
            f"Using attachment: {filename} ({content_type}), size: {len(file_content)} bytes"
        )
        print_debug(f"Using prompt: {prompt}")

        # Prepare payload for the chat API endpoint
        payload = {
            "bot": bot_name,
            "message": prompt,
            "protocol_version": "2",
            "attachments": [
                {"name": filename, "content_type": content_type, "content": file_content_b64}
            ],
        }

        # Send request to the bot
        response = requests.post(
            url,
            headers=headers,
            json=payload,
        )

        status_code = response.status_code
        print_debug(f"Response status code: {status_code}")

        # Response content will be Server-Sent Events
        content = response.text
        content_type = "sse"

        return {
            "status_code": status_code,
            "content": content,
            "content_type": content_type,
        }
    except Exception as e:
        print(f"Error testing bot: {str(e)}")
        import traceback

        print(traceback.format_exc())
        return {"status_code": 0, "error": str(e), "content": None, "content_type": "error"}


def main():
    """Main entry point."""
    print("\n=== Gemini Image Processing API Test ===\n")

    # Check if we're using auto-generated image
    if args.auto:
        print("Auto-generating test image...")
        image_data = create_test_image()
        image_src = "auto-generated"
    else:
        image_data = args.image_path
        image_src = args.image_path

    print(f"Testing with image: {image_src}")
    print(f"Bot name: {args.bot}")
    print(f"API URL: {args.host}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Prompt: {args.prompt}")
    print("\nSending request...")

    # Select the appropriate endpoint testing function
    if args.endpoint == "api/chat":
        result = test_api_chat_endpoint(image_data, args.bot, args.prompt, args.host)
    else:
        # Default to original bot endpoint
        result = test_image_processing(
            image_data if not args.auto else args.image_path, args.bot, args.prompt, args.host
        )

    print(f"\nStatus Code: {result['status_code']}")

    if result["content_type"] == "json":
        print("\nResponse JSON:")
        bot_message = None

        # Extract the text response from the JSON structure
        if isinstance(result["content"], dict) and "messages" in result["content"]:
            messages = result["content"]["messages"]
            if messages and len(messages) > 0:
                bot_message = messages[0].get("content", "No response content")

        if bot_message:
            print("\nBot Response:")
            print(f"{bot_message}")
        else:
            print(json.dumps(result["content"], indent=2))
    elif result["content_type"] in ["text", "sse"]:
        print("\nResponse Content:")
        # Handle Server-Sent Events format
        if "data:" in result["content"]:
            extracted_text = extract_text_from_sse(result["content"])
            print(extracted_text)

            # Check if the response mentions the image
            image_terms = ["image", "picture", "photo", "see", "shows", "blue", "red", "color"]
            if any(term in extracted_text.lower() for term in image_terms):
                print("\n✅ SUCCESS: The bot appears to be processing the image correctly!")
            else:
                print("\n⚠️ WARNING: The bot response doesn't seem to be describing the image.")
        else:
            print(result["content"])
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""
Improved test script for Gemini bot image handling.

This script tests the Gemini bot's ability to process and analyze images.
It directly uses the Gemini20FlashBot class to ensure proper model selection
for multimodal content.

Usage:
    python test_gemini_image_improved.py [path/to/image.jpg]
"""

import argparse
import os
import sys
import base64
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("gemini_test")

# Add the project root to path so modules can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the Gemini bot class
from bots.gemini import Gemini20FlashBot
from utils.api_keys import get_api_key

# Set up argument parser
parser = argparse.ArgumentParser(description="Test Gemini bot image processing")
parser.add_argument("image_path", nargs="?", default="assets/test_images/duck.jpg", 
                   help="Path to image file (default: assets/test_images/duck.jpg)")
parser.add_argument("--prompt", default="Describe what you see in this image in detail.", 
                   help="Text prompt to send with the image")
parser.add_argument("--debug", action="store_true", help="Enable debug output")
args = parser.parse_args()

def print_debug(message: str):
    """Print debug message if debug flag is set."""
    if args.debug:
        logger.debug(message)

async def test_image_processing():
    """Test image processing with the Gemini bot."""
    print("\n=== Gemini Image Processing Test ===\n")
    print(f"Testing with image: {args.image_path}")
    print(f"Prompt: {args.prompt}")
    print(f"Using multimodal model: {Gemini20FlashBot.multimodal_model_name}")
    print("\nSending request...\n")
    
    try:
        # Check if image file exists
        if not os.path.exists(args.image_path):
            print(f"Error: Image file not found: {args.image_path}")
            return
            
        # Get file type from extension
        import mimetypes
        content_type, _ = mimetypes.guess_type(args.image_path)
        if not content_type:
            content_type = "application/octet-stream"
        
        # Read file contents
        with open(args.image_path, "rb") as f:
            file_content = f.read()
        
        print(f"Image size: {len(file_content)} bytes")
        print(f"Image type: {content_type}")
        
        # Initialize the Gemini bot
        bot = Gemini20FlashBot()
        
        # Verify API key
        api_key = get_api_key("GOOGLE_API_KEY")
        if not api_key:
            print("ERROR: GOOGLE_API_KEY not found in environment")
            return
        else:
            print(f"Found Google API key: {api_key[:5]}...")
        
        # Create a mock attachment
        class MockAttachment:
            def __init__(self, name, content_type, content):
                self.name = name
                self.content_type = content_type
                self.content = content
                self.url = f"file://{name}"
        
        filename = os.path.basename(args.image_path)
        attachment = MockAttachment(filename, content_type, file_content)
        
        # Process the attachment
        print("Processing attachment...")
        media_data = bot._process_media_attachment(attachment)
        
        if not media_data:
            print("ERROR: Failed to process media attachment")
            return
            
        print(f"Media data extracted successfully: {media_data['mime_type']}, {len(media_data['data'])} bytes")
            
        # Prepare media parts
        print("Preparing media parts...")
        media_parts = bot._prepare_media_parts([attachment])
        
        if not media_parts:
            print("ERROR: Failed to prepare media parts")
            return
            
        print(f"Media parts prepared successfully: {len(media_parts)} parts")
        
        # Prepare content
        print("Preparing content with media...")
        content = bot._prepare_content(args.prompt, media_parts)
        
        print(f"Content prepared successfully: {len(content)} items")
        
        # Get the client for the multimodal model
        print(f"Getting client for multimodal model: {bot.multimodal_model_name}")
        from bots.gemini import get_client
        client = get_client(bot.multimodal_model_name)
        
        if not client:
            print(f"ERROR: Failed to get client for model {bot.multimodal_model_name}")
            return
            
        print("Client initialized successfully")
        
        # Create a mock query request
        class MockQueryRequest:
            def __init__(self, message_id="test123"):
                self.message_id = message_id
                self.query = []
        
        query = MockQueryRequest()
        
        # Process the request
        print("Processing request with Gemini API...")
        print("\n=== Response from Gemini API ===\n")
        
        # Use direct API call to avoid streaming complexities
        response = client.generate_content(content)
        
        if hasattr(response, "text"):
            print(response.text)
        elif hasattr(response, "parts"):
            for part in response.parts:
                if hasattr(part, "text"):
                    print(part.text)
        else:
            print("No text found in response")
            
    except Exception as e:
        import traceback
        print(f"Error: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    # Run the async function
    import asyncio
    asyncio.run(test_image_processing())
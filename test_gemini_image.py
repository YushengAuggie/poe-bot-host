#!/usr/bin/env python
"""
Test script for Gemini bot image handling.

This script tests the Gemini bot's ability to process and analyze images.
It sends an image to the Gemini bot and prints the response.

Usage:
    python test_gemini_image.py [path/to/image.jpg]
"""

import argparse
import os
import sys
import json
import base64
from typing import Dict, Any

import requests

# Add the project root to path so modules can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up argument parser
parser = argparse.ArgumentParser(description="Test Gemini bot image processing")
parser.add_argument("image_path", nargs="?", default="assets/test_images/duck.jpg",
                   help="Path to image file (default: assets/test_images/duck.jpg)")
parser.add_argument("--bot", default="gemini20flashbot", 
                   help="Bot name to test (default: gemini20flashbot)")
parser.add_argument("--prompt", default="Describe what you see in this image in detail.", 
                   help="Text prompt to send with the image")
parser.add_argument("--host", default="http://localhost:8000",
                   help="Host URL (default: http://localhost:8000)")
parser.add_argument("--debug", action="store_true", help="Enable debug output")
args = parser.parse_args()

def print_debug(message: str):
    """Print debug message if debug flag is set."""
    if args.debug:
        print(f"DEBUG: {message}")

def test_image_processing(image_path: str, bot_name: str, prompt: str, base_url: str) -> Dict[str, Any]:
    """Test image processing with the Gemini bot.
    
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
        "Authorization": "Bearer dummytoken"  # Since allow_without_key=True in our bot
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
        file_content_b64 = base64.b64encode(file_content).decode('utf-8')
        
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
                    "content": file_content_b64
                }
            ]
        }
        
        print_debug(f"Added attachment: {filename} ({content_type}), size: {len(file_content)} bytes")
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

def main():
    """Main entry point."""
    print("\n=== Gemini Image Processing Test ===\n")
    print(f"Testing with image: {args.image_path}")
    print(f"Bot name: {args.bot}")
    print(f"API URL: {args.host}")
    print(f"Prompt: {args.prompt}")
    print("\nSending request...")
    
    result = test_image_processing(
        args.image_path,
        args.bot,
        args.prompt,
        args.host
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
    elif result["content_type"] == "text":
        print("\nResponse Content:")
        print(result["content"])
    else:
        print(f"\nError: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    main()
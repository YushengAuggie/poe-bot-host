#!/usr/bin/env python
"""
Test script for direct Gemini API access with image input.

This script bypasses the bot framework and directly tests the Gemini API
with the provided Google API key.
"""

import os
import sys
from typing import Optional

# Add the project root to path so utils module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_keys import get_api_key


def test_gemini_api_with_image():
    """Test Gemini API with image input directly."""
    print("Testing Gemini API with image input directly...")

    # Get the API key
    api_key = get_api_key("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found in environment variables")
        return False

    print(f"Found GOOGLE_API_KEY (starts with {api_key[:5]}...)")

    # Try to import google.generativeai
    try:
        import google.generativeai as genai

        print("Successfully imported google.generativeai module")
    except ImportError:
        print("ERROR: Failed to import google.generativeai module")
        print("Please install it with: pip install google-generativeai")
        return False

    # Configure API key
    genai.configure(api_key=api_key)

    # Check if we can list models
    try:
        models = list(genai.list_models())
        print(f"Successfully listed {len(models)} models")

        # Print available model names
        print("Available models:")
        for model in models:
            print(f"  - {model.name}")
    except Exception as e:
        print(f"ERROR: Failed to list models: {str(e)}")

    # Try to load image
    image_path = os.path.join(os.path.dirname(__file__), "duck.jpg")
    if not os.path.exists(image_path):
        print(f"ERROR: Image file not found: {image_path}")
        return False

    print(f"Loading image file: {image_path}")
    try:
        with open(image_path, "rb") as f:
            image_data = f.read()
        print(f"Successfully loaded image: {len(image_data)} bytes")
    except Exception as e:
        print(f"ERROR: Failed to load image: {str(e)}")
        return False

    # Try to create a dictionary-based format (old API style)
    try:
        # Check the genai package version
        if hasattr(genai, "__version__"):
            print(f"Google Generative AI package version: {genai.__version__}")

        # Try to find a suitable vision model
        vision_model_names = [
            "gemini-1.5-flash",  # New recommended model
            "gemini-1.5-pro",
            "gemini-2.0-flash",
            "gemini-2.0-pro",
            "gemini-2.5-flash-preview",
            "gemini-pro-vision",  # Deprecated but kept as fallback
        ]

        vision_model_name = None
        for name in vision_model_names:
            for model in models:
                if name in model.name:
                    vision_model_name = model.name
                    break
            if vision_model_name:
                break

        if not vision_model_name:
            print("WARNING: No vision model found, using gemini-pro-vision")
            vision_model_name = "gemini-pro-vision"

        print(f"Using vision model: {vision_model_name}")

        # Create a content dict in the old format first
        content = {
            "contents": [
                {
                    "parts": [
                        {"inline_data": {"mime_type": "image/jpeg", "data": image_data}},
                        {"text": "What do you see in this image?"},
                    ]
                }
            ]
        }

        print("Created content dictionary for API request")

        # Try the new model API first
        try:
            model = genai.GenerativeModel(vision_model_name)
            print("Initialized GenerativeModel")

            # Format for newer API
            content_parts = [
                {"inline_data": {"mime_type": "image/jpeg", "data": image_data}},
                "What do you see in this image?",
            ]

            print("Sending request to API...")
            response = model.generate_content(content_parts)

            print("\n=== API Response ===")
            print(response.text)
            return True
        except Exception as e:
            print(f"ERROR with model API: {str(e)}")
            print("Trying alternative API approach...")

            # Fallback to direct API call
            try:
                response = genai.generate_content(model=vision_model_name, contents=content_parts)

                print("\n=== API Response (alternate API) ===")
                print(response.text)
                return True
            except Exception as e2:
                print(f"ERROR with alternate API approach: {str(e2)}")
                return False

    except Exception as e:
        print(f"ERROR: General failure in API call: {str(e)}")
        return False


if __name__ == "__main__":
    success = test_gemini_api_with_image()
    if success:
        print("\nTest completed successfully!")
    else:
        print("\nTest failed!")

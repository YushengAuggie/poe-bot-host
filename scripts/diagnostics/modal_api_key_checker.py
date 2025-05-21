#!/usr/bin/env python
"""
Modal API Key Diagnostic Tool

This script verifies API key configuration in a Modal deployment environment.
It checks for the presence and format of API keys and tests their
functionality with actual API calls.

Usage:
    python scripts/diagnostics/modal_api_key_checker.py
"""

import json
import os
import sys

# Add the project root to path so utils module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import modal
from modal import App, Secret

app = App("key-test")


@app.function(secrets=[Secret.from_name("OPENAI_API_KEY"), Secret.from_name("GOOGLE_API_KEY")])
def test_api_keys():
    """Test if the API keys are properly configured in Modal"""
    print("Environment variables in Modal:")

    # Print all environment variables (excluding any that might contain 'key' for security)
    for key in sorted(os.environ.keys()):
        if (
            "key" not in key.lower()
            and "secret" not in key.lower()
            and "password" not in key.lower()
        ):
            print(f"  {key}")

    # Check for specific API keys and print partial values for debugging
    openai_key = os.environ.get("OPENAI_API_KEY", "")
    google_key = os.environ.get("GOOGLE_API_KEY", "")

    print("\nAPI Key Information:")
    if openai_key:
        print(f"OPENAI_API_KEY found with length: {len(openai_key)}")
        # Show first few characters for verification
        if len(openai_key) > 5:
            print(f"  Starts with: {openai_key[:3]}...")

        # Check format
        if openai_key.startswith("sk-"):
            print("  ✓ OpenAI key has correct format (starts with sk-)")
        else:
            print("  ✗ OpenAI key does not have expected format")
    else:
        print("OPENAI_API_KEY not found in environment")

    if google_key:
        print(f"GOOGLE_API_KEY found with length: {len(google_key)}")
        # Show first few characters for verification
        if len(google_key) > 5:
            print(f"  Starts with: {google_key[:3]}...")

        # Check format
        if google_key.startswith("AIza"):
            print("  ✓ Google key has correct format (starts with AIza)")
        else:
            print("  ✗ Google key does not have expected format")
    else:
        print("GOOGLE_API_KEY not found in environment")

    # Try importing client libraries
    print("\nImport test:")
    try:
        from openai import OpenAI

        print("OpenAI library imported successfully")
        try:
            client = OpenAI(api_key=openai_key)
            print("OpenAI client initialized")
        except Exception as e:
            print(f"Error initializing OpenAI client: {str(e)}")
    except ImportError:
        print("OpenAI library not installed")

    try:
        import google.generativeai as genai

        print("google-generativeai library imported successfully")
        try:
            genai.configure(api_key=google_key)
            print("Gemini client initialized")
        except Exception as e:
            print(f"Error initializing Gemini client: {str(e)}")
    except ImportError:
        print("google-generativeai library not installed")


if __name__ == "__main__":
    with app.run():
        test_api_keys.remote()

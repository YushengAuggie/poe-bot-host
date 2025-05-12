#!/usr/bin/env python
"""
Check if GOOGLE_API_KEY exists and is configured properly
"""

import os
import sys

# Add the project root to path so utils module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.api_keys import get_api_key


def main():
    """Check for Google API key"""
    print("Checking for Google API key...")

    # Check environment directly
    env_key = os.environ.get("GOOGLE_API_KEY")
    if env_key:
        print(f"Found GOOGLE_API_KEY in environment: {env_key[:5]}...")
    else:
        print("GOOGLE_API_KEY not found in environment variables")

    # Try using get_api_key function
    api_key = get_api_key("GOOGLE_API_KEY")
    if api_key:
        print(f"get_api_key found GOOGLE_API_KEY: {api_key[:5]}...")
    else:
        print("get_api_key could not find GOOGLE_API_KEY")

    # Try importing google.generativeai
    try:
        import google.generativeai as genai

        print("google.generativeai package is installed")

        if api_key:
            print("Testing configuration with API key...")
            genai.configure(api_key=api_key)
            try:
                models = genai.list_models()
                print(f"API key is valid - {len(models)} models available")
            except Exception as e:
                print(f"API key may be invalid: {str(e)}")

    except ImportError:
        print("google.generativeai package is NOT installed")


if __name__ == "__main__":
    main()

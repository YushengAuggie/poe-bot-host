#!/usr/bin/env python
"""
Modal Secret Access Tester

This script tests direct access to Modal secrets and compares different
methods of accessing secrets to help troubleshoot configuration issues.
It provides detailed output on which secrets are accessible and how they
can be accessed.

Usage:
    python scripts/diagnostics/secret_access_tester.py
"""

import json
import os
import sys

import modal
from modal import App, Secret

# Add the project root to path so utils module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

app = App("secret-test")

@app.function(
    secrets=[
        Secret.from_name("OPENAI_API_KEY"),
        Secret.from_name("GOOGLE_API_KEY")
    ]
)
def test_secrets():
    """Test if the API keys are properly accessible"""
    print("Testing access to secrets...")

    # Print all environment variables
    print("\nAll environment variables:")
    for key in sorted(os.environ.keys()):
        if 'key' not in key.lower() and 'secret' not in key.lower() and 'password' not in key.lower():
            print(f"  {key}")
        else:
            print(f"  {key}: [REDACTED]")

    # Try to read the API key using our utility function
    print("\nTesting API key utility function:")
    try:
        # Add the current directory to Python path so utils module can be found
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))

        from utils.api_keys import get_api_key

        try:
            openai_key = get_api_key("OPENAI_API_KEY")
            print(f"OpenAI API key retrieved: {bool(openai_key)}")
            if openai_key:
                print(f"  Length: {len(openai_key)}")
                print(f"  Starts with: {openai_key[:3]}...")
                print(f"  Ends with: ...{openai_key[-3:]}")
        except Exception as e:
            print(f"Error getting OpenAI API key: {str(e)}")

        try:
            google_key = get_api_key("GOOGLE_API_KEY")
            print(f"Google API key retrieved: {bool(google_key)}")
            if google_key:
                print(f"  Length: {len(google_key)}")
                print(f"  Starts with: {google_key[:3]}...")
                print(f"  Ends with: ...{google_key[-3:]}")
        except Exception as e:
            print(f"Error getting Google API key: {str(e)}")

    except ImportError as e:
        print(f"Import error: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

    # Try directly accessing the secrets
    print("\nAccessing secrets directly:")
    try:
        # Direct environment variable
        api_key_env = os.environ.get("api_key")
        if api_key_env:
            print(f"Found api_key in environment: {api_key_env[:3]}...")
        else:
            print("No api_key found in environment")

        # Try Secret API
        openai_secret = modal.Secret.from_name("OPENAI_API_KEY")
        google_secret = modal.Secret.from_name("GOOGLE_API_KEY")

        try:
            openai_value = openai_secret.get()
            print(f"OpenAI secret direct: {bool(openai_value)}")
            if openai_value:
                print(f"  Value: {openai_value[:3]}...")
        except Exception as e:
            print(f"Error getting OpenAI secret directly: {str(e)}")

        try:
            openai_dict = openai_secret.get_dict()
            print(f"OpenAI secret dict: {json.dumps(openai_dict)}")
        except Exception as e:
            print(f"Error getting OpenAI secret dict: {str(e)}")

        try:
            google_value = google_secret.get()
            print(f"Google secret direct: {bool(google_value)}")
            if google_value:
                print(f"  Value: {google_value[:3]}...")
        except Exception as e:
            print(f"Error getting Google secret directly: {str(e)}")

        try:
            google_dict = google_secret.get_dict()
            print(f"Google secret dict: {json.dumps(google_dict)}")
        except Exception as e:
            print(f"Error getting Google secret dict: {str(e)}")

    except Exception as e:
        print(f"Error accessing secrets: {str(e)}")

if __name__ == "__main__":
    with app.run():
        test_secrets.remote()

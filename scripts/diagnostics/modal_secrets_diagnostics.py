#!/usr/bin/env python
"""
Modal Secrets Diagnostic Tool

This script runs comprehensive diagnostics on Modal secrets configuration.
It checks environment variables, secret availability, and attempts to use
the secrets with actual API clients to verify full functionality.

Usage:
    python scripts/diagnostics/modal_secrets_diagnostics.py
"""

import json
import os
import sys

import modal
from modal import App, Secret

# Add the project root to path so utils module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

app = App("secrets-diagnostic")

# Create a Modal function that uses both secrets
@app.function(
    secrets=[
        Secret.from_name("OPENAI_API_KEY"),
        Secret.from_name("GOOGLE_API_KEY")
    ]
)
def test_secret_access():
    """Test detailed secret access to diagnose issues"""
    print("\n=== ENVIRONMENT VARIABLES ===")

    # Show all environment variables (filtering out sensitive values)
    for name, value in sorted(os.environ.items()):
        if 'key' in name.lower() or 'token' in name.lower() or 'secret' in name.lower():
            print(f"{name}: **REDACTED**")
        else:
            print(f"{name}: {value}")

    # Explicitly check for our keys
    print("\n=== CHECKING FOR SPECIFIC API KEYS ===")
    keys_to_check = [
        "OPENAI_API_KEY",
        "GOOGLE_API_KEY",
        # Check for local testing variants
        "LOCAL_OPENAI_API_KEY",
        "LOCAL_GOOGLE_API_KEY"
    ]

    for key_name in keys_to_check:
        value = os.environ.get(key_name)
        if value:
            # Show truncated value for verification
            preview = f"{value[:3]}...{value[-3:]}" if len(value) > 10 else "**too short**"
            print(f"✓ Found {key_name}: {preview}")
        else:
            print(f"✗ {key_name} not found")

    # Test the utility function
    try:
        print("\n=== TESTING get_api_key() FUNCTION ===")
        from utils.api_keys import get_api_key

        try:
            openai_key = get_api_key("OPENAI_API_KEY")
            preview = f"{openai_key[:3]}...{openai_key[-3:]}" if openai_key and len(openai_key) > 10 else "**invalid**"
            print(f"OpenAI API key via get_api_key(): {preview}")
        except Exception as e:
            print(f"Error getting OpenAI key: {str(e)}")

        try:
            google_key = get_api_key("GOOGLE_API_KEY")
            preview = f"{google_key[:3]}...{google_key[-3:]}" if google_key and len(google_key) > 10 else "**invalid**"
            print(f"Google API key via get_api_key(): {preview}")
        except Exception as e:
            print(f"Error getting Google key: {str(e)}")
    except ImportError:
        print("Could not import utils.api_keys module")

    # Test with OpenAI client
    try:
        print("\n=== TESTING OPENAI CLIENT ===")
        try:
            from openai import OpenAI

            try:
                from utils.api_keys import get_api_key
                api_key = get_api_key("OPENAI_API_KEY")
                print(f"Using API key: {api_key[:3]}... (length: {len(api_key)})")

                client = OpenAI(api_key=api_key)
                print("OpenAI client initialized successfully")

                try:
                    # Try a simple completion to verify the key works
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[{"role": "user", "content": "Say hello"}],
                        max_tokens=5
                    )
                    print(f"API response: {response.choices[0].message.content}")
                except Exception as e:
                    print(f"OpenAI API call failed: {str(e)}")
            except Exception as e:
                print(f"Could not get API key: {str(e)}")
        except ImportError:
            print("OpenAI package not installed")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    with app.run():
        test_secret_access.remote()

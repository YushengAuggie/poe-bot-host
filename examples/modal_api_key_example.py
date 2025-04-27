"""
Example of using Modal with the API key management system.

This demonstrates how to:
1. Use the Modal KEY secret to access API keys
2. Use service-specific secrets for backward compatibility
3. Check both local environment and Modal secrets

Run this script with:
```
modal run examples/modal_api_key_example.py
```

Or deploy with:
```
modal deploy examples/modal_api_key_example.py
```
"""

import os

import modal

from utils.api_keys import (
    create_modal_app,
    get_function_secrets,
    get_google_api_key,
    get_openai_api_key,
)

# Create a Modal app for the example
app = modal.App("api-key-example")

# Create a custom image with all required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "openai",
        "google-generativeai",
        "fastapi-poe>=0.0.21",
        "pydantic>=2.0.0",
        "python-dotenv>=1.0.0"
    ])
    .add_local_python_source("utils")
)

# Update app to use the image
app.image = image


@app.function(secrets=get_function_secrets(["openai", "google"]))
def check_api_keys():
    """
    Function that checks and prints available API keys from Modal secrets.
    """
    results = {}

    # Check for OpenAI API key
    try:
        openai_key = get_openai_api_key()
        # Only show prefix for security
        results["OPENAI_API_KEY"] = f"{openai_key[:5]}...{openai_key[-4:]}" if openai_key else None
    except ValueError as e:
        results["OPENAI_API_KEY"] = str(e)

    # Check for Google API key
    try:
        google_key = get_google_api_key()
        # Only show prefix for security
        results["GOOGLE_API_KEY"] = f"{google_key[:5]}...{google_key[-4:]}" if google_key else None
    except ValueError as e:
        results["GOOGLE_API_KEY"] = str(e)

    # Check environment variables directly
    for env_var in ["OPENAI_API_KEY", "GOOGLE_API_KEY"]:
        results[f"os.environ.get({env_var})"] = "Available" if env_var in os.environ else "Not found"

    return results


@app.function(secrets=[modal.Secret.from_name("KEY")])
def check_key_secret():
    """
    Function that directly checks the KEY secret from Modal.
    """
    results = {}
    for env_var in ["OPENAI_API_KEY", "GOOGLE_API_KEY"]:
        value = os.environ.get(env_var)
        if value:
            # Only show prefix for security
            results[env_var] = f"{value[:5]}...{value[-4:]}"
        else:
            results[env_var] = "Not found"

    return results


@app.local_entrypoint()
def main():
    """
    Entry point for the Modal app.
    """
    print("\n\n=== Testing API key access through Modal functions ===\n")
    results = check_api_keys.remote()
    for key, value in results.items():
        print(f"{key}: {value}")

    print("\n\n=== Testing direct KEY secret access ===\n")
    results = check_key_secret.remote()
    for key, value in results.items():
        print(f"{key}: {value}")

    print("\n\nTo set up the Modal KEY secret with both API keys:")
    print("1. modal secret create KEY")
    print("2. Add OPENAI_API_KEY=sk-... and GOOGLE_API_KEY=... when prompted")

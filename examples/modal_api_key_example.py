"""
Example of using Modal with API key management.

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
from typing import Dict, List, Any

import modal

# Create a Modal app for the example
app = modal.App("api-key-example")

# Create a custom image with all required dependencies
image = (
    modal.Image.debian_slim()
    .pip_install([
        "openai",
        "google-generativeai"
    ])
)

# Update app to use the image
app.image = image


def get_openai_api_key() -> str:
    """
    Get the OpenAI API key from environment variables.
    First checks local environment variables, then Modal secrets.

    Returns:
        str: The OpenAI API key

    Raises:
        ValueError: If the API key is not found
    """
    # First check local environment variables
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key

    # If not in local env, check if we're in a Modal environment with secrets
    try:
        is_modal_env = modal.is_local() is False
        if is_modal_env:
            # Try to find the key from different potential sources
            for env_var in ["OPENAI_API_KEY", "value"]:
                if env_var in os.environ:
                    return os.environ[env_var]
    except (ImportError, AttributeError):
        pass

    raise ValueError("OPENAI_API_KEY not found in environment or Modal secrets")


def get_google_api_key() -> str:
    """
    Get the Google API key from environment variables.
    First checks local environment variables, then Modal secrets.

    Returns:
        str: The Google API key

    Raises:
        ValueError: If the API key is not found
    """
    # First check local environment variables
    key = os.environ.get("GOOGLE_API_KEY")
    if key:
        return key

    # If not in local env, check if we're in a Modal environment with secrets
    try:
        is_modal_env = modal.is_local() is False
        if is_modal_env:
            # Try to find the key from different potential sources
            for env_var in ["GOOGLE_API_KEY", "value"]:
                if env_var in os.environ:
                    return os.environ[env_var]
    except (ImportError, AttributeError):
        pass

    raise ValueError("GOOGLE_API_KEY not found in environment or Modal secrets")


def get_function_secrets(services: List[str]) -> List[modal.Secret]:
    """
    Get the Modal secrets for the specified services.

    Args:
        services: List of service names (e.g., "openai", "google")

    Returns:
        list: List of Modal secrets
    """
    secrets_map = {
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "openai-secret": "openai-secret",
        "googlecloud-secret": "googlecloud-secret"
    }
    
    secrets = []
    
    # Add KEY secret which includes all API keys
    try:
        key_secret = modal.Secret.from_name("KEY")
        secrets.append(key_secret)
    except ValueError:
        # Continue if KEY secret not found
        pass
        
    # Also add service-specific secrets for backward compatibility
    for service in services:
        if service in secrets_map:
            try:
                service_secret = modal.Secret.from_name(secrets_map[service])
                secrets.append(service_secret)
            except ValueError:
                # Continue if this specific secret not found
                pass
    
    return secrets


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
import json
import os
from typing import Any, Dict

import modal


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
    if "OPENAI_API_KEY" in os.environ:
        return os.environ["OPENAI_API_KEY"]

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
    if "GOOGLE_API_KEY" in os.environ:
        return os.environ["GOOGLE_API_KEY"]

    raise ValueError("GOOGLE_API_KEY not found in environment or Modal secrets")


def get_google_credentials() -> Dict[str, Any]:
    """
    Get Google service account credentials from environment variables.
    First checks local environment variables, then Modal secrets.

    Returns:
        Dict[str, Any]: Service account credentials as a dictionary

    Raises:
        ValueError: If the credentials are not found
    """
    # First check if JSON string is in environment
    if "GOOGLE_APPLICATION_CREDENTIALS_JSON" in os.environ:
        return json.loads(os.environ["GOOGLE_APPLICATION_CREDENTIALS_JSON"])

    # Next, check if there's a path to a credentials file
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path and os.path.exists(cred_path):
        with open(cred_path, 'r') as f:
            return json.load(f)

    # Finally, check if we have the API key in Modal environment
    if "GOOGLE_API_KEY" in os.environ:
        # Return as a dict with the API key
        return {"api_key": os.environ["GOOGLE_API_KEY"]}

    raise ValueError("Google credentials not found in environment or Modal secrets")


def get_modal_secrets() -> Dict[str, str]:
    """
    Get the Modal secrets mapping for various services.

    Returns:
        Dict[str, str]: Map of service name to Modal secret name
    """
    return {
        "openai": "openai-secret",
        "google": "googlecloud-secret"
    }


def create_modal_app(name: str, packages: "list[str] | None" = None) -> modal.App:
    """
    Create a Modal app with the necessary configuration.

    Args:
        name: The name of the app
        packages: Additional packages to install

    Returns:
        modal.App: The configured Modal app
    """
    packages_to_install: "list[str]" = packages or []
    image = modal.Image.debian_slim().pip_install(packages_to_install)
    return modal.App(name, image=image)


def get_function_secrets(services: "list[str]") -> "list[modal.Secret]":
    """
    Get the Modal secrets for the specified services.

    Args:
        services: List of service names

    Returns:
        list: List of Modal secrets
    """
    secrets_map = get_modal_secrets()
    return [modal.Secret.from_name(secrets_map[service])
            for service in services if service in secrets_map]

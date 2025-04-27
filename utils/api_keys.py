import os
from typing import Any, Optional

try:
    import modal
except ImportError:
    modal = None


def get_api_key(key_name: str) -> str:
    """
    Get the API key from environment variables or Modal secrets.

    Args:
        key_name: Name of the API key environment variable

    Returns:
        str: The API key value

    Raises:
        ValueError: If the API key is not found
    """
    # First check environment variables
    key = os.environ.get(key_name)
    if key:
        return key

    # Then check Modal secrets if running in Modal
    if modal and hasattr(modal, "is_local") and not modal.is_local():
        try:
            secret = modal.Secret.from_name(key_name)
            return secret.get()
        except (ValueError, AttributeError):
            pass

    raise ValueError(f"{key_name} not found in environment variables or Modal secrets")


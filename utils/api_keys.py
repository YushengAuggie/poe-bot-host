import logging
import os
from typing import Any, Optional, Union, cast

# Set up detailed logging for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

try:
    import modal
    from modal import Secret

    logger.debug("Modal package imported successfully")
except ImportError:
    logger.warning("Modal package not available")
    modal = None

    # Type stub for when modal is not available
    class Secret:
        @classmethod
        def from_name(cls, name: str) -> "Secret":
            return cast("Secret", None)


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
    logger.debug(f"Attempting to get API key: {key_name}")

    # First check environment variables with the exact name (case-sensitive)
    key: Optional[str] = os.environ.get(key_name)
    if key:
        logger.debug(f"Found {key_name} in environment variables")
        return key

    # Try with lowercase (Modal sometimes provides env vars in lowercase)
    key_lowercase: Optional[str] = os.environ.get(key_name.lower())
    if key_lowercase:
        logger.debug(f"Found {key_name.lower()} in environment variables")
        return key_lowercase

    # Try looking for a local secret key for testing
    local_key_name: str = f"LOCAL_{key_name}"
    local_key: Optional[str] = os.environ.get(local_key_name)
    if local_key:
        logger.debug(f"Found {local_key_name} in environment")
        return local_key

    # Then check Modal secrets if running in Modal
    if modal:
        logger.debug("Modal is available, checking if we're running in Modal")

        # Debug Modal state
        is_local: bool = getattr(modal, "is_local", lambda: True)()
        logger.debug(f"Modal.is_local(): {is_local}")

        if not is_local:
            logger.debug(f"Running in Modal, trying to get secret {key_name}")
            try:
                # Try using Modal's Secret API, which may or may not be available
                # in the runtime context
                try:
                    secret: Secret = Secret.from_name(key_name)
                    logger.debug(f"Successfully retrieved secret reference for {key_name}")

                    # These methods may not be available in the runtime
                    try:
                        value: Optional[str] = getattr(secret, "get", lambda: None)()
                        if value:
                            logger.debug("Got value using secret.get()")
                            return value
                    except Exception as e:
                        logger.debug(f"Couldn't use secret.get(): {str(e)}")

                except Exception as e:
                    logger.error(f"Error accessing Modal Secret API: {str(e)}")
            except Exception as e:
                logger.error(f"Unexpected error getting Modal secret {key_name}: {str(e)}")

    # If we reach here, the key wasn't found
    logger.error(f"{key_name} not found in environment variables or Modal secrets")
    # For debugging: log all env vars (excluding values for security)
    logger.debug(f"Available environment variables: {list(os.environ.keys())}")

    raise ValueError(f"{key_name} not found in environment variables or Modal secrets")

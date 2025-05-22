import json
import logging
import time
from typing import Any, Optional, AsyncGenerator, Dict, Union

from fastapi_poe.types import (
    MetaResponse,
    PartialResponse,
    QueryRequest,
    SettingsRequest,
    SettingsResponse,
)

from utils.api_keys import get_api_key
from utils.base_bot import BaseBot

# Get the logger
logger = logging.getLogger(__name__)


# Define a stub class to simulate the Gemini client
# This avoids import errors but still allows the code to run
class GeminiClientStub:
    """Stub implementation for the Gemini client."""

    def __init__(self, model_name: Optional[str] = None, api_key: Optional[str] = None):
        self.model_name = model_name

    def generate_content(self, contents: Any, stream: bool = False):
        # Return an object with text property for compatibility
        model_name = self.model_name  # Store locally for use in inner class

        class StubResponse:
            def __init__(self):
                self.text = f"Gemini API ({model_name}) is not available. Please install the google-generativeai package."
                self.parts = []  # Add parts property for multimodal support

            def __iter__(self):
                # Make this iterable for streaming support
                yield self

            def __aiter__(self):
                # Make this async iterable for async streaming support
                return self

            async def __anext__(self):
                # This will yield one item then stop iteration
                if not hasattr(self, "_yielded"):
                    self._yielded = True
                    return self
                raise StopAsyncIteration

        return StubResponse()


# Initialize client globally
def get_client(model_name: str):
    """Get a Gemini client, falling back to a stub if not available.

    Args:
        model_name: The name of the Gemini model to use

    Returns:
        A Gemini client instance or stub if the real client is unavailable
    """
    try:
        # Try to import the actual Gemini client inside the function to avoid module-level import errors
        import google.generativeai as genai

        # Use our Google API key management
        api_key = get_api_key("GOOGLE_API_KEY")
        if not api_key:
            logger.error("Google API key not found")
            return None

        # Configure the API key at the module level
        genai.configure(api_key=api_key)  # type: ignore

        # Then create and return the model
        return genai.GenerativeModel(model_name=model_name)  # type: ignore
    except ImportError:
        logger.warning("Failed to import google.generativeai module")
        return GeminiClientStub(model_name=model_name)
    except Exception as e:
        logger.warning(f"Failed to initialize Gemini client: {str(e)}")
        return GeminiClientStub(model_name=model_name)

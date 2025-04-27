import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional, Union

from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest

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

    def generate_content(self, contents: str):
        # Return an object with text property for compatibility
        class StubResponse:
            def __init__(self):
                self.text = "Gemini API is not available. Please install the google-generativeai package."
        return StubResponse()


# Initialize client globally
def get_client():
    """Get a Gemini client, falling back to a stub if not available."""
    try:
        # Try to import the actual Gemini client inside the function to avoid module-level import errors
        import google.generativeai as genai

        # Use our Google API key management
        api_key = get_api_key("GOOGLE_API_KEY")
        
        # Configure the API key at the module level
        genai.configure(api_key=api_key)
        
        # Then create and return the model
        return genai.GenerativeModel(model_name="gemini-2.0-flash")
    except (ImportError, Exception) as e:
        logger.warning(f"Failed to initialize Gemini client: {str(e)}")
        return GeminiClientStub()


client = None


class GeminiBot(BaseBot):
    """A simple gemini bot."""

    # Override the bot name
    bot_name = "GeminiBot"

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        """Process the query and generate a response.

        Args:
            query: The query from the user

        Yields:
            Response chunks as PartialResponse or MetaResponse objects
        """
        try:
            # Extract the query contents
            user_message = self._extract_message(query)

            # Log the extracted message
            logger.debug(f"[{self.bot_name}] Received message: {user_message}")
            logger.debug(f"[{self.bot_name}] Message type: {type(user_message)}")
            logger.debug(f"[{self.bot_name}] Query type: {type(query)}")
            logger.debug(f"[{self.bot_name}] Query contents: {query.query}")

            # Add metadata about the bot if requested
            if user_message.lower().strip() == "bot info":
                metadata = self._get_bot_metadata()
                yield PartialResponse(text=json.dumps(metadata, indent=2))
                return

            # Initialize client if not already done
            global client
            if client is None:
                client = get_client()
                if client is None:
                    yield PartialResponse(text="Error: Google API key is not configured.")
                    return

            try:
                # For now, use non-streaming for compatibility with older API versions
                response = client.generate_content(f"{user_message}")

                # Simulate streaming by breaking up the response
                full_text = response.text
                # Break into ~10 character chunks
                chunk_size = 10
                for i in range(0, len(full_text), chunk_size):
                    chunk = full_text[i : i + chunk_size]
                    yield PartialResponse(text=chunk)
            except Exception as e:
                logger.error(f"Error calling Gemini API: {str(e)}")
                yield PartialResponse(text=f"Error: Could not get response from Gemini: {str(e)}")

        except Exception as e:
            # Let the parent class handle errors
            async for resp in super().get_response(query):
                yield resp

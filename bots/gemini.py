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
        model_name = self.model_name  # Store locally for use in inner class

        class StubResponse:
            def __init__(self):
                self.text = f"Gemini API ({model_name}) is not available. Please install the google-generativeai package."
        return StubResponse()


# Initialize client globally
def get_client(model_name: str):
    """Get a Gemini client, falling back to a stub if not available."""
    try:
        # Try to import the actual Gemini client inside the function to avoid module-level import errors
        import google.generativeai as genai

        # Use our Google API key management
        api_key = get_api_key("GOOGLE_API_KEY")

        # Configure the API key at the module level
        genai.configure(api_key=api_key)

        # Then create and return the model
        return genai.GenerativeModel(model_name=model_name)
    except (ImportError, Exception) as e:
        logger.warning(f"Failed to initialize Gemini client: {str(e)}")
        return GeminiClientStub(model_name=model_name)


# Base Gemini bot class that other model-specific bots will inherit from
class GeminiBaseBot(BaseBot):
    """Base class for Gemini bots."""

    model_name = "gemini-2.0-flash"  # Default model
    bot_name = "GeminiBaseBot"
    bot_description = "Base Gemini model bot."

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
                metadata["model_name"] = self.model_name
                yield PartialResponse(text=json.dumps(metadata, indent=2))
                return

            # Initialize client for this specific model
            client = get_client(self.model_name)
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


# Original Gemini Flash model (2.0)
class GeminiBot(GeminiBaseBot):
    """Original Gemini bot implementation (uses 2.0 Flash model)."""

    model_name = "gemini-2.0-flash"
    bot_name = "GeminiBot"
    bot_description = "Original Gemini bot using Gemini 2.0 Flash model."


# Gemini 2.0 Series
class Gemini20FlashBot(GeminiBaseBot):
    """Gemini 2.0 Flash model - optimized for speed and efficiency."""

    model_name = "gemini-2.0-flash"
    bot_name = "Gemini20FlashBot"
    bot_description = "Fast and efficient Gemini 2.0 Flash model, optimized for speed and next-gen features."


class Gemini20ProBot(GeminiBaseBot):
    """Gemini 2.0 Pro model - balanced performance."""

    model_name = "gemini-2.0-pro"
    bot_name = "Gemini20ProBot"
    bot_description = "Balanced Gemini 2.0 Pro model with enhanced capabilities."


# Gemini 2.5 Series
class Gemini25FlashBot(GeminiBaseBot):
    """Gemini 2.5 Flash Preview - optimized for adaptive thinking and cost efficiency."""

    model_name = "gemini-2.5-flash-preview-04-17"
    bot_name = "Gemini25FlashBot"
    bot_description = "Advanced Gemini 2.5 Flash Preview model for adaptive thinking and cost efficiency."


class Gemini25ProExpBot(GeminiBaseBot):
    """Gemini 2.5 Pro Preview - premium model for complex reasoning."""

    model_name = "gemini-2.5-pro-preview-03-25"
    bot_name = "Gemini25ProExpBot"
    bot_description = "Premium Gemini 2.5 Pro Experimental model for enhanced reasoning, multimodal understanding, and advanced coding."


# Experimental Models
class Gemini20FlashExpBot(GeminiBaseBot):
    """Gemini 2.0 Flash Experimental model."""

    model_name = "gemini-2.0-flash-exp"
    bot_name = "Gemini20FlashExpBot"
    bot_description = "Experimental Gemini 2.0 Flash model with latest features."


class Gemini20FlashThinkingBot(GeminiBaseBot):
    """Gemini 2.0 Flash Thinking Experimental model."""

    model_name = "gemini-2.0-flash-thinking-exp-01-21"
    bot_name = "Gemini20FlashThinkingBot"
    bot_description = "Experimental Gemini 2.0 Flash Thinking model with enhanced reasoning capabilities."


class Gemini20ProExpBot(GeminiBaseBot):
    """Gemini 2.0 Pro Experimental model."""

    model_name = "gemini-2.0-pro-exp-02-05"
    bot_name = "Gemini20ProExpBot"
    bot_description = "Experimental Gemini 2.0 Pro model with latest capabilities."

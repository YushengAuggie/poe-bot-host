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


# Base Gemini bot class that other model-specific bots will inherit from
class GeminiBaseBot(BaseBot):
    """Base class for Gemini bots."""

    model_name = "gemini-2.0-flash"  # Default model
    bot_name = "GeminiBaseBot"
    bot_description = "Base Gemini model bot."
    supports_image_input = True  # Enable image input by default

    def __init__(self, **kwargs):
        """Initialize the GeminiBaseBot."""
        super().__init__(**kwargs)
        self.supported_image_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]

    def _extract_attachments(self, query: QueryRequest) -> list:
        """Extract attachments from the query.

        Args:
            query: The query from the user

        Returns:
            List of attachments
        """
        attachments = []
        if isinstance(query.query, list) and query.query:
            last_message = query.query[-1]
            if hasattr(last_message, "attachments"):
                attachments = last_message.attachments
        return attachments

    def _process_image_attachment(self, attachment) -> Optional[Dict[str, Any]]:
        """Process an image attachment for Gemini.

        Args:
            attachment: The attachment object

        Returns:
            Dictionary with mime_type and data, or None if unsupported
        """
        # Check if attachment is an image and supported
        if (
            not hasattr(attachment, "content_type")
            or attachment.content_type not in self.supported_image_types
        ):
            logger.warning(
                f"Unsupported attachment type: {getattr(attachment, 'content_type', 'unknown')}"
            )
            return None

        # Access content via __dict__ to satisfy type checker (content is added by Poe but not in type definition)
        if hasattr(attachment, "content") and attachment.__dict__.get("content"):
            return {"mime_type": attachment.content_type, "data": attachment.__dict__["content"]}
        return None

    async def _handle_bot_info_request(self) -> PartialResponse:
        """Handle a request for bot information.

        Returns:
            Formatted response with bot metadata
        """
        metadata = self._get_bot_metadata()
        metadata["model_name"] = self.model_name
        metadata["supports_image_input"] = self.supports_image_input
        return PartialResponse(text=json.dumps(metadata, indent=2))

    def _prepare_image_parts(self, attachments: list) -> list:
        """Process image attachments into parts for Gemini API.

        Args:
            attachments: List of attachments from the query

        Returns:
            List of image parts formatted for Gemini API
        """
        image_parts = []

        for attachment in attachments:
            image_data = self._process_image_attachment(attachment)
            if image_data:
                try:
                    # We only need to validate the import is available
                    # Direct import used to verify the package is installed
                    _ = __import__("google.generativeai")
                    image_parts.append(
                        {
                            "mime_type": image_data["mime_type"],
                            "data": image_data["data"],
                        }
                    )
                except ImportError:
                    logger.warning("Could not import google.generativeai for image processing")

        return image_parts

    def _format_chat_history(self, query: QueryRequest) -> list[dict[str, object]]:
        """Extract and format chat history from the query for Gemini API.

        Args:
            query: The query from the user

        Returns:
            List of message dictionaries formatted for Gemini API
        """
        chat_history: list[dict[str, object]] = []

        if not isinstance(query.query, list):
            return chat_history

        for message in query.query:
            # Skip messages without proper attributes
            if not (hasattr(message, "role") and hasattr(message, "content")):
                continue

            # Map roles: user -> user, bot/assistant -> model
            role = (
                "user"
                if message.role == "user"
                else "model" if message.role in ["bot", "assistant"] else None
            )

            # Skip messages with unsupported roles
            if not role:
                continue

            chat_history.append({"role": role, "parts": [{"text": message.content}]})

        return chat_history

    def _prepare_content(
        self,
        user_message: str,
        image_parts: list,
        chat_history: Optional[list[dict[str, object]]] = None,
    ) -> list:
        """Prepare content for the Gemini API (text and/or images).

        Args:
            user_message: The user's text message
            image_parts: List of processed image parts
            chat_history: Optional chat history list

        Returns:
            Content formatted for Gemini API
        """
        # For multimodal queries (with images), we can't use chat history
        if image_parts:
            return [{"inline_data": part} for part in image_parts] + [{"text": user_message}]

        # For text-only queries, use chat history if available
        if chat_history:
            # If the last message is from the user, update it
            # Otherwise, add a new user message
            if chat_history and chat_history[-1]["role"] == "user":
                chat_history[-1]["parts"] = [{"text": user_message}]
            else:
                chat_history.append({"role": "user", "parts": [{"text": user_message}]})
            return chat_history

        # Single-turn text-only query
        return [{"text": user_message}]

    async def _process_streaming_response(self, response) -> AsyncGenerator[PartialResponse, None]:
        """Process a streaming text response from Gemini.

        Args:
            response: The response from Gemini API (streaming or not)

        Yields:
            Response chunks as PartialResponse objects
        """
        try:
            # Non-streaming case: If response has text attribute directly and isn't an iterable
            if hasattr(response, "text"):
                try:
                    if not hasattr(response, "__iter__") or not callable(response.__iter__):
                        logger.debug("Processing direct text response")
                        yield PartialResponse(text=response.text)
                        return
                except Exception:
                    # If checking the attribute fails, treat it as a direct response
                    logger.debug("Processing direct text response after attribute error")
                    yield PartialResponse(text=response.text)
                    return

            # Process streaming response
            logger.debug(f"Processing streaming response of type {type(response)}")

            # Process async iterator responses (for async streaming)
            if hasattr(response, "__aiter__"):
                try:
                    if callable(response.__aiter__):
                        logger.debug("Processing async iterable streaming response")
                        async for chunk in response:
                            if hasattr(chunk, "text") and chunk.text:
                                yield PartialResponse(text=chunk.text)
                            elif hasattr(chunk, "parts") and chunk.parts:
                                for part in chunk.parts:
                                    if hasattr(part, "text") and part.text:
                                        yield PartialResponse(text=part.text)
                        return
                except Exception as e:
                    logger.warning(f"Error during async iteration: {str(e)}")
                    # Return error response for this specific case
                    yield PartialResponse(text=f"Error streaming from Gemini: {str(e)}")
                    return

            # Process regular iterator responses (for sync streaming)
            if hasattr(response, "__iter__"):
                try:
                    if callable(response.__iter__):
                        logger.debug("Processing sync iterable streaming response")
                        chunks = list(response)

                        # Yield each chunk as a separate response
                        for chunk in chunks:
                            if hasattr(chunk, "text") and chunk.text:
                                yield PartialResponse(text=chunk.text)
                            elif hasattr(chunk, "parts") and chunk.parts:
                                for part in chunk.parts:
                                    if hasattr(part, "text") and part.text:
                                        yield PartialResponse(text=part.text)
                        return
                except Exception as e:
                    logger.warning(f"Error during sync iteration: {str(e)}")
                    # Continue to try other methods

            # Fallback for any other type of response
            if hasattr(response, "text"):
                yield PartialResponse(text=response.text)
            else:
                logger.warning(f"Unknown response type: {type(response)}")
                yield PartialResponse(text="Unrecognized response format from Gemini")

        except Exception as e:
            logger.warning(f"Error during iteration: {str(e)}")
            logger.error(f"Error processing Gemini response: {str(e)}")
            yield PartialResponse(text=f"Error streaming from Gemini: {str(e)}")

    # Method removed - no longer needed

    def _get_extension_for_mime_type(self, mime_type: str) -> str:
        """Get the appropriate file extension for a MIME type.

        Args:
            mime_type: MIME type of the image

        Returns:
            File extension (without the dot)
        """
        if mime_type == "image/png":
            return "png"
        elif mime_type == "image/gif":
            return "gif"
        elif mime_type == "image/webp":
            return "webp"
        else:
            return "jpg"  # Default to jpg

    async def _handle_image_upload(
        self, image_data: bytes, mime_type: str, query: QueryRequest
    ) -> PartialResponse:
        """Handle uploading an image to Poe.

        Args:
            image_data: Raw image data
            mime_type: MIME type of the image
            query: Original query for message_id

        Returns:
            Response with image display information
        """
        # Check image size for performance reasons
        max_image_size = 10 * 1024 * 1024  # 10MB limit
        if len(image_data) > max_image_size:
            logger.warning(f"Image too large ({len(image_data)} bytes), skipping")
            return PartialResponse(text="[Image too large to display]")

        # Determine file extension
        extension = self._get_extension_for_mime_type(mime_type)
        filename = f"gemini_image.{extension}"

        try:
            # Use Poe's official attachment mechanism
            attachment_upload_response = await self.post_message_attachment(
                message_id=query.message_id,
                file_data=image_data,
                filename=filename,
                is_inline=True,
            )

            if (
                not hasattr(attachment_upload_response, "inline_ref")
                or not attachment_upload_response.inline_ref
            ):
                logger.error("Error uploading image: No inline_ref in response")
                return PartialResponse(text="[Error uploading image to Poe]")

            # Create markdown with the official Poe attachment reference
            output_md = f"![{filename}][{attachment_upload_response.inline_ref}]"
            return PartialResponse(text=output_md)
        except Exception as e:
            logger.error(f"Error uploading image: {str(e)}")
            # Fallback to base64 encoding for tests/local development
            try:
                import base64
                import io

                from PIL import Image

                # Resize the image if it's too large
                img = Image.open(io.BytesIO(image_data))
                img_format = img.format or extension.upper()

                # Save to buffer
                buffer = io.BytesIO()
                img.save(buffer, format=img_format)
                img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")

                # Return the image as base64 data URI with markdown formatting
                return PartialResponse(
                    text=f"![Gemini generated image](data:{mime_type};base64,{img_str})"
                )
            except Exception as nested_e:
                logger.error(f"Error with base64 fallback: {str(nested_e)}")
                return PartialResponse(text=f"[Error uploading image: {str(e)}]")

    # Method removed - no longer needed

    async def _process_images_in_response(
        self, response, query: QueryRequest
    ) -> AsyncGenerator[PartialResponse, None]:
        """Process images in the Gemini response.

        Args:
            response: The response from Gemini API
            query: The original query from the user

        Yields:
            Image responses as PartialResponse objects
        """
        if hasattr(response, "parts"):
            for part in response.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    try:
                        # Extract image data
                        image_data = part.inline_data.get("data")
                        mime_type = part.inline_data.get("mime_type")

                        if image_data and mime_type:
                            # Handle the image upload and display
                            yield await self._handle_image_upload(image_data, mime_type, query)
                    except Exception as e:
                        logger.error(f"Error processing image response: {str(e)}")
                        yield PartialResponse(text=f"[Error displaying image: {str(e)}]")

        # Handle text in the response after images
        if hasattr(response, "text") and response.text:
            yield PartialResponse(text=response.text)

    async def _process_multimodal_content(
        self, client, contents: list, query: QueryRequest
    ) -> AsyncGenerator[PartialResponse, None]:
        """Process multimodal content (text + images).

        Args:
            client: The Gemini API client
            contents: The formatted contents for the API
            query: The original query from the user

        Yields:
            Responses as PartialResponse objects
        """
        # For multimodal content with images, we need a complete response for proper processing
        response = client.generate_content(contents)

        # Process any images in the response
        async for partial_response in self._process_images_in_response(response, query):
            yield partial_response

    async def _process_user_query(
        self, client, user_message: str, query: QueryRequest
    ) -> AsyncGenerator[PartialResponse, None]:
        """Process the user query and generate appropriate response.

        This method is used by all Gemini model versions (2.0 and 2.5+) and handles
        different response formats appropriately.

        Args:
            client: The Gemini API client
            user_message: The user's text message
            query: The original query object

        Yields:
            Response chunks as PartialResponse objects
        """
        # Process any image attachments
        attachments = self._extract_attachments(query)
        image_parts = self._prepare_image_parts(attachments)

        # Format chat history for context (only for text-only conversations)
        chat_history = self._format_chat_history(query) if not image_parts else None

        # Prepare content (with chat history for text-only, without for multimodal)
        # Type ignore: pyright complains about None not being compatible with the list type
        contents = self._prepare_content(user_message, image_parts, chat_history)  # type: ignore

        # Log chat history use if applicable
        if chat_history and len(chat_history) > 1:
            logger.info(f"Using chat history with {len(chat_history)} messages")

        try:
            # Verify the package is installed
            _ = __import__("google.generativeai")

            # For multimodal content (with images), we use non-streaming mode
            if image_parts:
                async for partial_response in self._process_multimodal_content(
                    client, contents, query
                ):
                    yield partial_response
                return

            # For text-only content, use streaming mode
            logger.info(f"Using streaming mode for model: {self.model_name}")

            # Make the API call with streaming enabled
            response = client.generate_content(contents, stream=True)

            # Case 1: Response has resolve() method (Gemini 2.5+ models)
            if hasattr(response, "resolve"):
                try:
                    logger.debug(
                        f"Using chunked iteration with resolve() method for model: {self.model_name}"
                    )

                    # Process response chunks as they come in using synchronous iteration
                    for chunk in response:
                        if hasattr(chunk, "text") and chunk.text:
                            yield PartialResponse(text=chunk.text)
                        elif hasattr(chunk, "parts") and chunk.parts:
                            for part in chunk.parts:
                                if hasattr(part, "text") and part.text:
                                    yield PartialResponse(text=part.text)
                except Exception as e:
                    logger.error(f"Error with chunked iteration: {str(e)}")
                    yield PartialResponse(text=f"Error: Streaming error from Gemini: {str(e)}")

            # Case 2: Response is async iterable or regular iterable
            else:
                logger.debug(f"Using standard streaming for model: {self.model_name}")
                async for partial_response in self._process_streaming_response(response):
                    yield partial_response

        except ImportError:
            logger.warning("Failed to import google.generativeai")
            yield PartialResponse(
                text="Google Generative AI package is not available. Please install it with: pip install google-generativeai"
            )
        except Exception as e:
            logger.error(f"Error with Gemini API: {str(e)}")
            yield PartialResponse(text=f"Error: Could not get response from Gemini: {str(e)}")

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

            # Log the extracted message (simplified logging)
            logger.debug(f"[{self.bot_name}] Received message: {user_message}")

            # Handle bot info request
            if user_message.lower().strip() == "bot info":
                yield await self._handle_bot_info_request()
                return

            # Initialize client for this specific model
            client = get_client(self.model_name)
            if client is None:
                yield PartialResponse(text="Error: Google API key is not configured.")
                return

            try:
                # Process query and generate response
                async for partial_response in self._process_user_query(client, user_message, query):
                    yield partial_response
            except Exception as e:
                logger.error(f"Error calling Gemini API: {str(e)}")
                yield PartialResponse(text=f"Error: Could not get response from Gemini: {str(e)}")

        except Exception as e:
            # Let the parent class handle errors
            async for resp in super().get_response(query):
                yield resp


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
    bot_description = (
        "Fast and efficient Gemini 2.0 Flash model, optimized for speed and next-gen features."
    )


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
    bot_description = (
        "Advanced Gemini 2.5 Flash Preview model for adaptive thinking and cost efficiency."
    )


class Gemini25ProExpBot(GeminiBaseBot):
    """Gemini 2.5 Pro Experimental - premium model for complex reasoning."""

    model_name = "gemini-2.5-pro-exp-03-25"
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
    bot_description = (
        "Experimental Gemini 2.0 Flash Thinking model with enhanced reasoning capabilities."
    )


class Gemini20ProExpBot(GeminiBaseBot):
    """Gemini 2.0 Pro Experimental model."""

    model_name = "gemini-2.0-pro-exp-02-05"
    bot_name = "Gemini20ProExpBot"
    bot_description = "Experimental Gemini 2.0 Pro model with latest capabilities."

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

    def generate_content(self, contents: Any):
        # Return an object with text property for compatibility
        model_name = self.model_name  # Store locally for use in inner class

        class StubResponse:
            def __init__(self):
                self.text = f"Gemini API ({model_name}) is not available. Please install the google-generativeai package."
                self.parts = []  # Add parts property for multimodal support

        return StubResponse()

    def generate_content_stream(self, contents: Any):
        # Return an iterable that yields one chunk for the stub message
        model_name = self.model_name  # Store locally for use in inner class

        class StubStreamResponse:
            def __init__(self):
                self.text = f"Gemini API ({model_name}) is not available. Please install the google-generativeai package."
                self.parts = []  # Add parts property for multimodal support

        # Return a list with a single item, which is compatible with the 'for chunk in response:' pattern
        return [StubStreamResponse()]


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
        genai.configure(api_key=api_key)

        # Then create and return the model
        return genai.GenerativeModel(model_name=model_name)
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
                    # Import inside try block to avoid module-level import errors
                    import google.generativeai as genai

                    image_parts.append({
                        "mime_type": image_data["mime_type"],
                        "data": image_data["data"],
                    })
                except ImportError:
                    logger.warning("Could not import google.generativeai for image processing")

        return image_parts

    def _prepare_content(self, user_message: str, image_parts: list) -> list:
        """Prepare content for the Gemini API (text and/or images).

        Args:
            user_message: The user's text message
            image_parts: List of processed image parts

        Returns:
            Content list formatted for Gemini API
        """
        contents = []

        # Add images first if present
        for image_part in image_parts:
            contents.append({"inline_data": image_part})

        # Add text prompt
        contents.append({"text": user_message})

        return contents

    async def _process_streaming_response(self, response_stream) -> AsyncGenerator[PartialResponse, None]:
        """Process a streaming text response from Gemini.

        Args:
            response_stream: The streaming response from Gemini API

        Yields:
            Response chunks as PartialResponse objects
        """
        try:
            # Stream text responses in real-time
            for chunk in response_stream:
                if hasattr(chunk, "text") and chunk.text:
                    yield PartialResponse(text=chunk.text)
        except Exception as e:
            logger.error(f"Error streaming from Gemini API: {str(e)}")
            yield PartialResponse(text=f"Error streaming from Gemini: {str(e)}")

    async def _resize_image_if_needed(self, image_data: bytes, max_size_bytes: int = 1024 * 1024) -> bytes:
        """Resize an image if it exceeds the specified size limit.

        Args:
            image_data: Raw image data
            max_size_bytes: Maximum size in bytes (default 1MB)

        Returns:
            Resized image data if needed, or original data if small enough
        """
        if len(image_data) <= max_size_bytes:
            return image_data

        try:
            # Try to resize the image if possible
            import io

            from PIL import Image

            img = Image.open(io.BytesIO(image_data))
            # Calculate new dimensions while maintaining aspect ratio
            max_size = 800
            ratio = min(max_size / img.width, max_size / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))

            # Resize and save to bytes
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            img.save(buffer, format=img.format or "JPEG")
            resized_data = buffer.getvalue()
            logger.info(f"Resized image from {len(image_data)} to {len(resized_data)} bytes")
            return resized_data
        except Exception as resize_err:
            logger.warning(f"Failed to resize image: {str(resize_err)}")
            # Continue with original image
            return image_data

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

    async def _handle_image_upload(self, image_data: bytes, mime_type: str, query: QueryRequest) -> PartialResponse:
        """Handle uploading an image to Poe with fallbacks.

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
            # Try using Poe's official attachment mechanism
            attachment_upload_response = await self.post_message_attachment(
                message_id=query.message_id,
                file_data=image_data,
                filename=filename,
                is_inline=True,
            )

            if (not hasattr(attachment_upload_response, "inline_ref") or
                not attachment_upload_response.inline_ref):
                logger.error("Error uploading image: No inline_ref in response")
                return PartialResponse(text="[Error uploading image to Poe]")

            # Create markdown with the official Poe attachment reference
            output_md = f"![{filename}][{attachment_upload_response.inline_ref}]"
            return PartialResponse(text=output_md)
        except AttributeError as ae:
            # Fallback if post_message_attachment is not available
            logger.warning(f"post_message_attachment not available: {str(ae)}")
            return await self._create_base64_image_response(image_data, mime_type)
        except Exception as e:
            # Handle other errors with base64 fallback
            logger.error(f"Error uploading image: {str(e)}")
            return await self._create_base64_image_response(image_data, mime_type)

    async def _create_base64_image_response(self, image_data: bytes, mime_type: str) -> PartialResponse:
        """Create a base64-encoded image response for fallback.

        Args:
            image_data: Raw image data
            mime_type: MIME type of the image

        Returns:
            Response with base64-encoded image
        """
        # Built-in library import placed at function level to avoid module-level dependencies
        import base64

        # If image is too large for base64 encoding in markdown, resize it
        image_data = await self._resize_image_if_needed(image_data, 1024 * 1024)

        # Convert to base64
        b64_data = base64.b64encode(image_data).decode('utf-8')
        image_markdown = f"![Gemini generated image](data:{mime_type};base64,{b64_data})"
        return PartialResponse(text=image_markdown)

    async def _process_images_in_response(self, response, query: QueryRequest) -> AsyncGenerator[PartialResponse, None]:
        """Process images in the Gemini response.

        Args:
            response: The response from Gemini API
            query: The original query from the user

        Yields:
            Image responses as PartialResponse objects
        """
        has_images = False

        if hasattr(response, "parts"):
            for part in response.parts:
                if hasattr(part, "inline_data") and part.inline_data:
                    has_images = True
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

    async def _process_multimodal_content(self, client, contents: list, query: QueryRequest) -> AsyncGenerator[PartialResponse, None]:
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

    async def _process_user_query(self, client, user_message: str, query: QueryRequest) -> AsyncGenerator[PartialResponse, None]:
        """Process the user query and generate appropriate response.

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

        # Prepare content (text-only or multimodal)
        contents = self._prepare_content(user_message, image_parts)

        # Generate streaming content
        try:
            # Import inside try block to avoid module-level import errors
            import google.generativeai as genai
            response_stream = client.generate_content_stream(contents)
        except ImportError:
            # Fall back to text-only if imports fail
            logger.warning("Failed to import google.generativeai for streaming")
            response_stream = client.generate_content_stream(
                f"{user_message} (Note: Your image was uploaded but cannot be processed)"
            )

        # Process the response appropriately
        if image_parts:
            # For multimodal content (with images), we need special processing
            async for partial_response in self._process_multimodal_content(client, contents, query):
                yield partial_response
        else:
            # For text-only content, we can use streaming directly
            async for partial_response in self._process_streaming_response(response_stream):
                yield partial_response

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

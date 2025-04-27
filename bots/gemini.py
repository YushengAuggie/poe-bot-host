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
                metadata["supports_image_input"] = self.supports_image_input
                yield PartialResponse(text=json.dumps(metadata, indent=2))
                return

            # Initialize client for this specific model
            client = get_client(self.model_name)
            if client is None:
                yield PartialResponse(text="Error: Google API key is not configured.")
                return

            try:
                # Check for image attachments
                attachments = self._extract_attachments(query)
                image_parts = []

                # Process image attachments if present
                if attachments:
                    for attachment in attachments:
                        image_data = self._process_image_attachment(attachment)
                        if image_data:
                            try:
                                # Import inside try block to avoid module-level import errors
                                import google.generativeai as genai

                                image_parts.append(
                                    {
                                        "mime_type": image_data["mime_type"],
                                        "data": image_data["data"],
                                    }
                                )
                            except ImportError:
                                logger.warning(
                                    "Could not import google.generativeai for image processing"
                                )

                # Create content parts for multimodal input
                if image_parts:
                    try:
                        # Import inside try block to avoid module-level import errors
                        import google.generativeai as genai

                        # Prepare multimodal content
                        contents = []
                        # Add images first
                        for image_part in image_parts:
                            contents.append({"inline_data": image_part})
                        # Add text prompt
                        contents.append({"text": user_message})

                        # Generate content with multimodal input
                        # For multimodal generation, we'll prepare for streaming
                        # But handle it later depending on whether there are images in the response
                        response_stream = client.generate_content_stream(contents)
                    except ImportError:
                        # Fall back to text-only if imports fail
                        response_stream = client.generate_content_stream(
                            f"{user_message} (Note: Your image was uploaded but cannot be processed)"
                        )
                else:
                    # Text-only generation with streaming
                    response_stream = client.generate_content_stream(f"{user_message}")

                # For handling streaming response
                if image_parts:
                    # When we have image inputs, we need to use generate_content to handle images properly
                    # We cannot rely only on streaming for multimodal content with image inputs
                    response = client.generate_content(contents)
                    has_images = False

                    # For multimodal input, use the full response to handle any images in response
                    if hasattr(response, "parts"):
                        for part in response.parts:
                            if hasattr(part, "inline_data") and part.inline_data:
                                has_images = True

                        # Process images in response
                        if hasattr(response, "parts"):
                            for part in response.parts:
                                if hasattr(part, "inline_data") and part.inline_data:
                                    has_images = True
                                    try:
                                        # Extract image data
                                        image_data = part.inline_data.get("data")
                                        mime_type = part.inline_data.get("mime_type")

                                        if image_data and mime_type:
                                            # Check image size for performance reasons
                                            max_image_size = 10 * 1024 * 1024  # 10MB limit
                                            if len(image_data) > max_image_size:
                                                logger.warning(
                                                    f"Image too large ({len(image_data)} bytes), skipping"
                                                )
                                                yield PartialResponse(
                                                    text="[Image too large to display]"
                                                )
                                                continue

                                            # Use Poe's official attachment upload mechanism
                                            try:
                                                # Determine file extension from mime type
                                                extension = "jpg"
                                                if mime_type == "image/png":
                                                    extension = "png"
                                                elif mime_type == "image/gif":
                                                    extension = "gif"
                                                elif mime_type == "image/webp":
                                                    extension = "webp"

                                                filename = f"gemini_image.{extension}"

                                                # Upload the attachment to Poe
                                                # This requires the fastapi_poe client's post_message_attachment method
                                                # which is available through self if using PoeBot as a base class
                                                attachment_upload_response = (
                                                    await self.post_message_attachment(
                                                        message_id=query.message_id,
                                                        file_data=image_data,
                                                        filename=filename,
                                                        is_inline=True,
                                                    )
                                                )

                                                if (
                                                    not hasattr(
                                                        attachment_upload_response, "inline_ref"
                                                    )
                                                    or not attachment_upload_response.inline_ref
                                                ):
                                                    logger.error(
                                                        "Error uploading image: No inline_ref in response"
                                                    )
                                                    yield PartialResponse(
                                                        text="[Error uploading image to Poe]"
                                                    )
                                                else:
                                                    # Create markdown with the official Poe attachment reference
                                                    output_md = f"![{filename}][{attachment_upload_response.inline_ref}]"
                                                    yield PartialResponse(text=output_md)
                                            except AttributeError as ae:
                                                # Fallback if post_message_attachment is not available
                                                logger.warning(
                                                    f"post_message_attachment not available: {str(ae)}"
                                                )
                                                # Convert binary data to base64
                                                import base64

                                                b64_data = base64.b64encode(image_data).decode(
                                                    "utf-8"
                                                )

                                                # Create markdown image format
                                                image_markdown = f"![Gemini generated image](data:{mime_type};base64,{b64_data})"

                                                # Yield the image as markdown
                                                yield PartialResponse(text=image_markdown)
                                            except Exception as e:
                                                logger.error(f"Error uploading image: {str(e)}")
                                                # Fallback to base64 method
                                                import base64

                                                # If image is too large for base64 encoding in markdown, resize it
                                                if (
                                                    len(image_data) > 1024 * 1024
                                                ):  # 1MB is already large for base64
                                                    try:
                                                        # Try to resize the image if possible
                                                        import io

                                                        from PIL import Image

                                                        img = Image.open(io.BytesIO(image_data))
                                                        # Calculate new dimensions while maintaining aspect ratio
                                                        max_size = 800
                                                        ratio = min(
                                                            max_size / img.width,
                                                            max_size / img.height,
                                                        )
                                                        new_size = (
                                                            int(img.width * ratio),
                                                            int(img.height * ratio),
                                                        )

                                                        # Resize and save to bytes
                                                        img = img.resize(
                                                            new_size, Image.Resampling.LANCZOS
                                                        )
                                                        buffer = io.BytesIO()
                                                        img.save(
                                                            buffer, format=img.format or "JPEG"
                                                        )
                                                        image_data = buffer.getvalue()
                                                        logger.info(
                                                            f"Resized image from {len(image_data)} bytes"
                                                        )
                                                    except Exception as resize_err:
                                                        logger.warning(
                                                            f"Failed to resize image: {str(resize_err)}"
                                                        )
                                                        # Continue with original image

                                                # Convert to base64
                                                b64_data = base64.b64encode(image_data).decode(
                                                    "utf-8"
                                                )
                                                image_markdown = f"![Gemini generated image](data:{mime_type};base64,{b64_data})"
                                                yield PartialResponse(text=image_markdown)
                                    except Exception as e:
                                        logger.error(f"Error processing image response: {str(e)}")
                                        yield PartialResponse(
                                            text=f"[Error displaying image: {str(e)}]"
                                        )

                        # Handle text in the non-streaming response if images were present
                        if hasattr(response, "text") and response.text:
                            yield PartialResponse(text=response.text)
                else:
                    # For text-only responses, use real streaming
                    try:
                        # Stream text responses in real-time
                        for chunk in response_stream:
                            if hasattr(chunk, "text") and chunk.text:
                                yield PartialResponse(text=chunk.text)
                    except Exception as e:
                        logger.error(f"Error streaming from Gemini API: {str(e)}")
                        yield PartialResponse(text=f"Error streaming from Gemini: {str(e)}")
            except Exception as e:
                logger.error(f"Error calling Gemini API: {str(e)}")
                yield PartialResponse(text=f"Error: Could not get response from Gemini: {str(e)}")

        except Exception as e:
            # Let the parent class handle errors
            async for resp in super().get_response(query):
                yield resp

    async def get_settings(self, settings_request):
        """Get bot settings.

        Override to enable attachments for image uploads.

        Args:
            settings_request: The settings request

        Returns:
            Settings response with attachments enabled
        """
        # Get settings from parent class
        settings = await super().get_settings(settings_request)

        # Enable attachments
        settings.allow_attachments = True

        return settings


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

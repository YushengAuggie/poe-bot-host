import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional, Union

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


# Base Gemini bot class that other model-specific bots will inherit from
class GeminiBaseBot(BaseBot):
    """Base class for Gemini bots."""

    model_name = "gemini-2.0-flash"  # Default model
    bot_name = "GeminiBaseBot"
    bot_description = "Base Gemini model bot."
    supports_image_input = True  # Enable image input by default
    supports_video_input = True  # Enable video input by default
    supports_audio_input = True  # Enable audio input by default
    supports_grounding = False  # Default value, will be set based on model

    def __init__(self, **kwargs):
        """Initialize the GeminiBaseBot."""
        super().__init__(**kwargs)
        self.supported_image_types = ["image/jpeg", "image/png", "image/webp", "image/gif"]
        # Add support for video content types
        self.supported_video_types = ["video/mp4", "video/quicktime", "video/webm"]
        # Add support for audio content types
        self.supported_audio_types = [
            "audio/mp3",
            "audio/mpeg",
            "audio/wav",
            "audio/x-wav",
            "audio/ogg",
        ]

        # Image generation support - defaults to False unless overridden in subclass
        self.supports_image_generation = getattr(self, "supports_image_generation", False)

        # Determine grounding support based on model
        self._set_grounding_support()

        # Default grounding settings (only effective if the model supports grounding)
        self.grounding_enabled = kwargs.get("grounding_enabled", self.supports_grounding)
        self.citations_enabled = kwargs.get("citations_enabled", True)
        self.grounding_sources = []

    def _set_grounding_support(self):
        """Set grounding support based on the model name.

        Research indicates that only Pro models fully support grounding.
        """
        # Models known to support grounding based on Google's documentation
        grounding_supported_models = [
            # Pro models
            "gemini-2.0-pro",
            "gemini-2.0-pro-",  # Prefix match for pro experimental models
            "gemini-2.5-pro",
            # 2.5 models generally support more advanced features
            "gemini-2.5-",
        ]

        # Check if the current model supports grounding
        self.supports_grounding = any(
            self.model_name.startswith(supported_model)
            for supported_model in grounding_supported_models
        )

        # Log the grounding support status
        if self.supports_grounding:
            logger.info(f"Model {self.model_name} supports grounding")
        else:
            logger.info(f"Model {self.model_name} does not support grounding")

    def _extract_attachments(self, query: QueryRequest) -> list:
        """Extract attachments from the query.

        Args:
            query: The query from the user

        Returns:
            List of attachments
        """
        attachments = []
        try:
            if isinstance(query.query, list) and query.query:
                last_message = query.query[-1]
                logger.info(
                    f"Message content: {last_message.content if hasattr(last_message, 'content') else 'No content'}"
                )
                logger.info(
                    f"Message has attachments attribute: {hasattr(last_message, 'attachments')}"
                )

                if hasattr(last_message, "attachments") and last_message.attachments:
                    attachments = last_message.attachments
                    logger.info(f"Found {len(attachments)} attachments")

                    # Add more detailed debugging
                    logger.info(f"Attachment types: {[type(att).__name__ for att in attachments]}")
                    logger.info(f"Attachment dir: {[dir(att) for att in attachments]}")

            # Debug attachment details with more information
            for i, attachment in enumerate(attachments):
                logger.info(f"Attachment {i} details:")

                # Check all possible attributes
                logger.info(f"  type={getattr(attachment, 'content_type', 'unknown')}")
                logger.info(f"  name={getattr(attachment, 'name', 'unknown')}")

                if hasattr(attachment, "url"):
                    logger.info(f"  URL: {attachment.url}")

                # Log more details for content attribute
                if hasattr(attachment, "content"):
                    # The content attribute is added by Poe but not in type definitions
                    content = getattr(attachment, "content")
                    content_size = len(content) if content else 0
                    content_type = type(content).__name__
                    logger.info(
                        f"  content: found with type={content_type}, size={content_size} bytes"
                    )
                    if content_size > 0 and content_type == "str" and content_size > 50:
                        # Might be base64, check beginning
                        logger.info(f"  content preview: {content[:30]}...")

                # Check __dict__ for content
                if hasattr(attachment, "__dict__"):
                    dict_content = attachment.__dict__.get("content")
                    if dict_content is not None:
                        logger.info(
                            f"  __dict__ content: found with size={len(dict_content) if dict_content else 0} bytes"
                        )

            # Ensure content attribute is directly accessible for each attachment
            for attachment in attachments:
                if not hasattr(attachment, "content") and "content" in attachment.__dict__:
                    # Add content attribute directly to the attachment object
                    content = attachment.__dict__["content"]
                    # Use setattr to make content accessible as attribute even for Pydantic models
                    object.__setattr__(attachment, "content", content)
                    logger.debug(f"Fixed attachment content accessibility: {attachment.name}")

        except Exception as e:
            logger.error(f"Error extracting attachments: {str(e)}")
            import traceback

            logger.error(traceback.format_exc())

        return attachments

    def _process_media_attachment(self, attachment) -> Optional[Dict[str, Any]]:
        """Process a media (image, video, or audio) attachment for Gemini.

        Args:
            attachment: The attachment object

        Returns:
            Dictionary with mime_type and data, or None if unsupported
        """
        # Check if attachment exists
        if not attachment:
            logger.warning("Received null attachment")
            return None

        # Log attachment details for debugging
        logger.debug("=== PROCESSING ATTACHMENT ===")
        logger.debug(f"Attachment type: {type(attachment).__name__}")
        logger.debug(f"Attachment attributes: {dir(attachment)}")

        # Try to extract content_type
        content_type = "unknown"
        if hasattr(attachment, "content_type"):
            content_type = attachment.content_type
            logger.debug(f"Found content_type attribute: {content_type}")
        else:
            logger.debug("No content_type attribute found")

        # Log more attachment details
        if hasattr(attachment, "name"):
            logger.debug(f"Attachment name: {attachment.name}")
        if hasattr(attachment, "url"):
            logger.debug(f"Attachment URL: {attachment.url}")

        # Dump dict if available
        if hasattr(attachment, "__dict__"):
            logger.debug(f"Attachment __dict__: {attachment.__dict__}")

        # Check if attachment is a supported media type
        is_supported_image = content_type in self.supported_image_types
        is_supported_video = content_type in self.supported_video_types
        is_supported_audio = content_type in self.supported_audio_types

        logger.debug(f"Is supported image: {is_supported_image}")
        logger.debug(f"Is supported video: {is_supported_video}")
        logger.debug(f"Is supported audio: {is_supported_audio}")

        if not hasattr(attachment, "content_type") or (
            not is_supported_image and not is_supported_video and not is_supported_audio
        ):
            logger.warning(f"Unsupported attachment type: {content_type}")
            return None

        # Determine media type for logging
        media_type = "image" if is_supported_image else "video" if is_supported_video else "audio"
        logger.debug(f"Attachment is supported {media_type}")

        # FIXED: If attachment has content in __dict__ but not as attribute,
        # make it accessible as attribute (fix for Pydantic models in the fastapi-poe library)
        if (
            (not hasattr(attachment, "content") or getattr(attachment, "content", None) is None)
            and hasattr(attachment, "__dict__")
            and "content" in attachment.__dict__
        ):
            logger.debug("Fixing content attribute accessibility")
            try:
                content_from_dict = attachment.__dict__["content"]
                # Use object.__setattr__ to bypass Pydantic validation
                object.__setattr__(attachment, "content", content_from_dict)
                logger.debug("Successfully fixed content attribute accessibility")
            except Exception as e:
                logger.warning(f"Failed to fix content attribute accessibility: {e}")

        # Track what method we used to extract content
        content_extraction_method = "none"
        content = None

        # First try accessing content directly via normal attribute access
        if hasattr(attachment, "content") and getattr(attachment, "content", None):
            content = getattr(attachment, "content")
            content_extraction_method = "direct_attribute"
            logger.debug(
                f"Accessing content via direct attribute with type: {type(content).__name__}"
            )
            logger.debug(f"Content size: {len(content) if content else 0} bytes")

            if isinstance(content, str):
                logger.debug(f"Content string preview: {content[:30]}...")

            # Check if content is base64-encoded string (CLI tool sends base64 strings)
            if isinstance(content, str) and content.startswith(("/", "+", "i")):
                try:
                    import base64

                    logger.debug("Converting base64 string to bytes")
                    binary_data = base64.b64decode(content)
                    logger.debug(f"Decoded binary data size: {len(binary_data)} bytes")
                    content = binary_data
                    content_extraction_method = "direct_attribute_base64"
                except Exception as e:
                    logger.error(f"Error decoding base64: {str(e)}")
                    # Continue to try other methods

        # Then try via __dict__ to satisfy type checker (content is added by Poe but not in type definition)
        elif hasattr(attachment, "__dict__") and attachment.__dict__.get("content"):
            content = attachment.__dict__["content"]
            content_extraction_method = "dict_attribute"
            logger.debug(f"Accessing content via __dict__ with type: {type(content).__name__}")
            logger.debug(f"Content size: {len(content) if content else 0} bytes")

            # Check if content is base64-encoded string
            if isinstance(content, str) and content.startswith(("/", "+", "i")):
                try:
                    import base64

                    logger.debug("Converting base64 string to bytes")
                    binary_data = base64.b64decode(content)
                    logger.debug(f"Decoded binary data size: {len(binary_data)} bytes")
                    content = binary_data
                    content_extraction_method = "dict_attribute_base64"
                except Exception as e:
                    logger.error(f"Error decoding base64 from __dict__: {str(e)}")

        # Look for _content attribute which some Poe clients might use
        elif hasattr(attachment, "_content") and getattr(attachment, "_content", None):
            content = getattr(attachment, "_content")
            content_extraction_method = "underscore_attribute"
            logger.debug(
                f"Accessing content via _content attribute with type: {type(content).__name__}"
            )
            logger.debug(f"Content size: {len(content) if content else 0} bytes")

            # Check if content is base64-encoded string
            if isinstance(content, str) and content.startswith(("/", "+", "i")):
                try:
                    import base64

                    logger.debug("Converting base64 string to bytes")
                    binary_data = base64.b64decode(content)
                    logger.debug(f"Decoded binary data size: {len(binary_data)} bytes")
                    content = binary_data
                    content_extraction_method = "underscore_attribute_base64"
                except Exception as e:
                    logger.error(f"Error decoding base64 from _content: {str(e)}")

        # Try with url if content is not available
        elif hasattr(attachment, "url") and attachment.url:
            logger.debug(f"Downloading attachment from URL: {attachment.url}")
            try:
                import requests

                # Handle file:// URLs as a special case (used by CLI tool)
                if attachment.url.startswith("file://"):
                    logger.debug("Handling file:// URL - checking for content elsewhere")
                    # We expect content to be available in another field, but include a fallback
                    if not content:
                        logger.debug("No content found for file:// URL - will return None")
                        return None
                else:
                    # Normal URL handling
                    # Adjust timeout based on media type - longer for video/audio
                    timeout = 30 if is_supported_video or is_supported_audio else 20
                    response = requests.get(attachment.url, timeout=timeout)
                    if response.status_code == 200:
                        content = response.content
                        content_extraction_method = "url_download"
                        logger.debug(f"Successfully downloaded {len(content)} bytes from URL")
                    else:
                        logger.warning(f"Failed to download from URL: HTTP {response.status_code}")
            except Exception as e:
                logger.warning(f"Failed to download media from URL: {str(e)}")

        if not content:
            logger.error("Could not access attachment content through any method")
            return None

        logger.debug(f"Successfully extracted content using method: {content_extraction_method}")
        logger.debug(f"Final content type: {type(content).__name__}")
        logger.debug(f"Final content size: {len(content)} bytes")
        logger.debug("=== END PROCESSING ATTACHMENT ===")

        return {"mime_type": content_type, "data": content}

    async def _handle_bot_info_request(self) -> PartialResponse:
        """Handle a request for bot information.

        Returns:
            Formatted response with bot metadata
        """
        metadata = self._get_bot_metadata()
        metadata["model_name"] = self.model_name
        metadata["supports_image_input"] = self.supports_image_input
        metadata["supports_video_input"] = self.supports_video_input
        metadata["supports_audio_input"] = self.supports_audio_input
        metadata["supports_image_generation"] = self.supports_image_generation
        metadata["supports_grounding"] = self.supports_grounding
        metadata["grounding_enabled"] = self.grounding_enabled
        metadata["citations_enabled"] = self.citations_enabled
        metadata["supported_image_types"] = self.supported_image_types
        metadata["supported_video_types"] = self.supported_video_types
        metadata["supported_audio_types"] = self.supported_audio_types
        return PartialResponse(text=json.dumps(metadata, indent=2))

    def _prepare_media_parts(self, attachments: list) -> list:
        """Process media attachments (images, videos, audio) into parts for Gemini API.

        Args:
            attachments: List of attachments from the query

        Returns:
            List of media parts formatted for Gemini API
        """
        logger.debug("=== PREPARING MEDIA PARTS ===")
        logger.debug(f"Processing {len(attachments)} attachments")
        media_parts = []

        for i, attachment in enumerate(attachments):
            logger.debug(f"Processing attachment {i+1} of {len(attachments)}")
            media_data = self._process_media_attachment(attachment)

            if media_data:
                logger.debug(f"Got media data with mime type: {media_data['mime_type']}")
                logger.debug(f"Media data size: {len(media_data['data'])} bytes")
                logger.debug(f"Media data type: {type(media_data['data'])}")

                format_method = "unknown"
                try:
                    # Try to import the google.generativeai.types module for advanced formatting
                    has_types_module = False
                    try:
                        import google.generativeai

                        logger.debug(
                            f"Google Generative AI version: {getattr(google.generativeai, '__version__', 'unknown')}"
                        )

                        # Check if types module exists and has Part.from_bytes attribute
                        if hasattr(google.generativeai, "types"):
                            types_module = google.generativeai.types
                            logger.debug(f"Found types module: {types_module}")

                            if hasattr(types_module, "Part"):
                                part_class = types_module.Part
                                logger.debug(f"Found Part class: {part_class}")

                                if hasattr(part_class, "from_bytes"):
                                    logger.debug("Found Part.from_bytes method")
                                    has_types_module = True
                                else:
                                    logger.debug("Part class doesn't have from_bytes method")
                            else:
                                logger.debug("types module doesn't have Part class")
                        else:
                            logger.debug("google.generativeai doesn't have types module")

                    except (ImportError, AttributeError) as e:
                        logger.debug(
                            f"Error importing/checking google.generativeai.types: {str(e)}"
                        )
                        has_types_module = False

                    if has_types_module:
                        from google.generativeai import types

                        try:
                            # Create a proper Part object from bytes
                            logger.debug("Attempting to create Part from bytes")
                            part = types.Part.from_bytes(  # type: ignore
                                data=media_data["data"], mime_type=media_data["mime_type"]
                            )

                            logger.debug(f"Created Part object: {type(part).__name__}")
                            media_parts.append(part)
                            format_method = "types.Part"
                            logger.debug(f"Successfully added attachment {i+1} using types.Part")
                        except Exception as part_error:
                            logger.debug(f"Error creating Part object: {str(part_error)}")
                            # Fallback to dictionary format
                            has_types_module = False

                    if not has_types_module:
                        # Fall back to dictionary format if types module is not available
                        logger.debug("Using dictionary format for media part")

                        # Format according to Google Gemini API requirements
                        formatted_part = {
                            "inline_data": {
                                "mime_type": media_data["mime_type"],
                                "data": media_data["data"],
                            }
                        }

                        logger.debug(f"Created dictionary part: {list(formatted_part.keys())}")
                        logger.debug(
                            f"inline_data keys: {list(formatted_part['inline_data'].keys())}"
                        )
                        media_parts.append(formatted_part)
                        format_method = "dictionary"
                        logger.debug(f"Successfully added attachment {i+1} using dictionary format")

                    # Log the type of media we're processing
                    if media_data["mime_type"].startswith("video/"):
                        logger.debug(f"Processed video attachment: {media_data['mime_type']}")
                    elif media_data["mime_type"].startswith("audio/"):
                        logger.debug(f"Processed audio attachment: {media_data['mime_type']}")
                    else:
                        logger.debug(f"Processed image attachment: {media_data['mime_type']}")

                except Exception as e:
                    logger.error(f"Failed to format media for Gemini API: {str(e)}")
                    import traceback

                    logger.error(traceback.format_exc())
            else:
                logger.warning(f"Failed to process attachment {i+1}, no media data returned")

        logger.debug(f"Final media parts: {len(media_parts)} total")
        logger.debug(f"Media parts types: {[type(part).__name__ for part in media_parts]}")
        logger.debug("=== END PREPARING MEDIA PARTS ===")
        return media_parts

    # For backward compatibility with tests
    def _prepare_image_parts(self, attachments: list) -> list:
        """Alias for _prepare_media_parts for backward compatibility with tests.

        Args:
            attachments: List of attachments from the query

        Returns:
            List of image parts formatted for Gemini API
        """
        return self._prepare_media_parts(attachments)

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
                else "model"
                if message.role in ["bot", "assistant"]
                else None
            )

            # Skip messages with unsupported roles
            if not role:
                continue

            chat_history.append({"role": role, "parts": [{"text": message.content}]})

        return chat_history

    def add_grounding_source(self, source: Dict[str, Any]) -> None:
        """Add a grounding source for the bot to use.

        Args:
            source: A dictionary containing source information with keys:
                - title: Title of the source
                - url: URL of the source
                - content: The content to use for grounding
        """
        if not isinstance(source, dict):
            logger.warning(f"Invalid grounding source: {source}")
            return

        required_keys = ["title", "content"]
        if not all(key in source for key in required_keys):
            logger.warning(f"Grounding source missing required keys: {required_keys}")
            return

        self.grounding_sources.append(source)
        logger.info(f"Added grounding source: {source.get('title')}")

    def clear_grounding_sources(self) -> None:
        """Clear all grounding sources."""
        self.grounding_sources = []
        logger.info("Cleared all grounding sources")

    def set_grounding_enabled(self, enabled: bool) -> None:
        """Enable or disable grounding.

        Args:
            enabled: Whether grounding should be enabled
        """
        # Only enable if the model supports it
        if enabled and not self.supports_grounding:
            logger.warning(f"Model {self.model_name} does not support grounding. Request ignored.")
            return

        self.grounding_enabled = enabled
        logger.info(f"Grounding {'enabled' if enabled else 'disabled'}")

    def set_citations_enabled(self, enabled: bool) -> None:
        """Enable or disable citations in grounding responses.

        Args:
            enabled: Whether citations should be included in responses
        """
        self.citations_enabled = enabled
        logger.info(f"Citations {'enabled' if enabled else 'disabled'}")

    def _prepare_grounding_config(self) -> Optional[Dict[str, Any]]:
        """Prepare the grounding configuration for the Gemini API.

        Returns:
            A dictionary with grounding configuration or None if grounding is disabled
            or no sources are available
        """
        # Check if grounding is supported and enabled
        if not self.supports_grounding:
            logger.debug(f"Model {self.model_name} does not support grounding")
            return None

        if not self.grounding_enabled or not self.grounding_sources:
            return None

        try:
            # Verify the package is installed
            _ = __import__("google.generativeai")

            ground_sources: list[Dict[str, str]] = []
            for source in self.grounding_sources:
                ground_source: Dict[str, str] = {
                    "title": source.get("title", ""),
                    "content": source.get("content", ""),
                }

                # Add optional URL if present
                if "url" in source:
                    ground_source["uri"] = source["url"]

                ground_sources.append(ground_source)

            # Explicitly define the type of the returned dictionary
            result: Dict[str, Any] = {
                "groundingEnabled": True,
                "groundingSources": ground_sources,
            }

            # Add citation configuration if enabled
            if self.citations_enabled:
                result["includeCitations"] = True

            return result
        except ImportError:
            logger.warning("Failed to import google.generativeai for grounding")
            return None
        except Exception as e:
            logger.error(f"Error preparing grounding config: {str(e)}")
            return None

    def _prepare_content(
        self,
        user_message: str,
        media_parts: list,
        chat_history: Optional[list[dict[str, object]]] = None,
    ) -> list:
        """Prepare content for the Gemini API (text and/or images/videos).

        Args:
            user_message: The user's text message
            media_parts: List of processed media parts (images or videos)
            chat_history: Optional chat history list

        Returns:
            Content formatted for Gemini API
        """
        logger.info(
            f"Preparing content with {len(media_parts)} media parts and chat history: {chat_history is not None}"
        )

        # For multimodal queries (with images or videos), we can't use chat history
        if media_parts:
            logger.info("Preparing multimodal content")

            # Format each media part correctly for Gemini API
            formatted_parts = []
            for i, part in enumerate(media_parts):
                try:
                    logger.info(f"Formatting media part {i+1}")
                    # Check if the part already has inline_data structure
                    if isinstance(part, dict) and "inline_data" in part:
                        formatted_parts.append(part)  # Use as is
                    else:
                        formatted_part = {"inline_data": part}
                        formatted_parts.append(formatted_part)
                    logger.info(f"Media part {i+1} formatted successfully")
                except Exception as e:
                    logger.error(f"Error formatting media part {i+1}: {str(e)}")

            # Add the text part at the end
            formatted_parts.append({"text": user_message})
            logger.info(f"Final multimodal content has {len(formatted_parts)} parts")

            # Log the formatted content structure (without the actual binary data)
            safe_parts = []
            for part in formatted_parts:
                if "inline_data" in part:
                    # Don't log binary data, just its presence and size
                    mime_type = part["inline_data"].get("mime_type", "unknown")
                    data_size = (
                        len(part["inline_data"].get("data", b""))
                        if "data" in part["inline_data"]
                        else 0
                    )
                    safe_parts.append(
                        {"inline_data": {"mime_type": mime_type, "data_size": data_size}}
                    )
                else:
                    safe_parts.append(part)

            logger.info(f"Content structure: {safe_parts}")
            return formatted_parts

        # For text-only queries, use chat history if available
        if chat_history:
            logger.info("Preparing content with chat history")
            # If the last message is from the user, update it
            # Otherwise, add a new user message
            if chat_history and chat_history[-1]["role"] == "user":
                chat_history[-1]["parts"] = [{"text": user_message}]
                logger.info("Updated last user message in chat history")
            else:
                chat_history.append({"role": "user", "parts": [{"text": user_message}]})
                logger.info("Added new user message to chat history")

            logger.info(f"Final chat history has {len(chat_history)} messages")
            return chat_history

        # Single-turn text-only query
        logger.info("Preparing simple text-only content")
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
        """Get the appropriate file extension for an image MIME type.

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

    def _get_extension_for_video_mime_type(self, mime_type: str) -> str:
        """Get the appropriate file extension for a video MIME type.

        Args:
            mime_type: MIME type of the video

        Returns:
            File extension (without the dot)
        """
        if mime_type == "video/webm":
            return "webm"
        elif mime_type == "video/quicktime":
            return "mov"
        else:
            return "mp4"  # Default to mp4

    async def _handle_media_upload(
        self, media_data: bytes, mime_type: str, query: QueryRequest, is_video: bool = False
    ) -> PartialResponse:
        """Handle uploading an image or video to Poe.

        Args:
            media_data: Raw image or video data
            mime_type: MIME type of the media
            query: Original query for message_id
            is_video: Whether the media is a video

        Returns:
            Response with media display information
        """
        # Check media size for performance reasons
        max_media_size = (
            50 * 1024 * 1024 if is_video else 10 * 1024 * 1024
        )  # 50MB for video, 10MB for images
        if len(media_data) > max_media_size:
            logger.warning(f"Media too large ({len(media_data)} bytes), skipping")
            return PartialResponse(
                text=f"[{'Video' if is_video else 'Image'} too large to display]"
            )

        # Determine file extension and create an appropriate filename
        if is_video:
            extension = self._get_extension_for_video_mime_type(mime_type)
            filename = f"gemini_video_{int(time.time())}.{extension}"
        else:
            extension = self._get_extension_for_mime_type(mime_type)
            filename = f"gemini_image_{int(time.time())}.{extension}"

        try:
            # Use Poe's official attachment mechanism
            attachment_upload_response = await self.post_message_attachment(
                message_id=query.message_id,
                file_data=media_data,
                filename=filename,
                is_inline=True,
            )

            if (
                not hasattr(attachment_upload_response, "inline_ref")
                or not attachment_upload_response.inline_ref
            ):
                logger.error("Error uploading media: No inline_ref in response")
                return PartialResponse(
                    text=f"[Error uploading {'video' if is_video else 'image'} to Poe]"
                )

            # Create markdown with the official Poe attachment reference
            # For videos we don't use an exclamation mark
            if is_video:
                output_md = f"[{filename}][{attachment_upload_response.inline_ref}]"
            else:
                output_md = f"![{filename}][{attachment_upload_response.inline_ref}]"
            return PartialResponse(text=output_md)
        except Exception as e:
            logger.error(f"Error uploading media: {str(e)}")

            # For video, there's no fallback, just return error
            if is_video:
                return PartialResponse(text=f"[Error uploading video: {str(e)}]")

            # Image fallback to base64 encoding for tests/local development
            try:
                import base64
                import io

                from PIL import Image

                # Resize the image if it's too large
                img = Image.open(io.BytesIO(media_data))
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
                return PartialResponse(text=f"[Error uploading media: {str(e)}]")

    # Method removed - no longer needed

    async def _process_media_in_response(
        self, response, query: QueryRequest
    ) -> AsyncGenerator[PartialResponse, None]:
        """Process media (images and videos) in the Gemini response.

        Args:
            response: The response from Gemini API
            query: The original query from the user

        Yields:
            Media responses as PartialResponse objects
        """
        logger.info(f"Processing media in response: {type(response).__name__}")
        logger.info(f"Response attributes: {dir(response)}")

        # Add detailed debug info
        if hasattr(response, "__dict__"):
            logger.info(f"Response __dict__: {response.__dict__}")

        if hasattr(response, "parts"):
            logger.info(f"Response has {len(response.parts)} parts")
            for i, part in enumerate(response.parts):
                logger.info(f"Part {i} type: {type(part).__name__}")
                logger.info(
                    f"Part {i} dict: {part.__dict__ if hasattr(part, '__dict__') else 'No __dict__'}"
                )

                if hasattr(part, "inline_data"):
                    logger.info(f"Part {i} inline_data type: {type(part.inline_data).__name__}")
                    logger.info(f"Part {i} inline_data: {part.inline_data}")

        # Track if we've yielded anything to handle empty responses
        yielded_content = False

        # The model will automatically decide whether to generate images
        # We just need to check if we have image generation capability
        can_generate_images = getattr(self, "supports_image_generation", False)

        # Check if response has parts (multimodal response)
        if hasattr(response, "parts"):
            logger.info(f"Response has {len(response.parts)} parts")

            for i, part in enumerate(response.parts):
                logger.info(f"Processing response part {i+1}")

                # Check for inline_data (media content)
                if hasattr(part, "inline_data") and part.inline_data:
                    logger.info(f"Part {i+1} has inline_data")

                    try:
                        # Extract media data
                        media_data = None
                        mime_type = None

                        # Handle inline_data as either object or dict
                        if isinstance(part.inline_data, dict):
                            media_data = part.inline_data.get("data")
                            mime_type = part.inline_data.get("mime_type")
                        else:
                            # Try to handle as object with attributes
                            if hasattr(part.inline_data, "data"):
                                media_data = part.inline_data.data
                            if hasattr(part.inline_data, "mime_type"):
                                mime_type = part.inline_data.mime_type
                            elif hasattr(part.inline_data, "get_bytes") and callable(
                                part.inline_data.get_bytes
                            ):
                                media_data = part.inline_data.get_bytes()
                                mime_type = "image/png"  # Default

                        logger.info(
                            f"Extracted media_data type: {type(media_data).__name__ if media_data else 'None'}"
                        )
                        logger.info(f"Extracted mime_type: {mime_type}")

                        if media_data and mime_type:
                            logger.info(f"Found {mime_type} data of size {len(media_data)} bytes")

                            # Check if this is a video
                            is_video = mime_type.startswith("video/")

                            # Add caption for generated image (when the model outputs images)
                            if can_generate_images and mime_type.startswith("image/"):
                                # The image was generated by the model, add a caption
                                yield PartialResponse(text="📷 *Generated image:*\n\n")
                                yielded_content = True

                            # Handle the media upload and display
                            media_response = await self._handle_media_upload(
                                media_data, mime_type, query, is_video=is_video
                            )

                            yield media_response
                            yielded_content = True
                            logger.info(f"Successfully processed {mime_type} media")
                        else:
                            logger.warning(
                                f"Part {i+1} has inline_data but missing media_data or mime_type"
                            )
                    except Exception as e:
                        logger.error(
                            f"Error processing media in part {i+1}: {str(e)}", exc_info=True
                        )
                        yield PartialResponse(text=f"[Error displaying media: {str(e)}]")
                        yielded_content = True

                # Check for text content in the part
                if hasattr(part, "text") and part.text:
                    # Special handling for policy violations in responses (commonly happens with image requests)
                    if can_generate_images and (
                        "violates" in part.text.lower()
                        or "cannot generate" in part.text.lower()
                        or "policy" in part.text.lower()
                        or "not allowed" in part.text.lower()
                    ):
                        logger.info("Image generation request returned policy violation")
                        yield PartialResponse(
                            text="⚠️ *Image generation request declined:*\n\n" + part.text
                        )
                    else:
                        logger.info(f"Part {i+1} has text content: {part.text[:100]}...")
                        yield PartialResponse(text=part.text)
                    yielded_content = True

        # Handle text in the response after media (only if not already yielded from parts)
        if hasattr(response, "text") and response.text:
            logger.info(f"Response has direct text attribute: {response.text[:100]}...")

            # Special handling for policy violations in responses (commonly happens with image requests)
            if (
                not yielded_content
                and can_generate_images
                and (
                    "violates" in response.text.lower()
                    or "cannot generate" in response.text.lower()
                    or "policy" in response.text.lower()
                    or "not allowed" in response.text.lower()
                )
            ):
                logger.info("Image generation request returned policy violation")
                yield PartialResponse(
                    text="⚠️ *Image generation request declined:*\n\n" + response.text
                )
                yielded_content = True
            # Only yield the text if we haven't yielded any content yet
            # to avoid duplicating text that might have been in parts
            elif not yielded_content:
                logger.info("Yielding direct text response")
                yield PartialResponse(text=response.text)
                yielded_content = True

        # If we haven't yielded any content, yield a placeholder message
        if not yielded_content:
            logger.warning("No content found in response")
            yield PartialResponse(text="No content found in response from Gemini.")

        logger.info("Completed processing media in response")

    async def _process_multimodal_content(
        self, client, contents: list, query: QueryRequest
    ) -> AsyncGenerator[PartialResponse, None]:
        """Process multimodal content (text + images/videos).

        Args:
            client: The Gemini API client
            contents: The formatted contents for the API
            query: The original query from the user

        Yields:
            Responses as PartialResponse objects
        """
        logger.info("=== PROCESSING MULTIMODAL CONTENT ===")
        logger.info(f"Client model: {getattr(client, 'model_name', 'unknown')}")
        logger.info(f"Client type: {type(client).__name__}")
        logger.info(f"Contents length: {len(contents)} items")

        # Check for gemini client version
        try:
            import google.generativeai as genai

            logger.info(f"Google Generative AI version: {getattr(genai, '__version__', 'unknown')}")
        except ImportError:
            logger.info("Google Generative AI package not available")
        except Exception as e:
            logger.info(f"Error checking genai version: {str(e)}")

        # Prepare grounding configuration if enabled
        grounding_config = self._prepare_grounding_config()
        if grounding_config:
            logger.info(
                f"Using grounding with {len(grounding_config.get('groundingSources', []))} sources for multimodal content"
            )
        else:
            logger.info("No grounding configuration used")

        # Configure generation with grounding if available
        generation_config: Dict[str, Any] = {}
        logger.info("Initializing generation_config")

        # Apply grounding configuration if available
        if grounding_config:
            try:
                # Just make sure the module is available
                _ = __import__("google.generativeai")
                logger.info("google.generativeai successfully imported")

                # Add grounding directly to the generation_config dict
                # In the API, this nests under generation_config
                generation_config["generation_config"] = {}
                logger.info("Added empty generation_config to dict")

                # Add grounding to the generation config dictionary
                generation_config["grounding_config"] = grounding_config
                logger.info("Added grounding_config to generation_config")
            except ImportError:
                logger.warning("Failed to import google.generativeai for grounding config")
            except Exception as e:
                logger.warning(f"Error setting up grounding configuration: {str(e)}")

        try:
            # Check if we have valid contents before sending to the API
            if not contents:
                logger.error("Empty contents provided for multimodal request")
                yield PartialResponse(text="Error: No content to process")
                return

            # Log content structure (excluding binary data)
            safe_contents = []
            logger.info("Examining content structure:")
            for i, item in enumerate(contents):
                logger.info(f"Content item {i+1} type: {type(item).__name__}")

                if isinstance(item, dict):
                    logger.info(f"Content item {i+1} keys: {list(item.keys())}")

                if "inline_data" in item:
                    # Don't log binary data, just its presence and size
                    inline_data = item["inline_data"]
                    logger.info(f"  inline_data type: {type(inline_data).__name__}")

                    if isinstance(inline_data, dict):
                        logger.info(f"  inline_data keys: {list(inline_data.keys())}")

                    mime_type = inline_data.get("mime_type", "unknown")
                    data = inline_data.get("data", None)
                    data_type = type(data).__name__ if data is not None else "None"
                    data_size = len(data) if data is not None else 0

                    logger.info(f"  mime_type: {mime_type}")
                    logger.info(f"  data type: {data_type}")
                    logger.info(f"  data size: {data_size} bytes")

                    safe_contents.append(
                        {
                            "inline_data": {
                                "mime_type": mime_type,
                                "data_size": data_size,
                                "data_type": data_type,
                            }
                        }
                    )
                elif "text" in item:
                    logger.info(f"  text content: {item['text'][:50]}...")
                    safe_contents.append(item)
                else:
                    logger.info(f"  Unknown item format: {item}")
                    safe_contents.append({"unknown_format": str(type(item))})

            logger.info(f"Final safe content structure for logging: {safe_contents}")

            # Check API capabilities
            logger.info("Checking Google Generative AI capabilities")
            has_types_module = False
            has_part_from_bytes = False

            try:
                import google.generativeai as genai

                logger.info(
                    f"Successfully imported google.generativeai (version: {getattr(genai, '__version__', 'unknown')})"
                )

                if hasattr(genai, "types"):
                    logger.info("Found types module")
                    has_types_module = True

                    if hasattr(genai.types, "Part") and hasattr(genai.types.Part, "from_bytes"):
                        logger.info("Found Part.from_bytes method")
                        has_part_from_bytes = True
                    else:
                        logger.info("Part.from_bytes method not found")
                else:
                    logger.info("types module not found in google.generativeai")
            except ImportError:
                logger.info("Failed to import google.generativeai")
            except Exception as e:
                logger.info(f"Error checking Google Generative AI: {str(e)}")

            # For multimodal content with images or videos, we need a complete response for proper processing
            try:
                # Check if we can use types.Part (newer API)
                if has_types_module and has_part_from_bytes:
                    logger.info("=== USING PART OBJECT FORMAT ===")
                    from google.generativeai import types

                    # Convert to Part objects for new Google API requirements if needed
                    google_compatible_contents = []
                    for i, item in enumerate(contents):
                        logger.info(f"Processing content item {i+1} for Part conversion")

                        if "inline_data" in item:
                            # Check if we have nested inline_data
                            data_source = item["inline_data"]
                            logger.info(f"Found inline_data in item {i+1}")

                            # Extract data and mime_type correctly, handling both formats
                            if (
                                isinstance(data_source, dict)
                                and "data" in data_source
                                and "mime_type" in data_source
                            ):
                                # Standard format
                                data = data_source["data"]
                                mime_type = data_source["mime_type"]
                                logger.info(f"Standard format found with mime_type: {mime_type}")
                            else:
                                # This shouldn't happen with our fixes, but just in case
                                logger.warning(
                                    f"Unexpected inline_data format for item {i+1}, trying to recover..."
                                )
                                # Try to find data and mime_type in nested structure
                                if isinstance(data_source, dict) and "inline_data" in data_source:
                                    nested = data_source["inline_data"]
                                    data = nested.get("data", b"")
                                    mime_type = nested.get("mime_type", "application/octet-stream")
                                    logger.info(
                                        f"Recovered from nested format with mime_type: {mime_type}"
                                    )
                                else:
                                    # Can't parse the structure, skip this item
                                    logger.error(
                                        f"Cannot process malformed inline_data for item {i+1}: {data_source}"
                                    )
                                    continue

                            # Create a proper Part from bytes for media
                            try:
                                logger.info(
                                    f"Creating Part.from_bytes for {mime_type} content of size {len(data)} bytes"
                                )
                                # Use type: ignore for the dynamic type
                                part = types.Part.from_bytes(  # type: ignore
                                    data=data, mime_type=mime_type
                                )
                                logger.info(
                                    f"Successfully created Part object of type: {type(part).__name__}"
                                )
                                google_compatible_contents.append(part)
                                logger.info(
                                    f"Added Part for {mime_type} content to compatible contents"
                                )
                            except Exception as e:
                                logger.error(f"Error creating Part object for item {i+1}: {str(e)}")
                                logger.error("Skipping this item and continuing")
                                # Try to continue with other items
                        elif "text" in item:
                            # Add text content
                            text_content = item["text"]
                            logger.info(f"Adding text content: {text_content[:50]}...")
                            google_compatible_contents.append(text_content)
                            logger.info("Added text content to compatible contents")
                        else:
                            logger.warning(f"Unknown content format for item {i+1}, skipping")

                    logger.info(
                        f"Final google_compatible_contents has {len(google_compatible_contents)} items"
                    )
                    logger.info(
                        f"Content types: {[type(item).__name__ for item in google_compatible_contents]}"
                    )
                    logger.info("Making API call to Gemini with types.Part format")

                    # Log generation config
                    logger.info(f"Generation config: {generation_config}")

                    # Make the API call
                    response = client.generate_content(
                        google_compatible_contents, **generation_config
                    )
                    logger.info("API call with Part objects completed successfully")
                else:
                    # Fall back to dictionary format
                    logger.info("=== USING DICTIONARY FORMAT ===")
                    logger.info(
                        f"Making API call with dictionary format (has_types_module={has_types_module}, has_part_from_bytes={has_part_from_bytes})"
                    )
                    logger.info("Making API call to Gemini for multimodal content")

                    # Log generation config
                    logger.info(f"Generation config: {generation_config}")

                    # Make the API call
                    response = client.generate_content(contents, **generation_config)
                    logger.info("API call with dictionary format completed successfully")
            except (ImportError, AttributeError) as e:
                # Fall back to old format if types module is not available
                logger.info(f"Falling back to dictionary format due to: {str(e)}")
                logger.info(
                    "Making API call to Gemini for multimodal content with dictionary format"
                )

                # Log generation config
                logger.info(f"Generation config: {generation_config}")

                # Make the API call
                response = client.generate_content(contents, **generation_config)
                logger.info("API call with dictionary format completed successfully")

            logger.info("=== RECEIVED RESPONSE ===")
            logger.info(f"Response type: {type(response).__name__}")
            logger.info(f"Response dir: {dir(response)}")

            # Debug response structure
            if hasattr(response, "text"):
                logger.info(f"Response has text attribute: {response.text[:100]}...")
            else:
                logger.info("Response does not have text attribute")

            if hasattr(response, "parts"):
                logger.info(f"Response has {len(response.parts)} parts")

                # Log more details about parts
                for i, part in enumerate(response.parts):
                    logger.info(f"Part {i+1} type: {type(part).__name__}")
                    logger.info(f"Part {i+1} dir: {dir(part)}")

                    if hasattr(part, "text"):
                        logger.info(f"Part {i+1} has text: {part.text[:100]}...")

                    if hasattr(part, "inline_data"):
                        mime_type = getattr(part.inline_data, "mime_type", "unknown")
                        logger.info(f"Part {i+1} has inline_data with mime_type: {mime_type}")
            else:
                logger.info("Response does not have parts attribute")

            logger.info("=== HANDLING RESPONSE CONTENT ===")
            # Track if we yielded anything for debug purposes
            response_yielded = False
            yield_count = 0

            # Process any images or videos in the response
            logger.info("Processing media in response through _process_media_in_response")
            async for partial_response in self._process_media_in_response(response, query):
                logger.info(
                    f"Got partial response from _process_media_in_response: {partial_response.text[:100] if hasattr(partial_response, 'text') else 'No text'}"
                )
                response_yielded = True
                yield_count += 1
                yield partial_response

            logger.info(f"Media response processing complete, yielded {yield_count} responses")

            # If no response was yielded above and we have text, yield it now
            if not response_yielded and hasattr(response, "text") and response.text:
                # Check if we've already yielded this text from parts
                logger.info(
                    f"No response yielded yet, yielding direct text response: {response.text[:100]}..."
                )
                yield PartialResponse(text=response.text)
                yield_count += 1

            # For debugging - check if we yielded anything at all
            if yield_count == 0:
                logger.warning("No content was yielded from multimodal processing!")
                # Check if response contains any indication of error or rejection
                error_found = False

                if hasattr(response, "text") and response.text:
                    text = response.text.lower()
                    if (
                        ("cannot" in text and "image" in text)
                        or "sorry" in text
                        or "unable" in text
                    ):
                        logger.warning(f"Found potential rejection message: {response.text[:200]}")
                        error_found = True

                if not error_found:
                    logger.info("No error message found in response, yielding generic message")
                    yield PartialResponse(
                        text="I received your image but couldn't process it properly. The model may not fully support this kind of image analysis."
                    )

            logger.info(f"Total responses yielded: {yield_count}")

        except Exception as e:
            logger.error(f"Error processing multimodal content: {str(e)}", exc_info=True)
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            yield PartialResponse(text=f"Error processing image: {str(e)}")

        logger.info("=== COMPLETED MULTIMODAL CONTENT PROCESSING ===")

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
        # The model will automatically decide whether to generate images
        # We don't need special detection logic

        # Process any image or video attachments
        attachments = self._extract_attachments(query)
        media_parts = self._prepare_media_parts(attachments)

        # Format chat history for context (only for text-only conversations)
        chat_history = self._format_chat_history(query) if not media_parts else None

        # Prepare content (with chat history for text-only, without for multimodal)
        # Type ignore: pyright complains about None not being compatible with the list type
        contents = self._prepare_content(user_message, media_parts, chat_history)  # type: ignore

        # Log chat history use if applicable
        if chat_history and len(chat_history) > 1:
            logger.info(f"Using chat history with {len(chat_history)} messages")

        try:
            # Verify the package is installed
            _ = __import__("google.generativeai")

            # For multimodal content (with images or videos), we use non-streaming mode
            if media_parts:
                logger.info(f"Using multimodal mode with {len(media_parts)} media parts")
                async for partial_response in self._process_multimodal_content(
                    client, contents, query
                ):
                    yield partial_response
                return

            # For text-only content, use streaming mode
            logger.info(f"Using streaming mode for model: {self.model_name}")

            # Prepare grounding configuration if enabled
            grounding_config = self._prepare_grounding_config()
            if grounding_config:
                logger.info(
                    f"Using grounding with {len(grounding_config.get('groundingSources', []))} sources"
                )

            # Make the API call with streaming enabled and grounding config if available
            generation_config: Dict[str, Any] = {"stream": True}

            # Apply grounding configuration if available
            if grounding_config:
                try:
                    # Just make sure the module is available
                    _ = __import__("google.generativeai")

                    # Create or use existing generation config dictionary
                    if "generation_config" not in generation_config:
                        generation_config["generation_config"] = {}

                    # Add grounding to the generation config at the appropriate level
                    generation_config["grounding_config"] = grounding_config
                except ImportError:
                    logger.warning("Failed to import google.generativeai for grounding config")

            # If we support image generation, we need to disable streaming to get complete responses with media
            if getattr(self, "supports_image_generation", False):
                # Disable streaming for possible image responses
                generation_config["stream"] = False

            response = client.generate_content(contents, **generation_config)

            # If streaming is disabled (for image-capable models), use media processing
            if not generation_config.get("stream", True):
                logger.info("Processing potential media response")
                # Process any media (images) in the response
                async for partial_response in self._process_media_in_response(response, query):
                    yield partial_response
                return

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

    async def get_settings(self, settings_request: SettingsRequest) -> SettingsResponse:
        """Get bot settings including the rate card.

        Args:
            settings_request: The settings request

        Returns:
            Settings response with rate card and cost label
        """
        # Set a flat fee of 50 points per message
        rate_card = "50 points / message"
        cost_label = "Message Cost"

        # Include settings from the parent class
        settings = await super().get_settings(settings_request)

        # Add rate card and cost label
        settings.rate_card = rate_card
        settings.cost_label = cost_label

        return settings

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

            # Check if the API key exists
            api_key = get_api_key("GOOGLE_API_KEY")
            if not api_key:
                logger.error("GOOGLE_API_KEY not found in environment variables")
                yield PartialResponse(
                    text="Error: Google API key is not configured. Please set the GOOGLE_API_KEY environment variable."
                )
                return
            else:
                logger.info(f"Found GOOGLE_API_KEY in environment (starts with: {api_key[:3]}...)")

            # Handle bot info request
            if user_message.lower().strip() == "bot info":
                yield await self._handle_bot_info_request()
                return

            # Debug query structure
            logger.debug(f"Query type: {type(query)}")
            logger.debug(f"Query attributes: {dir(query)}")
            logger.debug(f"Query has query attribute: {hasattr(query, 'query')}")

            # Check for multimodal content in more detail
            has_attachments = False
            if hasattr(query, "query"):
                logger.debug(f"Query.query type: {type(query.query)}")
                if isinstance(query.query, list):
                    logger.debug(f"Query.query length: {len(query.query)}")
                    if query.query:
                        last_msg = query.query[-1]
                        logger.debug(f"Last message type: {type(last_msg)}")
                        logger.debug(f"Last message attributes: {dir(last_msg)}")
                        logger.debug(
                            f"Last message has attachments: {hasattr(last_msg, 'attachments')}"
                        )
                        if hasattr(last_msg, "attachments"):
                            logger.debug(f"Attachments: {last_msg.attachments}")
                            logger.debug(f"Attachments type: {type(last_msg.attachments)}")
                            logger.debug(
                                f"Attachments length: {len(last_msg.attachments) if last_msg.attachments else 0}"
                            )
                            if last_msg.attachments:
                                has_attachments = True
                                logger.debug("ATTACHMENT FOUND!")

            # For multimodal content, we need a different model that supports images
            if has_attachments:
                logger.debug("Using multimodal content flow")
                # Check if we have a specific multimodal model defined
                multimodal_model = getattr(self, "multimodal_model_name", None)
                logger.debug(f"Multimodal model attribute: {multimodal_model}")

                if multimodal_model:
                    logger.info(
                        f"Using multimodal model: {multimodal_model} for attachment handling"
                    )
                    client = get_client(multimodal_model)
                    logger.debug(f"Got multimodal client: {client is not None}")
                    if client is None:
                        logger.warning(
                            f"Failed to initialize multimodal client: {multimodal_model}, falling back to default model"
                        )
                        # Fall back to default model
                        client = get_client(self.model_name)
                else:
                    # Use default model
                    logger.debug(
                        f"No multimodal_model_name defined, using default: {self.model_name}"
                    )
                    client = get_client(self.model_name)
            else:
                # Use default model for text-only messages
                logger.debug(f"Using default model for text-only: {self.model_name}")
                client = get_client(self.model_name)

            if client is None:
                yield PartialResponse(
                    text="Error: Failed to initialize Gemini client with Google API key."
                )
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
    # Override model for images/multimodal content
    multimodal_model_name = "gemini-1.5-flash-latest"
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

    model_name = "gemini-2.5-pro-preview-05-06"
    bot_name = "Gemini25ProExpBot"
    bot_description = "Premium Gemini 2.5 Pro Experimental model for enhanced reasoning, multimodal understanding, and advanced coding."


# Experimental Models
class Gemini20FlashExpBot(GeminiBaseBot):
    """Gemini 2.0 Flash Experimental model."""

    model_name = "gemini-2.0-flash-exp"
    bot_name = "Gemini20FlashExpBot"
    bot_description = (
        "Experimental Gemini 2.0 Flash model with latest features, including image generation."
    )
    supports_image_generation = True  # Enable image generation

    # This model can generate images directly in responses


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


class GeminiImageGenerationBot(GeminiBaseBot):
    """Gemini Image Generation bot that creates images from text prompts."""

    model_name = "gemini-2.0-flash-preview-image-generation"
    bot_name = "GeminiImageGenerationBot"
    bot_description = (
        "Generates images from text descriptions using Gemini's image generation capabilities."
    )
    supports_image_generation = True  # Enable image generation

    # Prompt templates help guide the model to generate better images
    image_prompt_template = (
        "{prompt}\n\n"
        "Please generate a high-quality, visually detailed image based on this description. "
        "Include vibrant colors, clear subjects, and good composition."
    )

    async def get_settings(self, settings_request):
        """Get bot settings including the rate card.

        Args:
            settings_request: The settings request

        Returns:
            Settings response with rate card and cost label
        """
        from fastapi_poe.types import SettingsResponse

        # Create a new settings response
        settings = SettingsResponse(
            allow_attachments=True,
            expand_text_attachments=True,
            rate_card="100 points / image",
            cost_label="Image Generation Cost",
        )

        return settings

    async def _generate_image(
        self, prompt: str, query: QueryRequest
    ) -> AsyncGenerator[PartialResponse, None]:
        """Generate an image based on the text prompt using Gemini API.

        Args:
            prompt: The text prompt describing the image to generate
            query: The original query for message_id

        Yields:
            Response chunks as PartialResponse objects
        """
        try:
            # Import required modules
            import google.generativeai as genai

            # Get API key
            api_key = get_api_key("GOOGLE_API_KEY")
            if not api_key:
                logger.error("GOOGLE_API_KEY not found")
                # Try to get from environment directly as a fallback
                import os

                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    yield PartialResponse(
                        text="Error: Google API key is not configured. Please set the GOOGLE_API_KEY environment variable."
                    )
                    return
                logger.info("Found GOOGLE_API_KEY in direct environment")

            # Configure the API key
            genai.configure(api_key=api_key)
            logger.info(f"Configured Gemini with API key: {api_key[:5]}... (truncated)")

            # Send a status message
            yield PartialResponse(
                text=f'🖌️ Generating image from prompt: "{prompt}"\n\nPlease wait a moment...'
            )

            # Store all generated images/text
            generated_text = []
            generated_images = []

            # Apply the prompt template to enhance image generation
            enhanced_prompt = self.image_prompt_template.format(prompt=prompt)

            # Make API request with enhanced prompt
            logger.info(f"Making image generation request with original prompt: {prompt}")
            logger.info(f"Enhanced prompt: {enhanced_prompt}")
            logger.info(f"Using model: {self.model_name}")

            try:
                # Check if the newer API version with GenerationConfig.response_modalities is available
                has_newer_api = False
                try:
                    from google.generativeai import types

                    if hasattr(types, "GenerationConfig") and hasattr(
                        types.GenerationConfig, "response_modalities"
                    ):
                        has_newer_api = True
                except (ImportError, AttributeError):
                    logger.info(
                        "Newer API with types.GenerationConfig.response_modalities not available"
                    )

                # Initialize model
                model = genai.GenerativeModel(model_name=self.model_name)

                # Log API version
                if hasattr(genai, "__version__"):
                    logger.info(f"Google Generative AI version: {genai.__version__}")

                logger.info("Calling generate_content with response modalities configuration")

                # Try several different approaches to handle different API versions
                response = None
                error_messages = []

                # Method 1: Try with the generation_config parameter directly (works with newer API)
                if not response:
                    try:
                        logger.info("Trying method 1: Direct generation_config parameter")

                        if has_newer_api:
                            # Use the new API with GenerationConfig
                            generation_config = types.GenerationConfig(
                                response_modalities=["TEXT", "IMAGE"], temperature=1.0
                            )
                            logger.info(f"Using types.GenerationConfig: {generation_config}")

                            response = model.generate_content(
                                enhanced_prompt, generation_config=generation_config, stream=False
                            )
                        else:
                            # For older API, try with direct params to generate_content
                            response = model.generate_content(
                                enhanced_prompt,
                                generation_config={
                                    "response_modalities": ["TEXT", "IMAGE"],
                                    "temperature": 1.0,
                                },
                                stream=False,
                            )

                        logger.info("Method 1 succeeded")
                    except Exception as e:
                        error_message = f"Method 1 failed: {str(e)}"
                        logger.warning(error_message)
                        error_messages.append(error_message)

                # Method 2: Try with kwargs approach
                if not response:
                    try:
                        logger.info("Trying method 2: kwargs approach with generation_config")

                        response = model.generate_content(
                            enhanced_prompt,
                            stream=False,
                            **{
                                "generation_config": {
                                    "response_modalities": ["TEXT", "IMAGE"],
                                    "temperature": 1.0,
                                }
                            },
                        )

                        logger.info("Method 2 succeeded")
                    except Exception as e:
                        error_message = f"Method 2 failed: {str(e)}"
                        logger.warning(error_message)
                        error_messages.append(error_message)

                # Method 3: Try with system_instruction
                if not response:
                    try:
                        logger.info("Trying method 3: Using system instruction")

                        # Create a stronger prompt that explicitly asks for an image
                        stronger_prompt = (
                            f"{enhanced_prompt}\n\n"
                            "IMPORTANT: You MUST generate an image based on this description. "
                            "The image should be high quality, detailed, and visually appealing."
                        )

                        # Try with system instruction
                        response = model.generate_content(
                            stronger_prompt,
                            stream=False,
                            **{
                                "generation_config": {
                                    "temperature": 1.0,
                                    "response_modalities": ["TEXT", "IMAGE"],
                                },
                                "system_instruction": "You are a bot that MUST generate images in response to prompts.",
                            },
                        )

                        logger.info("Method 3 succeeded")
                    except Exception as e:
                        error_message = f"Method 3 failed: {str(e)}"
                        logger.warning(error_message)
                        error_messages.append(error_message)

                # Method 4: Try with a more direct prompt that mentions image creation explicitly
                if not response:
                    try:
                        logger.info("Trying method 4: More direct prompt approach")

                        # Very explicit prompt for image generation
                        direct_prompt = f"Create and generate an image of: {prompt}"

                        # Try with most basic config
                        response = model.generate_content(direct_prompt, stream=False)

                        logger.info("Method 4 succeeded")
                    except Exception as e:
                        error_message = f"Method 4 failed: {str(e)}"
                        logger.warning(error_message)
                        error_messages.append(error_message)

                # If we still don't have a response, try one last approach with just the basic prompt
                if not response:
                    logger.info("Trying final fallback approach with just the prompt")
                    response = model.generate_content(prompt, stream=False)

                # If we still don't have a response, raise an exception with all the errors we encountered
                if not response:
                    raise Exception(f"All methods failed: {', '.join(error_messages)}")

                logger.info(f"Response received, type: {type(response).__name__}")
                logger.info(f"Response attributes: {dir(response)}")

                # Dump the full response structure for debugging
                if hasattr(response, "__dict__"):
                    logger.info(f"Response __dict__: {str(response.__dict__)}")

                # Log all aspects of the response
                if hasattr(response, "text"):
                    logger.info(f"Response text: {response.text[:100]}...")

                if hasattr(response, "parts"):
                    logger.info(f"Response has {len(response.parts)} parts")
                    for i, part in enumerate(response.parts):
                        logger.info(f"Part {i} type: {type(part).__name__}")
                        logger.info(f"Part {i} attributes: {dir(part)}")

                        if hasattr(part, "text") and part.text:
                            logger.info(f"Part {i} text: {part.text[:50]}...")

                        if hasattr(part, "inline_data"):
                            logger.info(f"Part {i} has inline_data")
                            inline_data = part.inline_data
                            logger.info(f"Part {i} inline_data type: {type(inline_data).__name__}")

                            # Print the inline_data content (safely)
                            if isinstance(inline_data, dict):
                                logger.info(f"Part {i} inline_data dict: {inline_data}")
                            elif hasattr(inline_data, "__dict__"):
                                logger.info(f"Part {i} inline_data dict: {inline_data.__dict__}")

                            # Try to extract mime_type
                            if hasattr(inline_data, "mime_type"):
                                logger.info(f"Part {i} mime_type: {inline_data.mime_type}")
                            elif isinstance(inline_data, dict) and "mime_type" in inline_data:
                                logger.info(
                                    f"Part {i} mime_type (dict): {inline_data['mime_type']}"
                                )

                            # Try to extract data
                            if hasattr(inline_data, "data"):
                                logger.info(f"Part {i} data length: {len(inline_data.data)} bytes")
                            elif isinstance(inline_data, dict) and "data" in inline_data:
                                logger.info(
                                    f"Part {i} data length (dict): {len(inline_data['data'])} bytes"
                                )

                if hasattr(response, "candidates"):
                    logger.info(f"Response has {len(response.candidates)} candidates")
                    for i, candidate in enumerate(response.candidates):
                        logger.info(f"Candidate {i} type: {type(candidate).__name__}")
                        logger.info(f"Candidate {i} attributes: {dir(candidate)}")

                        if hasattr(candidate, "content"):
                            content = candidate.content
                            logger.info(f"Candidate {i} content type: {type(content).__name__}")
                            logger.info(f"Candidate {i} content attributes: {dir(content)}")

                            if hasattr(content, "parts"):
                                logger.info(f"Candidate {i} content has {len(content.parts)} parts")

                # Process the response logic...
                try:
                    logger.info("Beginning response processing with detailed debug info")
                    logger.info(f"Response type: {type(response).__name__}")
                    logger.info(f"Response dir: {dir(response)}")

                    if hasattr(response, "__dict__"):
                        logger.info(f"Response __dict__: {response.__dict__}")

                    # Try to handle text directly (may fail if there's image data)
                    try:
                        if hasattr(response, "text"):
                            logger.info("Response has text attribute, trying to access it safely")
                            try:
                                text = response.text
                                logger.info(
                                    f"Successfully accessed text: {text[:50] if text else 'empty'}"
                                )
                                if text:
                                    generated_text.append(text)
                                    logger.info(f"Found direct text: {text[:50]}...")
                            except Exception as text_err:
                                logger.error(f"Error accessing response.text: {str(text_err)}")
                                logger.error(f"Error type: {type(text_err).__name__}")
                    except ValueError as text_err:
                        # This is expected sometimes when there's only image data
                        logger.info(f"Could not get text from response: {str(text_err)}")

                    # Check for candidates structure (newer API)
                    if hasattr(response, "candidates") and response.candidates:
                        logger.info(f"Response has {len(response.candidates)} candidates")

                        for candidate_idx, candidate in enumerate(response.candidates):
                            logger.info(f"Processing candidate {candidate_idx+1}")

                            if hasattr(candidate, "content") and candidate.content:
                                content = candidate.content
                                logger.info(f"Content attributes: {dir(content)}")

                                # Process parts in content
                                if hasattr(content, "parts") and content.parts:
                                    logger.info(f"Content has {len(content.parts)} parts")

                                    for part_idx, part in enumerate(content.parts):
                                        logger.info(
                                            f"Processing part {part_idx+1}, type: {type(part).__name__}"
                                        )

                                        # Handle text parts
                                        try:
                                            if hasattr(part, "text") and part.text:
                                                text = part.text
                                                if text:
                                                    generated_text.append(text)
                                                    logger.info(f"Found part text: {text[:50]}...")
                                        except ValueError:
                                            # This is fine, might be an image part
                                            pass

                                        # Handle image parts
                                        if (
                                            hasattr(part, "inline_data")
                                            and part.inline_data is not None
                                        ):
                                            try:
                                                inline_data = part.inline_data
                                                logger.info(
                                                    f"Found inline_data: {type(inline_data).__name__}"
                                                )
                                                logger.info(
                                                    f"Inline data attributes: {dir(inline_data)}"
                                                )

                                                # Get mime type and data using different API formats
                                                mime_type = None
                                                data_buffer = None

                                                # Try to get mime_type first
                                                if hasattr(inline_data, "mime_type"):
                                                    mime_type = inline_data.mime_type
                                                elif (
                                                    isinstance(inline_data, dict)
                                                    and "mime_type" in inline_data
                                                ):
                                                    mime_type = inline_data["mime_type"]

                                                # Try different methods to get data
                                                if hasattr(inline_data, "data"):
                                                    # Old API format
                                                    data_buffer = inline_data.data
                                                elif (
                                                    isinstance(inline_data, dict)
                                                    and "data" in inline_data
                                                ):
                                                    # Dictionary format
                                                    data_buffer = inline_data["data"]
                                                elif hasattr(inline_data, "get_bytes") and callable(
                                                    inline_data.get_bytes
                                                ):
                                                    # New API format with get_bytes method
                                                    data_buffer = inline_data.get_bytes()
                                                    if not mime_type:
                                                        mime_type = (
                                                            "image/png"  # Default if not specified
                                                        )

                                                if not mime_type or not data_buffer:
                                                    logger.error(
                                                        "Could not extract mime_type or data from inline_data"
                                                    )
                                                    continue

                                                logger.info(
                                                    f"Successfully extracted image data with mime type: {mime_type}"
                                                )
                                            except Exception as extract_err:
                                                logger.error(
                                                    f"Error extracting image data: {str(extract_err)}"
                                                )
                                                continue

                                            # Now we have mime_type and data_buffer - continue with upload process
                                            logger.info(
                                                f"Extracted {len(data_buffer)} bytes for processing"
                                            )

                                            # Determine file extension
                                            extension = self._get_extension_for_mime_type(mime_type)
                                            filename = (
                                                f"gemini_generated_{int(time.time())}.{extension}"
                                            )

                                            # Upload to Poe
                                            logger.info(
                                                f"Uploading file {filename} with MIME type {mime_type}"
                                            )
                                            attachment_upload_response = (
                                                await self.post_message_attachment(
                                                    message_id=query.message_id,
                                                    file_data=data_buffer,
                                                    filename=filename,
                                                    is_inline=True,
                                                )
                                            )

                                            # Create markdown for display
                                            if (
                                                hasattr(attachment_upload_response, "inline_ref")
                                                and attachment_upload_response.inline_ref
                                            ):
                                                image_md = f"![{filename}][{attachment_upload_response.inline_ref}]"
                                                generated_images.append(image_md)
                                                logger.info(
                                                    f"Successfully processed and uploaded image with ref: {attachment_upload_response.inline_ref}"
                                                )
                                            else:
                                                logger.error(
                                                    "Error uploading image: No inline_ref in response"
                                                )
                                                generated_images.append(
                                                    "[Error uploading generated image]"
                                                )

                    # Fallback to older parts structure if no candidates
                    elif hasattr(response, "parts"):
                        logger.info(f"Response has {len(response.parts)} parts")

                        for part_idx, part in enumerate(response.parts):
                            logger.info(f"Processing part {part_idx+1}")

                            # Handle text content
                            try:
                                if hasattr(part, "text") and part.text:
                                    text = part.text.strip()
                                    if text:
                                        generated_text.append(text)
                                        logger.info(f"Found text part: {text[:30]}...")
                            except ValueError:
                                # This is fine, might be an image part
                                pass

                            # Handle image content
                            if hasattr(part, "inline_data") and part.inline_data is not None:
                                try:
                                    # Extract image data
                                    inline_data = part.inline_data
                                    logger.info(f"Inline data attributes: {dir(inline_data)}")

                                    # Get data using different API formats
                                    mime_type = None
                                    data_buffer = None

                                    # Try to get mime_type first
                                    if hasattr(inline_data, "mime_type"):
                                        mime_type = inline_data.mime_type
                                    elif (
                                        isinstance(inline_data, dict) and "mime_type" in inline_data
                                    ):
                                        mime_type = inline_data["mime_type"]

                                    # Try different methods to get data
                                    if hasattr(inline_data, "data"):
                                        data_buffer = inline_data.data
                                    elif isinstance(inline_data, dict) and "data" in inline_data:
                                        # Dictionary format
                                        data_buffer = inline_data["data"]
                                    elif hasattr(inline_data, "get_bytes") and callable(
                                        inline_data.get_bytes
                                    ):
                                        data_buffer = inline_data.get_bytes()
                                        if not mime_type:
                                            mime_type = "image/png"  # Default if not specified

                                    if not mime_type or not data_buffer:
                                        logger.error(
                                            "Could not extract mime_type or data from inline_data"
                                        )
                                        continue

                                    logger.info(f"Found inline data with mime type: {mime_type}")
                                    logger.info(
                                        f"Extracted {len(data_buffer)} bytes for processing"
                                    )

                                    # Determine file extension
                                    extension = self._get_extension_for_mime_type(mime_type)
                                    filename = f"gemini_generated_{int(time.time())}.{extension}"

                                    # Upload to Poe
                                    logger.info(
                                        f"Uploading file {filename} with MIME type {mime_type}"
                                    )
                                    attachment_upload_response = await self.post_message_attachment(
                                        message_id=query.message_id,
                                        file_data=data_buffer,
                                        filename=filename,
                                        is_inline=True,
                                    )

                                    # Create markdown for display
                                    if (
                                        hasattr(attachment_upload_response, "inline_ref")
                                        and attachment_upload_response.inline_ref
                                    ):
                                        image_md = f"![{filename}][{attachment_upload_response.inline_ref}]"
                                        generated_images.append(image_md)
                                        logger.info(
                                            f"Successfully processed and uploaded image with ref: {attachment_upload_response.inline_ref}"
                                        )
                                    else:
                                        logger.error(
                                            "Error uploading image: No inline_ref in response"
                                        )
                                        generated_images.append("[Error uploading generated image]")
                                except Exception as img_err:
                                    logger.error(f"Error processing image data: {str(img_err)}")
                                    generated_images.append(
                                        f"[Error processing image: {str(img_err)}]"
                                    )
                                    continue
                except Exception as parse_err:
                    logger.error(f"Error parsing response: {str(parse_err)}")
                    raise
            except Exception as api_err:
                # Check specifically for the "Could not convert `part.inline_data` to text" error
                error_str = str(api_err)
                logger.error(f"Error in Gemini API call: {error_str}", exc_info=True)

                if "Could not convert `part.inline_data` to text" in error_str:
                    logger.error(
                        "Detected the inline_data conversion error - providing helpful error message"
                    )
                    yield PartialResponse(
                        text="Sorry, I'm unable to generate the image at this time. The image generation model responded "
                        + "with content in a format that I couldn't process correctly.\n\n"
                        + "This is likely due to one of these reasons:\n"
                        + "1. The Google Gemini API version is not fully compatible with image generation\n"
                        + "2. The API key might not have permissions for image generation\n"
                        + "3. There might be a temporary service disruption\n\n"
                        + "You can try again later or with a different prompt."
                    )
                else:
                    yield PartialResponse(text=f"Error generating image: {error_str}\n")
                return

            # Yield the model's explanation first, if any
            if generated_text:
                yield PartialResponse(text="\n\n".join(generated_text))

            # Yield the generated images
            for image_md in generated_images:
                yield PartialResponse(text="\n\n" + image_md)

            # If no images were generated, inform the user and suggest alternative phrasing
            if not generated_images:
                suggest_text = (
                    "\n\nNo images were generated. This could be due to content policy restrictions "
                    "or a technical issue. Please try a different prompt with the following tips:\n\n"
                    "- Be more specific and descriptive (e.g. 'a photo of a red apple on a wooden table')\n"
                    "- Avoid potentially sensitive content\n"
                    "- Try simpler subjects (e.g. landscapes, objects, animals)\n"
                    "- Include details about style (e.g. 'digital art style', 'photorealistic', 'cartoon')"
                )
                yield PartialResponse(text=suggest_text)

        except Exception as e:
            logger.error(f"Error generating image: {str(e)}", exc_info=True)
            yield PartialResponse(text=f"Error generating image: {str(e)}")

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

            # Check if the API key exists
            api_key = get_api_key("GOOGLE_API_KEY")
            if not api_key:
                logger.error("GOOGLE_API_KEY not found in environment variables")
                yield PartialResponse(
                    text="Error: Google API key is not configured. Please set the GOOGLE_API_KEY environment variable."
                )
                return
            else:
                logger.info(f"Found GOOGLE_API_KEY in environment (starts with: {api_key[:3]}...)")

            # Handle bot info request
            if user_message.lower().strip() == "bot info":
                metadata = self._get_bot_metadata()
                metadata["model_name"] = self.model_name
                metadata["supports_image_generation"] = self.supports_image_generation
                yield PartialResponse(
                    text=f"Gemini Image Generation Bot\nModel: {self.model_name}\nCapabilities: Image generation from text prompts"
                )
                return

            # Check if the message is a help request
            if user_message.lower().startswith(("help", "info", "what can you do", "capabilities")):
                yield PartialResponse(
                    text=(
                        "I can generate images based on your text descriptions. "
                        "Simply describe the image you want to generate, and I'll create it for you.\n\n"
                        "Example prompts:\n"
                        "- A serene mountain lake at sunset with pine trees\n"
                        "- A futuristic cityscape with flying cars and neon lights\n"
                        "- A cute cartoon cat wearing a space helmet\n\n"
                        "Please note that I cannot generate images that violate Google's content policies."
                    )
                )
                return

            # Process as an image generation request
            async for response in self._generate_image(user_message, query):
                yield response

        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}", exc_info=True)
            yield PartialResponse(text=f"Error: {str(e)}")

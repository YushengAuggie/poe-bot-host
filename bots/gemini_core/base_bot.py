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

from .client import GeminiClientStub, get_client
from .model_config import get_model_capabilities
from .utils import get_extension_for_mime_type, get_extension_for_video_mime_type

# Get the logger
logger = logging.getLogger(__name__)


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

        # Get model capabilities from configuration
        model_capabilities = get_model_capabilities(self.model_name)
        self.supports_grounding = model_capabilities["supports_grounding"]
        self.supports_thinking = model_capabilities["supports_thinking"]

        # Log capability detection
        logger.info(
            f"Model {self.model_name} capabilities: grounding={self.supports_grounding}, thinking={self.supports_thinking}"
        )

        # Default grounding settings (only effective if the model supports grounding)
        self.grounding_enabled = kwargs.get("grounding_enabled", self.supports_grounding)
        self.citations_enabled = kwargs.get("citations_enabled", True)
        self.grounding_sources = []

        # Enable Google Search grounding by default for supported models
        self.google_search_grounding = kwargs.get(
            "google_search_grounding", self.supports_grounding
        )

        # Thinking budget configuration for supported models
        self.thinking_budget = kwargs.get(
            "thinking_budget", 8192 if self.supports_thinking else None
        )  # Default 8K tokens
        self.include_thoughts = kwargs.get("include_thoughts", False)

        # Update bot description with capability info
        self._update_bot_description(model_capabilities)

    def _update_bot_description(self, model_capabilities: dict):
        """Update bot description with capability information.

        Args:
            model_capabilities: Dictionary of model capabilities
        """
        # Add capability indicators to the description
        capability_indicators = []

        if model_capabilities["free_tier"]:
            capability_indicators.append("✅ Free tier available")
        else:
            capability_indicators.append("⚠️ Requires paid Google API plan")

        if model_capabilities["supports_grounding"]:
            capability_indicators.append("🔍 Google Search grounding")

        if model_capabilities["supports_thinking"]:
            capability_indicators.append("🧠 Thinking budget")

        if model_capabilities["supports_image_generation"]:
            capability_indicators.append("🎨 Image generation")

        # Update the description if we have capability indicators
        if capability_indicators and hasattr(self, "bot_description"):
            # Remove any existing capability indicators
            base_description = self.bot_description
            for indicator in [
                "✅ Free tier available",
                "⚠️ Requires paid Google API plan",
                "⚠️ Requires paid plan",
                "May require paid Google API plan",
            ]:
                base_description = base_description.replace(f" {indicator}", "").replace(
                    f". {indicator}", ""
                )

            # Add new capability indicators
            self.bot_description = (
                f"{base_description.rstrip('.')}. {' | '.join(capability_indicators)}"
            )

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
        metadata["supports_thinking"] = self.supports_thinking
        metadata["thinking_budget"] = self.thinking_budget
        metadata["include_thoughts"] = self.include_thoughts
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

    def set_google_search_grounding(self, enabled: bool) -> None:
        """Enable or disable Google Search grounding.

        Args:
            enabled: Whether Google Search grounding should be enabled
        """
        if enabled and not self.supports_grounding:
            logger.warning(f"Model {self.model_name} does not support grounding. Request ignored.")
            return

        self.google_search_grounding = enabled
        logger.info(f"Google Search grounding {'enabled' if enabled else 'disabled'}")

    def set_thinking_budget(self, budget: int) -> None:
        """Set the thinking budget for supported models.

        Args:
            budget: Number of tokens for thinking (1-24576, or 0 to disable)
        """
        if not self.supports_thinking:
            logger.warning(
                f"Model {self.model_name} does not support thinking budget. Request ignored."
            )
            return

        if budget < 0 or budget > 24576:
            logger.warning(f"Invalid thinking budget {budget}. Must be 0-24576. Request ignored.")
            return

        self.thinking_budget = budget
        logger.info(f"Thinking budget set to {budget} tokens")

    def set_include_thoughts(self, include: bool) -> None:
        """Enable or disable including thoughts in the response.

        Args:
            include: Whether to include the model's thinking process in responses
        """
        if not self.supports_thinking:
            logger.warning(
                f"Model {self.model_name} does not support thinking budget. Request ignored."
            )
            return

        self.include_thoughts = include
        logger.info(f"Include thoughts {'enabled' if include else 'disabled'}")

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

        # Return None if grounding is disabled and Google Search grounding is also disabled
        if not self.grounding_enabled and not self.google_search_grounding:
            return None

        # Also return None if grounding is enabled but no sources AND Google Search is disabled
        if (
            self.grounding_enabled
            and not self.grounding_sources
            and not self.google_search_grounding
        ):
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

            # Prepare the grounding configuration
            result: Dict[str, Any] = {}

            # Add Google Search grounding if enabled
            if self.google_search_grounding:
                result["tools"] = ["google_search_retrieval"]
                logger.info("Enabled Google Search grounding")

            # Add custom grounding sources if available
            if self.grounding_enabled and ground_sources:
                result.update(
                    {
                        "groundingEnabled": True,
                        "groundingSources": ground_sources,
                    }
                )
                logger.info(f"Added {len(ground_sources)} custom grounding sources")

            # Add citation configuration if enabled
            if self.citations_enabled:
                result["includeCitations"] = True

            return result if result else None
        except ImportError:
            logger.warning("Failed to import google.generativeai for grounding")
            return None
        except Exception as e:
            logger.error(f"Error preparing grounding config: {str(e)}")
            return None

    def _prepare_thinking_config(self) -> Optional[Dict[str, Any]]:
        """Prepare the thinking configuration for the Gemini API.

        Returns:
            A dictionary with thinking configuration or None if thinking is not supported/enabled
        """
        # Temporarily disable thinking budget configuration until we can research
        # the correct API parameter structure for Gemini 2.5 thinking models
        if not self.supports_thinking or self.thinking_budget is None:
            return None

        try:
            # NOTE: The thinking budget feature appears to be model-specific and may not
            # require explicit configuration in the API call. Many thinking models
            # automatically apply reasoning without explicit budget parameters.
            #
            # For now, we'll return None to avoid API errors while keeping the
            # infrastructure in place for future implementation once the correct
            # parameter structure is determined.

            logger.info(
                f"Thinking capability detected for {getattr(self, 'model_name', 'unknown')} but configuration disabled pending API research"
            )
            return None

            # Future implementation would go here once correct parameters are identified:
            # thinking_config = {}
            # if self.thinking_budget == 0:
            #     thinking_config["correct_thinking_param"] = 0
            # else:
            #     budget = max(1024, self.thinking_budget)
            #     thinking_config["correct_thinking_param"] = budget
            # if self.include_thoughts:
            #     thinking_config["correct_thoughts_param"] = True
            # return thinking_config

        except Exception as e:
            logger.error(f"Error preparing thinking config: {str(e)}")
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
            extension = get_extension_for_video_mime_type(mime_type)
            filename = f"gemini_video_{int(time.time())}.{extension}"
        else:
            extension = get_extension_for_mime_type(mime_type)
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
                # Provide helpful error message for missing access key
                if "access_key parameter is required" in str(e):
                    error_msg = (
                        "✅ **Video Generated Successfully!**\n\n"
                        "⚠️ **Display Issue:** To display videos inline, this bot needs a Poe access key.\n"
                        "**To fix:** Set environment variable with your Poe bot access key."
                    )
                else:
                    error_msg = f"[Error uploading video: {str(e)}]"
                return PartialResponse(text=error_msg)

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
                fallback_text = f"![Gemini generated image](data:{mime_type};base64,{img_str})"
                if "access_key parameter is required" in str(e):
                    fallback_text += "\n\n⚠️ **Note:** Image displayed using fallback method due to missing Poe access key. For inline display, set environment variable with your Poe bot access key."
                return PartialResponse(text=fallback_text)
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

                # Handle Google Search grounding (tools parameter)
                if "tools" in grounding_config:
                    generation_config["tools"] = grounding_config["tools"]
                    logger.info(
                        f"Added tools to multimodal generation config: {grounding_config['tools']}"
                    )

                # Handle custom grounding sources
                if "groundingEnabled" in grounding_config:
                    # Add grounding directly to the generation_config dict
                    # In the API, this nests under generation_config
                    generation_config["generation_config"] = {}
                    logger.info("Added empty generation_config to dict")

                    # Add grounding to the generation config dictionary
                    generation_config["grounding_config"] = {
                        k: v
                        for k, v in grounding_config.items()
                        if k not in ["tools"]  # Exclude tools from grounding_config
                    }
                    logger.info("Added grounding_config to generation_config")
            except ImportError:
                logger.warning("Failed to import google.generativeai for grounding config")
            except Exception as e:
                logger.warning(f"Error setting up grounding configuration: {str(e)}")

        # Apply thinking configuration if available
        thinking_config = self._prepare_thinking_config()
        if thinking_config:
            try:
                _ = __import__("google.generativeai")

                # Initialize generation_config if not already done
                if "generation_config" not in generation_config:
                    generation_config["generation_config"] = {}

                # Add thinking configuration to the generation_config dict
                generation_config["generation_config"].update(thinking_config)
                logger.info(f"Added thinking config to multimodal generation: {thinking_config}")
            except ImportError:
                logger.warning("Failed to import google.generativeai for thinking config")
            except Exception as e:
                logger.warning(f"Error setting up thinking configuration: {str(e)}")

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

                    # Handle Google Search grounding (tools parameter)
                    if "tools" in grounding_config:
                        generation_config["tools"] = grounding_config["tools"]
                        logger.info(
                            f"Added tools to generation config: {grounding_config['tools']}"
                        )

                    # Handle custom grounding sources
                    if "groundingEnabled" in grounding_config:
                        # Create or use existing generation config dictionary
                        if "generation_config" not in generation_config:
                            generation_config["generation_config"] = {}

                        # Add grounding to the generation config at the appropriate level
                        generation_config["grounding_config"] = {
                            k: v
                            for k, v in grounding_config.items()
                            if k not in ["tools"]  # Exclude tools from grounding_config
                        }
                except ImportError:
                    logger.warning("Failed to import google.generativeai for grounding config")

            # Apply thinking configuration if available
            thinking_config = self._prepare_thinking_config()
            if thinking_config:
                try:
                    _ = __import__("google.generativeai")

                    # Initialize generation_config if not already done
                    if "generation_config" not in generation_config:
                        generation_config["generation_config"] = {}

                    # Add thinking configuration to the generation_config dict
                    generation_config["generation_config"].update(thinking_config)
                    logger.info(f"Added thinking config to streaming generation: {thinking_config}")
                except ImportError:
                    logger.warning("Failed to import google.generativeai for thinking config")
                except Exception as e:
                    logger.warning(f"Error setting up thinking configuration: {str(e)}")

            # If we support image generation, we need to disable streaming and set response modalities
            if getattr(self, "supports_image_generation", False):
                # Disable streaming for possible image responses
                generation_config["stream"] = False

                # Add response modalities for image generation
                try:
                    import google.generativeai as genai

                    if hasattr(genai, "types") and hasattr(genai.types, "GenerateContentConfig"):
                        # Use proper config object
                        config = genai.types.GenerateContentConfig(
                            response_modalities=["TEXT", "IMAGE"], temperature=1.0
                        )
                        logger.info("Using GenerateContentConfig for image generation")
                        response = client.generate_content(contents, config=config)
                    else:
                        # Fallback to old method
                        generation_config["generation_config"] = {
                            "response_modalities": ["TEXT", "IMAGE"],
                            "temperature": 1.0,
                        }
                        logger.info("Using fallback config for image generation")
                        response = client.generate_content(contents, **generation_config)
                except ImportError:
                    logger.warning("Failed to import google.generativeai for image generation")
                    response = client.generate_content(contents, **generation_config)
            else:
                response = client.generate_content(contents, **generation_config)

            # If streaming is disabled (for image-capable models), use media processing
            if not generation_config.get("stream", True):
                logger.info("Processing potential media response")
                # Process any media (images) in the response
                try:
                    async for partial_response in self._process_media_in_response(response, query):
                        yield partial_response
                except ValueError as e:
                    if "Could not convert `part.inline_data` to text" in str(e):
                        logger.info(
                            "Successfully generated image but got inline_data conversion error - this is expected"
                        )
                        # The image was already processed successfully, just return
                        return
                    else:
                        raise e
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

            # Provide specific error messages for common issues
            error_str = str(e).lower()
            if "429" in str(e) and ("free quota" in error_str or "free tier" in error_str):
                error_msg = (
                    f"⚠️ **{self.model_name} requires a paid Google API plan**\n\n"
                    "Google has removed free tier access for some Gemini models. To use this model:\n"
                    "1. Visit [Google AI Studio](https://ai.google.dev/)\n"
                    "2. Set up billing for your Google Cloud project\n"
                    "3. Enable paid API access for Gemini models\n\n"
                    "**Alternative:** Try using a free Gemini model like `Gemini20FlashBot` instead."
                )
            elif "search grounding is not supported" in error_str or "grounding" in error_str:
                error_msg = (
                    f"⚠️ **Google Search grounding not supported for {self.model_name}**\n\n"
                    "This model doesn't support Google Search grounding features. \n"
                    "**Alternative models with grounding support:**\n"
                    "- `Gemini20ProBot` (Gemini 2.0 Pro)\n"
                    "- `Gemini20ProExpBot` (Gemini 2.0 Pro Experimental)\n"
                    "- `Gemini25ProExpBot` (Gemini 2.5 Pro - requires paid plan)\n\n"
                    "The bot will work normally without grounding features."
                )
            elif "quota" in error_str or "rate limit" in error_str:
                error_msg = (
                    f"⚠️ **API quota exceeded for {self.model_name}**\n\n"
                    "You've reached the usage limits for this model. Please:\n"
                    "1. Wait a few minutes and try again\n"
                    "2. Check your quota limits in [Google AI Studio](https://ai.google.dev/)\n"
                    "3. Consider upgrading to a paid plan for higher limits"
                )
            elif "api key" in error_str or "authentication" in error_str:
                error_msg = (
                    "🔑 **API key issue**\n\n"
                    "Please check that your Google API key is:\n"
                    "1. Correctly configured in Modal secrets\n"
                    "2. Valid and not expired\n"
                    "3. Has access to Gemini API\n\n"
                    "You can get an API key at [Google AI Studio](https://ai.google.dev/)"
                )
            else:
                error_msg = f"Error: Could not get response from Gemini: {str(e)}"

            yield PartialResponse(text=error_msg)

    async def get_settings(self, settings_request: SettingsRequest) -> SettingsResponse:
        """Get bot settings.

        Args:
            settings_request: The settings request

        Returns:
            Settings response with appropriate settings
        """
        # Use parent class settings
        return await super().get_settings(settings_request)

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

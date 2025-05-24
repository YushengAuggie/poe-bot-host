"""Media processing utilities extracted from GeminiBaseBot."""

import logging
from typing import Any, Dict, List, Optional

from .validators import MediaValidator

logger = logging.getLogger(__name__)


class MediaProcessor:
    """Handles media attachment processing for AI model consumption."""

    def __init__(
        self,
        supported_image_types: Optional[List[str]] = None,
        supported_video_types: Optional[List[str]] = None,
        supported_audio_types: Optional[List[str]] = None,
    ):
        """
        Initialize the media processor.

        Args:
            supported_image_types: Override default image types
            supported_video_types: Override default video types
            supported_audio_types: Override default audio types
        """
        self.validator = MediaValidator(
            supported_image_types, supported_video_types, supported_audio_types
        )
        self.supported_image_types = self.validator.supported_image_types
        self.supported_video_types = self.validator.supported_video_types
        self.supported_audio_types = self.validator.supported_audio_types

    def extract_attachments(self, query) -> List[Any]:
        """
        Extract attachments from a query request.

        Args:
            query: The query request object

        Returns:
            List of attachment objects
        """
        attachments = []
        try:
            if isinstance(query.query, list) and query.query:
                last_message = query.query[-1]

                if hasattr(last_message, "attachments") and last_message.attachments:
                    attachments = last_message.attachments
                    logger.info(f"Found {len(attachments)} attachments")

                    # Fix content attribute accessibility for Pydantic models
                    for attachment in attachments:
                        self._fix_content_accessibility(attachment)

        except Exception as e:
            logger.error(f"Error extracting attachments: {str(e)}")

        return attachments

    def _fix_content_accessibility(self, attachment) -> None:
        """Fix content attribute accessibility for Pydantic models."""
        if (
            (not hasattr(attachment, "content") or getattr(attachment, "content", None) is None)
            and hasattr(attachment, "__dict__")
            and "content" in attachment.__dict__
        ):
            try:
                content_from_dict = attachment.__dict__["content"]
                object.__setattr__(attachment, "content", content_from_dict)
                logger.debug("Fixed content attribute accessibility")
            except Exception as e:
                logger.warning(f"Failed to fix content attribute: {e}")

    def process_attachment(self, attachment) -> Optional[Dict[str, Any]]:
        """
        Process a single media attachment.

        Args:
            attachment: The attachment object

        Returns:
            Dictionary with mime_type and data, or None if processing failed
        """
        # Validate the attachment
        is_valid, error_msg, media_type = self.validator.validate_attachment(attachment)
        if not is_valid:
            logger.warning(f"Invalid attachment: {error_msg}")
            return None

        # Extract content
        content = self._extract_content(attachment)
        if not content:
            logger.error("Could not extract content from attachment")
            return None

        content_type = getattr(attachment, "content_type", "application/octet-stream")

        logger.debug(f"Successfully processed {media_type} attachment: {content_type}")
        return {"mime_type": content_type, "data": content}

    def _extract_content(self, attachment) -> Optional[bytes]:
        """Extract binary content from an attachment."""
        content = None
        extraction_method = "none"

        # Try direct attribute access
        if hasattr(attachment, "content") and getattr(attachment, "content", None):
            content = getattr(attachment, "content")
            extraction_method = "direct_attribute"

        # Try __dict__ access
        elif hasattr(attachment, "__dict__") and attachment.__dict__.get("content"):
            content = attachment.__dict__["content"]
            extraction_method = "dict_attribute"

        # Try _content attribute
        elif hasattr(attachment, "_content") and getattr(attachment, "_content", None):
            content = getattr(attachment, "_content")
            extraction_method = "underscore_attribute"

        # Try URL download
        elif hasattr(attachment, "url") and attachment.url:
            content = self._download_from_url(attachment.url)
            extraction_method = "url_download"

        if not content:
            return None

        # Handle base64 content
        if isinstance(content, str) and content.startswith(("/", "+", "i")):
            try:
                import base64

                decoded_content = base64.b64decode(content)
                extraction_method += "_base64"
                logger.debug(f"Content extracted using method: {extraction_method}")
                return decoded_content
            except Exception as e:
                logger.error(f"Error decoding base64: {str(e)}")
                return None

        # Ensure we return bytes
        if isinstance(content, str):
            logger.debug(f"Content extracted using method: {extraction_method}")
            return content.encode("utf-8")
        elif isinstance(content, bytes):
            logger.debug(f"Content extracted using method: {extraction_method}")
            return content
        else:
            logger.warning(f"Unexpected content type: {type(content)}")
            return None

    def _download_from_url(self, url: str) -> Optional[bytes]:
        """Download content from a URL."""
        if url.startswith("file://"):
            logger.debug("File:// URL detected, content should be in other fields")
            return None

        try:
            import requests

            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                logger.debug(f"Downloaded {len(response.content)} bytes from URL")
                return response.content
            else:
                logger.warning(f"Failed to download from URL: HTTP {response.status_code}")
        except Exception as e:
            logger.warning(f"Failed to download from URL: {str(e)}")

        return None

    def prepare_media_parts(self, attachments: List[Any]) -> List[Dict[str, Any]]:
        """
        Process multiple attachments into media parts for AI model consumption.

        Args:
            attachments: List of attachment objects

        Returns:
            List of formatted media parts
        """
        media_parts = []

        for i, attachment in enumerate(attachments):
            logger.debug(f"Processing attachment {i+1} of {len(attachments)}")

            media_data = self.process_attachment(attachment)
            if media_data:
                # Try to use Google Generative AI types if available
                formatted_part = self._format_for_api(media_data)
                if formatted_part:
                    media_parts.append(formatted_part)
                    logger.debug(f"Successfully added attachment {i+1}")
            else:
                logger.warning(f"Failed to process attachment {i+1}")

        logger.debug(f"Prepared {len(media_parts)} media parts")
        return media_parts

    def _format_for_api(self, media_data: Dict[str, Any]) -> Optional[Any]:
        """Format media data for API consumption."""
        try:
            # Try to use Google Generative AI types
            import google.generativeai as genai  # type: ignore

            if hasattr(genai, "types") and hasattr(genai.types, "Part"):
                # Type ignore because genai.types is dynamic
                part = genai.types.Part.from_bytes(  # type: ignore
                    data=media_data["data"], mime_type=media_data["mime_type"]
                )
                logger.debug("Created Part object using types.Part")
                return part
        except (ImportError, AttributeError, Exception) as e:
            logger.debug(f"Could not use types.Part: {e}")

        # Fallback to dictionary format
        return {
            "inline_data": {
                "mime_type": media_data["mime_type"],
                "data": media_data["data"],
            }
        }

    def has_media(self, query) -> bool:
        """Check if query contains media attachments."""
        attachments = self.extract_attachments(query)
        return len(attachments) > 0

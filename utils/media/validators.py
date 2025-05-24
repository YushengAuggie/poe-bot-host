"""Media validation utilities."""

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class MediaValidator:
    """Validates media attachments for bot processing."""

    # Supported media types
    SUPPORTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp", "image/gif"]
    SUPPORTED_VIDEO_TYPES = ["video/mp4", "video/quicktime", "video/webm"]
    SUPPORTED_AUDIO_TYPES = ["audio/mp3", "audio/mpeg", "audio/wav", "audio/x-wav", "audio/ogg"]

    # Size limits (in bytes)
    MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
    MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_AUDIO_SIZE = 25 * 1024 * 1024  # 25MB

    def __init__(
        self,
        supported_image_types: Optional[List[str]] = None,
        supported_video_types: Optional[List[str]] = None,
        supported_audio_types: Optional[List[str]] = None,
    ):
        """
        Initialize the validator with custom supported types.

        Args:
            supported_image_types: Override default image types
            supported_video_types: Override default video types
            supported_audio_types: Override default audio types
        """
        self.supported_image_types = supported_image_types or self.SUPPORTED_IMAGE_TYPES
        self.supported_video_types = supported_video_types or self.SUPPORTED_VIDEO_TYPES
        self.supported_audio_types = supported_audio_types or self.SUPPORTED_AUDIO_TYPES

    def validate_attachment(self, attachment) -> Tuple[bool, Optional[str], str]:
        """
        Validate a media attachment.

        Args:
            attachment: The attachment object to validate

        Returns:
            Tuple of (is_valid, error_message, media_type)
        """
        if not attachment:
            return False, "Attachment is null or empty", "unknown"

        # Get content type
        content_type = getattr(attachment, "content_type", "unknown")
        if content_type == "unknown":
            return False, "Unknown content type", "unknown"

        # Determine media type and validate
        media_type = self._get_media_type(content_type)
        if media_type == "unknown":
            return False, f"Unsupported media type: {content_type}", media_type

        # Validate size if content is available
        content = self._get_attachment_content(attachment)
        if content:
            size_valid, size_error = self._validate_size(content, media_type)
            if not size_valid:
                return False, size_error, media_type

        return True, None, media_type

    def _get_media_type(self, content_type: str) -> str:
        """Determine the media type category."""
        if content_type in self.supported_image_types:
            return "image"
        elif content_type in self.supported_video_types:
            return "video"
        elif content_type in self.supported_audio_types:
            return "audio"
        return "unknown"

    def _validate_size(self, content: bytes, media_type: str) -> Tuple[bool, Optional[str]]:
        """Validate media size based on type."""
        size = len(content)

        size_limits = {
            "image": self.MAX_IMAGE_SIZE,
            "video": self.MAX_VIDEO_SIZE,
            "audio": self.MAX_AUDIO_SIZE,
        }

        max_size = size_limits.get(media_type, self.MAX_IMAGE_SIZE)

        if size > max_size:
            max_mb = max_size / (1024 * 1024)
            current_mb = size / (1024 * 1024)
            return (
                False,
                f"{media_type.title()} too large: {current_mb:.1f}MB (max: {max_mb:.1f}MB)",
            )

        return True, None

    def _get_attachment_content(self, attachment) -> Optional[bytes]:
        """Extract content from attachment for validation."""
        try:
            # Try direct attribute access
            if hasattr(attachment, "content") and getattr(attachment, "content"):
                content = getattr(attachment, "content")
                if isinstance(content, str):
                    # Might be base64
                    try:
                        import base64

                        return base64.b64decode(content)
                    except Exception:
                        return content.encode("utf-8")
                return content

            # Try __dict__ access
            if hasattr(attachment, "__dict__") and "content" in attachment.__dict__:
                content = attachment.__dict__["content"]
                if isinstance(content, str):
                    try:
                        import base64

                        return base64.b64decode(content)
                    except Exception:
                        return content.encode("utf-8")
                return content

        except Exception as e:
            logger.debug(f"Could not extract content for validation: {e}")

        return None

    def get_supported_types(self) -> Dict[str, List[str]]:
        """Get all supported media types."""
        return {
            "image": self.supported_image_types,
            "video": self.supported_video_types,
            "audio": self.supported_audio_types,
        }

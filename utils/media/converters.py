"""Media format conversion utilities."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class MediaConverter:
    """Handles media format conversions and optimizations."""

    @staticmethod
    def get_extension_for_mime_type(mime_type: str) -> str:
        """
        Get file extension for a given MIME type.

        Args:
            mime_type: The MIME type string

        Returns:
            File extension (without dot)
        """
        mime_to_ext = {
            "image/jpeg": "jpg",
            "image/jpg": "jpg",
            "image/png": "png",
            "image/webp": "webp",
            "image/gif": "gif",
            "video/mp4": "mp4",
            "video/quicktime": "mov",
            "video/webm": "webm",
            "audio/mp3": "mp3",
            "audio/mpeg": "mp3",
            "audio/wav": "wav",
            "audio/x-wav": "wav",
            "audio/ogg": "ogg",
        }

        return mime_to_ext.get(mime_type.lower(), "bin")

    @staticmethod
    def get_extension_for_video_mime_type(mime_type: str) -> str:
        """
        Get file extension for video MIME types.

        Args:
            mime_type: The video MIME type

        Returns:
            Video file extension
        """
        return MediaConverter.get_extension_for_mime_type(mime_type)

    @staticmethod
    def optimize_image_size(
        image_data: bytes, max_size: int = 10 * 1024 * 1024, quality: int = 85
    ) -> Optional[bytes]:
        """
        Optimize image size by reducing quality or dimensions.

        Args:
            image_data: Original image data
            max_size: Maximum size in bytes
            quality: JPEG quality (1-100)

        Returns:
            Optimized image data or None if optimization failed
        """
        if len(image_data) <= max_size:
            return image_data

        try:
            import io

            from PIL import Image

            # Load image
            img = Image.open(io.BytesIO(image_data))
            original_format = img.format or "JPEG"

            # Try reducing quality first
            buffer = io.BytesIO()
            img.save(buffer, format=original_format, quality=quality, optimize=True)
            optimized_data = buffer.getvalue()

            if len(optimized_data) <= max_size:
                logger.debug(
                    f"Optimized image from {len(image_data)} to {len(optimized_data)} bytes"
                )
                return optimized_data

            # If still too large, resize image
            width, height = img.size
            scale_factor = (max_size / len(optimized_data)) ** 0.5
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            buffer = io.BytesIO()
            resized_img.save(buffer, format=original_format, quality=quality, optimize=True)
            final_data = buffer.getvalue()

            logger.debug(
                f"Resized and optimized image from {len(image_data)} to {len(final_data)} bytes"
            )
            return final_data

        except Exception as e:
            logger.error(f"Failed to optimize image: {e}")
            return None

    @staticmethod
    def convert_to_base64_data_uri(data: bytes, mime_type: str) -> str:
        """
        Convert binary data to base64 data URI.

        Args:
            data: Binary data
            mime_type: MIME type of the data

        Returns:
            Base64 data URI string
        """
        import base64

        b64_data = base64.b64encode(data).decode("utf-8")
        return f"data:{mime_type};base64,{b64_data}"

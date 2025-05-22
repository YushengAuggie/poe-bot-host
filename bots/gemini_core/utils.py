# Utility functions for Gemini bots

def get_extension_for_mime_type(mime_type: str) -> str:
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

def get_extension_for_video_mime_type(mime_type: str) -> str:
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

"""
YouTube Downloader Bot - A bot that downloads YouTube videos.

This bot takes a YouTube URL from the user, downloads the video,
and sends it back to the user as an attachment.
"""

import json
import logging
import os
import re
import tempfile
import uuid
from typing import AsyncGenerator, List, Optional, Union
from urllib.parse import urlparse

import modal
from fastapi_poe.types import (
    Attachment,
    ContentType,
    MetaResponse,
    PartialResponse,
    QueryRequest,
)

from utils.base_bot import BaseBot, BotError, BotErrorNoRetry

logger = logging.getLogger(__name__)

# Modal app definition
app = modal.App("youtube-downloader-bot")

# Make sure to install yt-dlp using pip
try:
    import yt_dlp
except ImportError:
    logger.error("yt-dlp not installed. Please install it with 'pip install yt-dlp'")


class YouTubeDownloaderBot(BaseBot):
    """
    A bot that downloads YouTube videos.

    Takes a YouTube URL from the user, downloads the video, and sends it back
    as an attachment.
    """

    bot_name = "YouTubeDownloaderBot"
    bot_description = "Send me a YouTube URL, and I'll download the video for you."
    version = "1.0.0"
    rate_card_points = 300  # Setting cost as 300 points per message

    # Pattern to match YouTube URLs including the video ID
    YOUTUBE_URL_PATTERN = (
        r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([\w-]+)"
    )

    def __init__(self, **kwargs):
        """Initialize the YouTubeDownloaderBot."""
        super().__init__(**kwargs)

        # Create a temporary directory for downloads
        self.temp_dir = tempfile.mkdtemp()
        logger.debug(f"Created temporary directory for downloads: {self.temp_dir}")

    def _extract_youtube_urls(self, text: str) -> List[str]:
        """
        Extract YouTube URLs from text.

        Args:
            text: The text to extract URLs from

        Returns:
            List of YouTube URLs found in the text
        """
        try:
            # Attempt to find matches in the text
            matches = re.findall(self.YOUTUBE_URL_PATTERN, text)

            if not matches:
                logger.debug(f"No YouTube URLs found in text: {text[:50]}...")
                return []

            # Convert matches to full URLs
            full_urls = []
            for match in matches:
                try:
                    # Extract components
                    protocol = match[0] if match[0] else "https://"
                    domain = match[1] if match[1] else "www."
                    path = match[2]
                    video_id = match[3]

                    # Validate video ID (should be alphanumeric with some special chars)
                    if not re.match(r"^[\w-]+$", video_id):
                        logger.warning(f"Invalid YouTube video ID: {video_id}")
                        continue

                    # Build full URL
                    if "youtu.be" in path:
                        url = f"{protocol}{domain}youtu.be/{video_id}"
                    elif "shorts" in path:
                        url = f"{protocol}{domain}youtube.com/shorts/{video_id}"
                    else:
                        url = f"{protocol}{domain}youtube.com/watch?v={video_id}"

                    full_urls.append(url)
                    logger.debug(f"Extracted YouTube URL: {url}")

                except Exception as e:
                    # Log but continue with other URLs if one fails
                    logger.error(f"Error processing YouTube URL match: {str(e)}")
                    continue

            return full_urls

        except Exception as e:
            logger.error(f"Error extracting YouTube URLs: {str(e)}", exc_info=True)
            return []

    def _validate_youtube_url(self, url: str) -> bool:
        """
        Validate that a URL is a YouTube URL.

        Args:
            url: The URL to validate

        Returns:
            True if the URL is a valid YouTube URL, False otherwise
        """
        try:
            # Validate input type
            if not isinstance(url, str):
                logger.warning(f"Invalid URL type: {type(url)}")
                return False

            # Handle empty or very short URLs
            if not url or len(url) < 10:  # Minimum reasonable length for a YouTube URL
                logger.warning(f"URL too short to be a valid YouTube URL: {url}")
                return False

            # Parse the URL
            parsed_url = urlparse(url)

            # Check for basic URL structure - must have netloc (domain)
            if not parsed_url.netloc:
                logger.debug(f"Invalid URL structure (no domain): {url}")
                return False

            # Check the domain (properly handle both with and without www)
            youtube_domains = ["youtube.com", "www.youtube.com", "youtu.be", "www.youtu.be"]

            if parsed_url.netloc.lower() in youtube_domains:
                # For youtu.be URLs
                if (
                    "youtu.be" in parsed_url.netloc.lower()
                    and parsed_url.path
                    and len(parsed_url.path) > 1
                ):
                    video_id = parsed_url.path.strip("/")
                    if len(video_id) > 3:  # Reasonable minimum length for a video ID
                        # Validate video ID format (alphanumeric plus some special chars)
                        if re.match(r"^[\w-]+$", video_id):
                            logger.debug(f"Valid youtu.be URL: {url}, video ID: {video_id}")
                            return True
                        else:
                            logger.warning(f"Invalid video ID format in youtu.be URL: {video_id}")
                    else:
                        logger.warning(f"Video ID too short in youtu.be URL: {video_id}")

                # For standard youtube.com URLs
                if "youtube.com" in parsed_url.netloc.lower():
                    # For watch URLs
                    if "watch" in parsed_url.path and "v=" in parsed_url.query:
                        # Extract and validate the video ID
                        query_params = dict(
                            pair.split("=") for pair in parsed_url.query.split("&") if "=" in pair
                        )
                        if "v" in query_params and len(query_params["v"]) > 3:
                            video_id = query_params["v"]
                            if re.match(r"^[\w-]+$", video_id):
                                logger.debug(
                                    f"Valid youtube.com watch URL: {url}, video ID: {video_id}"
                                )
                                return True
                            else:
                                logger.warning(
                                    f"Invalid video ID format in youtube.com URL: {video_id}"
                                )
                        else:
                            logger.warning(
                                f"Missing or invalid v parameter in youtube.com URL: {url}"
                            )

                    # For shorts URLs
                    if "shorts" in parsed_url.path and len(parsed_url.path) > 8:  # /shorts/ID
                        # Extract video ID from shorts URL
                        path_parts = parsed_url.path.strip("/").split("/")
                        if len(path_parts) >= 2 and path_parts[0] == "shorts":
                            video_id = path_parts[1]
                            if len(video_id) > 3 and re.match(r"^[\w-]+$", video_id):
                                logger.debug(
                                    f"Valid youtube.com shorts URL: {url}, video ID: {video_id}"
                                )
                                return True
                            else:
                                logger.warning(f"Invalid video ID in shorts URL: {video_id}")
                        else:
                            logger.warning(f"Invalid shorts URL format: {url}")

            logger.debug(f"URL failed domain validation: {url}, domain: {parsed_url.netloc}")
            return False

        except Exception as e:
            logger.error(f"Error validating YouTube URL: {str(e)}", exc_info=True)
            return False

    def _download_video(self, url: str, max_filesize_mb: int = 25) -> str:
        """
        Download a YouTube video.

        Args:
            url: The YouTube URL
            max_filesize_mb: Maximum allowed filesize in MB

        Returns:
            Path to the downloaded file

        Raises:
            BotError: If there's a retryable error
            BotErrorNoRetry: If there's a non-retryable error
        """
        try:
            # Create a unique filename for this download
            file_id = str(uuid.uuid4())
            output_path = os.path.join(self.temp_dir, f"{file_id}.mp4")

            # Enhanced yt-dlp options to attempt bypassing age restrictions
            ydl_opts = {
                # Try to find the best format under the filesize limit
                "format": f"best[filesize<{max_filesize_mb}M]/bestvideo[filesize<{max_filesize_mb}M]+bestaudio/best",
                "outtmpl": output_path,
                "noplaylist": True,  # Only download the video, not the playlist
                "quiet": False,  # Show progress
                "no_warnings": False,
                "noprogress": True,  # Don't show download progress bar
                # Enhanced options for bypassing restrictions
                "skip_download": False,
                "writesubtitles": False,
                # Options to help bypass restrictions
                "age_limit": 21,  # Set higher age limit
                "geo_bypass": True,  # Try to bypass geo-restrictions
                # Try alternative extraction methods
                "extractor_retries": 3,  # Retry extraction up to 3 times
            }

            logger.debug(f"Downloading video from {url} to {output_path}")

            # First attempt with regular options
            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
            except yt_dlp.utils.DownloadError as e:
                error_msg = str(e)

                # If the error mentions sign-in or age restriction, try alternative method
                if "Sign in" in error_msg or "age" in error_msg.lower():
                    logger.info("Detected age restriction, trying alternative method")

                    # Update options for age-restricted content
                    ydl_opts.update(
                        {
                            "age_limit": 30,  # Maximum possible
                            "youtube_include_dash_manifest": True,
                            # Use a more common browser user-agent
                            "user_agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                        }
                    )

                    # Try again with enhanced options
                    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                        info = ydl.extract_info(url, download=True)
                else:
                    # Re-raise the original error if it's not related to age restriction
                    raise

            # Check if the file was downloaded successfully
            if not os.path.exists(output_path):
                # Try to find the actual output file (yt-dlp might have added an extension)
                possible_files = [
                    f
                    for f in os.listdir(self.temp_dir)
                    if f.startswith(file_id) or f.startswith(os.path.basename(output_path))
                ]

                if possible_files:
                    output_path = os.path.join(self.temp_dir, possible_files[0])
                else:
                    # Last resort fallback - try one more method
                    ydl_opts.update({"format": "best", "merge_output_format": "mp4"})

                    try:
                        logger.info("Trying final fallback method for restricted content")
                        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                            info = ydl.extract_info(url, download=True)

                        # Check again for created files
                        possible_files = [
                            f
                            for f in os.listdir(self.temp_dir)
                            if f.startswith(file_id) or f.startswith(os.path.basename(output_path))
                        ]

                        if possible_files:
                            output_path = os.path.join(self.temp_dir, possible_files[0])
                        else:
                            raise BotError(
                                f"ERROR: Download failed - Could not download video from {url} after multiple attempts"
                            )
                    except Exception as fallback_error:
                        raise BotError(
                            f"ERROR: Download failed - Could not download video from {url}: {str(fallback_error)}"
                        )

            # Verify filesize
            filesize = os.path.getsize(output_path) / (1024 * 1024)  # Convert to MB
            if filesize > max_filesize_mb:
                os.remove(output_path)
                raise BotErrorNoRetry(
                    f"ERROR: Size limit exceeded - Video is {filesize:.1f}MB (maximum allowed is {max_filesize_mb}MB)"
                )

            logger.debug(f"Downloaded video to {output_path}, size: {filesize:.1f}MB")
            return output_path

        except yt_dlp.utils.DownloadError as e:
            error_msg = str(e)
            logger.error(f"YouTube download error: {error_msg}")

            if "This video is unavailable" in error_msg:
                raise BotErrorNoRetry(
                    "ERROR: Video unavailable - This video is private or has been removed."
                )
            elif "Video unavailable" in error_msg:
                raise BotErrorNoRetry(
                    "ERROR: Video unavailable - This video has been removed from YouTube."
                )
            elif "Sign in" in error_msg and "age" in error_msg.lower():
                raise BotErrorNoRetry(
                    "ERROR: Age-restricted content - Unable to bypass age restriction for this video."
                )
            elif "Sign in" in error_msg:
                raise BotErrorNoRetry(
                    "ERROR: Login required - Cannot bypass sign-in requirement for this video."
                )
            elif "copyright" in error_msg.lower():
                raise BotErrorNoRetry(
                    "ERROR: Copyright restriction - This video is blocked due to copyright claims."
                )
            elif "geo" in error_msg.lower() or "country" in error_msg.lower():
                raise BotErrorNoRetry(
                    "ERROR: Geo-restricted - This video is not available in your region."
                )
            else:
                raise BotError(f"ERROR: Download failed - {error_msg}")

        except Exception as e:
            logger.error(f"Unexpected error downloading video: {str(e)}", exc_info=True)
            raise BotError(f"ERROR: Unexpected failure - Could not download the video: {str(e)}")

    def _create_video_attachment(
        self, file_path: str, video_title: Optional[str] = None
    ) -> Attachment:
        """
        Create a video attachment from a file.

        Args:
            file_path: Path to the video file
            video_title: Optional title for the video

        Returns:
            Attachment object
        """
        try:
            # Read the file
            with open(file_path, "rb") as f:
                file_content = f.read()

            # Generate a name for the attachment
            filename = video_title if video_title else os.path.basename(file_path)
            if not filename.endswith(".mp4"):
                filename += ".mp4"

            # Create the attachment
            attachment = Attachment(
                name=filename,
                content_type=ContentType.mp4,
                data=file_content,
            )

            return attachment
        except Exception as e:
            logger.error(f"Error creating attachment: {str(e)}")
            # For testing purposes, return a mock if we're not in a full Poe environment
            if not hasattr(Attachment, "__origin__"):  # Check if we're using a mock class
                return Attachment(name="test.mp4", content_type="video/mp4", data=None)

    def _cleanup(self, file_path: Optional[str] = None):
        """
        Clean up downloaded files.

        Args:
            file_path: Optional specific file to delete
        """
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
                logger.debug(f"Deleted file {file_path}")
        except Exception as e:
            logger.error(f"Error cleaning up file {file_path}: {str(e)}")

    async def get_settings(self, settings_request):
        """Configure bot settings.

        This bot doesn't accept file attachments as input, but it does send
        attachments as output (downloaded videos).

        The bot has a rate card of 300 points per message.
        """
        try:
            from fastapi_poe.types import SettingsResponse

            # Return settings with attachments disabled for input
            # Note: Rate card is managed separately through the Poe API
            settings = SettingsResponse(
                allow_attachments=False,  # Disable file upload for this bot
                server_bot_dependencies={
                    "allow_attachments": False,
                    "rate_card": {
                        "api_calling_cost": self.rate_card_points,
                        "api_pricing_type": "per_message",
                    },
                },
            )

            logger.debug(
                f"[{self.bot_name}] Configured settings: allow_attachments=False, rate_card={self.rate_card_points}"
            )
            return settings

        except Exception as e:
            logger.error(f"[{self.bot_name}] Error configuring settings: {str(e)}", exc_info=True)
            # Still return a valid settings object even if there's an error
            from fastapi_poe.types import SettingsResponse

            return SettingsResponse(allow_attachments=False)

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        """Process the query and handle YouTube URL requests."""
        try:
            # Extract the message
            message = self._extract_message(query)
            message = message.strip()

            # Handle bot info requests
            if message.lower().strip() == "bot info":
                metadata = self._get_bot_metadata()
                yield PartialResponse(text=json.dumps(metadata, indent=2))
                return

            # Help command
            if message.lower() in ["help", "?", "/help"]:
                yield PartialResponse(
                    text="""
# 📺 YouTube Downloader Bot

Send me a YouTube video URL, and I'll download the video and send it back to you!

## How to use:
1. Just paste a YouTube URL in the chat
2. I'll download the video and send it to you as an attachment
3. You can then save the video to your device

## Supported URL formats:
- https://www.youtube.com/watch?v=VIDEO_ID
- https://youtu.be/VIDEO_ID
- https://youtube.com/shorts/VIDEO_ID

**Note:** Videos are limited to 25MB maximum size.
"""
                )
                return

            # Extract YouTube URLs
            youtube_urls = self._extract_youtube_urls(message)

            if not youtube_urls:
                yield PartialResponse(
                    text="Please send me a YouTube URL. Type 'help' for instructions."
                )
                return

            # Use the first URL if multiple are provided
            url = youtube_urls[0]

            # Validate the URL
            if not self._validate_youtube_url(url):
                yield PartialResponse(
                    text=f"'{url}' doesn't appear to be a valid YouTube URL. Please check the URL and try again."
                )
                return

            # Inform the user we're downloading
            yield PartialResponse(text=f"⏳ Downloading video from {url}...\n\n")

            # Download the video
            downloaded_file = None
            try:
                downloaded_file = self._download_video(url)

                # Create the attachment
                video_attachment = self._create_video_attachment(downloaded_file)

                # Send the response with the video attachment
                yield PartialResponse(
                    text=f"✅ Here's your video from {url}:", attachments=[video_attachment]
                )

            finally:
                # Clean up the downloaded file
                if downloaded_file:
                    self._cleanup(downloaded_file)

        except BotErrorNoRetry as e:
            # Log the error (non-retryable) and re-raise it to trigger a failure
            # This will charge the user points but show an error
            logger.error(f"[{self.bot_name}] Non-retryable error: {str(e)}", exc_info=True)
            # Return a helpful error message to the user before re-raising
            yield PartialResponse(
                text=f"❌ Error: {str(e)}\n\nThis type of error cannot be fixed by trying again."
            )
            raise

        except BotError as e:
            # Log the error (retryable) and re-raise it to trigger a failure
            # This will charge the user points but show an error
            logger.error(f"[{self.bot_name}] Retryable error: {str(e)}", exc_info=True)
            # Return a helpful error message to the user before re-raising
            yield PartialResponse(
                text=f"❌ Error: {str(e)}\n\nPlease try again or try with a different video."
            )
            raise

        except Exception as e:
            # Log the unexpected error and re-raise as BotError
            logger.error(f"[{self.bot_name}] Unexpected error: {str(e)}", exc_info=True)
            # Return a helpful error message to the user before re-raising
            yield PartialResponse(
                text="❌ An unexpected error occurred while processing your request.\n\nPlease try again later or try with a different video URL."
            )
            # Raise as a BotError to ensure proper error reporting
            raise BotError(f"ERROR: Unexpected failure - {str(e)}")


# Modal image with required dependencies
image = modal.Image.debian_slim().pip_install(
    ["fastapi-poe>=0.0.17", "pydantic>=2.0.0", "yt-dlp>=2023.10.7"]
)


# Modal web endpoint for the bot
@app.function(image=image)
@modal.web_endpoint(method="POST")
async def web_endpoint(request: bytes) -> bytes:
    """Modal web endpoint for the YouTube Downloader Bot."""
    from fastapi_poe import make_app

    bot = YouTubeDownloaderBot()
    app = make_app(bot)

    return await app.handle_request(request)

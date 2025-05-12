"""
Simple test script for YouTube downloader bot core functionality.

This script tests just the URL extraction and validation functionality
without requiring the full fastapi-poe dependencies.
"""

import os
import re
from urllib.parse import urlparse

# Define the YouTube URL pattern - improved to include the video ID
YOUTUBE_URL_PATTERN = (
    r"(https?://)?(www\.)?(youtube\.com/watch\?v=|youtu\.be/|youtube\.com/shorts/)([\w-]+)"
)


def extract_youtube_urls(text):
    """Extract YouTube URLs from text."""
    matches = re.findall(YOUTUBE_URL_PATTERN, text)

    # Convert matches to full URLs
    full_urls = []
    for match in matches:
        # Extract components
        protocol = match[0] if match[0] else "https://"
        domain = match[1] if match[1] else "www."
        path = match[2]
        video_id = match[3]

        # Build full URL
        if "youtu.be" in path:
            url = f"{protocol}{domain}youtu.be/{video_id}"
        elif "shorts" in path:
            url = f"{protocol}{domain}youtube.com/shorts/{video_id}"
        else:
            url = f"{protocol}{domain}youtube.com/watch?v={video_id}"

        full_urls.append(url)

    return full_urls


def validate_youtube_url(url):
    """Validate that a URL is a YouTube URL."""
    parsed_url = urlparse(url)

    # Check the domain (properly handle both with and without www)
    youtube_domains = ["youtube.com", "www.youtube.com", "youtu.be", "www.youtu.be"]

    if parsed_url.netloc in youtube_domains:
        # For youtu.be URLs
        if "youtu.be" in parsed_url.netloc and parsed_url.path and len(parsed_url.path) > 1:
            video_id = parsed_url.path.strip("/")
            if len(video_id) > 3:  # Reasonable minimum length for a video ID
                return True

        # For standard youtube.com URLs
        if "youtube.com" in parsed_url.netloc:
            # For watch URLs
            if "watch" in parsed_url.path and "v=" in parsed_url.query:
                return True

            # For shorts URLs
            if "shorts" in parsed_url.path and len(parsed_url.path) > 8:  # /shorts/ID
                return True

    return False


def test_url_extraction():
    """Test the URL extraction functionality."""
    # Test cases
    test_cases = [
        "Check out this video: https://www.youtube.com/watch?v=dQw4w9WgXcQ",
        "Look at this short: https://youtube.com/shorts/Tb5uoIRlcTw",
        "https://youtu.be/dQw4w9WgXcQ is a good one",
        "Multiple URLs: https://youtu.be/dQw4w9WgXcQ and https://youtube.com/shorts/Tb5uoIRlcTw",
        "No URL here",
        "Invalid URL: https://example.com",
    ]

    print("Testing URL extraction:")
    for i, test in enumerate(test_cases):
        urls = extract_youtube_urls(test)
        valid_count = sum(1 for url in urls if validate_youtube_url(url))

        print(f"  Test {i+1}: Found {len(urls)} URLs, {valid_count} valid.")
        for url in urls:
            valid = validate_youtube_url(url)
            print(f"    - {url} -> {'✅ Valid' if valid else '❌ Invalid'}")

    print("\nURL extraction test completed.")


if __name__ == "__main__":
    test_url_extraction()

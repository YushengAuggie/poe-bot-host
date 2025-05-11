# Testing Gemini Bot Image Handling

This document provides instructions for testing the Gemini bot's image handling capabilities after applying the fix for the fastapi-poe and Pydantic model attribute access issue.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Testing Options](#testing-options)
  - [1. Manual Testing with Local Server](#1-manual-testing-with-local-server)
  - [2. Automated API Testing (Recommended)](#2-automated-api-testing-recommended)
  - [3. Direct Testing of the Fix](#3-direct-testing-of-the-fix)
- [What to Look For](#what-to-look-for)
- [Troubleshooting](#troubleshooting)
- [Additional Notes](#additional-notes)

## Prerequisites

1. Make sure you have the Google API key set as an environment variable:
   ```bash
   export GOOGLE_API_KEY=your_api_key_here
   ```

2. Required Python packages:
   ```bash
   pip install pillow requests
   ```

## Testing Options

There are multiple ways to test the Gemini bot's image handling:

### 1. Manual Testing with Local Server

The simplest approach is to start the local server and use a web browser to test:

```bash
# Start the local server
python run_local.py
```

Then open http://localhost:8000 in your web browser, and:
1. Select the "Gemini20FlashBot" bot
2. Type a message like "Hello"
3. Send an image with a prompt like "What do you see in this image?"

### 2. Automated API Testing (Recommended)

For more systematic testing, use the provided test script that can test both API endpoints:

```bash
# Auto-generate a test image and test the chat API endpoint
python test_gemini_image_api.py --auto --endpoint api/chat --bot Gemini20FlashBot

# Test with your own image
python test_gemini_image_api.py path/to/your/image.jpg --endpoint api/chat --bot Gemini20FlashBot
```

### 3. Direct Testing of the Fix

To directly test the fix implementation without the API layer, use:

```bash
python direct_gemini_fix_test.py
```

This will test the specific fix for content accessibility in the attachment processing code.

## What to Look For

Successful image processing will show:

1. No errors related to content attributes or `__dict__` access
2. A response that clearly describes the image content
3. The bot correctly identifying visual elements in the image

## Troubleshooting

If you encounter issues:

1. **Check API Key**: Make sure your `GOOGLE_API_KEY` environment variable is correctly set:
   ```bash
   echo $GOOGLE_API_KEY
   ```

2. **Check Logs**: Look at the server logs for any error messages:
   ```bash
   # Add DEBUG=1 to see more detailed logs
   DEBUG=1 python run_local.py
   ```

3. **Verify Fix Implementation**: Ensure the fix was applied in both:
   - `_extract_attachments` method (around line 200)
   - `_process_media_attachment` method (around line 272)

4. **Test with Simple Image**: Try with a small, simple image first

## Additional Notes

- The Gemini20FlashBot uses "gemini-1.5-flash-latest" as its multimodal model for image processing
- Image handling is fixed to work with both direct attribute access and `__dict__` access patterns
- The fix should work for all supported attachment types (images, videos, audio)

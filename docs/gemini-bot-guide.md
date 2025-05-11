# Gemini Bot Guide

This guide provides an overview of the Gemini bot implementations, capabilities, and usage within the Poe Bot framework.

## Table of Contents
- [Overview](#overview)
- [Available Models](#available-models)
- [Key Features](#key-features)
- [Image Processing Support](#image-processing-support)
  - [Recent Image Handling Fix](#recent-image-handling-fix)
- [Usage](#usage)
  - [Setup Requirements](#setup-requirements)
  - [Basic Usage](#basic-usage)
  - [Advanced Features](#advanced-features)
- [Deployment](#deployment)
- [Troubleshooting](#troubleshooting)
- [Resources](#resources)

## Overview

The Gemini bots integrate Google's Gemini models into the Poe platform, providing advanced AI capabilities including multimodal content understanding (text, images, videos, and audio).

## Available Models

The framework includes several Gemini bot implementations:

| Bot Class | Model | Description |
|-----------|-------|-------------|
| `GeminiBot` | gemini-2.0-flash | Original Gemini implementation |
| `Gemini20FlashBot` | gemini-2.0-flash | Optimized for speed and efficiency |
| `Gemini20ProBot` | gemini-2.0-pro | Balanced performance model |
| `Gemini25FlashBot` | gemini-2.5-flash-preview-04-17 | Advanced adaptive thinking model |
| `Gemini25ProExpBot` | gemini-2.5-pro-preview-05-06 | Premium model for complex reasoning |
| `Gemini20FlashExpBot` | gemini-2.0-flash-exp | Experimental with image generation |
| `Gemini20FlashThinkingBot` | gemini-2.0-flash-thinking-exp-01-21 | Enhanced reasoning model |
| `Gemini20ProExpBot` | gemini-2.0-pro-exp-02-05 | Experimental pro model |

## Key Features

- **Multimodal Understanding**: Process and analyze text, images, videos, and audio
- **Image Generation**: Some models support image generation capabilities
- **Grounding Support**: Advanced models include grounding capabilities
- **Streaming Responses**: Responses are streamed for better user experience
- **Customizable Behavior**: Configurable response settings and behavior

## Image Processing Support

All Gemini bots support processing images with the following capabilities:

1. **Image Analysis**: Can describe and analyze the content of images
2. **Format Support**: JPEG, PNG, WebP, GIF
3. **Image Generation**: Some models can generate images based on text prompts
4. **Multimodal Model**: For image processing, models automatically use `multimodal_model_name` ("gemini-1.5-flash-latest" by default)

### Recent Image Handling Fix

The framework includes a fix for image handling in the Gemini bot that addresses content accessibility issues when attachments come through the Poe API:

- Fixed in both `_extract_attachments` and `_process_media_attachment` methods
- Ensures content is correctly accessible regardless of how the attachment is transmitted
- See [GEMINI_IMAGE_FIX.md](/GEMINI_IMAGE_FIX.md) for technical details
- For testing procedures, refer to [gemini_image_testing.md](/gemini_image_testing.md)

## Usage

### Setup Requirements

Before using the Gemini bots, you need:

1. A Google API key set as an environment variable:
   ```bash
   export GOOGLE_API_KEY=your_key_here
   ```

2. The Google Generative AI package:
   ```bash
   pip install google-generativeai
   ```

### Basic Usage

The simplest way to test the Gemini bot:

```bash
# Start the local server
./run_local.sh

# Test with the test_bot.py tool
python test_bot.py --bot Gemini20FlashBot --message "What can you do?"

# Test image handling
python test_gemini_image_api.py --auto --endpoint api/chat --bot Gemini20FlashBot
```

### Advanced Features

#### Using Grounding

For models that support grounding (typically Pro and 2.5 series models):

```python
# Example of using grounding sources
bot = Gemini25ProBot()
bot.add_grounding_source({
    "title": "Important Document",
    "url": "https://example.com/doc",
    "content": "This is factual content the bot should reference."
})
```

#### Image Generation

For models with image generation capabilities:

```bash
# Test image generation (Gemini20FlashExpBot supports this)
python test_bot.py --bot Gemini20FlashExpBot --message "Generate an image of a sunset over mountains"
```

## Deployment

When deploying Gemini bots to Modal, ensure:

1. Your API key is set as a secret:
   ```bash
   modal secret create GOOGLE_API_KEY "your_api_key_here"
   ```

2. You set your bot's access key as a Modal secret:
   ```bash
   modal secret create GEMINI20FLASHBOT_ACCESS_KEY "psk_your_bot_access_key"
   ```

## Troubleshooting

If you encounter issues with the Gemini bot:

1. **API Key Issues**: Verify your GOOGLE_API_KEY is correctly set
2. **Image Handling**: Use test scripts in `gemini_image_testing.md` to diagnose
3. **Attachment Issues**: Run with `DEBUG=1 ./run_local.sh` and check logs
4. **Model Availability**: Some models may be deprecated or renamed by Google

## Resources

- [Google Generative AI Documentation](https://ai.google.dev/docs)
- [Gemini API Overview](https://ai.google.dev/api/rest/v1beta/models/gemini)
- [Local Testing Guide](/gemini_image_testing.md)
- [Image Handling Fix](/GEMINI_IMAGE_FIX.md)

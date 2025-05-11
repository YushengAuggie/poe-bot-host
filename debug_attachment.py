#!/usr/bin/env python
"""
Debug script for Gemini bot attachment handling.

This script isolates the attachment handling process in the Gemini bot
to identify why images aren't being correctly processed through the API.
"""

import base64
import json
import logging
import os
import sys
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("debug_attachment")

# Add the project root to path so modules can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules
from bots.gemini import Gemini20FlashBot
from fastapi_poe.types import ProtocolMessage, QueryRequest, Attachment

def create_mock_attachment(image_path: str) -> Attachment:
    """Create a mock attachment from an image file.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        A mock attachment
    """
    # Check if image file exists
    if not os.path.exists(image_path):
        logger.error(f"Image file not found: {image_path}")
        sys.exit(1)
        
    # Get file type from extension
    import mimetypes
    content_type, _ = mimetypes.guess_type(image_path)
    if not content_type:
        content_type = "application/octet-stream"
    
    # Read file contents
    with open(image_path, "rb") as f:
        file_content = f.read()
    
    # Get filename
    filename = os.path.basename(image_path)
    
    # Create mock attachment with various attributes that might be missing
    mock_attachment = Attachment(
        url=f"file://{filename}",
        name=filename,
        content_type=content_type
    )
    
    # Add content attribute (not in the type definition but added by Poe)
    mock_attachment.__dict__["content"] = file_content
    
    return mock_attachment

def create_mock_query(message: str, attachment: Attachment) -> QueryRequest:
    """Create a mock query with an attachment.
    
    Args:
        message: The message text
        attachment: The attachment
        
    Returns:
        A mock query request
    """
    protocol_message = ProtocolMessage(
        role="user",
        content=message,
        attachments=[attachment]
    )
    
    return QueryRequest(
        version="1.0",
        type="query",
        query=[protocol_message],
        user_id="test_user",
        conversation_id="test_convo_123",
        message_id="test_msg_123",
    )

async def test_bot_with_attachment(image_path: str, message: str):
    """Test the Gemini bot with an attachment.
    
    Args:
        image_path: Path to the image file
        message: The message text
    """
    logger.info(f"Testing with image: {image_path}")
    logger.info(f"Message: {message}")
    
    # Create a mock attachment
    attachment = create_mock_attachment(image_path)
    logger.debug(f"Created mock attachment: {attachment}")
    logger.debug(f"Attachment type: {type(attachment)}")
    logger.debug(f"Attachment has content: {hasattr(attachment, 'content')}")
    logger.debug(f"Attachment content in __dict__: {'content' in attachment.__dict__}")
    if 'content' in attachment.__dict__:
        logger.debug(f"Attachment content type: {type(attachment.__dict__['content'])}")
        logger.debug(f"Attachment content size: {len(attachment.__dict__['content'])} bytes")
    
    # Create a mock query
    query = create_mock_query(message, attachment)
    logger.debug(f"Created mock query: {query}")
    
    # Create a Gemini bot
    bot = Gemini20FlashBot()
    logger.info(f"Created bot: {bot.bot_name}")
    logger.info(f"Model name: {bot.model_name}")
    logger.info(f"Multimodal model name: {bot.multimodal_model_name}")
    
    # Extract attachments using the bot's method
    attachments = bot._extract_attachments(query)
    logger.info(f"Extracted {len(attachments)} attachments")
    
    if attachments:
        # Process the attachment
        media_data = bot._process_media_attachment(attachments[0])
        if media_data:
            logger.info(f"Processed media data: {media_data['mime_type']}, {len(media_data['data'])} bytes")
            
            # Prepare media parts
            media_parts = bot._prepare_media_parts(attachments)
            logger.info(f"Prepared {len(media_parts)} media parts")
            
            # Print media parts for inspection
            logger.debug("Media parts keys:")
            for i, part in enumerate(media_parts):
                logger.debug(f"  Part {i+1} type: {type(part)}")
                if isinstance(part, dict):
                    logger.debug(f"  Part {i+1} keys: {part.keys()}")
                    if "inline_data" in part:
                        logger.debug(f"  inline_data keys: {part['inline_data'].keys()}")
            
            # Prepare content
            content = bot._prepare_content(message, media_parts)
            logger.info(f"Prepared content with {len(content)} parts")
            
            # Print content for inspection
            logger.debug("Content structure:")
            for i, item in enumerate(content):
                logger.debug(f"  Item {i+1} type: {type(item)}")
                if isinstance(item, dict):
                    logger.debug(f"  Item {i+1} keys: {item.keys()}")
                
            # Use multimodal model
            from bots.gemini import get_client
            client = get_client(bot.multimodal_model_name)
            if client:
                logger.info(f"Got client for model: {bot.multimodal_model_name}")
                
                try:
                    # Test sending the request directly
                    logger.info("Sending direct request to Gemini API...")
                    response = client.generate_content(content)
                    if hasattr(response, "text"):
                        logger.info(f"API response: {response.text}")
                    elif hasattr(response, "parts"):
                        for part in response.parts:
                            if hasattr(part, "text"):
                                logger.info(f"API response part: {part.text}")
                    else:
                        logger.warning("Response doesn't have text or parts attributes")
                except Exception as e:
                    logger.error(f"Error calling Gemini API: {str(e)}")
            else:
                logger.error(f"Failed to get client for model {bot.multimodal_model_name}")
        else:
            logger.error("Failed to process attachment")
    else:
        logger.error("No attachments extracted")

async def main():
    """Main entry point."""
    # Get image path from command line
    import argparse
    parser = argparse.ArgumentParser(description="Debug Gemini bot attachment handling")
    parser.add_argument("image_path", nargs="?", default="assets/test_images/duck2.jpg", 
                       help="Path to image file (default: assets/test_images/duck2.jpg)")
    parser.add_argument("--message", default="Describe what you see in this image in detail.", 
                       help="Text prompt to send with the image")
    args = parser.parse_args()
    
    # Test with image
    await test_bot_with_attachment(args.image_path, args.message)

if __name__ == "__main__":
    # Run the async function
    import asyncio
    asyncio.run(main())
#!/usr/bin/env python
"""
Debug and fix script for Gemini bot attachment handling via API.

This script reproduces the fastapi-poe request/response flow to identify
why image attachments aren't being properly processed through the API and
proposes a solution.
"""

import asyncio
import base64
import json
import logging
import os
import sys
from typing import AsyncGenerator, Dict, Any, List, Union

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("fix_api_attachment")

# Add the project root to path so modules can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import necessary modules
from bots.gemini import Gemini20FlashBot
from fastapi_poe.types import ProtocolMessage, QueryRequest, Attachment, PartialResponse, MetaResponse, ContentType

class MockQueryWithAttachment(QueryRequest):
    """A mock query with an attachment for testing."""
    
    @classmethod
    def create_with_attachment(cls, message: str, image_path: str) -> "MockQueryWithAttachment":
        """Create a mock query with an attachment.
        
        Args:
            message: The message text
            image_path: Path to the image file
            
        Returns:
            A mock query request with an attachment
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
        
        # Create a real attachment with content
        attachment = Attachment(
            url=f"file://{filename}",
            name=filename,
            content_type=content_type
        )
        
        # Add content attribute to __dict__ (this is what Poe does)
        attachment.__dict__["content"] = file_content
        
        # Create the protocol message with attachment
        protocol_message = ProtocolMessage(
            role="user",
            content=message,
            attachments=[attachment]
        )
        
        # Create the mock query
        return cls(
            version="1.0",
            type="query",
            query=[protocol_message],
            user_id="test_user",
            conversation_id="test_convo_123",
            message_id="test_msg_123",
        )

async def trace_bot_call(bot: Gemini20FlashBot, query: QueryRequest) -> List[str]:
    """Trace through the full bot call with detailed logging.
    
    Args:
        bot: The Gemini bot instance
        query: The query with attachment
        
    Returns:
        List of response text chunks
    """
    logger.info("=== Starting API call trace ===")
    
    # Log query details
    logger.info(f"Query type: {type(query)}")
    logger.info(f"Query has attachments: {hasattr(query.query[-1], 'attachments')}")
    if hasattr(query.query[-1], "attachments"):
        attachments = query.query[-1].attachments
        logger.info(f"Number of attachments: {len(attachments)}")
        for i, attachment in enumerate(attachments):
            logger.info(f"Attachment {i+1} type: {type(attachment)}")
            logger.info(f"Attachment {i+1} content_type: {attachment.content_type}")
            logger.info(f"Attachment {i+1} has content: {hasattr(attachment, 'content')}")
            logger.info(f"Attachment {i+1} __dict__: {list(attachment.__dict__.keys())}")
            if "content" in attachment.__dict__:
                logger.info(f"Attachment {i+1} content size: {len(attachment.__dict__['content'])} bytes")
    
    # Process the get_response call and capture all responses
    responses = []
    try:
        async for response in bot.get_response(query):
            logger.info(f"Response chunk type: {type(response)}")
            if isinstance(response, PartialResponse):
                logger.info(f"Response chunk text: {response.text[:50]}...")
                responses.append(response.text)
            elif isinstance(response, MetaResponse):
                logger.info(f"Meta response: {response}")
    except Exception as e:
        logger.error(f"Error in get_response: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    logger.info(f"Total response chunks: {len(responses)}")
    logger.info("=== Completed API call trace ===")
    
    return responses

def patch_attachment(attachment):
    """Add the content attribute directly to the attachment.
    
    This function monkeypatches the attachment object to ensure
    the content attribute is accessible via direct attribute access,
    not just via __dict__.
    
    Args:
        attachment: The attachment to patch
    """
    if not hasattr(attachment, "content") and "content" in attachment.__dict__:
        content = attachment.__dict__["content"]
        # Add setter method to pydantic model
        orig_setattr = attachment.__class__.__setattr__
        
        def allow_content_setattr(self, name, value):
            if name == "content":
                self.__dict__[name] = value
            else:
                orig_setattr(self, name, value)
        
        # Replace the setattr method to allow setting content
        attachment.__class__.__setattr__ = allow_content_setattr
        
        # Set the content attribute
        setattr(attachment, "content", content)
        
        return True
    return False

def apply_monkey_patch(bot_class):
    """Apply monkey patch to the bot class.
    
    Patches the bot's _extract_attachments method to ensure
    content is accessible directly from the attachment.
    
    Args:
        bot_class: The bot class to patch
    """
    original_extract_attachments = bot_class._extract_attachments
    
    def patched_extract_attachments(self, query):
        attachments = original_extract_attachments(self, query)
        
        # Apply patch to make content directly accessible
        for attachment in attachments:
            patched = patch_attachment(attachment)
            if patched:
                logger.info(f"Patched attachment: {attachment.name}")
                logger.info(f"Attachment now has content attribute: {hasattr(attachment, 'content')}")
            
        return attachments
    
    # Replace the method
    bot_class._extract_attachments = patched_extract_attachments
    logger.info(f"Applied monkey patch to {bot_class.__name__}._extract_attachments")

async def test_with_patch(image_path: str, message: str):
    """Test the bot with the patch applied.
    
    Args:
        image_path: Path to the image file
        message: The message text
    """
    logger.info(f"Testing with image: {image_path}")
    logger.info(f"Message: {message}")
    
    # Create the mock query
    query = MockQueryWithAttachment.create_with_attachment(message, image_path)
    logger.info(f"Created mock query with attachment")
    
    # Create a Gemini bot
    unpatched_bot = Gemini20FlashBot()
    logger.info(f"Created unpatched bot: {unpatched_bot.bot_name}")
    
    # Test without patch
    logger.info("=== Testing without patch ===")
    unpatched_responses = await trace_bot_call(unpatched_bot, query)
    
    # Create a Gemini bot with patch
    logger.info("=== Applying patch ===")
    apply_monkey_patch(Gemini20FlashBot)
    patched_bot = Gemini20FlashBot()
    logger.info(f"Created patched bot: {patched_bot.bot_name}")
    
    # Test with patch
    logger.info("=== Testing with patch ===")
    patched_responses = await trace_bot_call(patched_bot, query)
    
    # Compare results
    logger.info("=== Results comparison ===")
    logger.info(f"Unpatched responses: {len(unpatched_responses)}")
    logger.info(f"First unpatched response: {unpatched_responses[0] if unpatched_responses else 'None'}")
    logger.info(f"Patched responses: {len(patched_responses)}")
    logger.info(f"First patched response: {patched_responses[0][:100] if patched_responses else 'None'}...")
    
    # Print recommendation
    print("\n=== Recommendation ===")
    if len(patched_responses) > 0 and "see" not in patched_responses[0].lower():
        print("\nThe patch successfully fixed the image attachment handling!")
        print("\nTo fix the issue, modify the Gemini bot's _extract_attachments method in bots/gemini.py:")
        print("\nAdd this code at the end of _extract_attachments method:")
        print("""
        # Ensure content attribute is directly accessible for each attachment
        for attachment in attachments:
            if not hasattr(attachment, "content") and "content" in attachment.__dict__:
                # Add content attribute directly to the attachment object
                content = attachment.__dict__["content"]
                setattr(attachment, "content", content)
        """)
    else:
        print("\nThe patch did not fully resolve the issue, further investigation is needed.")
    
    # Print full successful response
    if len(patched_responses) > 0 and "see" not in patched_responses[0].lower():
        print("\n=== Successful Response ===")
        print("\n".join(patched_responses))

async def main():
    """Main entry point."""
    # Get image path from command line
    import argparse
    parser = argparse.ArgumentParser(description="Debug and fix Gemini bot attachment handling")
    parser.add_argument("image_path", nargs="?", default="assets/test_images/duck2.jpg", 
                       help="Path to image file (default: assets/test_images/duck2.jpg)")
    parser.add_argument("--message", default="Describe what you see in this image in detail.", 
                       help="Text prompt to send with the image")
    args = parser.parse_args()
    
    # Test with image
    await test_with_patch(args.image_path, args.message)

if __name__ == "__main__":
    # Run the async function
    asyncio.run(main())
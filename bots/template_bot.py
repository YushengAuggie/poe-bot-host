"""
Template for creating a new Poe bot.

Copy this file and modify it to create your own bot.
"""

from typing import AsyncGenerator
from fastapi_poe.types import PartialResponse, QueryRequest
from utils.base_bot import BaseBot

class TemplateBot(BaseBot):
    """Template for creating a new bot.
    
    Copy this class and modify it to create your own bot.
    This is a basic template that you can use as a starting point.
    """
    
    # Override the bot name and description
    bot_name = "TemplateBot"
    bot_description = "Template for creating a new bot"
    version = "1.0.0"
    
    # Override default settings if needed
    max_message_length = 2000
    stream_response = True
    
    async def _process_message(self, message: str, query: QueryRequest) -> AsyncGenerator[PartialResponse, None]:
        """Process the message and generate a response.
        
        Args:
            message: The extracted message from the user
            query: The original query object
            
        Yields:
            Response chunks as PartialResponse objects
        """
        # TODO: Implement your bot's logic here
        # This is just a placeholder implementation
        response = f"You said: {message}\n\nThis is a template bot. Customize me!"
        
        # You can yield multiple chunks for streaming responses
        yield PartialResponse(text=response)
        
        # Or you can yield a single chunk for non-streaming responses
        # yield PartialResponse(text=response)
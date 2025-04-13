from typing import AsyncGenerator
from fastapi_poe.types import PartialResponse, QueryRequest
from utils.base_bot import BaseBot

class ReverseBot(BaseBot):
    """A bot that reverses the user's message."""
    
    # Override the bot name
    bot_name = "ReverseBot"
    
    async def _process_message(self, message: str, query: QueryRequest) -> AsyncGenerator[PartialResponse, None]:
        """Process the message and generate a response.
        
        Args:
            message: The extracted message from the user
            query: The original query object
            
        Yields:
            Response chunks as PartialResponse objects
        """
        # Reverse the user's message
        reversed_message = message[::-1]
        yield PartialResponse(text=reversed_message)
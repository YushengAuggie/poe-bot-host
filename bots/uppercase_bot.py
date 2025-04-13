from typing import AsyncGenerator
from fastapi_poe.types import PartialResponse, QueryRequest
from utils.base_bot import BaseBot

class UppercaseBot(BaseBot):
    """A bot that converts the user's message to uppercase."""
    
    # Override the bot name
    bot_name = "UppercaseBot"
    
    async def _process_message(self, message: str, query: QueryRequest) -> AsyncGenerator[PartialResponse, None]:
        """Process the message and generate a response.
        
        Args:
            message: The extracted message from the user
            query: The original query object
            
        Yields:
            Response chunks as PartialResponse objects
        """
        # Convert the user's message to uppercase
        uppercase_message = message.upper()
        yield PartialResponse(text=uppercase_message)
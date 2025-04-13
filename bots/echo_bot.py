from typing import AsyncGenerator
from fastapi_poe.types import PartialResponse, QueryRequest
from utils.base_bot import BaseBot

class EchoBot(BaseBot):
    """A simple bot that echoes back the user's message."""
    
    # Override the bot name
    bot_name = "EchoBot"
    
    async def _process_message(self, message: str, query: QueryRequest) -> AsyncGenerator[PartialResponse, None]:
        """Process the message and generate a response.
        
        Args:
            message: The extracted message from the user
            query: The original query object
            
        Yields:
            Response chunks as PartialResponse objects
        """
        # Simply echo back the user's message
        yield PartialResponse(text=message)
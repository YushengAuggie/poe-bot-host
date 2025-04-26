import json
import logging
from typing import AsyncGenerator, Union

from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest

from utils.base_bot import BaseBot

# Get the logger
logger = logging.getLogger(__name__)

class EchoBot(BaseBot):
    """A simple bot that echoes back the user's message."""

    # Override the bot name
    bot_name = "EchoBot"

    async def get_response(self, query: QueryRequest) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        """Process the query and generate a response.

        Args:
            query: The query from the user

        Yields:
            Response chunks as PartialResponse or MetaResponse objects
        """
        try:
            # Extract the query contents
            user_message = self._extract_message(query)

            # Log the extracted message
            logger.debug(f"[{self.bot_name}] Received message: {user_message}")
            logger.debug(f"[{self.bot_name}] Message type: {type(user_message)}")
            logger.debug(f"[{self.bot_name}] Query type: {type(query)}")
            logger.debug(f"[{self.bot_name}] Query contents: {query.query}")

            # Add metadata about the bot if requested
            if user_message.lower().strip() == "bot info":
                metadata = self._get_bot_metadata()
                yield PartialResponse(text=json.dumps(metadata, indent=2))
                return

            # Simply echo back the user's message - extract content string
            # Make sure we're handling string or object correctly
            yield PartialResponse(text=user_message)

        except Exception:
            # Let the parent class handle errors
            async for resp in super().get_response(query):
                yield resp

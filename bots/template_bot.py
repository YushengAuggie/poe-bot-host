"""
Template for creating a new Poe bot.

Copy this file and modify it to create your own bot.
"""

import json
import logging
from typing import AsyncGenerator, Union

from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest

from utils.base_bot import BaseBot

# Get the logger
logger = logging.getLogger(__name__)

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

            # Add metadata about the bot if requested
            if user_message.lower().strip() == "bot info":
                metadata = self._get_bot_metadata()
                yield PartialResponse(text=json.dumps(metadata, indent=2))
                return

            # TODO: Implement your bot's logic here
            # This is just a placeholder implementation
            response = f"You said: {user_message}\n\nThis is a template bot. Customize me!"

            # You can yield multiple chunks for streaming responses
            yield PartialResponse(text=response)

            # Or you can yield a single chunk for non-streaming responses
            # yield PartialResponse(text=response)

        except Exception:
            # Let the parent class handle errors
            async for resp in super().get_response(query):
                yield resp

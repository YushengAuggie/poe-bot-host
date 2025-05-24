import json
import logging
from typing import AsyncGenerator, Union

from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest

from utils.base_bot import BaseBot
from utils.mixins import ErrorHandlerMixin

# Get the logger
logger = logging.getLogger(__name__)


class EchoBot(BaseBot, ErrorHandlerMixin):
    """A simple bot that echoes back the user's message."""

    # Override the bot name
    bot_name = "EchoBot"

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        """Process the query and generate a response."""

        async def _process_echo():
            # Extract the query contents
            user_message = self._extract_message(query)
            logger.debug(f"[{self.bot_name}] Received message: {user_message}")

            # Add metadata about the bot if requested
            if user_message.lower().strip() == "bot info":
                metadata = self._get_bot_metadata()
                yield PartialResponse(text=json.dumps(metadata, indent=2))
                return

            # Simply echo back the user's message
            yield PartialResponse(text=user_message)

        # Use error handling mixin
        async for response in self.handle_common_errors(query, _process_echo):
            yield response

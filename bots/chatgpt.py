import json
import logging
import os
from typing import AsyncGenerator, Union

from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest
from openai import OpenAI

from utils.base_bot import BaseBot

# Get the logger
logger = logging.getLogger(__name__)


# Initialize client globally
def get_client():
    try:
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception as e:
        logger.warning(f"Failed to initialize OpenAI client: {str(e)}")
        return None


client = None


class ChatgptBot(BaseBot):
    """A simple chatgpt bot."""

    # Override the bot name
    bot_name = "ChatgptBot"

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
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

            # Initialize client if not already done
            global client
            if client is None:
                client = get_client()
                if client is None:
                    yield PartialResponse(text="Error: OpenAI API key is not configured.")
                    return

            try:
                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": f"{user_message}"}],
                    stream=True,
                )
                # Process streaming response
                for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield PartialResponse(text=chunk.choices[0].delta.content)
            except Exception as e:
                logger.error(f"Error calling OpenAI API: {str(e)}")
                yield PartialResponse(text=f"Error: Could not get response from OpenAI: {str(e)}")

        except Exception as e:
            # Let the parent class handle errors
            async for resp in super().get_response(query):
                yield resp

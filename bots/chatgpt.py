import json
import logging
from typing import AsyncGenerator, Union

from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest
from openai import OpenAI

from utils.api_keys import get_api_key
from utils.base_bot import BaseBot

# Get the logger
logger = logging.getLogger(__name__)


# Initialize client globally
def get_client():
    try:
        return OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
    except Exception as e:
        logger.warning(f"Failed to initialize OpenAI client: {str(e)}")
        return None


client = None


class ChatgptBot(BaseBot):
    """A ChatGPT bot using GPT-4.1 nano model."""

    # Override the bot name
    bot_name = "ChatgptBot"

    def _format_chat_history(self, query: QueryRequest) -> list[dict[str, str]]:
        """Extract and format chat history from the query for OpenAI API.

        Args:
            query: The query from the user

        Returns:
            List of message dictionaries formatted for OpenAI API
        """
        chat_history = []

        if not isinstance(query.query, list):
            return chat_history

        for message in query.query:
            # Skip messages without proper attributes
            if not (hasattr(message, "role") and hasattr(message, "content")):
                continue

            # Map roles: user -> user, bot/assistant -> assistant
            role = (
                "user"
                if message.role == "user"
                else "assistant" if message.role in ["bot", "assistant"] else None
            )

            # Skip messages with unsupported roles
            if not role:
                continue

            chat_history.append({"role": role, "content": message.content})

        return chat_history

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
                # Format the messages with chat history
                messages = self._format_chat_history(query)

                # If there's chat history, use it
                if messages:
                    # Make sure the last message is from the user with the current message
                    if messages and messages[-1]["role"] == "user":
                        messages[-1]["content"] = user_message
                    else:
                        messages.append({"role": "user", "content": user_message})

                    # Log that we're using chat history
                    if len(messages) > 1:
                        logger.info(f"Using chat history with {len(messages)} messages")
                else:
                    # Single message with no history
                    messages = [{"role": "user", "content": user_message}]

                # Make the API call with streaming enabled
                # Type ignore because the typing for OpenAI client is external
                response = client.chat.completions.create(  # type: ignore
                    model="gpt-4.1-nano-2025-04-14",  # Use the newest GPT-4.1 nano model
                    messages=messages,  # type: ignore
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

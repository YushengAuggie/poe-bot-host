"""
Standalone deployment example for a single Poe bot.

This file shows how to deploy a single bot without using the multi-bot framework.
"""

import json
import logging
from typing import AsyncGenerator

from fastapi_poe import PoeBot, make_app
from fastapi_poe.types import PartialResponse, QueryRequest
from modal import App, Image, asgi_app

# Configure logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("echobot")


class EchoBot(PoeBot):
    """A simple bot that echoes back the user's message.

    This is an example of a standalone bot deployment without
    using the multi-bot framework.
    """

    async def get_response(self, query: QueryRequest) -> AsyncGenerator[PartialResponse, None]:
        """Process the query and generate a response.

        Args:
            query: The query from the user

        Yields:
            Response chunks as PartialResponse objects
        """
        try:
            # Log the exact query we received for debugging
            logger.debug(f"Query: {query}")
            logger.debug(f"Query type: {type(query.query)}")

            # Extract user's message - handle all possible formats
            if isinstance(query.query, list) and len(query.query) > 0:
                # Handle structured messages (newer format)
                last_message = query.query[-1]
                if isinstance(last_message, dict) and "content" in last_message:
                    user_message = last_message["content"]
                else:
                    user_message = str(last_message)
            elif isinstance(query.query, str):
                # Handle string messages (older format)
                user_message = query.query
            else:
                # Handle any other format
                if hasattr(query.query, "__dict__"):
                    user_message = json.dumps(query.query)
                else:
                    user_message = str(query.query)

            logger.debug(f"Extracted message: {user_message}")

            # Just echo back the user's message
            yield PartialResponse(text=user_message)

        except Exception as e:
            # Log the error
            logger.error(f"Error in get_response: {str(e)}", exc_info=True)

            # Return an error message
            yield PartialResponse(text=f"Echo bot error: {str(e)}")


# Create and expose the API
logger.info("Creating FastAPI app for EchoBot")
api = make_app([EchoBot()], allow_without_key=True)

# For Modal deployment
app = App("echobot")

# Create a custom image with required dependencies
image = Image.debian_slim().pip_install_from_requirements("requirements.txt")


@app.function(image=image)
@asgi_app()
def fastapi_app():
    """Create and return the FastAPI app for Modal deployment."""
    logger.info("Starting EchoBot for Modal deployment")
    return api


# This allows the app to be run locally with 'python echobot.py'
if __name__ == "__main__":
    import uvicorn

    logger.info("Starting EchoBot locally")
    uvicorn.run(api, host="0.0.0.0", port=8000)

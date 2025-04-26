"""
Web Search Bot - A bot that can search the web using the Google Search API.

This bot demonstrates how to make external API calls to provide web search functionality.
"""

import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, Union

import httpx
from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest

from utils.base_bot import BaseBot, BotError, BotErrorNoRetry

logger = logging.getLogger(__name__)


class WebSearchBot(BaseBot):
    """
    A bot that can search the web using the Google Search API.

    Note: You need to set SERP_API_KEY in your environment variables for this to work.
    You can get a free API key from https://serpapi.com/
    """

    bot_name = "WebSearchBot"
    bot_description = "A bot that can search the web. Just enter your search query."
    version = "1.0.0"

    def __init__(self, **kwargs):
        """Initialize the WebSearchBot."""
        super().__init__(**kwargs)
        self.api_key = os.environ.get("SERP_API_KEY", "")
        if not self.api_key:
            logger.warning("SERP_API_KEY not set. Web search will not work.")

    async def _search_web(self, query: str, num_results: int = 5) -> Dict[str, Any]:
        """
        Search the web using the SerpAPI Google Search API.

        Args:
            query: The search query
            num_results: Number of results to return (default: 5)

        Returns:
            Dictionary containing search results
        """
        if not self.api_key:
            raise BotErrorNoRetry("Search API key not configured. Please set SERP_API_KEY.")

        try:
            # Use a mock response for testing if no API key
            # Remove this for production or when you have an API key
            if self.api_key == "":
                return self._get_mock_response(query)

            # Use the SerpAPI for web search
            url = "https://serpapi.com/search"
            params = {"q": query, "api_key": self.api_key, "engine": "google", "num": num_results}

            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params)
                response.raise_for_status()
                return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error in web search: {e.response.text}")
            raise BotError(f"Search error: HTTP {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error in web search: {str(e)}")
            raise BotError(f"Search failed: {str(e)}")

    def _get_mock_response(self, query: str) -> Dict[str, Any]:
        """Return a mock response for testing without an API key."""
        return {
            "search_metadata": {
                "status": "Success",
                "created_at": "2023-04-13 12:34:56 UTC",
            },
            "search_parameters": {
                "q": query,
                "engine": "google",
            },
            "organic_results": [
                {
                    "position": 1,
                    "title": f"Example result 1 for '{query}'",
                    "link": "https://example.com/1",
                    "snippet": f"This is a mock result for the query '{query}'. In a real implementation, this would be an actual search result.",
                },
                {
                    "position": 2,
                    "title": f"Example result 2 for '{query}'",
                    "link": "https://example.com/2",
                    "snippet": "Another mock result for testing purposes. Replace this with actual API integration.",
                },
                {
                    "position": 3,
                    "title": f"Example result 3 for '{query}'",
                    "link": "https://example.com/3",
                    "snippet": "This is a mock result. Please set SERP_API_KEY to get real search results.",
                },
            ],
            "mock_response": True,  # Flag to indicate this is a mock response
        }

    def _format_search_results(self, results: Dict[str, Any], query: str) -> str:
        """Format search results into a readable markdown response."""
        if results.get("mock_response", False):
            formatted = f"## 🔍 Search Results for '{query}'\n\n"
            formatted += (
                "⚠️ **Note:** Using mock results. Set SERP_API_KEY for real search results.\n\n"
            )
        else:
            formatted = f"## 🔍 Search Results for '{query}'\n\n"

        organic_results = results.get("organic_results", [])

        if not organic_results:
            return formatted + "No results found."

        for i, result in enumerate(organic_results[:5], 1):
            title = result.get("title", "No title")
            link = result.get("link", "#")
            snippet = result.get("snippet", "No description available.")

            formatted += f"### {i}. [{title}]({link})\n"
            formatted += f"{snippet}\n\n"

        return formatted

    async def _process_message(
        self, message: str, query: QueryRequest
    ) -> AsyncGenerator[PartialResponse, None]:
        """Process the user's search query and return search results. (Deprecated)"""
        logger.warning(
            f"[{self.bot_name}] _process_message is deprecated, use get_response instead. This method will be removed in a future version."
        )

        # For backward compatibility, delegate to get_response
        async for response in self.get_response(query):
            if isinstance(response, PartialResponse):
                yield response

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        """Process the query and return search results."""
        try:
            # Extract the message
            message = self._extract_message(query)
            message = message.strip()

            # Handle bot info requests
            if message.lower().strip() == "bot info":
                metadata = self._get_bot_metadata()
                yield PartialResponse(text=json.dumps(metadata, indent=2))
                return

            # Help command
            if message.lower() in ["help", "?", "/help"]:
                yield PartialResponse(
                    text="""
## 🔍 Web Search Bot

Enter any search query and I'll search the web for information.

Examples:
- `climate change latest news`
- `python programming tutorials`
- `best restaurants in new york`

Note: For the best experience, be specific in your search queries.
"""
                )
                return

            # Empty query
            if not message:
                yield PartialResponse(
                    text="Please enter a search query. Type 'help' for instructions."
                )
                return

            # Perform the search
            yield PartialResponse(text=f"Searching for '{message}'...\n\n")

            search_results = await self._search_web(message)
            formatted_results = self._format_search_results(search_results, message)

            yield PartialResponse(text=formatted_results)

        except BotErrorNoRetry as e:
            # Log the error (non-retryable)
            logger.error(f"[{self.bot_name}] Non-retryable error: {str(e)}", exc_info=True)
            yield PartialResponse(text=f"Error (please try something else): {str(e)}")

        except BotError as e:
            # Log the error (retryable)
            logger.error(f"[{self.bot_name}] Retryable error: {str(e)}", exc_info=True)
            yield PartialResponse(text=f"Error (please try again): {str(e)}")

        except Exception as e:
            # Log the unexpected error
            logger.error(f"[{self.bot_name}] Unexpected error: {str(e)}", exc_info=True)
            # Return a generic error message
            error_msg = "An unexpected error occurred. Please try again later."
            yield PartialResponse(text=error_msg)

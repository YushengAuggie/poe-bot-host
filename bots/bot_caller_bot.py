"""
Bot Caller Bot - A bot that can call other bots in the framework.

This bot demonstrates how to make requests to other bots in the framework,
allowing for composing multiple bots together for more complex functionalities.
"""

import logging
import json
import httpx
from typing import AsyncGenerator, Dict, Any, List, Optional
from fastapi_poe.types import PartialResponse, QueryRequest
from utils.base_bot import BaseBot, BotError, BotErrorNoRetry

logger = logging.getLogger(__name__)

class BotCallerBot(BaseBot):
    """
    A bot that can call other bots in the framework.
    
    Commands:
    - list: List available bots
    - call <bot_name> <message>: Call the specified bot with the message
    - echo <message>: Echo the message (for testing)
    """
    
    bot_name = "BotCallerBot"
    bot_description = "A bot that can call other bots in the framework. Use 'list' to see available bots, or 'call <bot_name> <message>' to call a specific bot."
    version = "1.0.0"
    
    # Default base URL for local development
    base_url = "http://localhost:8000"
    
    def __init__(self, **kwargs):
        """Initialize the BotCallerBot."""
        super().__init__(**kwargs)
        # If we're running on Modal, we'd use a different URL pattern
        # This could be configured via environment variables in the future
    
    async def _list_available_bots(self) -> Dict[str, str]:
        """List all available bots in the framework."""
        try:
            async with httpx.AsyncClient() as client:
                # Call the /bots endpoint to get a list of available bots
                response = await client.get(f"{self.base_url}/bots")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Error listing bots: {str(e)}")
            raise BotError(f"Failed to list bots: {str(e)}")
    
    async def _call_bot(self, bot_name: str, message: str, user_id: str, conversation_id: str) -> AsyncGenerator[PartialResponse, None]:
        """Call another bot with the given message."""
        try:
            # Ensure bot name is lowercase as expected by the API
            bot_name = bot_name.lower()
            
            # Create a QueryRequest payload similar to what the framework uses
            payload = {
                "version": "1.0",
                "type": "query",
                "query": [
                    {"role": "user", "content": message}
                ],
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message_id": f"botcaller-{conversation_id}"
            }
            
            # Make a POST request to the bot's endpoint
            async with httpx.AsyncClient() as client:
                async with client.stream(
                    "POST", 
                    f"{self.base_url}/{bot_name}", 
                    json=payload, 
                    timeout=30
                ) as response:
                    # Raise exception for HTTP errors
                    response.raise_for_status()
                    
                    # Stream the response back
                    async for chunk in response.aiter_raw():
                        # Handle different response formats (text, JSON, etc.)
                        chunk_str = chunk.decode('utf-8', 'replace')
                        # Skip empty lines
                        if not chunk_str.strip():
                            continue
                            
                        # Try to parse as JSON
                        try:
                            data = json.loads(chunk_str.replace('data: ', ''))
                            if 'text' in data:
                                yield PartialResponse(text=data['text'])
                        except json.JSONDecodeError:
                            # Handle non-JSON responses
                            if chunk_str.startswith('data: '):
                                chunk_str = chunk_str.replace('data: ', '')
                            yield PartialResponse(text=chunk_str)
                    
                    # Return one final response if no data was yielded
                    yield PartialResponse(text=f"Called {bot_name} (empty response)")
                    
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling bot {bot_name}: {e.response.text}")
            raise BotError(f"Error calling bot {bot_name}: HTTP {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error calling bot {bot_name}: {str(e)}")
            raise BotError(f"Failed to call bot {bot_name}: {str(e)}")
    
    async def _process_message(self, message: str, query: QueryRequest) -> AsyncGenerator[PartialResponse, None]:
        """
        Process the user message and call appropriate bot or action.
        
        Commands:
        - list: List available bots
        - call <bot_name> <message>: Call the specified bot with the message
        - echo <message>: Echo the message (for testing)
        """
        message = message.strip()
        user_id = query.user_id
        conversation_id = query.conversation_id
        
        # Handle the 'list' command
        if message.lower() == "list":
            try:
                bots = await self._list_available_bots()
                bot_list = []
                for bot_name, description in bots.items():
                    bot_list.append(f"- **{bot_name}**: {description}")
                
                yield PartialResponse(text="## Available Bots\n\n" + "\n".join(bot_list))
                yield PartialResponse(text="\n\nTo call a bot, use: `call <bot_name> <message>`")
                return
            except Exception as e:
                yield PartialResponse(text=f"Error listing bots: {str(e)}")
                return
        
        # Handle the 'call' command
        if message.lower().startswith("call "):
            # Parse the command: call <bot_name> <message>
            parts = message[5:].strip().split(" ", 1)
            
            if len(parts) < 2:
                yield PartialResponse(text="Error: Please provide both a bot name and a message.\nExample: `call EchoBot Hello, world!`")
                return
                
            bot_name, bot_message = parts
            yield PartialResponse(text=f"Calling {bot_name}...\n\n")
            
            try:
                async for response_chunk in self._call_bot(bot_name, bot_message, user_id, conversation_id):
                    yield response_chunk
            except Exception as e:
                yield PartialResponse(text=f"Error calling {bot_name}: {str(e)}")
            return
        
        # Handle the 'echo' command (for testing)
        if message.lower().startswith("echo "):
            echo_text = message[5:].strip()
            yield PartialResponse(text=f"Echo: {echo_text}")
            return
        
        # Handle help or unknown commands
        yield PartialResponse(text="""
## Bot Caller Bot

I can call other bots in the framework. Here are my commands:

- `list` - List all available bots
- `call <bot_name> <message>` - Call a specific bot with a message  
- `echo <message>` - Echo a message (for testing)

Example: `call EchoBot Hello, world!`
""")
        return
import json
import logging
import traceback
from typing import Any, AsyncGenerator, Dict, Optional, Tuple, Type, TypeVar, Union

from fastapi_poe import PoeBot
from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class BotError(Exception):
    """Base exception for bot errors that should be retried."""

    pass


class BotErrorNoRetry(Exception):
    """Exception for bot errors that should not be retried."""

    pass


T = TypeVar("T", bound="BaseBot")


class BaseBot(PoeBot):
    """Base class for all Poe bots with common functionality."""

    bot_name: str = "BaseBot"
    bot_description: str = "Base bot implementation with common functionality."
    version: str = "1.0.0"

    # Default settings - can be overridden by subclasses
    max_message_length: int = 2000
    stream_response: bool = True

    def __init__(
        self,
        path: Optional[str] = None,
        access_key: Optional[str] = None,
        bot_name: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize the bot with a unique path based on the bot name.

        Args:
            path: Optional path to use (defaults to bot_name in lowercase)
            access_key: Optional access key for authentication
            bot_name: Optional bot name to override class attribute
            settings: Optional dictionary of settings to override defaults
            **kwargs: Additional arguments to pass to the parent class
        """
        # Make a copy of class attributes to ensure they're set before they're used
        if hasattr(self.__class__, "bot_name"):
            self.bot_name = self.__class__.bot_name
        else:
            self.bot_name = self.__class__.__name__

        # Override with parameter if provided
        if bot_name is not None:
            self.bot_name = bot_name

        # Generate a path based on the bot name if not provided
        if path is None:
            path = f"/{self.bot_name.lower()}"

        # Apply custom settings if provided
        if settings:
            for key, value in settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)

        # Initialize the parent class
        super().__init__(path=path, access_key=access_key, **kwargs)

        logger.info(f"Initialized {self.bot_name} (v{self.version}) at path: {path}")

    @classmethod
    def create(cls: Type[T], **kwargs) -> T:
        """Factory method to create a new instance of the bot.

        Args:
            **kwargs: Arguments to pass to the constructor

        Returns:
            A new instance of the bot
        """
        return cls(**kwargs)

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        """Process the query and generate a response.

        Subclasses should override this method to implement their logic directly.
        The default implementation provides error handling and basic command processing.

        Args:
            query: The query from the user

        Yields:
            Response chunks as PartialResponse or MetaResponse objects
        """
        start_time = None
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

            # Default implementation just echoes the message
            # Subclasses should override this method to implement their custom logic
            yield PartialResponse(text=f"{user_message}")

        except BotErrorNoRetry as e:
            # Log the error (non-retryable)
            logger.error(f"[{self.bot_name}] Non-retryable error: {str(e)}", exc_info=True)

            # Return an error message
            yield PartialResponse(text=f"Error (please try something else): {str(e)}")

        except BotError as e:
            # Log the error (retryable)
            logger.error(f"[{self.bot_name}] Retryable error: {str(e)}", exc_info=True)

            # Return an error message with retry suggestion
            yield PartialResponse(text=f"Error (please try again): {str(e)}")

        except Exception as e:
            # Log the unexpected error
            logger.error(f"[{self.bot_name}] Unexpected error: {str(e)}", exc_info=True)

            # Return a generic error message
            error_msg = "An unexpected error occurred. Please try again later."
            if logger.level <= logging.DEBUG:  # Only include details in debug mode
                error_msg += f"\n\nDetails: {str(e)}\n{traceback.format_exc()}"

            yield PartialResponse(text=error_msg)

    def _extract_message(self, query: QueryRequest) -> str:
        """Extract the user's message from the query.

        Args:
            query: The query from the user

        Returns:
            The extracted message as a string
        """
        try:
            # Log the exact query we received for debugging
            logger.debug(f"[{self.bot_name}] Query: {query}")
            logger.debug(f"[{self.bot_name}] Query type: {type(query.query)}")

            # Extract user's message - handle all possible formats
            if isinstance(query.query, list) and len(query.query) > 0:
                # Handle structured messages (newer format)
                last_message = query.query[-1]
                # Handle ProtocolMessage object with content attribute
                if hasattr(last_message, "content"):
                    return last_message.content
                # Handle dict with content key
                elif isinstance(last_message, dict) and "content" in last_message:
                    return last_message["content"]
                # Fall back to string representation if needed
                else:
                    return str(last_message)
            elif isinstance(query.query, str):
                # Handle string messages (older format)
                return query.query
            else:
                # Handle any other format
                if hasattr(query.query, "__dict__"):
                    return json.dumps(query.query.__dict__)
                elif hasattr(query.query, "content") and query.query is not None:
                    content = getattr(query.query, "content", None)
                    if content is not None:
                        return content
                    else:
                        return str(query.query)
                else:
                    return str(query.query)
        except Exception as e:
            logger.error(f"[{self.bot_name}] Error extracting message: {str(e)}", exc_info=True)
            return f"[Error extracting message: {str(e)}]"

    def _get_bot_metadata(self) -> Dict[str, Any]:
        """Get metadata about the bot.

        Returns:
            Dictionary of bot metadata
        """
        # Use class attributes if instance attributes are not set
        bot_name = (
            self.bot_name
            if hasattr(self, "bot_name") and self.bot_name is not None
            else self.__class__.bot_name
        )
        bot_description = (
            self.bot_description
            if hasattr(self, "bot_description") and self.bot_description is not None
            else self.__class__.bot_description
        )
        version = (
            self.version
            if hasattr(self, "version") and self.version is not None
            else self.__class__.version
        )

        return {
            "name": bot_name,
            "description": bot_description,
            "version": version,
            "settings": {
                "max_message_length": self.max_message_length,
                "stream_response": self.stream_response,
            },
        }

    async def _process_message(
        self, message: str, query: QueryRequest
    ) -> AsyncGenerator[PartialResponse, None]:
        """[DEPRECATED] Process the message and generate a response.

        DEPRECATED: This method is deprecated in favor of directly overriding get_response.
        It is kept for backward compatibility but will be removed in a future version.

        Args:
            message: The extracted message from the user
            query: The original query object

        Yields:
            Response chunks as PartialResponse objects
        """
        logger.warning(
            f"[{self.bot_name}] _process_message is deprecated, override get_response instead"
        )
        # Default implementation for backward compatibility, just forwards to get_response
        # This will use the get_response of the superclass (BaseBot), not of the subclass,
        # which is why subclasses should override get_response directly instead.
        modified_query = (
            query  # In a real backward compatibility layer, we might need to modify the query
        )
        async for response_chunk in super().get_response(modified_query):
            if isinstance(response_chunk, PartialResponse):
                yield response_chunk

    def _validate_message(self, message: str) -> Tuple[bool, Optional[str]]:
        """Validate the user's message.

        Args:
            message: The message to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if the message is too long
        if len(message) > self.max_message_length:
            return (
                False,
                f"Message is too long. Maximum length is {self.max_message_length} characters.",
            )

        # Message is valid
        return True, None

"""
Mixins for common bot functionality.

This module provides reusable mixins that bots can inherit to get
common functionality without duplicating code.
"""

import logging
from typing import AsyncGenerator, Callable, Union

from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest

from .base_bot import BotError, BotErrorNoRetry

logger = logging.getLogger(__name__)


class ErrorHandlerMixin:
    """Mixin that provides standardized error handling for bots."""

    async def handle_common_errors(
        self,
        query: QueryRequest,
        error_callback: Callable[[], AsyncGenerator[Union[PartialResponse, MetaResponse], None]],
    ) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        """
        Handle common error patterns consistently across bots.

        Args:
            query: The original query request
            error_callback: Async generator function to execute with error handling

        Yields:
            Response chunks with proper error handling
        """
        try:
            async for response in error_callback():
                yield response
        except BotErrorNoRetry as e:
            logger.error(f"Non-retryable error: {str(e)}")
            yield PartialResponse(text=f"Error (please try something else): {str(e)}")
        except BotError as e:
            logger.error(f"Retryable error: {str(e)}")
            yield PartialResponse(text=f"Error (please try again): {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            # Fall back to generic error message
            yield PartialResponse(text="An unexpected error occurred. Please try again later.")


class ResponseMixin:
    """Mixin that provides standardized response patterns."""

    def _format_help_response(self, help_text: str) -> str:
        """
        Format help text consistently across bots.

        Args:
            help_text: The help text to format

        Returns:
            Formatted help response
        """
        bot_name = getattr(self, "bot_name", "Bot")
        return f"## 🤖 {bot_name} Help\n\n{help_text}"

    def _format_error_response(self, error_msg: str, is_retryable: bool = True) -> str:
        """
        Format error messages consistently.

        Args:
            error_msg: The error message
            is_retryable: Whether the user should try again

        Returns:
            Formatted error response
        """
        action = "please try again" if is_retryable else "please try something else"
        return f"❌ Error ({action}): {error_msg}"

    def _format_success_response(self, content: str, prefix: str = "✅") -> str:
        """
        Format success messages consistently.

        Args:
            content: The success content
            prefix: Emoji or prefix for the message

        Returns:
            Formatted success response
        """
        return f"{prefix} {content}"


class BotInfoMixin:
    """Mixin that provides standardized bot info responses."""

    def get_enhanced_bot_metadata(self) -> dict:
        """
        Get enhanced metadata about the bot.

        Returns:
            Dictionary of enhanced bot metadata
        """
        base_metadata = {}
        if hasattr(self, "_get_bot_metadata") and callable(getattr(self, "_get_bot_metadata")):
            base_metadata = getattr(self, "_get_bot_metadata")()

        # Add enhanced information
        enhanced = {
            **base_metadata,
            "capabilities": self._get_bot_capabilities(),
            "usage_tips": self._get_usage_tips(),
            "version_info": self._get_version_info(),
        }

        return enhanced

    def _get_bot_capabilities(self) -> list:
        """Get list of bot capabilities."""
        capabilities = []

        if getattr(self, "supports_image_input", False):
            capabilities.append("Image processing")
        if getattr(self, "supports_video_input", False):
            capabilities.append("Video processing")
        if getattr(self, "supports_audio_input", False):
            capabilities.append("Audio processing")
        if getattr(self, "supports_grounding", False):
            capabilities.append("Web grounding")
        if getattr(self, "supports_image_generation", False):
            capabilities.append("Image generation")

        return capabilities

    def _get_usage_tips(self) -> list:
        """Get usage tips for the bot."""
        return ["Type 'help' for detailed instructions", "Type 'bot info' for technical details"]

    def _get_version_info(self) -> dict:
        """Get version information."""
        return {
            "version": getattr(self, "version", "1.0.0"),
            "framework": "fastapi-poe",
            "python_version": "3.7+",
        }

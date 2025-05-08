"""
Example implementation of monetization for Gemini bots using fastapi_poe CostItem.

This example demonstrates:
1. Setting up a rate card for token-based pricing
2. Authorizing costs at the beginning of processing
3. Capturing actual costs once processing is complete
4. Handling insufficient funds gracefully
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, Optional, Union

from fastapi_poe import CostItem
from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest, SettingsResponse

from utils.base_bot import BaseBot

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants for pricing (in milli-cents, 1/1000th of a cent)
BASE_COST = 500  # Base cost of 0.5 cents per query
INPUT_TOKEN_COST = 10  # 0.01 cents per 1k input tokens
OUTPUT_TOKEN_COST = 30  # 0.03 cents per 1k output tokens
IMAGE_INPUT_COST = 1000  # 1 cent per image input
IMAGE_OUTPUT_COST = 3000  # 3 cents per image output


class MonetizedGeminiBot(BaseBot):
    """Example Gemini bot with monetization features."""

    model_name = "gemini-2.0-pro"
    bot_name = "MonetizedGeminiBot"
    bot_description = "Gemini Pro model with token-based monetization"
    supports_image_input = True
    supports_image_generation = False  # Set to True for models that support image generation

    def __init__(self, **kwargs):
        """Initialize the bot with monetization settings."""
        super().__init__(**kwargs)

        # Define token rates and other monetization settings
        self.base_cost = kwargs.get("base_cost", BASE_COST)
        self.input_token_cost = kwargs.get("input_token_cost", INPUT_TOKEN_COST)
        self.output_token_cost = kwargs.get("output_token_cost", OUTPUT_TOKEN_COST)
        self.image_input_cost = kwargs.get("image_input_cost", IMAGE_INPUT_COST)
        self.image_output_cost = kwargs.get("image_output_cost", IMAGE_OUTPUT_COST)

    async def get_settings(self, settings_request) -> SettingsResponse:
        """Get bot settings including monetization rate card.

        Args:
            settings_request: The settings request

        Returns:
            Settings response with rate card and cost label
        """
        # Create a markdown rate card that displays pricing
        rate_card = (
            "# Gemini Pro Pricing\n\n"
            "| Service | Cost |\n"
            "|---------|------|\n"
            f"| Base fee | [usd_milli_cents={self.base_cost}] points per message |\n"
            f"| Input text | [usd_milli_cents={self.input_token_cost}] points per 1k tokens |\n"
            f"| Output text | [usd_milli_cents={self.output_token_cost}] points per 1k tokens |\n"
        )

        # Add image pricing if supported
        if self.supports_image_input:
            rate_card += (
                f"| Image input | [usd_milli_cents={self.image_input_cost}] points per image |\n"
            )

        if self.supports_image_generation:
            rate_card += f"| Image generation | [usd_milli_cents={self.image_output_cost}] points per image |\n"

        # Define a cost label that shows in the bot's chat interface (next to the Send button)
        # This should be a short summary of the cost structure
        cost_label = f"[usd_milli_cents={self.base_cost}]+"

        # Return settings with monetization details
        return SettingsResponse(
            allow_attachments=self.supports_image_input,
            expand_text_attachments=True,
            rate_card=rate_card,
            cost_label=cost_label,
        )

    def _extract_attachments(self, query: QueryRequest) -> list:
        """Extract attachments from the query.

        Args:
            query: The query from the user

        Returns:
            List of attachments
        """
        attachments = []
        try:
            if isinstance(query.query, list) and query.query:
                last_message = query.query[-1]
                if hasattr(last_message, "attachments") and last_message.attachments:
                    attachments = last_message.attachments
                    logger.info(f"Found {len(attachments)} attachments")

        except Exception as e:
            logger.error(f"Error extracting attachments: {str(e)}")

        return attachments

    def _estimate_token_count(self, text: str) -> int:
        """Estimate the number of tokens in a text.

        This is a simple approximation. In production, use a proper tokenizer.

        Args:
            text: The text to estimate tokens for

        Returns:
            Estimated token count
        """
        # Simple estimation: assume ~4 characters per token
        return max(1, len(text) // 4)

    async def _authorize_initial_costs(
        self, query: QueryRequest, message: str, attachments: list
    ) -> None:
        """Authorize the initial costs for processing the query.

        Args:
            query: The query request
            message: The user message
            attachments: List of attachments

        Raises:
            InsufficientFundsError: If the user doesn't have enough funds
        """
        try:
            # Calculate base cost
            costs = [
                CostItem(amount_usd_milli_cents=self.base_cost, description="Base processing fee")
            ]

            # Add cost for input tokens
            input_tokens = self._estimate_token_count(message)
            token_cost = max(1, (input_tokens * self.input_token_cost) // 1000)
            costs.append(
                CostItem(
                    amount_usd_milli_cents=token_cost,
                    description=f"Input text ({input_tokens} tokens)",
                )
            )

            # Add cost for each image attachment
            if self.supports_image_input and attachments:
                image_costs = len(attachments) * self.image_input_cost
                costs.append(
                    CostItem(
                        amount_usd_milli_cents=image_costs,
                        description=f"Image input ({len(attachments)} images)",
                    )
                )

            # Authorize all costs
            logger.info(f"Authorizing initial costs: {costs}")
            await self.authorize_cost(query, costs)
            logger.info("Cost authorization successful")

        except Exception as e:
            logger.error(f"Error authorizing costs: {str(e)}")
            # Re-raise to be handled by the caller
            raise

    async def _capture_output_costs(
        self, query: QueryRequest, output_text: str, images_generated: int = 0
    ) -> None:
        """Capture the costs for the generated output.

        Args:
            query: The query request
            output_text: The generated text response
            images_generated: Number of images generated

        Raises:
            Exception: If there's an issue capturing costs
        """
        try:
            costs = []

            # Calculate and capture cost for output tokens
            output_tokens = self._estimate_token_count(output_text)
            token_cost = max(1, (output_tokens * self.output_token_cost) // 1000)

            costs.append(
                CostItem(
                    amount_usd_milli_cents=token_cost,
                    description=f"Output text ({output_tokens} tokens)",
                )
            )

            # Add cost for generated images if any
            if images_generated > 0 and self.supports_image_generation:
                image_costs = images_generated * self.image_output_cost
                costs.append(
                    CostItem(
                        amount_usd_milli_cents=image_costs,
                        description=f"Image generation ({images_generated} images)",
                    )
                )

            # Capture costs
            logger.info(f"Capturing output costs: {costs}")
            await self.capture_cost(query, costs)
            logger.info("Cost capture successful")

        except Exception as e:
            logger.error(f"Error capturing costs: {str(e)}")
            # In a real implementation, you might want to handle this more gracefully
            # and potentially record failed captures for later reconciliation
            raise

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        """Process the query and generate a response with monetization.

        Args:
            query: The query from the user

        Yields:
            Response chunks as PartialResponse or MetaResponse objects
        """
        try:
            # Extract the message and attachments
            user_message = self._extract_message(query)
            attachments = self._extract_attachments(query)

            # Log the extracted message (simplified logging)
            logger.debug(f"[{self.bot_name}] Received message: {user_message}")

            # Handle bot info request (no charge for this)
            if user_message.lower().strip() == "bot info":
                yield await self._handle_bot_info_request()
                return

            # Authorize the initial costs
            try:
                await self._authorize_initial_costs(query, user_message, attachments)
            except Exception as e:
                # Handle insufficient funds or other authorization errors
                yield PartialResponse(
                    text="⚠️ Unable to process your request due to insufficient funds. "
                    f"Please check your Poe subscription or try a simpler query.\n\nError: {str(e)}"
                )
                return

            # Process your query here...
            # For this example, we'll just simulate a response
            # In a real implementation, you would call the Gemini API here

            # Simulate a response with token count proportional to input
            response_text = f"This is a simulated response to your query: '{user_message}'\n\n"
            response_text += "In a real implementation, this would be processed by the Gemini API."

            # Simulate the number of generated images
            images_generated = 0

            # Capture costs for the output
            try:
                await self._capture_output_costs(query, response_text, images_generated)
            except Exception as e:
                # If cost capture fails, log it but still return the response to the user
                logger.error(f"Cost capture failed: {str(e)}")
                # In a production environment, you might want to record this for reconciliation

            # Return the response to the user
            yield PartialResponse(text=response_text)

        except Exception as e:
            # Let the parent class handle errors
            logger.error(f"Error in get_response: {str(e)}")
            yield PartialResponse(text=f"An error occurred: {str(e)}")

    async def _handle_bot_info_request(self) -> PartialResponse:
        """Handle a request for bot information.

        Returns:
            Formatted response with bot metadata
        """
        metadata = self._get_bot_metadata()
        metadata["model_name"] = self.model_name
        metadata["supports_image_input"] = self.supports_image_input
        metadata["supports_image_generation"] = self.supports_image_generation
        metadata["monetization"] = {
            "base_cost": self.base_cost,
            "input_token_cost": self.input_token_cost,
            "output_token_cost": self.output_token_cost,
            "image_input_cost": self.image_input_cost,
            "image_output_cost": self.image_output_cost,
        }
        return PartialResponse(text=json.dumps(metadata, indent=2))


# Example of how to run the bot using fastapi_poe
if __name__ == "__main__":
    import fastapi_poe as fp
    import uvicorn

    # Create an instance of the monetized bot
    bot = MonetizedGeminiBot()

    # Create a FastAPI app with the bot
    app = fp.make_app(bot, allow_without_key=True)

    # Run the app using uvicorn
    uvicorn.run(app, host="localhost", port=8080)

"""
Gemini Image Generation Bot implementation.

This bot uses Google's Gemini model to generate images based on text descriptions.
"""

import logging
import time
from typing import AsyncGenerator, Union

from fastapi_poe.types import (
    MetaResponse,
    PartialResponse,
    QueryRequest,
    SettingsRequest,
    SettingsResponse,
)

from utils.api_keys import get_api_key
from utils.base_bot import BaseBot

# Configure logging
logger = logging.getLogger(__name__)


class GeminiImageGenerationBot(BaseBot):
    """Gemini Image Generation bot that creates images from text prompts."""

    model_name = "gemini-2.0-flash-preview-image-generation"
    bot_name = "GeminiImageGenerationBot"
    bot_description = (
        "Generates images from text descriptions using Gemini's image generation capabilities."
    )
    supports_image_generation = True  # Enable image generation

    async def get_settings(self, settings_request: SettingsRequest) -> SettingsResponse:
        """Get bot settings including the rate card.

        Args:
            settings_request: The settings request

        Returns:
            Settings response with rate card and cost label
        """
        from fastapi_poe.types import SettingsResponse

        # Create a new settings response
        settings = SettingsResponse(
            allow_attachments=True,
            expand_text_attachments=True,
            rate_card="100 points / image",
            cost_label="Image Generation Cost",
        )

        return settings

    def _get_extension_for_mime_type(self, mime_type: str) -> str:
        """Get the appropriate file extension for an image MIME type.

        Args:
            mime_type: MIME type of the image

        Returns:
            File extension (without the dot)
        """
        if mime_type == "image/png":
            return "png"
        elif mime_type == "image/gif":
            return "gif"
        elif mime_type == "image/webp":
            return "webp"
        else:
            return "jpg"  # Default to jpg

    async def _generate_image(
        self, prompt: str, query: QueryRequest
    ) -> AsyncGenerator[PartialResponse, None]:
        """Generate an image based on the text prompt using Gemini API.

        Args:
            prompt: The text prompt describing the image to generate
            query: The original query for message_id

        Yields:
            Response chunks as PartialResponse objects
        """
        try:
            # Import required modules
            import google.generativeai as genai

            # Get API key
            api_key = get_api_key("GOOGLE_API_KEY")
            if not api_key:
                logger.error("GOOGLE_API_KEY not found")
                yield PartialResponse(
                    text="Error: Google API key is not configured. Please set the GOOGLE_API_KEY environment variable."
                )
                return

            # Configure the API key
            genai.configure(api_key=api_key)

            # Initialize the model for image generation
            model = genai.GenerativeModel(model_name=self.model_name)

            # Send a status message
            yield PartialResponse(
                text=f'🖌️ Generating image from prompt: "{prompt}"\n\nPlease wait a moment...'
            )

            # Store all generated images/text
            generated_text = []
            generated_images = []

            # Make the API request - use non-streaming for image generation
            logger.info(f"Making image generation request with prompt: {prompt}")

            # For image generation, we don't use special generation_config
            # The model itself handles this based on its capabilities
            try:
                # Remove any response_mime_type configuration as it's causing errors
                response = model.generate_content(prompt, stream=False)
            except Exception as api_err:
                logger.error(f"Error in Gemini API call: {str(api_err)}")
                yield PartialResponse(text=f"Error generating image: {str(api_err)}\n")
                return

            # Process the response
            # Check if response has text
            if hasattr(response, "text") and response.text:
                generated_text.append(response.text)

            # Check if response has parts
            if hasattr(response, "parts"):
                for part in response.parts:
                    # Text content
                    if hasattr(part, "text") and part.text:
                        text = part.text.strip()
                        if text:
                            generated_text.append(text)

                    # Image content
                    if hasattr(part, "inline_data"):
                        try:
                            # Extract image data
                            inline_data = part.inline_data
                            mime_type = inline_data.mime_type
                            data_buffer = inline_data.data

                            # Determine file extension
                            extension = self._get_extension_for_mime_type(mime_type)
                            filename = f"gemini_generated_{int(time.time())}.{extension}"

                            # Upload to Poe
                            attachment_upload_response = await self.post_message_attachment(
                                message_id=query.message_id,
                                file_data=data_buffer,
                                filename=filename,
                                is_inline=True,
                            )

                            # Create markdown for display
                            if (
                                hasattr(attachment_upload_response, "inline_ref")
                                and attachment_upload_response.inline_ref
                            ):
                                image_md = f"![{filename}][{attachment_upload_response.inline_ref}]"
                                generated_images.append(image_md)
                            else:
                                logger.error("Error uploading image: No inline_ref in response")
                                generated_images.append("[Error uploading generated image]")
                        except Exception as img_err:
                            logger.error(f"Error processing image data: {str(img_err)}")
                            generated_images.append(f"[Error processing image: {str(img_err)}]")

            # Yield the model's explanation first, if any
            if generated_text:
                yield PartialResponse(text="\n\n".join(generated_text))

            # Yield the generated images
            for image_md in generated_images:
                yield PartialResponse(text="\n\n" + image_md)

            # If no images were generated, inform the user
            if not generated_images:
                yield PartialResponse(
                    text="\n\nNo images were generated. This could be due to content policy restrictions or a technical issue. Please try a different prompt."
                )

        except Exception as e:
            logger.error(f"Error generating image: {str(e)}", exc_info=True)
            yield PartialResponse(text=f"Error generating image: {str(e)}")

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

            # Check if the API key exists
            api_key = get_api_key("GOOGLE_API_KEY")
            if not api_key:
                logger.error("GOOGLE_API_KEY not found in environment variables")
                yield PartialResponse(
                    text="Error: Google API key is not configured. Please set the GOOGLE_API_KEY environment variable."
                )
                return
            else:
                logger.info(f"Found GOOGLE_API_KEY in environment (starts with: {api_key[:3]}...)")

            # Handle bot info request
            if user_message.lower().strip() == "bot info":
                metadata = self._get_bot_metadata()
                metadata["model_name"] = self.model_name
                metadata["supports_image_generation"] = self.supports_image_generation
                yield PartialResponse(
                    text=f"Gemini Image Generation Bot\nModel: {self.model_name}\nCapabilities: Image generation from text prompts"
                )
                return

            # Check if the message is a help request
            if user_message.lower().startswith(("help", "info", "what can you do", "capabilities")):
                yield PartialResponse(
                    text=(
                        "I can generate images based on your text descriptions. "
                        "Simply describe the image you want to generate, and I'll create it for you.\n\n"
                        "Example prompts:\n"
                        "- A serene mountain lake at sunset with pine trees\n"
                        "- A futuristic cityscape with flying cars and neon lights\n"
                        "- A cute cartoon cat wearing a space helmet\n\n"
                        "Please note that I cannot generate images that violate Google's content policies."
                    )
                )
                return

            # Process as an image generation request
            async for response in self._generate_image(user_message, query):
                yield response

        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}", exc_info=True)
            yield PartialResponse(text=f"Error: {str(e)}")

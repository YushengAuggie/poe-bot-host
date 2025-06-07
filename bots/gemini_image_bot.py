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
from utils.bot_config import get_access_key_patterns

from .gemini_core.base_bot import GeminiBaseBot
from .gemini_core.utils import get_extension_for_mime_type

logger = logging.getLogger(__name__)


class GeminiImageGenerationBot(GeminiBaseBot):
    """Gemini Image Generation bot that creates images from text prompts."""

    model_name = "gemini-2.0-flash-preview-image-generation"
    bot_name = "GeminiImageGenerationBot"
    bot_description = (
        "Generates and edits images from text descriptions using Gemini's image generation capabilities. "
        "Supports multi-turn conversations for iterative image editing."
    )
    supports_image_generation = True

    def _create_access_key_error_message(
        self, filename: str, mime_type: str, data_size: int
    ) -> str:
        """Create a standardized error message for missing access keys."""
        # Get the expected patterns for better user guidance
        patterns = get_access_key_patterns(self.bot_name)
        primary_patterns = patterns[:3] if len(patterns) > 3 else patterns

        return (
            f"✅ **Image Generated Successfully!**\n\n"
            f"📁 **Image Details:**\n"
            f"- Filename: `{filename}`\n"
            f"- Format: `{mime_type}`\n"
            f"- Size: `{data_size} bytes`\n\n"
            f"⚠️ **Display Issue:** To display images inline, this bot needs a Poe access key.\n"
            f"**For local testing:** The image was generated but can't be displayed due to missing access key configuration.\n\n"
            f"**To fix:** Set one of these environment variables with your Poe bot access key:\n"
            f"- `{'` or `'.join(primary_patterns)}`\n\n"
            f"**Note:** {len(patterns)} different patterns are automatically tried for access key resolution."
        )

    def _extract_media_data(self, inline_data) -> tuple[str, bytes] | tuple[None, None]:
        """Extract mime_type and data from inline_data object."""
        mime_type = None
        data_buffer = None

        # Extract mime_type
        if hasattr(inline_data, "mime_type"):
            mime_type = inline_data.mime_type
        elif isinstance(inline_data, dict) and "mime_type" in inline_data:
            mime_type = inline_data["mime_type"]

        # Extract data
        if hasattr(inline_data, "data"):
            data_buffer = inline_data.data
        elif isinstance(inline_data, dict) and "data" in inline_data:
            data_buffer = inline_data["data"]
        elif hasattr(inline_data, "get_bytes") and callable(inline_data.get_bytes):
            data_buffer = inline_data.get_bytes()
            mime_type = mime_type or "image/png"

        return (mime_type, data_buffer) if mime_type and data_buffer else (None, None)

    async def _process_image_part(self, part, query: QueryRequest, generated_images: list) -> None:
        """Process a single image part from the response."""
        if not (hasattr(part, "inline_data") and part.inline_data):
            return

        try:
            mime_type, data_buffer = self._extract_media_data(part.inline_data)
            if not mime_type or not data_buffer:
                logger.error("Could not extract mime_type or data from inline_data")
                return

            # Create filename
            extension = get_extension_for_mime_type(mime_type)
            filename = f"gemini_generated_{int(time.time())}.{extension}"

            # Check access key
            access_key = getattr(self, "access_key", None) or self.get_access_key()
            if not access_key:
                error_msg = self._create_access_key_error_message(
                    filename, mime_type, len(data_buffer)
                )
                generated_images.append(error_msg)
                return

            # Upload image
            try:
                attachment_response = await self.post_message_attachment(
                    message_id=query.message_id,
                    file_data=data_buffer,
                    filename=filename,
                    is_inline=True,
                )

                if hasattr(attachment_response, "inline_ref") and attachment_response.inline_ref:
                    image_md = f"![{filename}][{attachment_response.inline_ref}]"
                    generated_images.append(image_md)
                    logger.info(f"Successfully uploaded image: {filename}")
                else:
                    generated_images.append("[Error uploading generated image]")

            except Exception as upload_err:
                if "access_key parameter is required" in str(upload_err):
                    error_msg = self._create_access_key_error_message(
                        filename, mime_type, len(data_buffer)
                    )
                else:
                    error_msg = f"[Error uploading image: {str(upload_err)}]"
                generated_images.append(error_msg)

        except Exception as e:
            logger.error(f"Error processing image data: {str(e)}")
            generated_images.append(f"[Error processing image: {str(e)}]")

    async def _process_parts(
        self, parts, query: QueryRequest, generated_text: list, generated_images: list
    ) -> None:
        """Process all parts from response (text and images)."""
        for part in parts:
            # Handle text parts
            try:
                if hasattr(part, "text") and part.text:
                    text = part.text.strip()
                    if text:
                        generated_text.append(text)
            except ValueError:
                pass  # Might be an image part

            # Handle image parts
            if hasattr(part, "inline_data") and part.inline_data:
                await self._process_image_part(part, query, generated_images)

    async def _generate_image(
        self, prompt: str, query: QueryRequest
    ) -> AsyncGenerator[PartialResponse, None]:
        """Generate an image based on the text prompt using Gemini API with conversation context."""
        try:
            import google.generativeai as genai

            # Get and configure API key
            api_key = get_api_key("GOOGLE_API_KEY")
            if not api_key:
                import os

                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    yield PartialResponse(text="Error: Google API key is not configured.")
                    return

            genai.configure(api_key=api_key)

            # Send status message
            yield PartialResponse(
                text=f'🖌️ Generating image from prompt: "{prompt}"\n\nPlease wait a moment...'
            )

            # Initialize model
            model = genai.GenerativeModel(model_name=self.model_name)

            # Extract any image attachments from the query
            attachments = self._extract_attachments(query)
            media_parts = self._prepare_media_parts(attachments)

            # Get conversation history for multi-turn support (only for text-only conversations)
            chat_history = self._format_chat_history(query) if not media_parts else None

            # Prepare content with conversation context and attachments
            if media_parts:
                # Multimodal content (text + image attachments)
                logger.info(f"Multimodal image generation with {len(media_parts)} attachments")

                # For multimodal, use the _prepare_content method from base class
                contents = self._prepare_content(prompt, media_parts, chat_history)

                # Status message for image editing
                yield PartialResponse(
                    text="\n\n🎨 **Image editing mode**: Processing your image with the prompt..."
                )
            elif chat_history and len(chat_history) > 1:
                # Multi-turn conversation - include history (text-only)
                logger.info(f"Multi-turn conversation detected with {len(chat_history)} messages")

                # Build conversation history with the new prompt as the latest message
                conversation_parts = chat_history[:-1]  # All previous messages
                current_message = {"role": "user", "parts": [{"text": prompt}]}
                contents = conversation_parts + [current_message]

                # Status message for multi-turn
                yield PartialResponse(
                    text="\n\n🔄 **Multi-turn editing**: Building on our previous conversation..."
                )
            else:
                # Single-turn conversation (text-only)
                contents = prompt

            # Try with proper config first, fallback to simple approach
            try:
                from google.generativeai import types

                if hasattr(types, "GenerateContentConfig"):
                    config = types.GenerateContentConfig(
                        response_modalities=["TEXT", "IMAGE"], temperature=1.0
                    )
                    response = model.generate_content(contents, config=config, stream=False)
                else:
                    raise ImportError("Fallback to older API")
            except (ImportError, AttributeError, Exception):
                # Fallback to older API
                response = model.generate_content(
                    contents,
                    generation_config={
                        "response_modalities": ["TEXT", "IMAGE"],
                        "temperature": 1.0,
                    },
                    stream=False,
                )

            # Process response
            generated_text = []
            generated_images = []

            # Check for candidates (newer API) or parts (older API)
            if hasattr(response, "candidates") and response.candidates:
                for candidate in response.candidates:
                    if (
                        hasattr(candidate, "content")
                        and candidate.content
                        and hasattr(candidate.content, "parts")
                    ):
                        await self._process_parts(
                            candidate.content.parts, query, generated_text, generated_images
                        )
            elif hasattr(response, "parts") and response.parts:
                await self._process_parts(response.parts, query, generated_text, generated_images)
            else:
                yield PartialResponse(
                    text="⚠️ Received an empty response from Gemini. Please try again."
                )
                return

            # Yield results
            if generated_text:
                yield PartialResponse(text="\n\n".join(generated_text))

            for image_md in generated_images:
                yield PartialResponse(text="\n\n" + image_md)

            # If no images were generated, provide helpful suggestions
            if not generated_images:
                yield PartialResponse(
                    text=(
                        "\n\nNo images were generated. Please try a different prompt:\n\n"
                        "- Be more specific (e.g. 'a photo of a red apple on a wooden table')\n"
                        "- Include style details (e.g. 'digital art style', 'photorealistic')\n"
                        "- Try simpler subjects (landscapes, objects, animals)"
                    )
                )

        except Exception as e:
            logger.error(f"Error generating image: {str(e)}")
            yield PartialResponse(text=f"Error generating image: {str(e)}")

    async def get_settings(self, settings_request: SettingsRequest) -> SettingsResponse:
        """Get bot settings."""
        return SettingsResponse(
            allow_attachments=True,
            expand_text_attachments=True,
        )

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        """Process the query and generate a response."""
        try:
            user_message = self._extract_message(query)

            # Check API key
            api_key = get_api_key("GOOGLE_API_KEY")
            if not api_key:
                yield PartialResponse(text="Error: Google API key is not configured.")
                return

            # Handle bot info request
            if user_message.lower().strip() == "bot info":
                metadata = self._get_bot_metadata()
                metadata.update(
                    {
                        "model_name": self.model_name,
                        "supports_image_generation": self.supports_image_generation,
                    }
                )
                yield PartialResponse(
                    text=f"Gemini Image Generation Bot\nModel: {self.model_name}\nCapabilities: Image generation from text prompts"
                )
                return

            # Handle help request
            if user_message.lower().startswith(("help", "info", "what can you do", "capabilities")):
                yield PartialResponse(
                    text=(
                        "I can generate images based on your text descriptions. "
                        "Simply describe the image you want to generate, and I'll create it for you.\n\n"
                        "🎨 **Features:**\n"
                        "- **Image Generation**: Create new images from text descriptions\n"
                        "- **Multi-turn Editing**: Continue our conversation to refine images\n"
                        "- **Iterative Improvements**: Ask me to modify previously generated images\n\n"
                        "📝 **Example prompts:**\n"
                        "- A serene mountain lake at sunset with pine trees\n"
                        "- A futuristic cityscape with flying cars and neon lights\n"
                        "- A cute cartoon cat wearing a space helmet\n\n"
                        "🔄 **Multi-turn editing examples:**\n"
                        '- "Make the cat bigger"\n'
                        '- "Change the background to a sunset"\n'
                        '- "Add a rainbow in the sky"\n\n'
                        "Please note that I cannot generate images that violate Google's content policies."
                    )
                )
                return

            # Process as image generation request
            async for response in self._generate_image(user_message, query):
                yield response

        except Exception as e:
            logger.error(f"Error in get_response: {str(e)}")
            yield PartialResponse(text=f"Error: {str(e)}")

import json
import logging
import time
from typing import Any, AsyncGenerator, Dict, Optional, Union

from fastapi_poe.types import (
    MetaResponse,
    PartialResponse,
    QueryRequest,
    SettingsRequest,
    SettingsResponse, 
)

from utils.api_keys import get_api_key
from .gemini_core.base_bot import GeminiBaseBot
from .gemini_core.utils import get_extension_for_mime_type

# Get the logger
logger = logging.getLogger(__name__)


class GeminiImageGenerationBot(GeminiBaseBot):
    """Gemini Image Generation bot that creates images from text prompts."""

    model_name = "gemini-2.0-flash-preview-image-generation"
    bot_name = "GeminiImageGenerationBot"
    bot_description = (
        "Generates images from text descriptions using Gemini's image generation capabilities."
    )
    supports_image_generation = True  # Enable image generation

    # Prompt templates help guide the model to generate better images
    image_prompt_template = (
        "{prompt}\n\n"
        "Please generate a high-quality, visually detailed image based on this description. "
        "Include vibrant colors, clear subjects, and good composition."
    )

    async def get_settings(self, settings_request: SettingsRequest) -> SettingsResponse: # Added SettingsRequest type hint
        """Get bot settings including the rate card.

        Args:
            settings_request: The settings request

        Returns:
            Settings response with rate card and cost label
        """
        # from fastapi_poe.types import SettingsResponse # Removed redundant import

        # Create a new settings response
        settings = SettingsResponse(
            allow_attachments=True,
            expand_text_attachments=True,
            rate_card="100 points / image",
            cost_label="Image Generation Cost",
        )

        return settings

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
                # Try to get from environment directly as a fallback
                import os

                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    yield PartialResponse(
                        text="Error: Google API key is not configured. Please set the GOOGLE_API_KEY environment variable."
                    )
                    return
                logger.info("Found GOOGLE_API_KEY in direct environment")

            # Configure the API key
            genai.configure(api_key=api_key)
            logger.info(f"Configured Gemini with API key: {api_key[:5]}... (truncated)")

            # Send a status message
            yield PartialResponse(
                text=f'🖌️ Generating image from prompt: "{prompt}"\n\nPlease wait a moment...'
            )

            # Store all generated images/text
            generated_text = []
            generated_images = []

            # Apply the prompt template to enhance image generation
            enhanced_prompt = self.image_prompt_template.format(prompt=prompt)

            # Make API request with enhanced prompt
            logger.info(f"Making image generation request with original prompt: {prompt}")
            logger.info(f"Enhanced prompt: {enhanced_prompt}")
            logger.info(f"Using model: {self.model_name}")

            try:
                # Check if the newer API version with GenerationConfig.response_modalities is available
                has_newer_api = False
                try:
                    from google.generativeai import types

                    if hasattr(types, "GenerationConfig") and hasattr(
                        types.GenerationConfig, "response_modalities"
                    ):
                        has_newer_api = True
                except (ImportError, AttributeError):
                    logger.info(
                        "Newer API with types.GenerationConfig.response_modalities not available"
                    )

                # Initialize model
                model = genai.GenerativeModel(model_name=self.model_name)

                # Log API version
                if hasattr(genai, "__version__"):
                    logger.info(f"Google Generative AI version: {genai.__version__}")

                logger.info("Calling generate_content with response modalities configuration")

                # Try several different approaches to handle different API versions
                response = None
                error_messages = []

                # Method 1: Try with the generation_config parameter directly (works with newer API)
                if not response:
                    try:
                        logger.info("Trying method 1: Direct generation_config parameter")

                        if has_newer_api:
                            # Use the new API with GenerationConfig
                            generation_config = types.GenerationConfig(
                                response_modalities=["TEXT", "IMAGE"], temperature=1.0
                            )
                            logger.info(f"Using types.GenerationConfig: {generation_config}")

                            response = model.generate_content(
                                enhanced_prompt, generation_config=generation_config, stream=False
                            )
                        else:
                            # For older API, try with direct params to generate_content
                            response = model.generate_content(
                                enhanced_prompt,
                                generation_config={
                                    "response_modalities": ["TEXT", "IMAGE"],
                                    "temperature": 1.0,
                                },
                                stream=False,
                            )

                        logger.info("Method 1 succeeded")
                    except Exception as e:
                        error_message = f"Method 1 failed: {str(e)}"
                        logger.warning(error_message)
                        error_messages.append(error_message)

                # Method 2: Try with kwargs approach
                if not response:
                    try:
                        logger.info("Trying method 2: kwargs approach with generation_config")

                        response = model.generate_content(
                            enhanced_prompt,
                            stream=False,
                            **{
                                "generation_config": {
                                    "response_modalities": ["TEXT", "IMAGE"],
                                    "temperature": 1.0,
                                }
                            },
                        )

                        logger.info("Method 2 succeeded")
                    except Exception as e:
                        error_message = f"Method 2 failed: {str(e)}"
                        logger.warning(error_message)
                        error_messages.append(error_message)

                # Method 3: Try with system_instruction
                if not response:
                    try:
                        logger.info("Trying method 3: Using system instruction")

                        # Create a stronger prompt that explicitly asks for an image
                        stronger_prompt = (
                            f"{enhanced_prompt}\n\n"
                            "IMPORTANT: You MUST generate an image based on this description. "
                            "The image should be high quality, detailed, and visually appealing."
                        )

                        # Try with system instruction
                        response = model.generate_content(
                            stronger_prompt,
                            stream=False,
                            **{
                                "generation_config": {
                                    "temperature": 1.0,
                                    "response_modalities": ["TEXT", "IMAGE"],
                                },
                                "system_instruction": "You are a bot that MUST generate images in response to prompts.",
                            },
                        )

                        logger.info("Method 3 succeeded")
                    except Exception as e:
                        error_message = f"Method 3 failed: {str(e)}"
                        logger.warning(error_message)
                        error_messages.append(error_message)

                # Method 4: Try with a more direct prompt that mentions image creation explicitly
                if not response:
                    try:
                        logger.info("Trying method 4: More direct prompt approach")

                        # Very explicit prompt for image generation
                        direct_prompt = f"Create and generate an image of: {prompt}"

                        # Try with most basic config
                        response = model.generate_content(direct_prompt, stream=False)

                        logger.info("Method 4 succeeded")
                    except Exception as e:
                        error_message = f"Method 4 failed: {str(e)}"
                        logger.warning(error_message)
                        error_messages.append(error_message)

                # If we still don't have a response, try one last approach with just the basic prompt
                if not response:
                    logger.info("Trying final fallback approach with just the prompt")
                    response = model.generate_content(prompt, stream=False)

                # If we still don't have a response, raise an exception with all the errors we encountered
                if not response:
                    raise Exception(f"All methods failed: {', '.join(error_messages)}")

                logger.info(f"Response received, type: {type(response).__name__}")
                logger.info(f"Response attributes: {dir(response)}")

                # Dump the full response structure for debugging
                if hasattr(response, "__dict__"):
                    logger.info(f"Response __dict__: {str(response.__dict__)}")

                # Log all aspects of the response
                if hasattr(response, "text"):
                    logger.info(f"Response text: {response.text[:100]}...")

                if hasattr(response, "parts"):
                    logger.info(f"Response has {len(response.parts)} parts")
                    for i, part in enumerate(response.parts):
                        logger.info(f"Part {i} type: {type(part).__name__}")
                        logger.info(f"Part {i} attributes: {dir(part)}")

                        if hasattr(part, "text") and part.text:
                            logger.info(f"Part {i} text: {part.text[:50]}...")

                        if hasattr(part, "inline_data"):
                            logger.info(f"Part {i} has inline_data")
                            inline_data = part.inline_data
                            logger.info(f"Part {i} inline_data type: {type(inline_data).__name__}")

                            # Print the inline_data content (safely)
                            if isinstance(inline_data, dict):
                                logger.info(f"Part {i} inline_data dict: {inline_data}")
                            elif hasattr(inline_data, "__dict__"):
                                logger.info(f"Part {i} inline_data dict: {inline_data.__dict__}")

                            # Try to extract mime_type
                            if hasattr(inline_data, "mime_type"):
                                logger.info(f"Part {i} mime_type: {inline_data.mime_type}")
                            elif isinstance(inline_data, dict) and "mime_type" in inline_data:
                                logger.info(
                                    f"Part {i} mime_type (dict): {inline_data['mime_type']}"
                                )

                            # Try to extract data
                            if hasattr(inline_data, "data"):
                                logger.info(f"Part {i} data length: {len(inline_data.data)} bytes")
                            elif isinstance(inline_data, dict) and "data" in inline_data:
                                logger.info(
                                    f"Part {i} data length (dict): {len(inline_data['data'])} bytes"
                                )

                if hasattr(response, "candidates"):
                    logger.info(f"Response has {len(response.candidates)} candidates")
                    for i, candidate in enumerate(response.candidates):
                        logger.info(f"Candidate {i} type: {type(candidate).__name__}")
                        logger.info(f"Candidate {i} attributes: {dir(candidate)}")

                        if hasattr(candidate, "content"):
                            content = candidate.content
                            logger.info(f"Candidate {i} content type: {type(content).__name__}")
                            logger.info(f"Candidate {i} content attributes: {dir(content)}")

                            if hasattr(content, "parts"):
                                logger.info(f"Candidate {i} content has {len(content.parts)} parts")

                # Process the response logic...
                try:
                    logger.info("Beginning response processing with detailed debug info")
                    logger.info(f"Response type: {type(response).__name__}")
                    logger.info(f"Response dir: {dir(response)}")

                    if hasattr(response, "__dict__"):
                        logger.info(f"Response __dict__: {response.__dict__}")

                    # Try to handle text directly (may fail if there's image data)
                    try:
                        if hasattr(response, "text"):
                            logger.info("Response has text attribute, trying to access it safely")
                            try:
                                text = response.text
                                logger.info(
                                    f"Successfully accessed text: {text[:50] if text else 'empty'}"
                                )
                                if text:
                                    generated_text.append(text)
                                    logger.info(f"Found direct text: {text[:50]}...")
                            except Exception as text_err:
                                logger.error(f"Error accessing response.text: {str(text_err)}")
                                logger.error(f"Error type: {type(text_err).__name__}")
                    except ValueError as text_err:
                        # This is expected sometimes when there's only image data
                        logger.info(f"Could not get text from response: {str(text_err)}")

                    # Check for candidates structure (newer API)
                    if hasattr(response, "candidates") and response.candidates:
                        logger.info(f"Response has {len(response.candidates)} candidates")

                        for candidate_idx, candidate in enumerate(response.candidates):
                            logger.info(f"Processing candidate {candidate_idx+1}")

                            if hasattr(candidate, "content") and candidate.content:
                                content = candidate.content
                                logger.info(f"Content attributes: {dir(content)}")

                                # Process parts in content
                                if hasattr(content, "parts") and content.parts:
                                    logger.info(f"Content has {len(content.parts)} parts")

                                    for part_idx, part in enumerate(content.parts):
                                        logger.info(
                                            f"Processing part {part_idx+1}, type: {type(part).__name__}"
                                        )

                                        # Handle text parts
                                        try:
                                            if hasattr(part, "text") and part.text:
                                                text = part.text
                                                if text:
                                                    generated_text.append(text)
                                                    logger.info(f"Found part text: {text[:50]}...")
                                        except ValueError:
                                            # This is fine, might be an image part
                                            pass

                                        # Handle image parts
                                        if (
                                            hasattr(part, "inline_data")
                                            and part.inline_data is not None
                                        ):
                                            try:
                                                inline_data = part.inline_data
                                                logger.info(
                                                    f"Found inline_data: {type(inline_data).__name__}"
                                                )
                                                logger.info(
                                                    f"Inline data attributes: {dir(inline_data)}"
                                                )

                                                # Get mime type and data using different API formats
                                                mime_type = None
                                                data_buffer = None

                                                # Try to get mime_type first
                                                if hasattr(inline_data, "mime_type"):
                                                    mime_type = inline_data.mime_type
                                                elif (
                                                    isinstance(inline_data, dict)
                                                    and "mime_type" in inline_data
                                                ):
                                                    mime_type = inline_data["mime_type"]

                                                # Try different methods to get data
                                                if hasattr(inline_data, "data"):
                                                    # Old API format
                                                    data_buffer = inline_data.data
                                                elif (
                                                    isinstance(inline_data, dict)
                                                    and "data" in inline_data
                                                ):
                                                    # Dictionary format
                                                    data_buffer = inline_data["data"]
                                                elif hasattr(inline_data, "get_bytes") and callable(
                                                    inline_data.get_bytes
                                                ):
                                                    # New API format with get_bytes method
                                                    data_buffer = inline_data.get_bytes()
                                                    if not mime_type:
                                                        mime_type = (
                                                            "image/png"  # Default if not specified
                                                        )

                                                if not mime_type or not data_buffer:
                                                    logger.error(
                                                        "Could not extract mime_type or data from inline_data"
                                                    )
                                                    continue

                                                logger.info(
                                                    f"Successfully extracted image data with mime type: {mime_type}"
                                                )
                                            except Exception as extract_err:
                                                logger.error(
                                                    f"Error extracting image data: {str(extract_err)}"
                                                )
                                                continue

                                            # Now we have mime_type and data_buffer - continue with upload process
                                            logger.info(
                                                f"Extracted {len(data_buffer)} bytes for processing"
                                            )

                                            # Determine file extension
                                            extension = get_extension_for_mime_type(mime_type) # Changed here
                                            filename = (
                                                f"gemini_generated_{int(time.time())}.{extension}"
                                            )

                                            # Upload to Poe
                                            logger.info(
                                                f"Uploading file {filename} with MIME type {mime_type}"
                                            )
                                            attachment_upload_response = (
                                                await self.post_message_attachment(
                                                    message_id=query.message_id,
                                                    file_data=data_buffer,
                                                    filename=filename,
                                                    is_inline=True,
                                                )
                                            )

                                            # Create markdown for display
                                            if (
                                                hasattr(attachment_upload_response, "inline_ref")
                                                and attachment_upload_response.inline_ref
                                            ):
                                                image_md = f"![{filename}][{attachment_upload_response.inline_ref}]"
                                                generated_images.append(image_md)
                                                logger.info(
                                                    f"Successfully processed and uploaded image with ref: {attachment_upload_response.inline_ref}"
                                                )
                                            else:
                                                logger.error(
                                                    "Error uploading image: No inline_ref in response"
                                                )
                                                generated_images.append(
                                                    "[Error uploading generated image]"
                                                )

                    # Fallback to older parts structure if no candidates
                    elif hasattr(response, "parts"):
                        logger.info(f"Response has {len(response.parts)} parts")

                        for part_idx, part in enumerate(response.parts):
                            logger.info(f"Processing part {part_idx+1}")

                            # Handle text content
                            try:
                                if hasattr(part, "text") and part.text:
                                    text = part.text.strip()
                                    if text:
                                        generated_text.append(text)
                                        logger.info(f"Found text part: {text[:30]}...")
                            except ValueError:
                                # This is fine, might be an image part
                                pass

                            # Handle image content
                            if hasattr(part, "inline_data") and part.inline_data is not None:
                                try:
                                    # Extract image data
                                    inline_data = part.inline_data
                                    logger.info(f"Inline data attributes: {dir(inline_data)}")

                                    # Get data using different API formats
                                    mime_type = None
                                    data_buffer = None

                                    # Try to get mime_type first
                                    if hasattr(inline_data, "mime_type"):
                                        mime_type = inline_data.mime_type
                                    elif (
                                        isinstance(inline_data, dict) and "mime_type" in inline_data
                                    ):
                                        mime_type = inline_data["mime_type"]

                                    # Try different methods to get data
                                    if hasattr(inline_data, "data"):
                                        data_buffer = inline_data.data
                                    elif isinstance(inline_data, dict) and "data" in inline_data:
                                        # Dictionary format
                                        data_buffer = inline_data["data"]
                                    elif hasattr(inline_data, "get_bytes") and callable(
                                        inline_data.get_bytes
                                    ):
                                        data_buffer = inline_data.get_bytes()
                                        if not mime_type:
                                            mime_type = "image/png"  # Default if not specified

                                    if not mime_type or not data_buffer:
                                        logger.error(
                                            "Could not extract mime_type or data from inline_data"
                                        )
                                        continue

                                    logger.info(f"Found inline data with mime type: {mime_type}")
                                    logger.info(
                                        f"Extracted {len(data_buffer)} bytes for processing"
                                    )

                                    # Determine file extension
                                    extension = get_extension_for_mime_type(mime_type) # Changed here
                                    filename = f"gemini_generated_{int(time.time())}.{extension}"

                                    # Upload to Poe
                                    logger.info(
                                        f"Uploading file {filename} with MIME type {mime_type}"
                                    )
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
                                        logger.info(
                                            f"Successfully processed and uploaded image with ref: {attachment_upload_response.inline_ref}"
                                        )
                                    else:
                                        logger.error(
                                            "Error uploading image: No inline_ref in response"
                                        )
                                        generated_images.append("[Error uploading generated image]")
                                except Exception as img_err:
                                    logger.error(f"Error processing image data: {str(img_err)}")
                                    generated_images.append(
                                        f"[Error processing image: {str(img_err)}]"
                                    )
                                    continue
                except Exception as parse_err:
                    logger.error(f"Error parsing response: {str(parse_err)}")
                    raise
            except Exception as api_err:
                # Check specifically for the "Could not convert `part.inline_data` to text" error
                error_str = str(api_err)
                logger.error(f"Error in Gemini API call: {error_str}", exc_info=True)

                if "Could not convert `part.inline_data` to text" in error_str:
                    logger.error(
                        "Detected the inline_data conversion error - providing helpful error message"
                    )
                    yield PartialResponse(
                        text="Sorry, I'm unable to generate the image at this time. The image generation model responded "
                        + "with content in a format that I couldn't process correctly.\n\n"
                        + "This is likely due to one of these reasons:\n"
                        + "1. The Google Gemini API version is not fully compatible with image generation\n"
                        + "2. The API key might not have permissions for image generation\n"
                        + "3. There might be a temporary service disruption\n\n"
                        + "You can try again later or with a different prompt."
                    )
                else:
                    yield PartialResponse(text=f"Error generating image: {error_str}\n")
                return

            # Yield the model's explanation first, if any
            if generated_text:
                yield PartialResponse(text="\n\n".join(generated_text))

            # Yield the generated images
            for image_md in generated_images:
                yield PartialResponse(text="\n\n" + image_md)

            # If no images were generated, inform the user and suggest alternative phrasing
            if not generated_images:
                suggest_text = (
                    "\n\nNo images were generated. This could be due to content policy restrictions "
                    "or a technical issue. Please try a different prompt with the following tips:\n\n"
                    "- Be more specific and descriptive (e.g. 'a photo of a red apple on a wooden table')\n"
                    "- Avoid potentially sensitive content\n"
                    "- Try simpler subjects (e.g. landscapes, objects, animals)\n"
                    "- Include details about style (e.g. 'digital art style', 'photorealistic', 'cartoon')"
                )
                yield PartialResponse(text=suggest_text)

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

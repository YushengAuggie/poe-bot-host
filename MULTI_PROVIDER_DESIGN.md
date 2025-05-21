# Multi-Provider Bot Framework Design Document

## Overview
This design extends the existing Poe Bots Framework to support multiple LLM providers (OpenAI, Google Gemini, Claude, Fireworks, Together, etc.) through a unified configuration system. Each provider must support the OpenAI-compatible API interface with potentially different URLs, API keys, and model settings.

## Requirements
1. Support multiple LLM providers with OpenAI-compatible APIs
2. Configure bots with different providers via a central configuration
3. Isolate failures (failed bot queries/launches shouldn't affect other bots)
4. Allow easy addition of new bot configurations

## Design

### 1. Provider Configuration Model
Create a configuration model for different LLM providers:

```python
# utils/providers.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    name: str
    api_base: str
    api_key_env: str  # Environment variable name to get API key
    api_version: Optional[str] = None
    organization_id: Optional[str] = None  # For OpenAI org id
    default_model: str  # Default model to use
    models: List[str] = []  # Available models for this provider
    timeout: int = 30  # Default timeout in seconds
    headers: Dict[str, str] = {}  # Custom headers if needed
    additional_params: Dict[str, Any] = {}  # Provider-specific params
```

### 2. Bot Configuration Model
Create a configuration model for bots:

```python
# utils/bot_config.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class BotConfig(BaseModel):
    """Configuration for a bot."""
    bot_name: str
    provider: str  # Provider name to use
    model: Optional[str] = None  # Override provider default model
    bot_description: str = ""
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: Optional[int] = None  # Override provider timeout
    system_message: Optional[str] = None
    settings: Dict[str, Any] = {}  # Additional bot-specific settings
```

### 3. Modified BaseBot with Provider Support
Extend the BaseBot to use provider-specific configuration:

```python
# utils/multi_provider_bot.py
from utils.base_bot import BaseBot, BotError, BotErrorNoRetry
from utils.providers import ProviderConfig
from utils.bot_config import BotConfig
import os
import logging
from typing import Dict, Any, Optional, ClassVar, Type

logger = logging.getLogger(__name__)

class MultiProviderBot(BaseBot):
    """Bot class with support for multiple LLM providers."""

    # Configuration for this bot
    bot_config: ClassVar[BotConfig] = None

    # Provider configuration
    provider_config: ClassVar[ProviderConfig] = None

    @classmethod
    def init_provider(cls, provider_config: ProviderConfig) -> None:
        """Initialize the provider configuration."""
        cls.provider_config = provider_config

    @classmethod
    def init_bot(cls, bot_config: BotConfig) -> None:
        """Initialize the bot configuration."""
        cls.bot_config = bot_config

    def __init__(self, **kwargs):
        """Initialize the bot with provider-specific configuration."""
        if not self.bot_config:
            raise ValueError("Bot configuration not initialized")
        if not self.provider_config:
            raise ValueError("Provider configuration not initialized")

        # Set bot attributes from config
        self.bot_name = self.bot_config.bot_name
        self.bot_description = self.bot_config.bot_description

        # Get API key from environment
        self.api_key = os.environ.get(self.provider_config.api_key_env, "")
        if not self.api_key:
            logger.warning(f"API key for provider {self.provider_config.name} not found")

        # Setup client parameters
        self.api_base = self.provider_config.api_base
        self.model = self.bot_config.model or self.provider_config.default_model
        self.temperature = self.bot_config.temperature
        self.max_tokens = self.bot_config.max_tokens
        self.timeout = self.bot_config.timeout or self.provider_config.timeout
        self.system_message = self.bot_config.system_message

        # Initialize the base class
        super().__init__(bot_name=self.bot_name, **kwargs)

        # Setup OpenAI-compatible client
        self._setup_client()

    def _setup_client(self):
        """Setup the OpenAI client with provider-specific configuration."""
        try:
            from openai import OpenAI

            # Configure the client
            client_params = {
                "api_key": self.api_key,
                "base_url": self.api_base,
                "timeout": self.timeout,
            }

            # Add API version if specified
            if self.provider_config.api_version:
                client_params["api_version"] = self.provider_config.api_version

            # Add headers if specified
            if self.provider_config.headers:
                client_params["default_headers"] = self.provider_config.headers

            # Add organization if specified
            if self.provider_config.organization_id:
                client_params["organization"] = self.provider_config.organization_id

            # Create the client
            self.client = OpenAI(**client_params)

        except Exception as e:
            logger.error(f"Error initializing OpenAI client for {self.bot_name}: {str(e)}")
            self.client = None
```

### 4. Provider Configuration Registry
Create a registry to manage provider configurations:

```python
# utils/provider_registry.py
from typing import Dict, List, Optional
from utils.providers import ProviderConfig

class ProviderRegistry:
    """Registry for LLM providers."""

    _providers: Dict[str, ProviderConfig] = {}

    @classmethod
    def register(cls, provider: ProviderConfig) -> None:
        """Register a provider."""
        cls._providers[provider.name] = provider

    @classmethod
    def get(cls, name: str) -> Optional[ProviderConfig]:
        """Get a provider by name."""
        return cls._providers.get(name)

    @classmethod
    def list(cls) -> List[str]:
        """List all registered providers."""
        return list(cls._providers.keys())

    @classmethod
    def load_providers(cls, config_path: str) -> None:
        """Load providers from a configuration file."""
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        for provider_data in config.get('providers', []):
            provider = ProviderConfig(**provider_data)
            cls.register(provider)
```

### 5. Bot Configuration Registry
Create a registry to manage bot configurations:

```python
# utils/bot_config_registry.py
from typing import Dict, List, Optional, Type
from utils.bot_config import BotConfig
from utils.multi_provider_bot import MultiProviderBot
from utils.provider_registry import ProviderRegistry
from utils.base_bot import BaseBot
import logging

logger = logging.getLogger(__name__)

class BotConfigRegistry:
    """Registry for bot configurations."""

    _bot_configs: Dict[str, BotConfig] = {}
    _bot_classes: Dict[str, Type[MultiProviderBot]] = {}

    @classmethod
    def register_config(cls, bot_config: BotConfig) -> None:
        """Register a bot configuration."""
        cls._bot_configs[bot_config.bot_name] = bot_config

    @classmethod
    def register_class(cls, bot_name: str, bot_class: Type[MultiProviderBot]) -> None:
        """Register a bot class."""
        cls._bot_classes[bot_name] = bot_class

    @classmethod
    def get_config(cls, name: str) -> Optional[BotConfig]:
        """Get a bot configuration by name."""
        return cls._bot_configs.get(name)

    @classmethod
    def get_class(cls, name: str) -> Optional[Type[MultiProviderBot]]:
        """Get a bot class by name."""
        return cls._bot_classes.get(name)

    @classmethod
    def list(cls) -> List[str]:
        """List all registered bot configurations."""
        return list(cls._bot_configs.keys())

    @classmethod
    def create_bot(cls, bot_name: str) -> Optional[BaseBot]:
        """Create a bot instance from configuration."""
        bot_config = cls.get_config(bot_name)
        if not bot_config:
            logger.error(f"Bot configuration not found: {bot_name}")
            return None

        # Get the provider configuration
        provider = ProviderRegistry.get(bot_config.provider)
        if not provider:
            logger.error(f"Provider not found: {bot_config.provider}")
            return None

        # Get the bot class
        bot_class = cls.get_class(bot_name)
        if not bot_class:
            logger.error(f"Bot class not found: {bot_name}")
            return None

        try:
            # Initialize the bot class with provider and bot config
            bot_class.init_provider(provider)
            bot_class.init_bot(bot_config)

            # Create the bot instance
            return bot_class()
        except Exception as e:
            logger.error(f"Error creating bot {bot_name}: {str(e)}")
            return None

    @classmethod
    def load_configs(cls, config_path: str) -> None:
        """Load bot configurations from a configuration file."""
        import yaml

        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        for bot_data in config.get('bots', []):
            bot_config = BotConfig(**bot_data)
            cls.register_config(bot_config)
```

### 6. Enhanced BotFactory
Enhance the BotFactory to create bots from configuration:

```python
# utils/enhanced_bot_factory.py
import logging
from typing import Dict, List, Type

from fastapi import FastAPI
from fastapi_poe import PoeBot, make_app

from utils.base_bot import BaseBot
from utils.bot_config_registry import BotConfigRegistry
from utils.provider_registry import ProviderRegistry

logger = logging.getLogger(__name__)

class EnhancedBotFactory:
    """Factory for creating and managing multi-provider Poe bots."""

    @staticmethod
    def create_app(allow_without_key: bool = True) -> FastAPI:
        """Create a FastAPI app with all configured bots.

        Args:
            allow_without_key: Whether to allow requests without an API key

        Returns:
            A FastAPI app with all configured bots
        """
        # Get all bot names from the registry
        bot_names = BotConfigRegistry.list()

        # Create bot instances
        bots = []
        for bot_name in bot_names:
            try:
                bot = BotConfigRegistry.create_bot(bot_name)
                if bot:
                    bots.append(bot)
                    logger.info(f"Created bot: {bot_name}")
                else:
                    logger.warning(f"Failed to create bot: {bot_name}")
            except Exception as e:
                logger.error(f"Error creating bot {bot_name}: {str(e)}")

        # Log the bots that were created
        logger.info(f"Created {len(bots)} bots out of {len(bot_names)} configurations")

        # Create and return the app
        return make_app(bots, allow_without_key=allow_without_key)

    @staticmethod
    def load_configurations(
        provider_config_path: str, bot_config_path: str
    ) -> None:
        """Load provider and bot configurations from files.

        Args:
            provider_config_path: Path to provider configuration file
            bot_config_path: Path to bot configuration file
        """
        # Load provider configurations
        ProviderRegistry.load_providers(provider_config_path)
        logger.info(f"Loaded providers: {ProviderRegistry.list()}")

        # Load bot configurations
        BotConfigRegistry.load_configs(bot_config_path)
        logger.info(f"Loaded bot configs: {BotConfigRegistry.list()}")
```

### 7. Example Provider Configuration
Create a YAML file for provider configurations:

```yaml
# config/providers.yaml
providers:
  - name: openai
    api_base: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    default_model: gpt-3.5-turbo
    models:
      - gpt-3.5-turbo
      - gpt-4
      - gpt-4-turbo
    timeout: 30

  - name: anthropic
    api_base: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
    default_model: claude-3-haiku-20240307
    models:
      - claude-3-haiku-20240307
      - claude-3-sonnet-20240229
      - claude-3-opus-20240229
    timeout: 30
    headers:
      anthropic-version: "2023-06-01"

  - name: gemini
    api_base: https://generativelanguage.googleapis.com/v1beta
    api_key_env: GEMINI_API_KEY
    default_model: gemini-pro
    models:
      - gemini-pro
      - gemini-ultra
    timeout: 30

  - name: fireworks
    api_base: https://api.fireworks.ai/inference/v1
    api_key_env: FIREWORKS_API_KEY
    default_model: llama-v3-8b-instruct
    models:
      - llama-v3-8b-instruct
      - llama-v3-70b-instruct
    timeout: 30

  - name: together
    api_base: https://api.together.xyz/v1
    api_key_env: TOGETHER_API_KEY
    default_model: mistralai/Mixtral-8x7B-Instruct-v0.1
    models:
      - mistralai/Mixtral-8x7B-Instruct-v0.1
      - togethercomputer/llama-2-70b-chat
    timeout: 30
```

### 8. Example Bot Configuration
Create a YAML file for bot configurations:

```yaml
# config/bots.yaml
bots:
  - bot_name: GPT35Bot
    provider: openai
    model: gpt-3.5-turbo
    bot_description: "A bot powered by GPT-3.5 Turbo"
    temperature: 0.7
    max_tokens: 1000
    system_message: "You are a helpful assistant."

  - bot_name: ClaudeBot
    provider: anthropic
    model: claude-3-haiku-20240307
    bot_description: "A bot powered by Claude 3 Haiku"
    temperature: 0.5
    max_tokens: 2000
    system_message: "You are Claude, a helpful AI assistant."

  - bot_name: GeminiBot
    provider: gemini
    model: gemini-pro
    bot_description: "A bot powered by Gemini Pro"
    temperature: 0.8
    max_tokens: 1500
    system_message: "You are a helpful AI assistant."

  - bot_name: LlamaBot
    provider: fireworks
    model: llama-v3-8b-instruct
    bot_description: "A bot powered by Llama 3"
    temperature: 0.7
    max_tokens: 1000
    system_message: "You are a helpful AI assistant."

  - bot_name: MixtralBot
    provider: together
    model: mistralai/Mixtral-8x7B-Instruct-v0.1
    bot_description: "A bot powered by Mixtral"
    temperature: 0.6
    max_tokens: 1200
    system_message: "You are a helpful AI assistant."
```

### 9. Implementation of a Generic Multi-provider Chat Bot
Create a generic chat bot class that uses the multi-provider framework:

```python
# bots/llm_chat_bot.py
import json
import logging
from typing import AsyncGenerator, Dict, Union

from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest

from utils.multi_provider_bot import MultiProviderBot, BotError, BotErrorNoRetry

logger = logging.getLogger(__name__)

class LLMChatBot(MultiProviderBot):
    """
    A generic chat bot that uses different LLM providers.

    This bot can be configured to use any provider that supports
    the OpenAI-compatible API interface.
    """

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        """Process the query and generate a response using the configured provider."""
        try:
            # Extract the query contents
            user_message = self._extract_message(query)

            # Log the extracted message
            logger.debug(f"[{self.bot_name}] Received message: {user_message}")

            # Add metadata about the bot if requested
            if user_message.lower().strip() == "bot info":
                metadata = self._get_extended_metadata()
                yield PartialResponse(text=json.dumps(metadata, indent=2))
                return

            # Ensure client is initialized
            if not hasattr(self, "client") or self.client is None:
                raise BotErrorNoRetry("LLM provider client not initialized")

            # Create messages payload
            messages = self._build_messages(user_message)

            # Call the API in streaming mode
            try:
                stream = True
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=stream
                )

                # Process streaming response
                if stream:
                    for chunk in response:
                        if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
                            content = chunk.choices[0].delta.content
                            if content:
                                yield PartialResponse(text=content)
                else:
                    # Non-streaming response
                    content = response.choices[0].message.content
                    yield PartialResponse(text=content)

            except Exception as e:
                logger.error(f"[{self.bot_name}] API error: {str(e)}")
                raise BotError(f"API error: {str(e)}")

        except BotError:
            # Let the parent class handle retryable errors
            raise
        except BotErrorNoRetry:
            # Let the parent class handle non-retryable errors
            raise
        except Exception as e:
            # Let the parent class handle unexpected errors
            logger.error(f"[{self.bot_name}] Unexpected error: {str(e)}", exc_info=True)
            async for resp in super().get_response(query):
                yield resp

    def _build_messages(self, user_message: str) -> list:
        """Build the messages list for the API request."""
        messages = []

        # Add system message if provided
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})

        # Add user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def _get_extended_metadata(self) -> Dict:
        """Get extended metadata about the bot including provider info."""
        metadata = self._get_bot_metadata()

        # Add provider information
        metadata["provider"] = {
            "name": self.provider_config.name,
            "model": self.model,
            "api_base": self.provider_config.api_base,
        }

        # Add settings
        metadata["settings"] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }

        return metadata
```

### 10. Updated app.py
Update the application entry point to use the enhanced bot factory:

```python
# app.py
import logging
import os

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from modal import App, Image, asgi_app

from utils.enhanced_bot_factory import EnhancedBotFactory
from utils.config import settings
from utils.bot_config_registry import BotConfigRegistry
from utils.provider_registry import ProviderRegistry

# Import chat bot implementation
from bots.llm_chat_bot import LLMChatBot

# Get configured logger
logger = logging.getLogger("poe_bots.app")

__version__ = "2.0.0"

# Default configuration paths
DEFAULT_PROVIDER_CONFIG = "config/providers.yaml"
DEFAULT_BOT_CONFIG = "config/bots.yaml"

def create_api(allow_without_key: bool = settings.ALLOW_WITHOUT_KEY) -> FastAPI:
    """Create and configure the FastAPI app with all available bots.

    Args:
        allow_without_key: Whether to allow requests without an API key

    Returns:
        Configured FastAPI app
    """
    # Load provider and bot configurations
    provider_config_path = os.environ.get("PROVIDER_CONFIG_PATH", DEFAULT_PROVIDER_CONFIG)
    bot_config_path = os.environ.get("BOT_CONFIG_PATH", DEFAULT_BOT_CONFIG)

    try:
        # Load configurations
        EnhancedBotFactory.load_configurations(provider_config_path, bot_config_path)

        # Register bot classes for each bot configuration
        for bot_name in BotConfigRegistry.list():
            BotConfigRegistry.register_class(bot_name, LLMChatBot)

        # Create a FastAPI app with all the bots
        api = EnhancedBotFactory.create_app(allow_without_key=allow_without_key)

    except Exception as e:
        logger.error(f"Error creating API: {str(e)}", exc_info=True)
        # Create an empty app if configuration failed
        from fastapi import FastAPI
        api = FastAPI()

    # Add custom error handling
    @api.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception occurred:")
        return JSONResponse(
            status_code=500,
            content={"error": "An internal server error occurred", "detail": str(exc)}
        )

    # Add a health check endpoint
    @api.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "ok",
            "version": __version__,
            "providers": ProviderRegistry.list(),
            "bots": BotConfigRegistry.list(),
            "bot_count": len(BotConfigRegistry.list()),
            "environment": {
                "debug": settings.DEBUG,
                "log_level": settings.LOG_LEVEL,
                "allow_without_key": settings.ALLOW_WITHOUT_KEY
            }
        }

    # Add a bot list endpoint
    @api.get("/bots")
    async def list_bots():
        """List all available bots."""
        result = {}
        for bot_name in BotConfigRegistry.list():
            config = BotConfigRegistry.get_config(bot_name)
            if config:
                result[bot_name] = {
                    "description": config.bot_description,
                    "provider": config.provider,
                    "model": config.model
                }
        return result

    # Add a providers endpoint
    @api.get("/providers")
    async def list_providers():
        """List all available providers."""
        result = {}
        for provider_name in ProviderRegistry.list():
            provider = ProviderRegistry.get(provider_name)
            if provider:
                result[provider_name] = {
                    "api_base": provider.api_base,
                    "default_model": provider.default_model,
                    "models": provider.models
                }
        return result

    return api
```

## Deployment Instructions

1. Create configuration files:
   - Create `config/providers.yaml` with provider configurations
   - Create `config/bots.yaml` with bot configurations

2. Set environment variables for API keys:
   - `OPENAI_API_KEY`
   - `ANTHROPIC_API_KEY`
   - `GEMINI_API_KEY`
   - `FIREWORKS_API_KEY`
   - `TOGETHER_API_KEY`

3. Install additional dependencies:
   - Add `pyyaml` to requirements.txt
   - Add the OpenAI SDK (latest version supports all providers)

4. Run the application:
   - `python run_local.py`

## Error Handling Strategy

1. Bot Initialization Errors:
   - Log errors but continue with other bots
   - Failed bots will not be registered in the app

2. API Call Errors:
   - Use BotError (retryable) for temporary issues
   - Use BotErrorNoRetry for permanent issues
   - Return readable error messages to users

3. Configuration Errors:
   - Log errors and provide clear messages
   - Fall back to default settings when possible

4. Environment Variable Errors:
   - Log warnings when API keys are missing
   - Continue initialization but bot will return error responses

## Testing Strategy

1. Create mock providers for testing
2. Add unit tests for configuration loading
3. Add integration tests for bot creation and responses
4. Test error handling with forced failures

## Future Enhancements

1. Support for provider-specific features via the additional_params field
2. Web UI for managing bot configurations
3. Dynamic model switching based on request parameters
4. Health monitoring for providers
5. Caching for API responses

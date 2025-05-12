# Final Multi-Provider Bot Framework Implementation Plan

After analyzing the latest changes to the codebase, including the new `chatgpt.py` and `gemini.py` bots and the API key management system, I've revised the implementation plan to build on these foundations rather than duplicating them.

## Current State Analysis

The codebase already has:
1. **API Key Management System**: `utils/api_keys.py` provides a unified way to access API keys from both environment variables and Modal secrets
2. **Provider-Specific Bots**: Individual bot implementations for OpenAI (`chatgpt.py`) and Gemini (`gemini.py`)
3. **Modal Integration**: Support for using API keys in Modal deployments

## Enhanced Design Approach

We'll build on the existing foundations to create a more scalable and configurable multi-provider system:

1. **Provider Configuration**: Add a YAML-based configuration system for defining provider details
2. **Bot Configuration**: Allow bots to be created from configuration files
3. **Provider Registry**: Centralized management of available providers
4. **Unified Bot Base Class**: A common base class that works with multiple providers
5. **Enhanced Bot Factory**: A factory that can create bots from configurations

## Implementation Plan

### Phase 1: Provider Registry and Configuration (Days 1-2)

1. Create a Provider Configuration Model:

```python
# utils/provider_config.py
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    name: str  # Provider name (e.g., "openai", "google", "anthropic")
    api_base: str  # API endpoint base URL
    api_key_env: str  # Environment variable name for the API key
    default_model: str  # Default model to use
    models: List[str] = []  # Available models
    api_version: Optional[str] = None  # Optional API version
    headers: Dict[str, str] = {}  # Optional custom headers
    params: Dict[str, Any] = {}  # Optional additional parameters
```

2. Create a Provider Registry:

```python
# utils/provider_registry.py
import yaml
import logging
import os
from typing import Dict, List, Optional
from utils.provider_config import ProviderConfig

logger = logging.getLogger(__name__)

class ProviderRegistry:
    """Registry for managing LLM providers."""

    _instance = None
    _providers: Dict[str, ProviderConfig] = {}

    def __new__(cls):
        """Singleton pattern to ensure a single registry instance."""
        if cls._instance is None:
            cls._instance = super(ProviderRegistry, cls).__new__(cls)
            cls._instance._providers = {}
        return cls._instance

    def register(self, provider: ProviderConfig) -> None:
        """Register a provider configuration."""
        self._providers[provider.name] = provider
        logger.info(f"Registered provider: {provider.name}")

    def get(self, name: str) -> Optional[ProviderConfig]:
        """Get a provider by name."""
        return self._providers.get(name)

    def list(self) -> List[str]:
        """List all registered provider names."""
        return list(self._providers.keys())

    def load_from_file(self, config_path: str) -> None:
        """Load providers from a YAML configuration file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            for provider_data in config.get('providers', []):
                provider = ProviderConfig(**provider_data)
                self.register(provider)

            logger.info(f"Loaded {len(config.get('providers', []))} providers from {config_path}")
        except Exception as e:
            logger.error(f"Error loading providers from {config_path}: {str(e)}")
            raise
```

### Phase 2: Unified Provider Client (Days 3-4)

Create a unified client that works with various providers:

```python
# utils/provider_client.py
import logging
from typing import Any, Dict, Optional

from utils.api_keys import get_api_key
from utils.provider_config import ProviderConfig

logger = logging.getLogger(__name__)

class ProviderClient:
    """A unified client for interacting with LLM providers."""

    def __init__(self, provider: ProviderConfig):
        """Initialize the provider client."""
        self.provider = provider
        self.api_key = None
        self.client = None
        self.initialized = False

    def initialize(self) -> bool:
        """Initialize the client with the provider's API key."""
        try:
            # Get the API key
            self.api_key = get_api_key(self.provider.api_key_env)

            # Initialize based on provider type
            if self.provider.name == "openai" or "openai" in self.provider.api_base:
                self._init_openai_client()
            elif self.provider.name == "anthropic" or "anthropic" in self.provider.api_base:
                self._init_anthropic_client()
            elif self.provider.name == "gemini" or "google" in self.provider.name:
                self._init_gemini_client()
            elif self.provider.name == "fireworks" or "fireworks" in self.provider.api_base:
                self._init_openai_compatible_client()
            elif self.provider.name == "together" or "together" in self.provider.api_base:
                self._init_openai_compatible_client()
            else:
                # Default to OpenAI-compatible client for unknown providers
                self._init_openai_compatible_client()

            self.initialized = True
            return True

        except Exception as e:
            logger.error(f"Failed to initialize client for provider {self.provider.name}: {str(e)}")
            self.initialized = False
            return False

    def _init_openai_client(self) -> None:
        """Initialize an OpenAI client."""
        try:
            from openai import OpenAI

            client_params = {
                "api_key": self.api_key,
                "base_url": self.provider.api_base,
            }

            # Add optional parameters
            if self.provider.api_version:
                client_params["api_version"] = self.provider.api_version

            if self.provider.headers:
                client_params["default_headers"] = self.provider.headers

            self.client = OpenAI(**client_params)
            logger.info(f"Initialized OpenAI client for provider {self.provider.name}")

        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai>=1.0.0")
            raise

    def _init_anthropic_client(self) -> None:
        """Initialize an Anthropic client."""
        try:
            # Prefer using OpenAI-compatible client if possible
            try:
                self._init_openai_compatible_client()
                return
            except:
                pass

            # Fall back to native Anthropic client if needed
            try:
                from anthropic import Anthropic

                self.client = Anthropic(api_key=self.api_key)
                logger.info(f"Initialized Anthropic client for provider {self.provider.name}")
            except ImportError:
                logger.error("Anthropic package not installed. Install with: pip install anthropic")
                raise

        except Exception as e:
            logger.error(f"Failed to initialize Anthropic client: {str(e)}")
            raise

    def _init_gemini_client(self) -> None:
        """Initialize a Google Gemini client."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)

            # Create a placeholder client to store the model name
            class GeminiClient:
                def __init__(self, model_name):
                    self.model_name = model_name
                    self.genai = genai

            self.client = GeminiClient(self.provider.default_model)
            logger.info(f"Initialized Gemini client for provider {self.provider.name}")

        except ImportError:
            logger.error("Google GenerativeAI package not installed. Install with: pip install google-generativeai")
            raise

    def _init_openai_compatible_client(self) -> None:
        """Initialize an OpenAI-compatible client (for Fireworks, Together, etc.)."""
        try:
            from openai import OpenAI

            client_params = {
                "api_key": self.api_key,
                "base_url": self.provider.api_base,
            }

            # Add optional parameters
            if self.provider.api_version:
                client_params["api_version"] = self.provider.api_version

            if self.provider.headers:
                client_params["default_headers"] = self.provider.headers

            self.client = OpenAI(**client_params)
            logger.info(f"Initialized OpenAI-compatible client for provider {self.provider.name}")

        except ImportError:
            logger.error("OpenAI package not installed. Install with: pip install openai>=1.0.0")
            raise

    async def generate_text(
        self,
        prompt: str,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        stream: bool = True,
        **kwargs
    ) -> Any:
        """Generate text from the LLM provider."""
        if not self.initialized:
            if not self.initialize():
                raise ValueError(f"Failed to initialize client for provider {self.provider.name}")

        # Use specified model or fall back to default
        model_name = model or self.provider.default_model

        # Handle based on provider type
        if self.provider.name == "openai" or isinstance(self.client, object) and hasattr(self.client, "chat"):
            # For OpenAI and OpenAI-compatible clients
            messages = []

            if system_message:
                messages.append({"role": "system", "content": system_message})

            messages.append({"role": "user", "content": prompt})

            return self.client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )

        elif self.provider.name == "anthropic" and hasattr(self.client, "messages"):
            # For native Anthropic client
            messages = []

            if system_message:
                messages.append({"role": "system", "content": system_message})

            messages.append({"role": "user", "content": prompt})

            return self.client.messages.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )

        elif self.provider.name == "gemini" or hasattr(self.client, "genai"):
            # For Gemini client
            client = self.client
            model_obj = client.genai.GenerativeModel(
                model_name=model_name,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                }
            )

            generation_content = prompt
            if system_message:
                generation_content = f"{system_message}\n\n{prompt}"

            if stream:
                return model_obj.generate_content(
                    generation_content,
                    stream=True
                )
            else:
                return model_obj.generate_content(generation_content)

        else:
            raise ValueError(f"Unsupported provider type: {self.provider.name}")
```

### Phase 3: Configurable LLM Bot (Days 5-6)

1. Create a Bot Configuration Model:

```python
# utils/bot_config.py
from pydantic import BaseModel
from typing import Dict, Any, Optional

class BotConfig(BaseModel):
    """Configuration for a bot."""
    bot_name: str
    provider: str
    model: Optional[str] = None
    system_message: Optional[str] = None
    bot_description: str = ""
    version: str = "1.0.0"
    temperature: float = 0.7
    max_tokens: int = 1000
    settings: Dict[str, Any] = {}
```

2. Create the Configurable LLM Bot:

```python
# utils/configurable_llm_bot.py
import json
import logging
from typing import AsyncGenerator, Dict, Optional, Union

from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest

from utils.base_bot import BaseBot, BotError, BotErrorNoRetry
from utils.provider_registry import ProviderRegistry
from utils.provider_client import ProviderClient
from utils.bot_config import BotConfig

logger = logging.getLogger(__name__)

class ConfigurableLLMBot(BaseBot):
    """Bot powered by LLM providers defined in configuration."""

    def __init__(
        self,
        path: Optional[str] = None,
        access_key: Optional[str] = None,
        bot_name: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        config: Optional[BotConfig] = None,
        **kwargs
    ):
        """Initialize the bot with configuration."""
        # Set the bot name from config if provided
        if config and not bot_name:
            bot_name = config.bot_name

        # Initialize the base bot
        super().__init__(
            path=path,
            access_key=access_key,
            bot_name=bot_name,
            settings=settings,
            **kwargs
        )

        # Store the configuration
        self.config = config
        self.provider_registry = ProviderRegistry()
        self.client = None

        # Initialize from config if provided
        if config:
            self.bot_name = config.bot_name
            self.bot_description = config.bot_description or f"Bot powered by {config.provider}"
            self.version = config.version
            self.provider_name = config.provider
            self.model = config.model
            self.system_message = config.system_message
            self.temperature = config.temperature
            self.max_tokens = config.max_tokens

            # Apply any additional settings
            for key, value in config.settings.items():
                if hasattr(self, key):
                    setattr(self, key, value)

    def _init_client(self) -> bool:
        """Initialize the provider client."""
        # Get the provider config
        provider_config = self.provider_registry.get(self.provider_name)
        if not provider_config:
            logger.error(f"Provider not found: {self.provider_name}")
            return False

        # Create and initialize the client
        self.client = ProviderClient(provider_config)
        return self.client.initialize()

    @classmethod
    def from_config(cls, config: BotConfig):
        """Create a bot from configuration."""
        return cls(config=config)

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        """Process the query and generate a response."""
        try:
            # Extract the user's message
            user_message = self._extract_message(query)

            # Log the received message
            logger.debug(f"[{self.bot_name}] Received message: {user_message}")

            # Handle bot info request
            if user_message.lower().strip() == "bot info":
                metadata = self._get_bot_metadata()
                # Add provider-specific info
                metadata["provider"] = {
                    "name": getattr(self, "provider_name", "unknown"),
                    "model": getattr(self, "model", "unknown"),
                }
                yield PartialResponse(text=json.dumps(metadata, indent=2))
                return

            # Initialize client if needed
            if not hasattr(self, "client") or self.client is None:
                if not self._init_client():
                    raise BotErrorNoRetry("Failed to initialize LLM provider client")

            try:
                # Generate text from the provider
                stream = True  # Default to streaming

                # Call the provider
                response = await self.client.generate_text(
                    prompt=user_message,
                    model=self.model,
                    system_message=self.system_message,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    stream=stream
                )

                # Handle the response based on provider
                if self.provider_name == "openai" or hasattr(response, "choices"):
                    # OpenAI and compatible providers
                    if stream:
                        for chunk in response:
                            if hasattr(chunk.choices[0], "delta") and hasattr(chunk.choices[0].delta, "content"):
                                content = chunk.choices[0].delta.content
                                if content:
                                    yield PartialResponse(text=content)
                    else:
                        content = response.choices[0].message.content
                        yield PartialResponse(text=content)

                elif self.provider_name == "gemini":
                    # Google Gemini
                    if stream:
                        for chunk in response:
                            if hasattr(chunk, "text"):
                                yield PartialResponse(text=chunk.text)
                    else:
                        yield PartialResponse(text=response.text)

                elif self.provider_name == "anthropic":
                    # Anthropic Claude
                    if stream:
                        for chunk in response:
                            if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                                yield PartialResponse(text=chunk.delta.text)
                    else:
                        yield PartialResponse(text=response.content[0].text)

                else:
                    # Generic fallback
                    if hasattr(response, "content"):
                        yield PartialResponse(text=str(response.content))
                    else:
                        yield PartialResponse(text=str(response))

            except Exception as e:
                logger.error(f"[{self.bot_name}] Provider API error: {str(e)}")
                raise BotError(f"Provider API error: {str(e)}")

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
```

### Phase 4: Bot Registry and Factory (Days 7-8)

1. Create a Bot Registry:

```python
# utils/bot_registry.py
import yaml
import logging
from typing import Dict, List, Optional, Type

from utils.bot_config import BotConfig
from utils.configurable_llm_bot import ConfigurableLLMBot

logger = logging.getLogger(__name__)

class BotRegistry:
    """Registry for managing bot configurations."""

    _instance = None
    _bot_configs: Dict[str, BotConfig] = {}
    _bot_classes: Dict[str, Type[ConfigurableLLMBot]] = {}

    def __new__(cls):
        """Singleton pattern to ensure a single registry instance."""
        if cls._instance is None:
            cls._instance = super(BotRegistry, cls).__new__(cls)
            cls._instance._bot_configs = {}
            cls._instance._bot_classes = {}
        return cls._instance

    def register_config(self, config: BotConfig) -> None:
        """Register a bot configuration."""
        self._bot_configs[config.bot_name] = config
        logger.info(f"Registered bot config: {config.bot_name}")

    def register_class(self, bot_name: str, bot_class: Type[ConfigurableLLMBot]) -> None:
        """Register a custom bot class for a bot name."""
        self._bot_classes[bot_name] = bot_class
        logger.info(f"Registered custom bot class for: {bot_name}")

    def get_config(self, name: str) -> Optional[BotConfig]:
        """Get a bot configuration by name."""
        return self._bot_configs.get(name)

    def get_class(self, name: str) -> Type[ConfigurableLLMBot]:
        """Get a bot class by name, or return the default class."""
        return self._bot_classes.get(name, ConfigurableLLMBot)

    def list(self) -> List[str]:
        """List all registered bot names."""
        return list(self._bot_configs.keys())

    def create_bot(self, bot_name: str) -> Optional[ConfigurableLLMBot]:
        """Create a bot instance from a registered configuration."""
        config = self.get_config(bot_name)
        if not config:
            logger.error(f"Bot configuration not found: {bot_name}")
            return None

        bot_class = self.get_class(bot_name)
        return bot_class.from_config(config)

    def load_from_file(self, config_path: str) -> None:
        """Load bot configurations from a YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            for bot_data in config.get('bots', []):
                bot_config = BotConfig(**bot_data)
                self.register_config(bot_config)

            logger.info(f"Loaded {len(config.get('bots', []))} bot configs from {config_path}")
        except Exception as e:
            logger.error(f"Error loading bot configs from {config_path}: {str(e)}")
            raise
```

2. Create an Enhanced Bot Factory:

```python
# utils/enhanced_bot_factory.py
import logging
import os
from typing import List

from fastapi import FastAPI
from fastapi_poe import PoeBot, make_app

from utils.bot_factory import BotFactory
from utils.bot_registry import BotRegistry

logger = logging.getLogger(__name__)

class EnhancedBotFactory:
    """Factory for creating and managing Poe bots from both code and configuration."""

    @staticmethod
    def create_app(
        allow_without_key: bool = True,
        include_traditional_bots: bool = True,
        include_config_bots: bool = True,
    ) -> FastAPI:
        """Create a FastAPI app with all available bots.

        Args:
            allow_without_key: Whether to allow requests without an API key
            include_traditional_bots: Whether to include traditionally coded bots
            include_config_bots: Whether to include config-driven bots

        Returns:
            A FastAPI app with all the bots
        """
        bots = []

        # Add traditional bots if requested
        if include_traditional_bots:
            try:
                bot_classes = BotFactory.load_bots_from_module("bots")
                traditional_bots = [bot_class() for bot_class in bot_classes]
                bots.extend(traditional_bots)
                logger.info(f"Loaded {len(traditional_bots)} traditional bots")
            except Exception as e:
                logger.error(f"Error loading traditional bots: {str(e)}")

        # Add config-driven bots if requested
        if include_config_bots:
            try:
                registry = BotRegistry()
                for bot_name in registry.list():
                    bot = registry.create_bot(bot_name)
                    if bot:
                        bots.append(bot)
                        logger.info(f"Created config-driven bot: {bot_name}")
                    else:
                        logger.warning(f"Failed to create config-driven bot: {bot_name}")
            except Exception as e:
                logger.error(f"Error creating config-driven bots: {str(e)}")

        # Log the total number of bots
        logger.info(f"Total bots created: {len(bots)}")

        # Create the FastAPI app
        return make_app(bots, allow_without_key=allow_without_key)
```

### Phase 5: Integration with Existing Code (Days 9-10)

1. Update app.py:

```python
# app.py (modified)
import os
import logging

from fastapi import FastAPI
from modal import App, Image, asgi_app

from utils.config import settings

# Get configured logger
logger = logging.getLogger("poe_bots.app")

__version__ = "1.0.0"

# Configuration for multi-provider mode
MULTI_PROVIDER_ENABLED = os.environ.get("MULTI_PROVIDER_ENABLED", "").lower() == "true"
PROVIDER_CONFIG_PATH = os.environ.get("PROVIDER_CONFIG_PATH", "config/providers.yaml")
BOT_CONFIG_PATH = os.environ.get("BOT_CONFIG_PATH", "config/bots.yaml")

# Initialize configurations if multi-provider mode is enabled
if MULTI_PROVIDER_ENABLED:
    try:
        from utils.provider_registry import ProviderRegistry
        from utils.bot_registry import BotRegistry

        # Load provider configurations
        if os.path.exists(PROVIDER_CONFIG_PATH):
            provider_registry = ProviderRegistry()
            provider_registry.load_from_file(PROVIDER_CONFIG_PATH)
            logger.info(f"Loaded provider configurations from {PROVIDER_CONFIG_PATH}")

        # Load bot configurations
        if os.path.exists(BOT_CONFIG_PATH):
            bot_registry = BotRegistry()
            bot_registry.load_from_file(BOT_CONFIG_PATH)
            logger.info(f"Loaded bot configurations from {BOT_CONFIG_PATH}")
    except Exception as e:
        logger.error(f"Error initializing multi-provider mode: {str(e)}")
        MULTI_PROVIDER_ENABLED = False

def create_api(allow_without_key: bool = settings.ALLOW_WITHOUT_KEY) -> FastAPI:
    """Create and configure the FastAPI app with all available bots."""

    # Use enhanced factory if multi-provider mode is enabled
    if MULTI_PROVIDER_ENABLED:
        try:
            from utils.enhanced_bot_factory import EnhancedBotFactory
            api = EnhancedBotFactory.create_app(
                allow_without_key=allow_without_key,
                include_traditional_bots=True,
                include_config_bots=True
            )
            logger.info("Created API with EnhancedBotFactory")
        except Exception as e:
            logger.error(f"Error creating API with EnhancedBotFactory: {str(e)}")
            # Fall back to traditional BotFactory
            from utils.bot_factory import BotFactory
            bot_classes = BotFactory.load_bots_from_module("bots")
            api = BotFactory.create_app(bot_classes, allow_without_key=allow_without_key)
            logger.info("Fell back to traditional BotFactory")
    else:
        # Use traditional BotFactory
        from utils.bot_factory import BotFactory
        bot_classes = BotFactory.load_bots_from_module("bots")
        api = BotFactory.create_app(bot_classes, allow_without_key=allow_without_key)
        logger.info("Created API with traditional BotFactory")

    # Rest of function remains the same...
    # ...

    return api
```

2. Update run_local.py with multi-provider flags:

```python
# Add these arguments to run_local.py
parser.add_argument("--multi-provider", action="store_true", help="Enable multi-provider mode")
parser.add_argument("--provider-config", default="config/providers.yaml", help="Path to provider config")
parser.add_argument("--bot-config", default="config/bots.yaml", help="Path to bot config")

# Later in the code:
if args.multi_provider:
    os.environ["MULTI_PROVIDER_ENABLED"] = "true"
    os.environ["PROVIDER_CONFIG_PATH"] = args.provider_config
    os.environ["BOT_CONFIG_PATH"] = args.bot_config
```

3. Create default configuration files:

```yaml
# config/providers.yaml
providers:
  - name: openai
    api_base: https://api.openai.com/v1
    api_key_env: OPENAI_API_KEY
    default_model: gpt-4o-mini
    models:
      - gpt-4o-mini
      - gpt-3.5-turbo
      - gpt-4

  - name: anthropic
    api_base: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
    default_model: claude-3-haiku-20240307
    models:
      - claude-3-haiku-20240307
      - claude-3-sonnet-20240229
    headers:
      anthropic-version: "2023-06-01"

  - name: gemini
    api_base: https://generativelanguage.googleapis.com
    api_key_env: GOOGLE_API_KEY
    default_model: gemini-2.0-flash
    models:
      - gemini-2.0-flash
      - gemini-2.0-pro
```

```yaml
# config/bots.yaml
bots:
  - bot_name: GPT4oMiniBot
    provider: openai
    model: gpt-4o-mini
    bot_description: "A bot powered by GPT-4o Mini model"
    temperature: 0.7
    max_tokens: 1000
    system_message: "You are a helpful assistant."

  - bot_name: ClaudeHaikuBot
    provider: anthropic
    model: claude-3-haiku-20240307
    bot_description: "A bot powered by Claude 3 Haiku model"
    temperature: 0.5
    max_tokens: 2000
    system_message: "You are Claude, a helpful AI assistant."

  - bot_name: GeminiFlashBot
    provider: gemini
    model: gemini-2.0-flash
    bot_description: "A bot powered by Gemini Flash model"
    temperature: 0.8
    max_tokens: 1500
    system_message: "You are a helpful AI assistant."
```

4. Create a configuration management CLI tool:

```python
# config_manager.py
import argparse
import os
import yaml
from pathlib import Path

def ensure_config_dir():
    """Ensure the config directory exists."""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    return config_dir

def add_provider(args):
    """Add a provider configuration."""
    config_dir = ensure_config_dir()
    provider_file = config_dir / "providers.yaml"

    # Create default file if it doesn't exist
    if not provider_file.exists():
        with open(provider_file, 'w') as f:
            yaml.dump({"providers": []}, f)

    # Load existing config
    with open(provider_file, 'r') as f:
        config = yaml.safe_load(f)

    # Check if provider already exists
    providers = config.get("providers", [])
    for i, provider in enumerate(providers):
        if provider.get("name") == args.name:
            if args.force:
                # Remove existing provider
                providers.pop(i)
                break
            else:
                print(f"Provider '{args.name}' already exists. Use --force to overwrite.")
                return

    # Create new provider config
    new_provider = {
        "name": args.name,
        "api_base": args.api_base,
        "api_key_env": args.api_key_env,
        "default_model": args.default_model,
    }

    # Add optional fields
    if args.models:
        new_provider["models"] = args.models

    if args.api_version:
        new_provider["api_version"] = args.api_version

    if args.headers:
        headers = {}
        for header in args.headers:
            if ":" in header:
                key, value = header.split(":", 1)
                headers[key.strip()] = value.strip()
        if headers:
            new_provider["headers"] = headers

    # Add the provider
    if "providers" not in config:
        config["providers"] = []

    config["providers"].append(new_provider)

    # Save the updated config
    with open(provider_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Provider '{args.name}' added to {provider_file}")

def add_bot(args):
    """Add a bot configuration."""
    config_dir = ensure_config_dir()
    bot_file = config_dir / "bots.yaml"

    # Create default file if it doesn't exist
    if not bot_file.exists():
        with open(bot_file, 'w') as f:
            yaml.dump({"bots": []}, f)

    # Load existing config
    with open(bot_file, 'r') as f:
        config = yaml.safe_load(f)

    # Check if bot already exists
    bots = config.get("bots", [])
    for i, bot in enumerate(bots):
        if bot.get("bot_name") == args.name:
            if args.force:
                # Remove existing bot
                bots.pop(i)
                break
            else:
                print(f"Bot '{args.name}' already exists. Use --force to overwrite.")
                return

    # Create new bot config
    new_bot = {
        "bot_name": args.name,
        "provider": args.provider,
    }

    # Add optional fields
    if args.model:
        new_bot["model"] = args.model

    if args.description:
        new_bot["bot_description"] = args.description

    if args.temperature is not None:
        new_bot["temperature"] = args.temperature

    if args.max_tokens is not None:
        new_bot["max_tokens"] = args.max_tokens

    if args.system_message:
        new_bot["system_message"] = args.system_message

    # Add the bot
    if "bots" not in config:
        config["bots"] = []

    config["bots"].append(new_bot)

    # Save the updated config
    with open(bot_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Bot '{args.name}' added to {bot_file}")

def list_configs(args):
    """List providers or bots."""
    config_dir = ensure_config_dir()

    if args.type == "providers":
        file_path = config_dir / "providers.yaml"
        if not file_path.exists():
            print("No provider configurations found.")
            return

        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)

        providers = config.get("providers", [])
        if not providers:
            print("No providers configured.")
            return

        print(f"Found {len(providers)} provider(s):")
        for provider in providers:
            print(f"  - {provider['name']}")
            print(f"    API Base: {provider['api_base']}")
            print(f"    API Key Env: {provider['api_key_env']}")
            print(f"    Default Model: {provider['default_model']}")
            if "models" in provider:
                print(f"    Models: {', '.join(provider['models'])}")
            print()

    elif args.type == "bots":
        file_path = config_dir / "bots.yaml"
        if not file_path.exists():
            print("No bot configurations found.")
            return

        with open(file_path, 'r') as f:
            config = yaml.safe_load(f)

        bots = config.get("bots", [])
        if not bots:
            print("No bots configured.")
            return

        print(f"Found {len(bots)} bot(s):")
        for bot in bots:
            print(f"  - {bot['bot_name']}")
            print(f"    Provider: {bot['provider']}")
            if "model" in bot:
                print(f"    Model: {bot['model']}")
            if "bot_description" in bot:
                print(f"    Description: {bot['bot_description']}")
            print()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Configure multi-provider bots")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add provider command
    add_provider_parser = subparsers.add_parser("add-provider", help="Add a provider")
    add_provider_parser.add_argument("--name", required=True, help="Provider name")
    add_provider_parser.add_argument("--api-base", required=True, help="API base URL")
    add_provider_parser.add_argument("--api-key-env", required=True, help="Environment variable for API key")
    add_provider_parser.add_argument("--default-model", required=True, help="Default model")
    add_provider_parser.add_argument("--models", nargs="+", help="Available models")
    add_provider_parser.add_argument("--api-version", help="API version")
    add_provider_parser.add_argument("--headers", nargs="+", help="HTTP headers as 'key: value' pairs")
    add_provider_parser.add_argument("--force", action="store_true", help="Force overwrite if provider exists")
    add_provider_parser.set_defaults(func=add_provider)

    # Add bot command
    add_bot_parser = subparsers.add_parser("add-bot", help="Add a bot")
    add_bot_parser.add_argument("--name", required=True, help="Bot name")
    add_bot_parser.add_argument("--provider", required=True, help="Provider name")
    add_bot_parser.add_argument("--model", help="Model name")
    add_bot_parser.add_argument("--description", help="Bot description")
    add_bot_parser.add_argument("--temperature", type=float, help="Temperature (0.0-1.0)")
    add_bot_parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    add_bot_parser.add_argument("--system-message", help="System message")
    add_bot_parser.add_argument("--force", action="store_true", help="Force overwrite if bot exists")
    add_bot_parser.set_defaults(func=add_bot)

    # List command
    list_parser = subparsers.add_parser("list", help="List providers or bots")
    list_parser.add_argument("--type", choices=["providers", "bots"], required=True, help="Type to list")
    list_parser.set_defaults(func=list_configs)

    # Parse arguments
    args = parser.parse_args()

    if args.command and hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
```

## Example Usage

1. Start with default configurations:

```bash
mkdir -p config
python config_manager.py add-provider --name openai --api-base https://api.openai.com/v1 --api-key-env OPENAI_API_KEY --default-model gpt-4o-mini --models gpt-4o-mini gpt-3.5-turbo
python config_manager.py add-provider --name anthropic --api-base https://api.anthropic.com/v1 --api-key-env ANTHROPIC_API_KEY --default-model claude-3-haiku-20240307 --models claude-3-haiku-20240307 --headers "anthropic-version: 2023-06-01"
python config_manager.py add-provider --name gemini --api-base https://generativelanguage.googleapis.com --api-key-env GOOGLE_API_KEY --default-model gemini-2.0-flash

python config_manager.py add-bot --name GPT4oMiniBot --provider openai --model gpt-4o-mini --description "A bot powered by GPT-4o Mini model" --system-message "You are a helpful assistant."
python config_manager.py add-bot --name ClaudeHaikuBot --provider anthropic --model claude-3-haiku-20240307 --description "A bot powered by Claude 3 Haiku model" --system-message "You are Claude, a helpful AI assistant."
python config_manager.py add-bot --name GeminiFlashBot --provider gemini --model gemini-2.0-flash --description "A bot powered by Gemini Flash model" --system-message "You are a helpful AI assistant."
```

2. Run in multi-provider mode:

```bash
python run_local.py --multi-provider
```

## Benefits of This Approach

1. **Builds on Existing Foundations**: Uses the existing API key management system

2. **Gradual Migration**: Works alongside existing bots, with no disruption

3. **Configuration-Driven**: No need to write code for new bots or providers

4. **Extensible**: Easy to add new providers and models

5. **Robust Error Handling**: Graceful fallback when providers fail

6. **Unified API**: Consistent interface for different providers

7. **Simplified Deployment**: Works with Modal for cloud deployment

## Testing Strategy

1. **Unit Tests**: Test each component in isolation
2. **Integration Tests**: Test bots with mock providers
3. **End-to-End Tests**: Test with actual API calls (using environment variables)
4. **Compatibility Tests**: Ensure existing bots and tests still work

## Migration Path

For existing bots, the migration path is straightforward:

1. Keep using the current bots as-is
2. Add new bots via configuration
3. Optionally convert existing bots to use the new system

The most significant advantage is that this solution doesn't require changing any existing code - it simply adds new capabilities while integrating with the existing architecture.

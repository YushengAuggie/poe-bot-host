# Revised Multi-Provider Bot Framework Implementation Plan

Based on a thorough analysis of the existing codebase, this document outlines a refined approach to integrate multiple provider support with minimal disruption to the current architecture.

## Key Design Decisions

1. **Extend Existing Architecture**: Build upon the current `BaseBot` class rather than replacing it
2. **Minimize Code Duplication**: Share logic between existing bots and new multi-provider bots
3. **Maintain Test Compatibility**: Ensure all tests continue to pass without modification
4. **Gradual Migration Path**: Allow both old and new bot types to coexist

## Implementation Strategy

### Phase 1: Core Provider Components (Days 1-2)

1. Create Provider Configuration Models:

```python
# utils/providers.py
from pydantic import BaseModel
from typing import Dict, Any, Optional, List

class ProviderConfig(BaseModel):
    """Configuration for an LLM provider."""
    name: str
    api_base: str
    api_key_env: str
    api_version: Optional[str] = None
    organization_id: Optional[str] = None
    default_model: str
    models: List[str] = []
    timeout: int = 30
    headers: Dict[str, str] = {}
    additional_params: Dict[str, Any] = {}
```

2. Create Provider Registry:

```python
# utils/provider_registry.py
import os
import yaml
import logging
from typing import Dict, List, Optional
from utils.providers import ProviderConfig

logger = logging.getLogger(__name__)

class ProviderRegistry:
    """Registry for LLM providers."""

    _instance = None
    _providers: Dict[str, ProviderConfig] = {}

    def __new__(cls):
        """Singleton pattern to ensure a single registry instance."""
        if cls._instance is None:
            cls._instance = super(ProviderRegistry, cls).__new__(cls)
            cls._instance._providers = {}
        return cls._instance

    def register(self, provider: ProviderConfig) -> None:
        """Register a provider."""
        self._providers[provider.name] = provider
        logger.info(f"Registered provider: {provider.name}")

    def get(self, name: str) -> Optional[ProviderConfig]:
        """Get a provider by name."""
        return self._providers.get(name)

    def list(self) -> List[str]:
        """List all registered providers."""
        return list(self._providers.keys())

    def get_api_key(self, provider_name: str) -> Optional[str]:
        """Get API key for a provider from environment."""
        provider = self.get(provider_name)
        if not provider:
            return None
        return os.environ.get(provider.api_key_env)

    def load_from_file(self, config_path: str) -> None:
        """Load providers from a configuration file."""
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

3. Create LLM Client Utility:

```python
# utils/llm_client.py
import logging
from typing import Dict, Any, Optional
from utils.providers import ProviderConfig

logger = logging.getLogger(__name__)

class LLMClient:
    """Utility for creating and managing OpenAI-compatible clients."""

    @staticmethod
    def create_client(provider: ProviderConfig, api_key: Optional[str] = None) -> Any:
        """Create an OpenAI-compatible client for the given provider."""
        try:
            from openai import OpenAI

            # Use provided API key or None (will raise appropriate error)
            client_params = {
                "api_key": api_key,
                "base_url": provider.api_base,
                "timeout": provider.timeout,
            }

            # Add API version if specified
            if provider.api_version:
                client_params["api_version"] = provider.api_version

            # Add headers if specified
            if provider.headers:
                client_params["default_headers"] = provider.headers

            # Add organization if specified
            if provider.organization_id:
                client_params["organization"] = provider.organization_id

            # Create the client
            return OpenAI(**client_params)

        except ImportError:
            logger.error("OpenAI package not installed. Please install it with: pip install openai>=1.0.0")
            return None
        except Exception as e:
            logger.error(f"Error creating client for provider {provider.name}: {str(e)}")
            return None
```

### Phase 2: LLM Bot Base Class (Days 3-4)

1. Create LLM Bot Config:

```python
# utils/llm_bot_config.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional

class LLMBotConfig(BaseModel):
    """Configuration for an LLM-based bot."""
    bot_name: str
    provider: str
    model: Optional[str] = None
    bot_description: str = ""
    version: str = "1.0.0"
    temperature: float = 0.7
    max_tokens: int = 1000
    timeout: Optional[int] = None
    system_message: Optional[str] = None
    settings: Dict[str, Any] = {}
```

2. Create LLM Bot Base Class:

```python
# utils/llm_bot.py
import json
import logging
import os
from typing import Any, AsyncGenerator, Dict, Optional, Union

from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest

from utils.base_bot import BaseBot, BotError, BotErrorNoRetry
from utils.llm_bot_config import LLMBotConfig
from utils.provider_registry import ProviderRegistry
from utils.llm_client import LLMClient
from utils.providers import ProviderConfig

logger = logging.getLogger(__name__)

class LLMBot(BaseBot):
    """Base class for bots powered by LLM providers with OpenAI-compatible APIs."""

    def __init__(
        self,
        path: Optional[str] = None,
        access_key: Optional[str] = None,
        bot_name: Optional[str] = None,
        settings: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
        model: Optional[str] = None,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        """Initialize a new LLM bot with optional provider overrides."""
        # Initialize base bot first
        super().__init__(
            path=path,
            access_key=access_key,
            bot_name=bot_name,
            settings=settings,
            **kwargs
        )

        # Setup provider configuration
        self.provider_name = provider_name
        self.provider_registry = ProviderRegistry()

        # Setup model parameters (can be overridden)
        self.model = model
        self.system_message = system_message
        self.temperature = temperature if temperature is not None else 0.7
        self.max_tokens = max_tokens if max_tokens is not None else 1000

        # Setup client (initialized on first use)
        self.client = None
        self.provider = None

    def _ensure_client_initialized(self) -> bool:
        """Ensure the LLM client is initialized."""
        if self.client is not None:
            return True

        # Get provider configuration
        if not self.provider_name:
            logger.error(f"[{self.bot_name}] Provider name not specified")
            return False

        self.provider = self.provider_registry.get(self.provider_name)
        if not self.provider:
            logger.error(f"[{self.bot_name}] Provider not found: {self.provider_name}")
            return False

        # Get API key
        api_key = self.provider_registry.get_api_key(self.provider_name)
        if not api_key:
            logger.warning(f"[{self.bot_name}] API key not found for provider: {self.provider_name}")
            return False

        # Create client
        self.client = LLMClient.create_client(self.provider, api_key)
        if not self.client:
            logger.error(f"[{self.bot_name}] Failed to create client for provider: {self.provider_name}")
            return False

        # Use default model if not specified
        if not self.model:
            self.model = self.provider.default_model

        # Log successful initialization
        logger.info(f"[{self.bot_name}] Initialized client for provider: {self.provider_name}, model: {self.model}")
        return True

    def _build_messages(self, user_message: str) -> list:
        """Build the messages list for the API request."""
        messages = []

        # Add system message if provided
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})

        # Add user message
        messages.append({"role": "user", "content": user_message})

        return messages

    def _get_extended_metadata(self) -> Dict[str, Any]:
        """Get extended metadata about the bot including provider info."""
        metadata = self._get_bot_metadata()

        # Add provider information
        metadata["provider"] = {
            "name": self.provider_name,
            "model": self.model,
        }

        if self.provider:
            metadata["provider"]["api_base"] = self.provider.api_base

        # Add LLM settings
        metadata["llm_settings"] = {
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "system_message": self.system_message,
        }

        return metadata

    @classmethod
    def from_config(cls, config: LLMBotConfig):
        """Create a new LLM bot from a configuration object."""
        return cls(
            bot_name=config.bot_name,
            provider_name=config.provider,
            model=config.model,
            system_message=config.system_message,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            settings=config.settings
        )

    async def get_response(
        self, query: QueryRequest
    ) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        """Process the query and generate a response using the configured LLM provider."""
        try:
            # Extract the user's message
            user_message = self._extract_message(query)

            # Log the received message
            logger.debug(f"[{self.bot_name}] Received message: {user_message}")

            # Handle bot info request
            if user_message.lower().strip() == "bot info":
                metadata = self._get_extended_metadata()
                yield PartialResponse(text=json.dumps(metadata, indent=2))
                return

            # Ensure client is initialized
            if not self._ensure_client_initialized():
                raise BotErrorNoRetry(f"LLM provider client not initialized for {self.provider_name}")

            # Create messages payload
            messages = self._build_messages(user_message)

            # Call the API in streaming mode
            try:
                stream = True  # Default to streaming

                # Indicate we're processing
                yield PartialResponse(text=f"Processing with {self.provider_name} ({self.model})...\n\n")

                # Create completion
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
```

3. Create Bot Registry:

```python
# utils/llm_bot_registry.py
import yaml
import logging
from typing import Dict, List, Optional, Type
from utils.llm_bot import LLMBot
from utils.llm_bot_config import LLMBotConfig

logger = logging.getLogger(__name__)

class LLMBotRegistry:
    """Registry for LLM bot configurations and instances."""

    _instance = None
    _bot_configs: Dict[str, LLMBotConfig] = {}
    _bot_classes: Dict[str, Type[LLMBot]] = {}

    def __new__(cls):
        """Singleton pattern to ensure a single registry instance."""
        if cls._instance is None:
            cls._instance = super(LLMBotRegistry, cls).__new__(cls)
            cls._instance._bot_configs = {}
            cls._instance._bot_classes = {}
        return cls._instance

    def register_config(self, config: LLMBotConfig) -> None:
        """Register a bot configuration."""
        self._bot_configs[config.bot_name] = config
        logger.info(f"Registered bot config: {config.bot_name}")

    def register_class(self, bot_name: str, bot_class: Type[LLMBot]) -> None:
        """Register a bot class for a bot name."""
        self._bot_classes[bot_name] = bot_class
        logger.info(f"Registered bot class for: {bot_name}")

    def get_config(self, name: str) -> Optional[LLMBotConfig]:
        """Get a bot configuration by name."""
        return self._bot_configs.get(name)

    def get_class(self, name: str) -> Type[LLMBot]:
        """Get a bot class by name or return default LLMBot."""
        return self._bot_classes.get(name, LLMBot)

    def list(self) -> List[str]:
        """List all registered bot names."""
        return list(self._bot_configs.keys())

    def create_bot(self, bot_name: str) -> Optional[LLMBot]:
        """Create a new bot instance from a registered configuration."""
        config = self.get_config(bot_name)
        if not config:
            logger.error(f"Bot configuration not found: {bot_name}")
            return None

        bot_class = self.get_class(bot_name)
        try:
            return bot_class.from_config(config)
        except Exception as e:
            logger.error(f"Error creating bot {bot_name}: {str(e)}")
            return None

    def load_from_file(self, config_path: str) -> None:
        """Load bot configurations from a file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)

            for bot_data in config.get('bots', []):
                bot_config = LLMBotConfig(**bot_data)
                self.register_config(bot_config)

            logger.info(f"Loaded {len(config.get('bots', []))} bot configs from {config_path}")
        except Exception as e:
            logger.error(f"Error loading bot configs from {config_path}: {str(e)}")
            raise
```

### Phase 3: Enhanced BotFactory (Days 5-6)

1. Create an enhanced bot factory that supports both traditional and LLM bots:

```python
# utils/enhanced_bot_factory.py
import logging
import os
from typing import Dict, List, Optional, Type

from fastapi import FastAPI
from fastapi_poe import PoeBot, make_app

from utils.base_bot import BaseBot
from utils.bot_factory import BotFactory
from utils.provider_registry import ProviderRegistry
from utils.llm_bot_registry import LLMBotRegistry

logger = logging.getLogger(__name__)

class EnhancedBotFactory:
    """Factory for creating and managing both traditional and LLM-based Poe bots."""

    @staticmethod
    def create_app(
        allow_without_key: bool = True,
        include_traditional_bots: bool = True,
        include_llm_bots: bool = True
    ) -> FastAPI:
        """Create a FastAPI app with all available bots.

        Args:
            allow_without_key: Whether to allow requests without an API key
            include_traditional_bots: Whether to include traditional bots
            include_llm_bots: Whether to include LLM-based bots

        Returns:
            A FastAPI app with all the bots
        """
        bots = []

        # Add traditional bots if requested
        if include_traditional_bots:
            try:
                # Use the existing BotFactory to load traditional bots
                bot_classes = BotFactory.load_bots_from_module("bots")
                traditional_bots = [bot_class() for bot_class in bot_classes]
                bots.extend(traditional_bots)
                logger.info(f"Loaded {len(traditional_bots)} traditional bots")
            except Exception as e:
                logger.error(f"Error loading traditional bots: {str(e)}")

        # Add LLM bots if requested
        if include_llm_bots:
            try:
                # Use the LLMBotRegistry to create LLM bots
                llm_bot_registry = LLMBotRegistry()
                for bot_name in llm_bot_registry.list():
                    bot = llm_bot_registry.create_bot(bot_name)
                    if bot:
                        bots.append(bot)
                        logger.info(f"Created LLM bot: {bot_name}")
                    else:
                        logger.warning(f"Failed to create LLM bot: {bot_name}")
            except Exception as e:
                logger.error(f"Error creating LLM bots: {str(e)}")

        # Log the total number of bots
        logger.info(f"Total bots created: {len(bots)}")

        # Create the FastAPI app
        return make_app(bots, allow_without_key=allow_without_key)

    @staticmethod
    def load_configurations(
        provider_config_path: Optional[str] = None,
        bot_config_path: Optional[str] = None
    ) -> None:
        """Load provider and bot configurations.

        Args:
            provider_config_path: Path to provider configuration
            bot_config_path: Path to bot configuration
        """
        # Load provider configurations if specified
        if provider_config_path and os.path.exists(provider_config_path):
            try:
                provider_registry = ProviderRegistry()
                provider_registry.load_from_file(provider_config_path)
                logger.info(f"Loaded provider configurations from {provider_config_path}")
            except Exception as e:
                logger.error(f"Error loading provider configurations: {str(e)}")

        # Load bot configurations if specified
        if bot_config_path and os.path.exists(bot_config_path):
            try:
                bot_registry = LLMBotRegistry()
                bot_registry.load_from_file(bot_config_path)
                logger.info(f"Loaded bot configurations from {bot_config_path}")
            except Exception as e:
                logger.error(f"Error loading bot configurations: {str(e)}")
```

2. Create example configuration files:

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
    timeout: 30

  - name: anthropic
    api_base: https://api.anthropic.com/v1
    api_key_env: ANTHROPIC_API_KEY
    default_model: claude-3-haiku-20240307
    models:
      - claude-3-haiku-20240307
      - claude-3-sonnet-20240229
    timeout: 30
    headers:
      anthropic-version: "2023-06-01"

  - name: gemini
    api_base: https://generativelanguage.googleapis.com/v1beta
    api_key_env: GEMINI_API_KEY
    default_model: gemini-pro
    models:
      - gemini-pro
    timeout: 30
```

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
```

### Phase 4: Application Integration (Days 7-8)

1. Update app.py (non-disruptive, feature-flag based):

```python
# app.py (modified portion)
import os
import logging

from fastapi import FastAPI
from modal import App, Image, asgi_app

from utils.config import settings

# Get configured logger
logger = logging.getLogger("poe_bots.app")

__version__ = "1.0.0"

# Check if multi-provider mode is enabled
MULTI_PROVIDER_ENABLED = os.environ.get("MULTI_PROVIDER_ENABLED", "").lower() == "true"

# Load provider configurations if multi-provider mode is enabled
if MULTI_PROVIDER_ENABLED:
    try:
        from utils.enhanced_bot_factory import EnhancedBotFactory

        # Default configuration paths
        PROVIDER_CONFIG_PATH = os.environ.get("PROVIDER_CONFIG_PATH", "config/providers.yaml")
        BOT_CONFIG_PATH = os.environ.get("BOT_CONFIG_PATH", "config/bots.yaml")

        # Load configurations if they exist
        if os.path.exists(PROVIDER_CONFIG_PATH) and os.path.exists(BOT_CONFIG_PATH):
            EnhancedBotFactory.load_configurations(PROVIDER_CONFIG_PATH, BOT_CONFIG_PATH)
            logger.info(f"Multi-provider mode enabled and configurations loaded")
        else:
            logger.warning(f"Multi-provider mode enabled but configuration files not found")
    except Exception as e:
        logger.error(f"Error initializing multi-provider mode: {str(e)}")
        MULTI_PROVIDER_ENABLED = False

def create_api(allow_without_key: bool = settings.ALLOW_WITHOUT_KEY) -> FastAPI:
    """Create and configure the FastAPI app with all available bots."""

    # Use enhanced factory if multi-provider mode is enabled
    if MULTI_PROVIDER_ENABLED:
        try:
            # Create FastAPI app with both traditional and LLM bots
            api = EnhancedBotFactory.create_app(
                allow_without_key=allow_without_key,
                include_traditional_bots=True,
                include_llm_bots=True
            )
            logger.info("Created API with EnhancedBotFactory (multi-provider mode)")
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
```

2. Update run_local.py:

```python
# run_local.py (add these arguments)
parser.add_argument("--multi-provider", action="store_true", help="Enable multi-provider mode")
parser.add_argument("--provider-config", default="config/providers.yaml", help="Path to provider config")
parser.add_argument("--bot-config", default="config/bots.yaml", help="Path to bot config")

# Later in the code, set environment variables based on args
if args.multi_provider:
    os.environ["MULTI_PROVIDER_ENABLED"] = "true"
    os.environ["PROVIDER_CONFIG_PATH"] = args.provider_config
    os.environ["BOT_CONFIG_PATH"] = args.bot_config
    print(f"Multi-provider mode enabled")
    print(f"Provider config: {args.provider_config}")
    print(f"Bot config: {args.bot_config}")
```

3. Create configuration management CLI:

```python
# config_manager.py
import argparse
import os
import yaml
from pathlib import Path
import sys

def ensure_config_dir():
    """Ensure the config directory exists."""
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    return config_dir

def add_provider(args):
    """Add a new provider configuration."""
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
    for provider in config.get("providers", []):
        if provider.get("name") == args.name:
            print(f"Provider '{args.name}' already exists. Use --force to overwrite.")
            if not args.force:
                return

            # Remove existing provider
            config["providers"].remove(provider)
            break

    # Create new provider config
    new_provider = {
        "name": args.name,
        "api_base": args.api_base,
        "api_key_env": args.api_key_env,
        "default_model": args.default_model,
    }

    # Add optional fields
    if args.api_version:
        new_provider["api_version"] = args.api_version

    if args.models:
        new_provider["models"] = args.models

    if args.timeout:
        new_provider["timeout"] = args.timeout

    if args.organization:
        new_provider["organization_id"] = args.organization

    # Add the provider
    if "providers" not in config:
        config["providers"] = []

    config["providers"].append(new_provider)

    # Save the updated config
    with open(provider_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Provider '{args.name}' added to {provider_file}")

def add_bot(args):
    """Add a new bot configuration."""
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
    for bot in config.get("bots", []):
        if bot.get("bot_name") == args.name:
            print(f"Bot '{args.name}' already exists. Use --force to overwrite.")
            if not args.force:
                return

            # Remove existing bot
            config["bots"].remove(bot)
            break

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

    if args.temperature:
        new_bot["temperature"] = args.temperature

    if args.max_tokens:
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
            print(f"    Default Model: {provider['default_model']}")
            print(f"    API Key Env: {provider['api_key_env']}")
            if provider.get("models"):
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
            if bot.get("model"):
                print(f"    Model: {bot['model']}")
            if bot.get("bot_description"):
                print(f"    Description: {bot['bot_description']}")
            print()

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Manage provider and bot configurations")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add provider command
    add_provider_parser = subparsers.add_parser("add-provider", help="Add a provider configuration")
    add_provider_parser.add_argument("--name", required=True, help="Provider name")
    add_provider_parser.add_argument("--api-base", required=True, help="API base URL")
    add_provider_parser.add_argument("--api-key-env", required=True, help="Environment variable for API key")
    add_provider_parser.add_argument("--default-model", required=True, help="Default model")
    add_provider_parser.add_argument("--api-version", help="API version")
    add_provider_parser.add_argument("--models", nargs="+", help="List of available models")
    add_provider_parser.add_argument("--timeout", type=int, help="Timeout in seconds")
    add_provider_parser.add_argument("--organization", help="Organization ID")
    add_provider_parser.add_argument("--force", action="store_true", help="Force overwrite if provider exists")
    add_provider_parser.set_defaults(func=add_provider)

    # Add bot command
    add_bot_parser = subparsers.add_parser("add-bot", help="Add a bot configuration")
    add_bot_parser.add_argument("--name", required=True, help="Bot name")
    add_bot_parser.add_argument("--provider", required=True, help="Provider name")
    add_bot_parser.add_argument("--model", help="Model name (default: provider's default model)")
    add_bot_parser.add_argument("--description", help="Bot description")
    add_bot_parser.add_argument("--temperature", type=float, help="Temperature (0.0-1.0)")
    add_bot_parser.add_argument("--max-tokens", type=int, help="Maximum tokens to generate")
    add_bot_parser.add_argument("--system-message", help="System message")
    add_bot_parser.add_argument("--force", action="store_true", help="Force overwrite if bot exists")
    add_bot_parser.set_defaults(func=add_bot)

    # List command
    list_parser = subparsers.add_parser("list", help="List configurations")
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

### Phase 5: Tests and Documentation (Days 9-10)

1. Add tests for provider and LLM bot components:

```python
# tests/test_provider_registry.py
import os
import pytest
import tempfile
import yaml

from utils.provider_registry import ProviderRegistry
from utils.providers import ProviderConfig

@pytest.fixture
def test_config_file():
    """Create a temporary provider configuration file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump({
            "providers": [
                {
                    "name": "test-provider",
                    "api_base": "https://api.test.com",
                    "api_key_env": "TEST_API_KEY",
                    "default_model": "test-model"
                }
            ]
        }, f)
        return f.name

@pytest.fixture
def clean_registry():
    """Ensure clean registry for tests."""
    registry = ProviderRegistry()
    registry._providers = {}
    return registry

def test_register_provider(clean_registry):
    """Test registering a provider."""
    registry = clean_registry
    provider = ProviderConfig(
        name="test-provider",
        api_base="https://api.test.com",
        api_key_env="TEST_API_KEY",
        default_model="test-model"
    )

    registry.register(provider)

    assert "test-provider" in registry.list()
    assert registry.get("test-provider") == provider

def test_load_from_file(clean_registry, test_config_file):
    """Test loading providers from a file."""
    registry = clean_registry
    registry.load_from_file(test_config_file)

    assert "test-provider" in registry.list()
    provider = registry.get("test-provider")
    assert provider.api_base == "https://api.test.com"
    assert provider.default_model == "test-model"

def test_get_api_key(clean_registry):
    """Test getting API key from environment."""
    registry = clean_registry
    provider = ProviderConfig(
        name="test-provider",
        api_base="https://api.test.com",
        api_key_env="TEST_API_KEY",
        default_model="test-model"
    )

    registry.register(provider)

    # Set environment variable
    os.environ["TEST_API_KEY"] = "test-key"

    assert registry.get_api_key("test-provider") == "test-key"

    # Clean up
    os.environ.pop("TEST_API_KEY", None)
```

```python
# tests/test_llm_bot.py
import json
import pytest
from unittest.mock import MagicMock, patch

from fastapi_poe.types import QueryRequest

from utils.llm_bot import LLMBot
from utils.provider_registry import ProviderRegistry
from utils.providers import ProviderConfig

@pytest.fixture
def mock_provider():
    """Create a mock provider configuration."""
    return ProviderConfig(
        name="test-provider",
        api_base="https://api.test.com",
        api_key_env="TEST_API_KEY",
        default_model="test-model"
    )

@pytest.fixture
def mock_registry(mock_provider):
    """Create a mock provider registry."""
    registry = ProviderRegistry()
    registry._providers = {}
    registry.register(mock_provider)

    # Mock get_api_key to return a test key
    registry.get_api_key = MagicMock(return_value="test-key")

    return registry

@pytest.fixture
def llm_bot(mock_registry):
    """Create an LLMBot instance for testing."""
    bot = LLMBot(
        bot_name="TestLLMBot",
        provider_name="test-provider",
        model="test-model",
        system_message="You are a test bot."
    )

    # Set provider registry to our mock
    bot.provider_registry = mock_registry

    return bot

@pytest.mark.asyncio
async def test_llm_bot_initialization(llm_bot):
    """Test LLMBot initialization."""
    assert llm_bot.bot_name == "TestLLMBot"
    assert llm_bot.provider_name == "test-provider"
    assert llm_bot.model == "test-model"
    assert llm_bot.system_message == "You are a test bot."

@pytest.mark.asyncio
async def test_build_messages(llm_bot):
    """Test building messages for API call."""
    messages = llm_bot._build_messages("Hello, bot!")

    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "You are a test bot."
    assert messages[1]["role"] == "user"
    assert messages[1]["content"] == "Hello, bot!"

@pytest.mark.asyncio
async def test_get_extended_metadata(llm_bot, mock_provider):
    """Test getting extended metadata."""
    # Set provider
    llm_bot.provider = mock_provider

    metadata = llm_bot._get_extended_metadata()

    assert metadata["name"] == "TestLLMBot"
    assert "provider" in metadata
    assert metadata["provider"]["name"] == "test-provider"
    assert metadata["provider"]["model"] == "test-model"
    assert "llm_settings" in metadata
    assert metadata["llm_settings"]["system_message"] == "You are a test bot."

@pytest.mark.asyncio
async def test_bot_info_request(llm_bot):
    """Test bot info request."""
    # Create a bot info query
    query = QueryRequest(
        version="1.0",
        type="query",
        query=[{"role": "user", "content": "bot info"}],
        user_id="test_user",
        conversation_id="test_conversation",
        message_id="test_message"
    )

    # Collect responses
    responses = []
    async for response in llm_bot.get_response(query):
        responses.append(response)

    # Should have one response with JSON metadata
    assert len(responses) == 1
    metadata = json.loads(responses[0].text)
    assert metadata["name"] == "TestLLMBot"
    assert metadata["provider"]["name"] == "test-provider"

@pytest.mark.asyncio
async def test_client_initialization_error(llm_bot):
    """Test error handling when client initialization fails."""
    # Create a mock client that returns None
    with patch('utils.llm_client.LLMClient.create_client', return_value=None):
        # Create a query
        query = QueryRequest(
            version="1.0",
            type="query",
            query=[{"role": "user", "content": "Hello, bot!"}],
            user_id="test_user",
            conversation_id="test_conversation",
            message_id="test_message"
        )

        # Collect responses
        responses = []
        async for response in llm_bot.get_response(query):
            responses.append(response)

        # Should have an error response
        assert len(responses) > 0
        error_text = responses[-1].text
        assert "Error" in error_text
        assert "not initialized" in error_text.lower()
```

2. Create documentation and examples:

```markdown
# Multi-Provider Bot Framework

This enhancement to the Poe Bots Framework allows you to create bots powered by multiple LLM providers through a unified configuration system.

## Supported Providers

The framework supports any provider with an OpenAI-compatible API interface, including:

- OpenAI
- Anthropic
- Google Gemini
- Fireworks
- Together
- And more!

## Getting Started

1. Create provider and bot configuration files:

```bash
mkdir -p config
```

2. Add provider configurations:

```bash
./config_manager.py add-provider \
  --name openai \
  --api-base https://api.openai.com/v1 \
  --api-key-env OPENAI_API_KEY \
  --default-model gpt-3.5-turbo \
  --models gpt-3.5-turbo gpt-4
```

3. Add bot configurations:

```bash
./config_manager.py add-bot \
  --name GPT35Bot \
  --provider openai \
  --model gpt-3.5-turbo \
  --description "A bot powered by GPT-3.5 Turbo" \
  --temperature 0.7 \
  --system-message "You are a helpful assistant."
```

4. Set API keys as environment variables:

```bash
export OPENAI_API_KEY=your_api_key_here
```

5. Run the server with multi-provider mode enabled:

```bash
./run_local.py --multi-provider
```

## Custom Bot Implementations

You can create custom bot implementations that leverage different providers:

```python
from utils.llm_bot import LLMBot

class CustomLLMBot(LLMBot):
    """Custom bot with specialized behavior."""

    async def get_response(self, query):
        # Custom pre-processing
        # ...

        # Use LLM provider
        async for response in super().get_response(query):
            yield response

        # Custom post-processing
        # ...
```
```

## Data Flow Diagram

```
┌─────────────┐       ┌───────────────┐       ┌───────────────┐
│             │       │               │       │               │
│  Provider   │───────▶  ProviderReg  │◀──────▶   LLMClient   │
│  Config     │       │               │       │               │
│             │       └───────┬───────┘       └───────┬───────┘
└─────────────┘               │                       │
                              │                       │
┌─────────────┐       ┌───────┴───────┐       ┌───────┴───────┐
│             │       │               │       │               │
│  Bot        │───────▶   LLMBot      │◀──────▶   API Call    │
│  Config     │       │               │       │               │
│             │       └───────┬───────┘       └───────────────┘
└─────────────┘               │
                              │
┌─────────────┐       ┌───────┴───────┐       ┌───────────────┐
│             │       │               │       │               │
│  BotReg     │◀──────▶ EnhancedBot  │───────▶    FastAPI    │
│             │       │  Factory     │       │     App       │
└─────────────┘       └───────────────┘       └───────────────┘
```

## Migration Strategy

The implementation is designed for a smooth migration path:

1. **Parallel Operation**: Traditional bots and LLM bots can coexist
2. **Feature Flag**: Control enabling multi-provider mode with environment variable
3. **Gradual Migration**: Move bots one by one to the new system
4. **Backward Compatibility**: All existing bots continue to work without changes
5. **Tests**: All existing tests continue to pass

## Development Roadmap

| Phase | Task | Est. Time |
|-------|------|-----------|
| 1 | Core Components | 2 days |
| 2 | LLM Bot Base | 2 days |
| 3 | Enhanced Factory | 2 days |
| 4 | App Integration | 2 days |
| 5 | Tests & Docs | 2 days |
| | **Total** | **10 days** |

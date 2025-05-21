# Multi-Provider Implementation Plan

This document outlines a step-by-step approach to integrate the multi-provider framework into the existing codebase.

## Implementation Steps

### 1. Create Directory Structure
```
/config
  - providers.yaml
  - bots.yaml
/utils
  - providers.py
  - bot_config.py
  - multi_provider_bot.py
  - provider_registry.py
  - bot_config_registry.py
  - enhanced_bot_factory.py
/bots
  - llm_chat_bot.py
```

### 2. Implementation Phases

#### Phase 1: Core Components (Days 1-2)
1. Create new configuration models
   - `utils/providers.py`
   - `utils/bot_config.py`
2. Create registries
   - `utils/provider_registry.py`
   - `utils/bot_config_registry.py`
3. Create configuration files
   - `config/providers.yaml`
   - `config/bots.yaml`
4. Add dependencies to `requirements.txt`
   - `pyyaml`
   - Ensure OpenAI SDK is up-to-date

#### Phase 2: Bot Framework (Days 3-4)
1. Create multi-provider bot base class
   - `utils/multi_provider_bot.py`
2. Create enhanced bot factory
   - `utils/enhanced_bot_factory.py`
3. Create generic chat bot implementation
   - `bots/llm_chat_bot.py`

#### Phase 3: Integration (Days 5-6)
1. Update `app.py` to use enhanced bot factory
2. Update existing CLI tools like `run_local.py` to support provider configuration
3. Create CLI tool for managing bot configurations

#### Phase 4: Testing & Documentation (Days 7-8)
1. Add tests for new components
2. Update documentation
3. Create examples and tutorials

## Integration Strategy

### Non-disruptive Approach
To ensure minimal disruption to the existing functionality:

1. **Parallel Implementation**: Create the new components alongside existing ones without replacing core functionality initially
2. **Feature Flag**: Use environment variables to toggle between old and new frameworks
3. **Gradual Migration**: Move bots one by one to the new framework

### Updates to Existing Files

1. `app.py`:
   - Add conditional logic to use either the existing BotFactory or the new EnhancedBotFactory
   - Use an environment variable like `USE_MULTI_PROVIDER=true` to control which system is used

```python
# app.py (modified)
import os
from utils.bot_factory import BotFactory
from utils.config import settings

# Import new components conditionally
if os.environ.get("USE_MULTI_PROVIDER", "").lower() == "true":
    from utils.enhanced_bot_factory import EnhancedBotFactory
    from utils.bot_config_registry import BotConfigRegistry
    from utils.provider_registry import ProviderRegistry

def create_api(allow_without_key: bool = settings.ALLOW_WITHOUT_KEY) -> FastAPI:
    """Create and configure the FastAPI app with all available bots."""

    # Use enhanced factory if multi-provider mode is enabled
    if os.environ.get("USE_MULTI_PROVIDER", "").lower() == "true":
        # Initialize provider and bot registries
        provider_config_path = os.environ.get("PROVIDER_CONFIG_PATH", "config/providers.yaml")
        bot_config_path = os.environ.get("BOT_CONFIG_PATH", "config/bots.yaml")

        try:
            # Load configurations
            EnhancedBotFactory.load_configurations(provider_config_path, bot_config_path)

            # Register bot classes
            from bots.llm_chat_bot import LLMChatBot
            for bot_name in BotConfigRegistry.list():
                BotConfigRegistry.register_class(bot_name, LLMChatBot)

            # Create the API
            api = EnhancedBotFactory.create_app(allow_without_key=allow_without_key)

        except Exception as e:
            logger.error(f"Error creating enhanced API: {str(e)}", exc_info=True)
            # Fall back to original implementation
            logger.info("Falling back to standard bot factory")
            bot_classes = BotFactory.load_bots_from_module("bots")
            api = BotFactory.create_app(bot_classes, allow_without_key=allow_without_key)
    else:
        # Use original implementation
        bot_classes = BotFactory.load_bots_from_module("bots")
        api = BotFactory.create_app(bot_classes, allow_without_key=allow_without_key)

    # Add endpoints (same as before, with additions for providers)
    # ...

    return api
```

2. `requirements.txt`:
   - Add new dependencies

```
# Add to requirements.txt
pyyaml>=6.0
openai>=1.0.0  # Ensure latest version for provider compatibility
```

3. `run_local.py`:
   - Add command-line arguments for multi-provider mode

```python
# Add to argument parser in run_local.py
parser.add_argument("--multi-provider", action="store_true", help="Enable multi-provider mode")
parser.add_argument("--provider-config", default="config/providers.yaml", help="Path to provider config")
parser.add_argument("--bot-config", default="config/bots.yaml", help="Path to bot config")

# Set environment variables based on args
if args.multi_provider:
    os.environ["USE_MULTI_PROVIDER"] = "true"
    os.environ["PROVIDER_CONFIG_PATH"] = args.provider_config
    os.environ["BOT_CONFIG_PATH"] = args.bot_config
```

## Migration Path for Existing Bots

For existing bots, there are two migration strategies:

### 1. Configuration-based Migration
Create configurations for each existing bot in `config/bots.yaml` that map to the LLMChatBot implementation:

```yaml
# Example migration for EchoBot
bots:
  - bot_name: EchoBot
    provider: openai  # Or default provider
    model: gpt-3.5-turbo
    bot_description: "A simple bot that echoes back the user's message."
    temperature: 0.7
    max_tokens: 1000
    system_message: "You are an echo bot. Simply repeat the user's message back to them."
```

### 2. Class-based Migration
For bots with custom logic, create specialized implementations that extend MultiProviderBot:

```python
# Example migration for WeatherBot
from utils.multi_provider_bot import MultiProviderBot
from typing import AsyncGenerator, Union
from fastapi_poe.types import MetaResponse, PartialResponse, QueryRequest

class WeatherBot(MultiProviderBot):
    """Weather bot that combines custom logic with provider capabilities."""

    async def get_response(self, query: QueryRequest) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        # Extract message
        user_message = self._extract_message(query)

        # Special handling for weather commands
        if "weather" in user_message.lower():
            location = user_message.replace("weather", "").strip()
            # Custom weather logic...
            yield PartialResponse(text=f"Weather for {location}...")
        else:
            # For non-weather queries, fall back to LLM provider
            # Use the client as in LLMChatBot
            # ...
```

## Command-line Tool for Configuration Management

Create a CLI tool to simplify configuration management:

```python
# config_manager.py
import argparse
import os
import yaml
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Manage bot and provider configurations")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Add provider command
    add_provider = subparsers.add_parser("add-provider", help="Add a new provider")
    add_provider.add_argument("--name", required=True, help="Provider name")
    add_provider.add_argument("--api-base", required=True, help="API base URL")
    add_provider.add_argument("--api-key-env", required=True, help="Environment variable for API key")
    add_provider.add_argument("--default-model", required=True, help="Default model")
    add_provider.add_argument("--models", nargs="+", help="Available models")

    # Add bot command
    add_bot = subparsers.add_parser("add-bot", help="Add a new bot")
    add_bot.add_argument("--name", required=True, help="Bot name")
    add_bot.add_argument("--provider", required=True, help="Provider name")
    add_bot.add_argument("--model", help="Model name (defaults to provider's default)")
    add_bot.add_argument("--description", help="Bot description")
    add_bot.add_argument("--system-message", help="System message")

    # List command
    list_cmd = subparsers.add_parser("list", help="List providers or bots")
    list_cmd.add_argument("--type", choices=["providers", "bots"], required=True,
                        help="Type to list")

    # Parse arguments
    args = parser.parse_args()

    # Process commands
    if args.command == "add-provider":
        add_provider_config(args)
    elif args.command == "add-bot":
        add_bot_config(args)
    elif args.command == "list":
        list_configs(args.type)
    else:
        parser.print_help()

# Implementation of commands...
```

## Risk Mitigation

1. **Backward Compatibility**: Ensure all existing functionality continues to work
2. **Fallback Mechanism**: If new framework fails, fall back to original framework
3. **Comprehensive Testing**: Test with various providers before deployment
4. **Graceful Degradation**: If a specific provider fails, other providers should continue working

## Timeline and Resource Requirements

- **Total Timeline**: 8-10 days for full implementation
- **Development Resources**: 1-2 developers
- **Testing Resources**: 1 QA engineer
- **Documentation**: Updates to README and examples

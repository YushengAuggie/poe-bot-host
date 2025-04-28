# API Key Management for Poe Bots

This guide explains how to manage API keys for your Poe bots, especially when deploying to Modal.

## Preventing Secret Leaks

This repository uses pre-commit hooks to prevent accidentally committing secrets:

1. Install pre-commit:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

2. The pre-commit hooks will automatically:
   - Detect private keys
   - Detect AWS credentials
   - Run Gitleaks to scan for other sensitive data patterns

If a commit is blocked due to potential secrets, carefully review the output and remove any sensitive information before committing again.

## API Key Setup

### Local Development

For local development, you can set API keys as environment variables:

```bash
# For bash/zsh
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
# Per-bot access keys (get these from your bot's settings page on Poe)
export MY_BOT_NAME_ACCESS_KEY="your-bot-specific-access-key"
export ANOTHER_BOT_ACCESS_KEY="another-bot-specific-access-key"

# For Windows Command Prompt
set OPENAI_API_KEY=your-openai-key
set GOOGLE_API_KEY=your-google-key
set MY_BOT_NAME_ACCESS_KEY=your-bot-specific-access-key

# For PowerShell
$env:OPENAI_API_KEY="your-openai-key"
$env:GOOGLE_API_KEY="your-google-key"
$env:MY_BOT_NAME_ACCESS_KEY="your-bot-specific-access-key"
```

Alternatively, create a `.env` file in the project root (don't commit this file):

```bash
# Third-party API keys
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-key

# Bot-specific access keys
# The system supports multiple naming formats - use any consistent style
MY_BOT_NAME_ACCESS_KEY=your-bot-specific-access-key
ANOTHER_BOT_ACCESS_KEY=another-bot-specific-access-key
GEMINI_2_5_FLASH_ACCESS_KEY=your-gemini-bot-access-key
ECHOBOT_ACCESS_KEY=your-echo-bot-access-key
```

### Modal Deployment

For Modal deployment, create secrets with the same name as your environment variables:

```bash
# Create secrets in Modal
modal secret create OPENAI_API_KEY "your-openai-key"
modal secret create GOOGLE_API_KEY "your-google-key"
# Per-bot access keys
modal secret create MY_BOT_NAME_ACCESS_KEY "your-bot-specific-access-key"
modal secret create ANOTHER_BOT_ACCESS_KEY "another-bot-specific-access-key"
```

## How API Key Management Works

### Third-Party Service Keys

For third-party services (OpenAI, Google, etc.):
- Check for an exact environment variable match (e.g., `OPENAI_API_KEY`)
- Check for a local testing key (e.g., `LOCAL_OPENAI_API_KEY`)
- If running in Modal, try to retrieve the key from Modal secrets

### Bot Access Keys

The system has been redesigned to use a flexible approach for managing bot-specific access keys:

1. **Bot Instance Method**: Each bot automatically looks for its own access key using:
   - The bot's name to generate potential environment variable names
   - Fuzzy matching to find similar names
   - Support for multiple naming conventions

2. **No Hardcoded Names**: There's no need to modify the code when adding new bots. Just:
   - Add your bot's access key to your `.env` file or environment
   - Name it following any of the supported patterns (see examples below)
   - The system will automatically find it

3. **Flexible Naming**: The system supports multiple naming formats for each bot:
   - `BOT_NAME_ACCESS_KEY` (standard format with original casing)
   - `BOT_NAME_UPPERCASED_ACCESS_KEY` (uppercase version)
   - `BOT_NAME_WITH_UNDERSCORES_ACCESS_KEY` (replacing dashes with underscores)
   - `BOTNAMEWITHOUTDASHES_ACCESS_KEY` (removing all dashes)
   - And many other variations

## .env File Example

Create a `.env` file in your project root to manage all your keys in one place:

```bash
# Example .env file for Poe Bots
# ==============================
# DO NOT COMMIT THIS FILE TO YOUR REPOSITORY!

# Third-party API keys
# --------------------
OPENAI_API_KEY=sk-...your-openai-key...
GOOGLE_API_KEY=AIza...your-google-key...

# Bot-specific access keys
# -----------------------
# Standard format (uppercase with underscores)
ECHO_BOT_ACCESS_KEY=psk_...your-access-key...
CALCULATOR_BOT_ACCESS_KEY=psk_...your-access-key...

# Alternative formats (all of these work)
WeatherBot_ACCESS_KEY=psk_...your-access-key...
WEATHERBOT_ACCESS_KEY=psk_...your-access-key...
WEATHER_BOT_ACCESS_KEY=psk_...your-access-key...

# Special case for Gemini bots
GEMINI_2_5_FLASH_ACCESS_KEY=psk_...your-access-key...
GEMINI_2_5_PRO_ACCESS_KEY=psk_...your-access-key...

# For branded bots (e.g. with initials like "JY-EchoBot")
JY_ECHOBOT_ACCESS_KEY=psk_...your-access-key...
```

For this to work, you need to load the `.env` file in your application. Add this to your main script:

```python
# At the top of your main script
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
```

Make sure to install the `python-dotenv` package:

```bash
pip install python-dotenv
```

## Troubleshooting

### Key Not Found

If you get a "Key not found" error:

1. **Local environment**: Check that the environment variable is set
   ```bash
   # Check if environment variables are set
   echo $OPENAI_API_KEY
   echo $GOOGLE_API_KEY
   # Check for bot-specific access keys
   env | grep ACCESS_KEY
   ```

2. **Modal deployment**: Verify secrets are correctly created
   ```bash
   # List Modal secrets
   modal secret list
   ```

3. **Debug logging**: Run the sync script with verbose logging
   ```bash
   python sync_bot_settings.py --bot YOUR_BOT_NAME -v
   ```

### Bot-Specific Access Keys

For each bot that needs to sync settings, you need to define its access key:

1. Obtain the bot-specific access key from the bot's settings page on Poe
2. Set it as an environment variable following one of these formats:
   - `BOT_NAME_ACCESS_KEY`
   - `BOTNAME_ACCESS_KEY`
   - `BOT_NAME` with underscores instead of hyphens

   Example:
   ```bash
   # For a bot named "my-cool-bot"
   export MY_COOL_BOT_ACCESS_KEY="your-bot-specific-access-key"
   # Alternative format
   export MYCOOLBOT_ACCESS_KEY="your-bot-specific-access-key"
   ```

3. For special bot names like those with Gemini, special formats are also supported:
   ```bash
   # For Gemini bots, both of these formats work
   export GEMINI_2_5_FLASH_ACCESS_KEY="..."
   export GEMINI25FLASH_ACCESS_KEY="..."
   ```

### Testing API Keys

Use the verification tools included in the project:

```bash
# Test locally
python scripts/diagnostics/modal_api_key_checker.py

# Test on Modal
modal run scripts/diagnostics/modal_secrets_diagnostics.py

# Test Gemini API key specifically
python scripts/diagnostics/gemini_api_key_checker.py
```

## Using API Keys in Your Bot

To use API keys in your bot, simply import the `get_api_key` function:

```python
from utils.api_keys import get_api_key

def initialize_client():
    # Get the API key
    api_key = get_api_key("OPENAI_API_KEY")

    # Initialize your client with the API key
    client = YourAPIClient(api_key=api_key)
    return client
```

## Setting Up New API Keys

To add a new API key:

1. Choose a consistent name for your API key (e.g., `CUSTOM_SERVICE_API_KEY`)
2. Add it to your environment for local development
3. Create a Modal secret with the same name
4. Add the secret to your Modal function deployment

Example Modal function with multiple API keys:

```python
@app.function(
    secrets=[
        modal.Secret.from_name("OPENAI_API_KEY"),
        modal.Secret.from_name("GOOGLE_API_KEY"),
        modal.Secret.from_name("CUSTOM_SERVICE_API_KEY")
    ]
)
def my_function():
    # All API keys are now available via get_api_key()
    pass
```

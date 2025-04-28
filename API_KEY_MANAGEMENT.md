# API Key Management for Poe Bots

This guide explains how to manage API keys for your Poe bots, especially when deploying to Modal.

## API Key Setup

### Local Development

For local development, you can set API keys as environment variables:

```bash
# For bash/zsh
export OPENAI_API_KEY="your-openai-key"
export GOOGLE_API_KEY="your-google-key"
export POE_ACCESS_KEY="your-poe-access-key"
export POE_API_KEY="your-poe-api-key"  # Get this from https://poe.com/api_key

# For Windows Command Prompt
set OPENAI_API_KEY=your-openai-key
set GOOGLE_API_KEY=your-google-key
set POE_ACCESS_KEY=your-poe-access-key
set POE_API_KEY=your-poe-api-key

# For PowerShell
$env:OPENAI_API_KEY="your-openai-key"
$env:GOOGLE_API_KEY="your-google-key"
$env:POE_ACCESS_KEY="your-poe-access-key"
$env:POE_API_KEY="your-poe-api-key"
```

Alternatively, create a `.env` file in the project root (don't commit this file):

```bash
OPENAI_API_KEY=your-openai-key
GOOGLE_API_KEY=your-google-key
POE_ACCESS_KEY=your-poe-access-key
POE_API_KEY=your-poe-api-key
```

### Modal Deployment

For Modal deployment, create secrets with the same name as your environment variables:

```bash
# Create secrets in Modal
modal secret create OPENAI_API_KEY "your-openai-key"
modal secret create GOOGLE_API_KEY "your-google-key"
modal secret create POE_ACCESS_KEY "your-poe-access-key"
modal secret create POE_API_KEY "your-poe-api-key"  # Get this from https://poe.com/api_key
```

## How API Key Management Works

The system uses a simple approach to find API keys:

1. Check for an exact environment variable match (e.g., `OPENAI_API_KEY`)
2. Check for a local testing key (e.g., `LOCAL_OPENAI_API_KEY`)
3. If running in Modal, try to retrieve the key from Modal secrets

## Troubleshooting

### Key Not Found

If you get a "Key not found" error:

1. **Local environment**: Check that the environment variable is set
   ```bash
   # Check if environment variables are set
   echo $OPENAI_API_KEY
   echo $GOOGLE_API_KEY
   ```

2. **Modal deployment**: Verify secrets are correctly created
   ```bash
   # List Modal secrets
   modal secret list
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

# API Key Management System

This guide describes how to use the API key management system for both local development and Modal deployment.

## Setting Up API Keys

### Local Development

For local development, you can set environment variables in a `.env` file:

```
# .env file example
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here

# For Google service account credentials, you can use one of these:
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/credentials.json
# or
GOOGLE_APPLICATION_CREDENTIALS_JSON={"key": "value", "other_key": "other_value"}
```

### Modal Deployment

For Modal deployment, set up secrets:

```bash
# Set up OpenAI secret
modal secret create openai-secret OPENAI_API_KEY=your_openai_key_here

# Set up Google secret
modal secret create googlecloud-secret GOOGLE_API_KEY=your_google_key_here
```

For Google service account credentials, you can also use:

```bash
# Setting the Google API key
modal secret create googlecloud-secret GOOGLE_API_KEY=your_google_api_key_here
```

## Using the API Keys

In your code, import and use the API key helpers:

```python
from utils.api_keys import get_openai_api_key, get_google_api_key

# Get OpenAI API key (checks local env then Modal secrets)
openai_key = get_openai_api_key()

# Get Google API key (checks local env then Modal secrets)
google_key = get_google_api_key()
```

For Modal apps, use the helper functions:

```python
from utils.api_keys import create_modal_app, get_function_secrets

# Create a Modal app with required packages
app = create_modal_app("your-app-name", ["openai", "google-generativeai"])

# Add secrets to functions
@app.function(secrets=get_function_secrets(["openai", "google"]))
def your_function():
    # Your code here
    pass
```

## Example: Modal Bot

See `modal_bot_example.py` for a complete example that demonstrates:

1. Running locally with environment variables
2. Running on Modal with secrets
3. Seamless switching between the two environments

```bash
# Run locally
python examples/modal_bot_example.py --local

# Deploy to Modal
modal deploy examples/modal_bot_example.py

# Run on Modal
modal run examples/modal_bot_example.py
```

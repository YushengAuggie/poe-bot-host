# Setting Up Modal Secrets for API Keys

This document explains how to set up Modal secrets for your API keys, which allows your bots to access external APIs like OpenAI and Google when deployed to Modal.

## What are Modal Secrets?

Modal secrets are a secure way to store sensitive information like API keys. When your code runs in Modal, these secrets are securely injected into the environment of your functions.

## Option 1: Using a Single KEY Secret (Recommended)

The simplest approach is to create a single `KEY` secret that contains all your API keys:

```bash
# Create the KEY secret
modal secret create KEY
```

When prompted, add your API keys in the format:

```
OPENAI_API_KEY=sk-...your-openai-key...
GOOGLE_API_KEY=...your-google-key...
```

Then use the secret in your Modal functions:

```python
import os
import modal

app = modal.App()

@app.function(secrets=[modal.Secret.from_name("KEY")])
def my_function():
    # Access keys from environment variables
    openai_key = os.environ["OPENAI_API_KEY"]
    google_key = os.environ["GOOGLE_API_KEY"]
    # Use the keys...
```

## Option 2: Using Service-Specific Secrets

For more granular control, you can create separate secrets for each service:

```bash
# Create OpenAI-specific secret
modal secret create OPENAI_API_KEY
# Enter your OpenAI API key when prompted

# Create Google-specific secret
modal secret create GOOGLE_API_KEY
# Enter your Google API key when prompted
```

Then use the secrets in your functions:

```python
@app.function(secrets=[
    modal.Secret.from_name("OPENAI_API_KEY"),
    modal.Secret.from_name("GOOGLE_API_KEY")
])
def my_function():
    # Access keys from environment variables
    openai_key = os.environ["OPENAI_API_KEY"]
    google_key = os.environ["GOOGLE_API_KEY"]
    # Use the keys...
```

## Using the API Key Management System

This repository includes a helper system in `utils/api_keys.py` that makes working with API keys easier:

```python
from utils.api_keys import get_function_secrets, get_openai_api_key, get_google_api_key

# Create the Modal app
app = modal.App()

# Add secrets to a function
@app.function(secrets=get_function_secrets(["openai", "google"]))
def my_function():
    # This will check local environment first, then Modal secrets
    openai_key = get_openai_api_key()
    google_key = get_google_api_key()
    # Use the keys...
```

The `get_function_secrets()` function automatically handles both the combined `KEY` secret and service-specific secrets, trying each approach.

## Testing Your Setup

Run the example script to verify your Modal secrets are working:

```bash
modal run examples/modal_api_key_example.py
```

This will show which API keys are available in your Modal environment.

## Debugging Common Issues

1. **Secret not found errors**: Make sure you've created the secrets in Modal with the exact names expected
2. **Empty API keys**: Verify that the values were properly set when creating the secrets
3. **Local vs remote environments**: Remember that local environment variables won't be available in Modal unless explicitly added as secrets

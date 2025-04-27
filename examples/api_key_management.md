# Simplified API Key Management

This guide describes how to use the simplified API key management system for both local development and Modal deployment.

## Setting Up API Keys

### Local Development

For local development, set environment variables in a `.env` file:

```
# .env file example
OPENAI_API_KEY=your_openai_key_here
GOOGLE_API_KEY=your_google_key_here
CUSTOM_SERVICE_API_KEY=your_custom_service_key_here
```

### Modal Deployment

For Modal deployment, create secrets with the same names as your environment variables:

```bash
# Set up API keys as Modal secrets
modal secret create OPENAI_API_KEY sk-your-openai-key-here
modal secret create GOOGLE_API_KEY your-google-key-here
modal secret create CUSTOM_SERVICE_API_KEY your-custom-service-key-here
```

## Using the API Keys in Your Code

Import and use the generic API key function:

```python
from utils.api_keys import get_api_key

# Get API keys by their environment variable names
openai_key = get_api_key("OPENAI_API_KEY")
google_key = get_api_key("GOOGLE_API_KEY")
custom_key = get_api_key("CUSTOM_SERVICE_API_KEY")
```

The `get_api_key` function:
1. First checks environment variables
2. Then checks Modal secrets if running in a Modal environment
3. Raises ValueError if the key is not found

## Example: Using API Keys with Service Clients

```python
import os
from openai import OpenAI
from utils.api_keys import get_api_key

# Initialize OpenAI client
def get_openai_client():
    try:
        return OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return None

# Initialize Google client
def get_google_client():
    try:
        import google.generativeai as genai
        genai.configure(api_key=get_api_key("GOOGLE_API_KEY"))
        return genai.GenerativeModel("gemini-1.0-pro")
    except Exception as e:
        print(f"Failed to initialize Google client: {e}")
        return None
```

## Modal Function with API Keys

For Modal applications, use Modal's secret mounting:

```python
import modal
from modal import App, Secret

app = App("my-app")

@app.function(secrets=[Secret.from_name("OPENAI_API_KEY")])
def function_using_openai():
    # The key will be available as an environment variable
    from utils.api_keys import get_api_key
    
    openai_key = get_api_key("OPENAI_API_KEY")
    # Use the key...
    
# Or use multiple secrets
@app.function(secrets=[
    Secret.from_name("OPENAI_API_KEY"), 
    Secret.from_name("GOOGLE_API_KEY")
])
def function_using_multiple_apis():
    # Both keys will be available
    from utils.api_keys import get_api_key
    
    openai_key = get_api_key("OPENAI_API_KEY")
    google_key = get_api_key("GOOGLE_API_KEY")
    # Use the keys...
```

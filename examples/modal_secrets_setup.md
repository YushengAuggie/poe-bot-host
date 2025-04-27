# Setting Up Modal Secrets for API Keys

This document explains how to set up Modal secrets for your API keys using the simplified API key management system.

## What are Modal Secrets?

Modal secrets are a secure way to store sensitive information like API keys. When your code runs in Modal, these secrets are securely injected into the environment of your functions.

## Creating API Key Secrets

With the simplified API key approach, create individual secrets with the same names as your environment variables:

```bash
# Create API key secrets directly
modal secret create OPENAI_API_KEY sk-...your-openai-key...
modal secret create GOOGLE_API_KEY ...your-google-key...
modal secret create CUSTOM_SERVICE_API_KEY ...your-custom-key...
```

This creates secrets that match the environment variable names used in your local development environment.

## Using API Key Secrets in Modal Functions

Use the secrets in your Modal functions:

```python
import modal
from modal import App, Secret

app = App("my-app")

# Use a single API key
@app.function(secrets=[Secret.from_name("OPENAI_API_KEY")])
def openai_function():
    from utils.api_keys import get_api_key
    
    # get_api_key checks environment variables first, then Modal secrets
    openai_key = get_api_key("OPENAI_API_KEY")
    # Use the key...

# Use multiple API keys
@app.function(secrets=[
    Secret.from_name("OPENAI_API_KEY"),
    Secret.from_name("GOOGLE_API_KEY")
])
def multi_api_function():
    from utils.api_keys import get_api_key
    
    # Access multiple API keys
    openai_key = get_api_key("OPENAI_API_KEY")
    google_key = get_api_key("GOOGLE_API_KEY")
    # Use the keys...
```

## Creating an App with All Secrets

If your app needs access to all API keys:

```python
import modal
from modal import App, Secret

# List all secrets your app needs
needed_secrets = [
    Secret.from_name("OPENAI_API_KEY"),
    Secret.from_name("GOOGLE_API_KEY"),
    # Add more as needed
]

app = App("my-app")

@app.function(secrets=needed_secrets)
def my_function():
    from utils.api_keys import get_api_key
    
    # All secrets are available
    openai_key = get_api_key("OPENAI_API_KEY")
    google_key = get_api_key("GOOGLE_API_KEY")
    # Use the keys...
```

## Testing Your Modal Secrets

Verify your Modal secrets are working:

```bash
# Create a test script
cat > test_modal_secrets.py << EOL
import modal
from modal import App, Secret
import os

app = App("secret-test")

@app.function(secrets=[
    Secret.from_name("OPENAI_API_KEY"),
    Secret.from_name("GOOGLE_API_KEY")
])
def test_secrets():
    print("Available API keys:")
    for key_name in ["OPENAI_API_KEY", "GOOGLE_API_KEY"]:
        if key_name in os.environ and os.environ[key_name]:
            print(f"- {key_name}: {os.environ[key_name][:5]}...")
        else:
            print(f"- {key_name}: NOT FOUND")

if __name__ == "__main__":
    with app.run():
        test_secrets.remote()
EOL

# Run the test
modal run test_modal_secrets.py
```

## Troubleshooting Modal Secrets

If you encounter issues:

1. **Secret not found errors**: 
   - Verify the secret exists: `modal secret list`
   - Check that secret names match exactly (case-sensitive)

2. **Access issues**: 
   - Make sure the secret is properly mounted: `@app.function(secrets=[Secret.from_name("SECRET_NAME")])`
   - Test with a simple print statement to ensure the secret is accessible

3. **Value problems**:
   - If your API key is visible but not working, update it: `modal secret update SECRET_NAME new-value`
   - Some secrets might have special characters that need proper escaping

4. **Local vs Modal environment**:
   - Remember that local environment variables are not automatically available in Modal
   - The `get_api_key` function will check both environments appropriately

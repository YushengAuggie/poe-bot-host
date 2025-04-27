# Scripts Directory

This directory contains utility scripts, tools, and diagnostic helpers for the Poe Bots framework.

## Bot Testing

- **`test_bot_cli.py`**: Command-line tool for testing bots running in the Poe Bots Framework
  ```bash
  # Test the first available bot
  python scripts/test_bot_cli.py
  
  # Test a specific bot
  python scripts/test_bot_cli.py --bot EchoBot
  
  # Test with a custom message
  python scripts/test_bot_cli.py --bot ReverseBot --message "Reverse this text"
  
  # Check API health
  python scripts/test_bot_cli.py --health
  ```

## Diagnostic Tools

These scripts help diagnose issues with API keys and Modal deployment:

### `diagnostics/modal_api_key_checker.py`

Verifies API key configuration in a Modal deployment:
```bash
python scripts/diagnostics/modal_api_key_checker.py
# Or deploy to Modal
modal deploy scripts/diagnostics/modal_api_key_checker.py
```

### `diagnostics/modal_secrets_diagnostics.py`

Runs comprehensive diagnostics on Modal secrets:
```bash
modal run scripts/diagnostics/modal_secrets_diagnostics.py
```

### `diagnostics/secret_access_tester.py`

Tests direct access to Modal secrets:
```bash
modal run scripts/diagnostics/secret_access_tester.py
```

### `diagnostics/gemini_api_key_checker.py`

Specifically tests Gemini API key configuration:
```bash
# Test locally
python scripts/diagnostics/gemini_api_key_checker.py
# Or deploy to Modal
modal deploy scripts/diagnostics/gemini_api_key_checker.py
```

## Creating New Scripts

When creating new utility scripts:
1. Place them in the appropriate directory based on purpose
2. Include a detailed docstring explaining the script's purpose
3. Add executable permissions: `chmod +x scripts/your_script.py`
4. Add usage examples to this README
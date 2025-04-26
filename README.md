# Poe Bot Host

English | [简体中文](README.zh-CN.md)

A multi-bot hosting framework for the Poe platform.

## Overview

The Poe Bot Host is a comprehensive platform for creating, testing, deploying, and managing multiple bots on the Poe platform. This framework simplifies the process of developing and hosting bots by providing:

- A unified API for hosting multiple bots
- A robust base bot architecture with error handling
- Automatic bot discovery and registration
- Comprehensive testing tools
- Simplified deployment process with Modal integration
- Standardized logging and error reporting
- Multiple example bots with different capabilities:
  - Basic bots (Echo, Reverse, Uppercase)
  - Advanced bots (BotCaller, Weather, WebSearch)
  - Functional bots (Calculator, Function Calling, File Analyzer)

### Documentation

- [QUICKSTART.md](QUICKSTART.md): Get started in 5 minutes with a minimal setup
- [DEPLOYMENT.md](DEPLOYMENT.md): Step-by-step guide with screenshots for deployment
- [examples/](examples/): Example bots and implementation guides
- This README: Complete documentation for the framework

## What is Poe?

[Poe](https://poe.com/) is a platform for interacting with AI models and custom bots. Poe allows developers to create custom bots that can be used by anyone on the platform. This framework makes it easy to create and deploy Poe bots.

## Getting Started

### Prerequisites

- Python 3.7+ installed
- A [Poe](https://poe.com/) account for testing and deployment
- [Modal](https://modal.com/) account for cloud deployment (optional)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/YushengAuggie/poe-bot-host.git
   cd poe-bot-host
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - macOS/Linux: `source venv/bin/activate`

4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Key Concepts

### Bot Architecture

- **BaseBot**: The abstract base class that provides common functionality
- **Bot Factory**: Automatically discovers and loads bots from the `bots` directory
- **API**: FastAPI-based API that hosts all bots

### Directory Structure

```
poe_bots/
├── app.py              # Main API application
├── run_local.py        # Script to run the platform locally
├── run_local.sh        # Helper script with CLI options
├── test_bot.py         # Script to test bots
├── Makefile            # Common commands for development
├── QUICKSTART.md       # Quick start guide
├── DEPLOYMENT.md       # Detailed deployment guide
├── .env.example        # Environment variable template
├── pyproject.toml      # Python project configuration
├── setup.py            # Package setup for compatibility
├── requirements.txt    # Python dependencies
├── bots/               # Bot implementations
│   ├── __init__.py     # Package initialization
│   ├── echo_bot.py     # Echo bot implementation
│   ├── reverse_bot.py  # Reverse bot implementation
│   ├── uppercase_bot.py # Uppercase bot implementation
│   └── template_bot.py # Template for creating new bots
├── examples/           # Example bots and guides
│   ├── README.md       # Examples documentation
│   ├── standalone_echobot.py # Standalone bot example
│   ├── weather_bot.py  # More complex bot example
│   └── add_weather_bot.md # Guide for adding weather bot
├── tests/              # Test suite
│   ├── __init__.py     # Test package initialization
│   ├── conftest.py     # Pytest configuration
│   ├── test_base_bot.py # Tests for BaseBot class
│   └── test_bot_factory.py # Tests for BotFactory class
└── utils/              # Utility modules
    ├── __init__.py     # Package initialization
    ├── base_bot.py     # Base bot class with common functionality
    ├── bot_factory.py  # Factory for discovering and creating bots
    └── config.py       # Configuration management
```

## Running the Platform Locally

You can run the platform locally for development and testing. There are two equivalent ways to start the server:

### Using the Shell Script

The shell script automatically activates your virtual environment before running:

```bash
# Basic run
./run_local.sh

# Development mode with auto-reload
./run_local.sh --reload

# Debug mode with verbose logging
./run_local.sh --debug

# Custom port
./run_local.sh --port 9000

# All options combined
./run_local.sh --debug --reload --port 9000 --host 127.0.0.1
```

### Using Python Directly

If you prefer to use Python directly (make sure your virtual environment is activated):

```bash
python run_local.py --debug --reload
```

### Available Options

Both methods support the same options:

| Option | Description |
|--------|-------------|
| `--host` | Host to bind to (default: 0.0.0.0) |
| `--port` | Port to bind to (default: 8000) |
| `--reload` | Enable auto-reload for development |
| `--debug` | Enable debug mode with verbose logging |
| `--log-level` | Set logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |
| `--no-banner` | Don't display the banner |
| `--help` | Show help message and exit |

The server will start at http://localhost:8000 by default with all discovered bots available.

## Testing Bots

The platform includes a comprehensive testing tool:

```bash
# Test the first available bot
python test_bot.py

# Test a specific bot
python test_bot.py --bot EchoBot

# Test with a custom message
python test_bot.py --bot ReverseBot --message "Reverse this text"

# Check API health
python test_bot.py --health

# Show API endpoints
python test_bot.py --schema
```

### Manual Testing with curl

To test your bots manually with curl, first make sure your server is running:

```bash
# Start the server in one terminal
source venv/bin/activate  # Make sure your virtual environment is activated
./run_local.sh  # Or python run_local.py
```

Then in a separate terminal, you can send requests:

```bash
# Get a list of available bots
curl http://localhost:8000/bots

# Check API health
curl http://localhost:8000/health

# Test a specific bot (replace echobot with your bot's name in lowercase)
curl -X POST "http://localhost:8000/echobot" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummytoken" \
  -d '{
    "version": "1.0",
    "type": "query",
    "query": [
        {"role": "user", "content": "Hello world!"}
    ],
    "user_id": "test_user",
    "conversation_id": "test_convo_123",
    "message_id": "test_msg_123",
    "protocol": "poe"
  }'
```

## Included Bots

The framework comes with several example bots that demonstrate different capabilities:

### Basic Bots

- **EchoBot**: Echoes back the user's message
- **ReverseBot**: Returns the user's message in reverse
- **UppercaseBot**: Converts the user's message to uppercase
- **TemplateBot**: A template for creating new bots

### Advanced Bots

- **BotCallerBot**: A bot that can call other bots in the framework
- **WeatherBot**: Provides weather information for any location
- **WebSearchBot**: Searches the web for information

### Functional Bots

- **CalculatorBot**: Performs various mathematical calculations
- **FunctionCallingBot**: Demonstrates function calling capabilities
- **FileAnalyzerBot**: Analyzes uploaded files and provides statistics

## Creating a New Bot

### Using the Template

The simplest way to create a new bot is to copy and modify the template:

1. Copy `bots/template_bot.py` to `bots/your_bot_name.py`
2. Modify the class name, bot name, and description
3. Implement your logic in the `get_response` method

Example:
```python
import json
from typing import AsyncGenerator, Union
from fastapi_poe.types import PartialResponse, QueryRequest, MetaResponse
from utils.base_bot import BaseBot

class WeatherBot(BaseBot):
    """Bot that provides weather information."""

    bot_name = "WeatherBot"
    bot_description = "Provides weather information for specified locations"
    version = "1.0.0"

    async def get_response(self, query: QueryRequest) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        # Extract the message
        user_message = self._extract_message(query)

        # Handle bot info requests
        if user_message.lower().strip() == "bot info":
            metadata = self._get_bot_metadata()
            yield PartialResponse(text=json.dumps(metadata, indent=2))
            return

        # Parse the message to get location
        location = user_message.strip()

        # In a real bot, you would call a weather API here
        weather_info = f"The weather in {location} is sunny with a high of 75°F."

        # Return the response
        yield PartialResponse(text=weather_info)
```

### Bot Configuration Options

Your bot can include various configuration options:

```python
class ConfigurableBot(BaseBot):
    bot_name = "ConfigurableBot"
    bot_description = "A bot with custom configuration"
    version = "1.0.0"

    # Custom settings
    max_message_length = 5000  # Override default (2000)
    stream_response = False    # Disable streaming

    # You can add your own settings too
    api_key = "default-key"   # Custom setting

    def __init__(self, **kwargs):
        # Initialize with settings from environment or kwargs
        settings = {
            "api_key": os.environ.get("MY_BOT_API_KEY", self.api_key)
        }
        super().__init__(settings=settings, **kwargs)
```

### Error Handling

The framework provides built-in error handling with two error types:

- `BotError`: Regular errors that can be retried
- `BotErrorNoRetry`: Errors that should not be retried

Example usage:

```python
from utils.base_bot import BaseBot, BotError, BotErrorNoRetry

class ErrorHandlingBot(BaseBot):
    # ...

    async def _process_message(self, message: str, query: QueryRequest):
        try:
            # Some code that might fail
            if not self._is_valid_input(message):
                # User error - don't retry
                raise BotErrorNoRetry("Invalid input format. Please try something else.")

            result = await self._fetch_external_data(message)
            if not result:
                # Service error - can retry
                raise BotError("Service unavailable. Please try again later.")

            yield PartialResponse(text=result)

        except Exception as e:
            # Handle unexpected errors
            self.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            raise  # Framework will handle it
```

## Deploying Bots

### Step-by-Step Deployment to Modal

[Modal](https://modal.com/) is a cloud platform for running Python code. Here's a detailed guide for deploying your Poe bots to Modal:

#### 1. Set Up Modal

1. Sign up for a Modal account at [modal.com](https://modal.com/signup)

2. Install the Modal client in your environment:
   ```bash
   pip install modal-client
   ```

3. Authenticate with Modal by running:
   ```bash
   modal token new
   ```
   - This will open a browser window
   - Log in to your Modal account
   - The CLI will automatically save your authentication token

#### 2. Deploy Your Bots

To deploy the entire framework with all your bots:

```bash
modal deploy app.py
```

You'll see output similar to this:

```
Modal app poe-bots deployed!
Endpoints:
→ web_endpoint: https://yourname--poe-bots-fastapi-app.modal.run
```

Make note of this URL - you'll need it to configure your bots on Poe.

If you want to deploy a standalone bot instead (using the example_standalone_bot.py), run:

```bash
modal deploy example_standalone_bot.py
```

This will output a URL specific to that standalone bot.

#### 3. Test Your Deployment

Before configuring on Poe, test that your deployment is working:

```bash
# Test health endpoint
curl https://yourname--poe-bots-fastapi-app.modal.run/health

# List available bots
curl https://yourname--poe-bots-fastapi-app.modal.run/bots
```

You should see a successful response with health information and a list of available bots.

### Setting Up Bots on Poe

Now that your API is deployed, you can create bots on Poe that connect to your API:

#### 1. Create a Bot on Poe

1. Go to the [Poe Creator Portal](https://creator.poe.com/)
2. Click "Create Bot" (or "Create a bot" button)
3. Fill in the basic details:
   - **Name**: A unique name for your bot (e.g., "EchoBot")
   - **Description**: A clear description of what your bot does
   - **Instructions/Prompt**: Optional instructions for the bot

#### 2. Configure Server Settings

1. In the bot creation form, scroll down to "Server Bot Settings" and select "Server bot" as the bot type
2. Configure the API:
   - **Server Endpoint**: Your Modal deployment URL + the specific bot path
     - Format: `https://yourusername--poe-bots-fastapi-app.modal.run/botname`
     - Example for EchoBot: `https://yourusername--poe-bots-fastapi-app.modal.run/echobot`
     - Note that the path is the lowercase version of your bot's name
   - **API Protocol**: Select "Poe Protocol"
   - **API Key Protection**: Select "No protection" (or configure an API key if you've set one up)

#### 3. Additional Settings (Optional)

1. **Message Feedback**: Choose whether to allow user feedback
2. **Sample Messages**: Add sample conversation starters
3. **Profile Picture**: Upload a custom image for your bot
4. **Knowledge Base**: Add documents for reference (if needed)

#### 4. Save and Test

1. Click "Create Bot" to save your configuration
2. After creation, you'll be redirected to a chat with your bot
3. Test your bot by sending a message
4. The message will be sent to your Modal-hosted API, processed by your bot, and the response will be displayed in the chat

### Troubleshooting Deployment

If you encounter issues with your deployment:

1. **Connection Issues**:
   - Verify the URL is correct in your Poe bot configuration
   - Ensure the bot name in the URL matches the lowercase version of your bot class name
   - Test the API directly with curl to confirm it's responding

2. **Error Responses**:
   - Check the Modal logs: `modal app logs poe-bots`
   - Look for error messages or exceptions

3. **Authentication Issues**:
   - If using API key protection, ensure the key is correct in both your code and Poe configuration

4. **Deployment Failures**:
   - Check your requirements.txt for compatible packages
   - Ensure your code doesn't have any syntax errors or import issues

### Updating Your Deployment

To update your bots after making changes:

1. Make your code changes locally
2. Test locally using `./run_local.sh`
3. When ready, redeploy with `modal deploy app.py`
4. The existing deployment will be updated with your changes

### Managing Costs

Modal offers a free tier that is sufficient for many Poe bots. If your usage grows:

1. Monitor your usage in the Modal dashboard
2. Set up billing if you exceed the free tier
3. Consider optimizing your code to reduce compute time and memory usage

## Maintenance and Development

This framework is designed for easy maintenance and extensibility. Here's how to keep it running smoothly and add new features.

### Version Management

The current version is 1.0.0. When making significant changes:

1. Update the version number in:
   - `pyproject.toml`
   - `app.py` (the `__version__` variable)
   - Any documentation references

2. Follow semantic versioning:
   - MAJOR: Breaking changes
   - MINOR: New features, backward compatible
   - PATCH: Bug fixes, backward compatible

### Adding New Features

To add new features to the framework:

1. For bot-specific features:
   - Add methods to the `BaseBot` class in `utils/base_bot.py`
   - Document them clearly with docstrings
   - Use type hints for better IDE support

2. For platform features:
   - Add endpoints to `app.py`
   - Update the `BotFactory` in `utils/bot_factory.py` as needed
   - Update tests to cover new functionality

3. For configuration options:
   - Add to `utils/config.py`
   - Update `.env.example` with the new options

### Logging and Monitoring

The framework uses Python's built-in logging. Configure log levels:

- In code: `logger.setLevel(logging.DEBUG)`
- Via environment: `DEBUG=true ./run_local.sh`
- Via CLI: `./run_local.sh --debug`

For production monitoring:
- Use Modal's built-in dashboards
- Consider adding structured logging for easier analysis
- Set up alerts for critical errors

### Testing Changes

Always test your changes:

1. Run the local server: `./run_local.sh --debug`
2. Test all bots: `python test_bot.py --all`
3. Test specific changes: `python test_bot.py --bot YourBot --message "Test message"`
4. Run automated tests: `make test`
5. Check linting: `make lint`
6. Verify formatting: `make format`

### Best Practices

1. **Bot Design**:
   - Keep bots focused on a single task
   - Use clear, descriptive names and documentation
   - Handle errors gracefully with specific error messages
   - Consider user experience in responses

2. **Code Organization**:
   - Follow Python's PEP 8 style guide
   - Use type hints for better IDE support
   - Document all public methods and classes
   - Keep related functionality together

3. **Performance**:
   - Use async functions for I/O-bound operations
   - Keep bots stateless where possible
   - Avoid unnecessary dependencies
   - Consider caching for repeated operations

4. **Security**:
   - Never store credentials in code
   - Use environment variables for sensitive information
   - Validate all user inputs
   - Keep dependencies updated

## Troubleshooting

### Common Issues

1. **Bot not found**:
   - Ensure the bot class inherits from `BaseBot`
   - Check that the file is in the `bots` directory
   - Verify that the bot name is unique

2. **Deployment errors**:
   - Check Modal credentials: `modal token new`
   - Verify requirements in `requirements.txt`

3. **Runtime errors**:
   - Run with debug mode: `./run_local.sh --debug`
   - Check logs for specific error messages

## Continuous Integration and Quality Assurance

The project uses GitHub Actions to run automated tests and code quality checks on every push and pull request.

### CI/CD Pipeline

The CI/CD pipeline runs:
- Unit tests with pytest on multiple Python versions
- Linting with ruff
- Type checking with pyright

You can check the status of the CI pipeline in the GitHub Actions tab of the repository.

### Pre-Commit Hooks

Pre-commit hooks are used to ensure code quality by automatically checking your code before each commit and push.

#### Setup Instructions

```bash
# Install the pre-commit tool
pip install pre-commit

# Install the pre-commit hooks
pre-commit install --install-hooks

# Also install pre-push hooks (for tests)
pre-commit install --hook-type pre-push
```

#### What the Hooks Check

On every commit:
- **Linting** (ruff): Checks code style and formatting
- **Type checking** (pyright): Verifies correct type usage
- **Security checks**: Detects private keys, debugging statements, etc.
- **File formatting**: Fixes trailing whitespace, line endings, etc.

On every push:
- **Tests** (pytest): Runs the entire test suite

#### Benefits

- Prevents committing code with errors or poor quality
- Provides immediate feedback on issues
- Ensures consistent code quality across the team
- Reduces the need for code review comments about style/formatting
- Some issues are fixed automatically

## Resources

- [Poe Documentation](https://creator.poe.com/docs)
- [fastapi-poe Documentation](https://github.com/poe-platform/fastapi-poe)
- [Modal Documentation](https://modal.com/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## License

MIT

## Author

Created with ❤️ by Yusheng Ding (auggie1024.d@gmail.com) for the Poe community.

---

*This framework is not officially affiliated with Poe, Modal, or Anthropic.*

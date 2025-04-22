# Poe Bot Host Quick Start Guide

This guide will help you quickly get started with the Poe Bot Host.

## Setup in 5 Minutes

### 1. Install

Clone the repository and install dependencies:

```bash
git clone https://github.com/YushengAuggie/poe-bot-host.git
cd poe-bot-host
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Locally

Start the server:

```bash
# Using the shell script (automatically activates venv)
./run_local.sh

# Or directly with Python (if venv is already activated)
python run_local.py
```

The server will start at http://localhost:8000 with a nice banner showing your configuration.

### 3. Test Your Bots

In a new terminal:

```bash
# List available bots
curl http://localhost:8000/bots

# Test a specific bot
python test_bot.py --bot EchoBot --message "Hello, world!"
```

### 4. Create Your Own Bot

Create a new file in the `bots` directory:

```bash
cp bots/template_bot.py bots/my_awesome_bot.py
```

Edit the file to implement your bot's logic:

```python
import json
from typing import AsyncGenerator, Union
from fastapi_poe.types import PartialResponse, QueryRequest, MetaResponse
from utils.base_bot import BaseBot

class MyAwesomeBot(BaseBot):
    """My awesome bot that does cool things."""
    
    bot_name = "MyAwesomeBot"
    bot_description = "A really cool bot that does awesome things"
    
    async def get_response(self, query: QueryRequest) -> AsyncGenerator[Union[PartialResponse, MetaResponse], None]:
        # Extract the user message
        user_message = self._extract_message(query)
        
        # Handle bot info requests (required)
        if user_message.lower().strip() == "bot info":
            metadata = self._get_bot_metadata()
            yield PartialResponse(text=json.dumps(metadata, indent=2))
            return
            
        # Your custom logic here
        response = f"You said: {user_message}\n\nHere's my awesome response!"
        yield PartialResponse(text=response)
```

Restart the server and test your new bot:

```bash
python test_bot.py --bot MyAwesomeBot
```

### 5. Deploy to Modal

```bash
# Authenticate with Modal
modal token new

# Deploy all bots
modal deploy app.py
```

Make note of the deployment URL.

### 6. Create a Bot on Poe

1. Go to [creator.poe.com](https://creator.poe.com/)
2. Click "Create Bot"
3. Set "Server Endpoint" to your Modal URL + bot name (lowercase)
   Example: `https://yourusername--poe-bot-host-fastapi-app.modal.run/myawesomebot`
4. Set API Protocol to "Poe Protocol"
5. Save and test your bot

## Next Steps

- Read [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions
- Explore the [README.md](README.md) for full documentation
- Check out the example bots in the `bots` directory for inspiration

## Common Commands

```bash
# Run locally
./run_local.sh

# Run with auto-reload for development
./run_local.sh --reload

# Run with debug logging
./run_local.sh --debug

# Run tests
make test

# Format code
make format

# Lint code
make lint

# Deploy to Modal
modal deploy app.py

# Check Modal logs
modal app logs poe-bot-host
```
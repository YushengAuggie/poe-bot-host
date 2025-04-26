"""
Package containing all the Poe bot implementations.

All bot classes are automatically loaded if they:
1. Inherit from BaseBot or PoeBot
2. Are defined in this package or submodules

To create a new bot:
1. Create a new file in this directory
2. Define a class that inherits from BaseBot
3. Override get_response to implement the bot's behavior

The bot will be automatically discovered and loaded when the app starts.
"""

# Import all bots for easy access
from .chatgpt import ChatgptBot
from .echo_bot import EchoBot
from .gemini import GeminiBot
from .reverse_bot import ReverseBot
from .uppercase_bot import UppercaseBot

# Add other bots here as you create them
# from .my_other_bot import MyOtherBot

# Export the bot classes
__all__ = ["EchoBot", "ReverseBot", "UppercaseBot", "ChatgptBot", "GeminiBot"]

import importlib
import inspect
import logging
import pkgutil
from typing import Dict, List, Type

from fastapi import FastAPI
from fastapi_poe import PoeBot, make_app

from utils.base_bot import BaseBot

logger = logging.getLogger(__name__)


class BotFactory:
    """Factory for creating and managing Poe bots."""

    @staticmethod
    def create_app(bot_classes: List[Type[PoeBot]], allow_without_key: bool = True) -> FastAPI:
        """Create a FastAPI app with the given bots.

        Args:
            bot_classes: List of bot classes to include in the app
            allow_without_key: Whether to allow requests without an API key

        Returns:
            A FastAPI app with the given bots
        """
        # Remove duplicates by class name to avoid path conflicts
        seen_classes = {}
        unique_bot_classes = []
        for bot_class in bot_classes:
            class_name = bot_class.__name__
            if class_name not in seen_classes:
                seen_classes[class_name] = True
                unique_bot_classes.append(bot_class)
            else:
                logger.warning(f"Duplicate bot class found: {class_name}, skipping")

        # Create bot instances
        bots = []
        for bot_class in unique_bot_classes:
            bot_instance = bot_class()

            # Ensure the bot has its access key set up properly
            if hasattr(bot_instance, "get_access_key"):
                access_key = bot_instance.get_access_key()
                if access_key and not hasattr(bot_instance, "access_key"):
                    # Set the access key on the bot instance so post_message_attachment can use it
                    bot_instance.access_key = access_key
                    logger.info(
                        f"Set access key for {getattr(bot_instance, 'bot_name', bot_class.__name__)}: {access_key[:10]}..."
                    )

            bots.append(bot_instance)

        # Log the bots that were created
        for bot in bots:
            if hasattr(bot, "bot_name"):
                logger.info(f"Created bot: {bot.bot_name}")
            else:
                logger.info(f"Created bot: {bot.__class__.__name__}")

        # Create and return the app
        return make_app(bots, allow_without_key=allow_without_key)

    @staticmethod
    def load_bots_from_module(module_name: str = "bots") -> List[Type[PoeBot]]:
        """Load all bot classes from a module.

        Args:
            module_name: The name of the module to load bots from

        Returns:
            A list of bot classes
        """
        try:
            # Import the module
            module = importlib.import_module(module_name)

            # Get all classes from the module that inherit from BaseBot or PoeBot directly
            bot_classes = []

            # First try to get all submodules and their classes
            for _, submodule_name, is_pkg in pkgutil.iter_modules(
                module.__path__, module.__name__ + "."
            ):
                if not is_pkg:  # If it's not a package but a module
                    try:
                        submodule = importlib.import_module(submodule_name)
                        for name, obj in inspect.getmembers(submodule):
                            # Skip BaseBot itself, only include concrete subclasses
                            if (
                                inspect.isclass(obj)
                                and obj is not BaseBot
                                and obj is not PoeBot  # Skip base classes
                                and (issubclass(obj, BaseBot) or issubclass(obj, PoeBot))
                            ):
                                bot_classes.append(obj)
                                logger.debug(f"Found bot class: {obj.__name__} in {submodule_name}")
                    except Exception as e:
                        logger.warning(f"Error loading submodule {submodule_name}: {str(e)}")

            # Also check for direct classes in the module
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    inspect.isclass(attr)
                    and attr is not BaseBot
                    and attr is not PoeBot  # Skip base classes
                    and (issubclass(attr, BaseBot) or issubclass(attr, PoeBot))
                ):
                    # Use class name to deduplicate
                    if attr.__name__ not in [bc.__name__ for bc in bot_classes]:
                        bot_classes.append(attr)
                        logger.debug(f"Found bot class: {attr.__name__} directly in {module_name}")

            # Log the number of bots found
            logger.info(f"Found {len(bot_classes)} bot classes in module {module_name}")
            for bot_class in bot_classes:
                logger.info(f"  - {bot_class.__name__}")

            # Return the bot classes
            return bot_classes
        except Exception as e:
            logger.error(f"Error loading bots from module {module_name}: {str(e)}", exc_info=True)
            return []

    @staticmethod
    def get_available_bots() -> Dict[str, str]:
        """Get information about all available bots.

        Returns:
            Dictionary mapping bot names to their descriptions
        """
        bot_classes = BotFactory.load_bots_from_module("bots")
        bot_info = {}

        for bot_class in bot_classes:
            name = getattr(bot_class, "bot_name", bot_class.__name__)
            description = bot_class.__doc__ or "No description available"
            bot_info[name] = description.strip()

        return bot_info

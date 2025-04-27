"""
Main application entry point for the Poe Bot Host.

This module creates and configures the FastAPI application for hosting
multiple Poe bots, both locally and deployed on Modal.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from modal import App, Image, asgi_app

from utils.bot_factory import BotFactory
from utils.config import settings

# Get configured logger
logger = logging.getLogger("poe_bots.app")

__version__ = "1.0.0"


def create_api(allow_without_key: bool = settings.ALLOW_WITHOUT_KEY) -> FastAPI:
    """Create and configure the FastAPI app with all available bots.

    Args:
        allow_without_key: Whether to allow requests without an API key

    Returns:
        Configured FastAPI app
    """
    # Load all bots from the 'bots' module
    logger.info("Loading bots from 'bots' module")
    bot_classes = BotFactory.load_bots_from_module("bots")

    if not bot_classes:
        logger.warning("No bots found in 'bots' module!")

    # Create a FastAPI app with all the bots
    api = BotFactory.create_app(bot_classes, allow_without_key=allow_without_key)

    # Add custom error handling
    @api.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception occurred:")
        return JSONResponse(
            status_code=500,
            content={"error": "An internal server error occurred", "detail": str(exc)},
        )

    # Add a health check endpoint
    @api.get("/health")
    async def health_check():
        """Health check endpoint."""
        bot_info = BotFactory.get_available_bots()
        return {
            "status": "ok",
            "version": __version__,
            "bots": list(bot_info.keys()),
            "bot_count": len(bot_info),
            "environment": {
                "debug": settings.DEBUG,
                "log_level": settings.LOG_LEVEL,
                "allow_without_key": settings.ALLOW_WITHOUT_KEY,
            },
        }

    # Add a bot list endpoint
    @api.get("/bots")
    async def list_bots():
        """List all available bots."""
        return BotFactory.get_available_bots()

    return api


# Create the API
api = create_api()

# For Modal deployment
app = App(settings.MODAL_APP_NAME)

# Create a custom image with required dependencies and local modules
image = (
    Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("utils", "/root/utils")
    .add_local_dir("bots", "/root/bots")
    .add_local_dir("tests", "/root/tests")
    .add_local_python_source("bots", "utils")
)


@app.function(image=image)
@asgi_app()
def fastapi_app():
    """Create and return the FastAPI app for Modal deployment."""
    logger.info("Starting FastAPI app for Modal deployment")
    return api


# This allows the app to be run locally with 'python app.py'
if __name__ == "__main__":
    import uvicorn

    logger.info(f"Starting FastAPI app locally on {settings.API_HOST}:{settings.API_PORT}")
    uvicorn.run(api, host=settings.API_HOST, port=settings.API_PORT)

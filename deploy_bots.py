"""
Deployment script for Poe bots using Modal.

This script creates and deploys the FastAPI application to Modal with proper configuration
of API secrets and dependencies for both OpenAI and Gemini bots.
"""

import logging
import os
import sys

import modal

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("deploy_bots")
logger.setLevel(logging.DEBUG)

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from utils.config import settings
except ImportError:
    logger.error("Cannot import settings. Make sure utils/config.py exists.")
    sys.exit(1)

# Create a Modal app
app = modal.App(settings.MODAL_APP_NAME)

# Create a custom image with required dependencies
logger.info("Creating Modal image with dependencies")
image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .pip_install("google-generativeai>=0.3.2")  # Ensure Gemini package is installed
    .add_local_dir("utils", "/root/utils")
    .add_local_dir("bots", "/root/bots")
    .add_local_dir("tests", "/root/tests")
    .add_local_python_source("utils")  # Add local Python modules explicitly
)

@app.function(
    image=image,
    secrets=[
        # Include all required API key secrets
        modal.Secret.from_name("OPENAI_API_KEY"),
        modal.Secret.from_name("GOOGLE_API_KEY")
    ]
)
@modal.asgi_app()
def fastapi_app():
    """Create and return the FastAPI app for Modal deployment."""
    logger.info("Starting FastAPI app for Modal deployment")

    # Import inside function to ensure Modal image context
    from utils.bot_factory import BotFactory

    # Create the FastAPI app
    logger.info("Loading bots from 'bots' module")
    try:
        bot_classes = BotFactory.load_bots_from_module("bots")
        if not bot_classes:
            logger.warning("No bots found in 'bots' module!")
    except Exception as e:
        logger.error(f"Error loading bots: {str(e)}")
        logger.exception("Exception details:")
        bot_classes = []

    # Create a FastAPI app with all the bots
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    api = BotFactory.create_app(bot_classes, allow_without_key=settings.ALLOW_WITHOUT_KEY)

    # Add custom error handling
    @api.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.exception("Unhandled exception occurred:")
        return JSONResponse(
            status_code=500,
            content={"error": "An internal server error occurred", "detail": str(exc)},
        )

    # Add health check endpoint
    @api.get("/health")
    async def health_check():
        """Health check endpoint."""
        __version__ = "1.0.0"
        return {
            "status": "ok",
            "version": __version__,
            "bots": list(BotFactory.get_available_bots().keys()),
            "bot_count": len(BotFactory.get_available_bots()),
            "environment": {
                "debug": settings.DEBUG,
                "log_level": settings.LOG_LEVEL,
                "allow_without_key": settings.ALLOW_WITHOUT_KEY,
            },
        }

    # Add bot listing endpoint
    @api.get("/bots")
    async def list_bots():
        """List all available bots."""
        return BotFactory.get_available_bots()

    # Add a debug endpoint
    @api.get("/debug_info")
    async def debug_info():
        """Get debug information about the deployment."""
        import inspect
        import sys

        result = {
            "python_version": sys.version,
            "python_path": sys.path,
            "module_path": inspect.getfile(BotFactory),
            "bots_directory": os.listdir("/root/bots") if os.path.exists("/root/bots") else [],
            "utils_directory": os.listdir("/root/utils") if os.path.exists("/root/utils") else [],
            "working_directory": os.getcwd(),
            "bot_classes": [str(bc) for bc in bot_classes] if bot_classes else [],
            "loaded_modules": list(sys.modules.keys())
        }
        return result

    # Add a diagnostic endpoint
    @api.get("/keys_configured")
    async def check_api_keys():
        """Check if API keys are configured."""
        from utils.api_keys import get_api_key

        result = {}

        # Check OpenAI key
        try:
            openai_key = get_api_key("OPENAI_API_KEY")
            result["openai"] = {
                "configured": bool(openai_key),
                "length": len(openai_key) if openai_key else 0,
                "starts_with": openai_key[:3] + "..." if openai_key and len(openai_key) > 5 else None
            }
        except Exception as e:
            result["openai"] = {"error": str(e)}

        # Check Google key
        try:
            google_key = get_api_key("GOOGLE_API_KEY")
            result["google"] = {
                "configured": bool(google_key),
                "length": len(google_key) if google_key else 0,
                "starts_with": google_key[:3] + "..." if google_key and len(google_key) > 5 else None
            }
        except Exception as e:
            result["google"] = {"error": str(e)}

        # Check environment variables (omitting sensitive values)
        env_vars = []
        for key in sorted(os.environ.keys()):
            # Only include non-sensitive keys
            if 'api' not in key.lower() and 'key' not in key.lower() and 'secret' not in key.lower():
                env_vars.append(key)

        result["environment"] = {
            "available_vars": env_vars,
            "api_keys": [k for k in os.environ.keys() if "api_key" in k.lower() or "key" in k.lower()]
        }

        return result

    logger.info("FastAPI app created successfully")
    return api

if __name__ == "__main__":
    print("\n=== Poe Bots Deployment Tool ===")
    print("\nThis script is not meant to be run directly with 'python deploy_bots.py'")
    print("\nTo deploy to Modal, use:")
    print("  modal deploy deploy_bots.py")
    print("\nTo test locally, use:")
    print("  modal serve deploy_bots.py")

    # Check if being run for deployment via modal CLI
    if "MODAL_ENVIRONMENT" in os.environ:
        print("\nModal environment detected - proceeding with deployment...")
        # This branch will be taken when running via 'modal deploy'
    elif "--serve" in sys.argv:
        # Add option to run for testing with argument
        print("\nRunning in local development mode (--serve)")
        with app.run():
            print("API is running locally. Press Ctrl+C to stop.")

            try:
                # Set a reasonable timeout (10 minutes) instead of infinite loop
                import time
                for _ in range(600):  # 10 minutes
                    time.sleep(1)
                print("\nAutomatic timeout after 10 minutes. Server stopped.")
            except KeyboardInterrupt:
                print("\nStopping local server...")
    else:
        print("\nExiting. Please use the modal CLI commands shown above.")
        # Exit with helpful message instead of hanging indefinitely

#!/usr/bin/env python
"""
Run the Poe Bot Host locally for testing.

This script provides a command-line interface for running the Poe bots
locally with options for debugging, auto-reload, and custom host/port.
"""

import argparse
import logging
import os
import sys

import uvicorn

from app import __version__
from utils.config import settings

# Get configured logger
logger = logging.getLogger("poe_bots.run_local")


def print_banner(args):
    """Print a nice banner with configuration information."""
    banner = [
        "=" * 60,
        "                    POE BOTS FRAMEWORK                   ",
        f"                       v{__version__}                       ",
        "=" * 60,
        f"Host:      {args.host}",
        f"Port:      {args.port}",
        f"Debug:     {args.debug}",
        f"Log Level: {args.log_level}",
        f"Reload:    {args.reload}",
        "=" * 60,
        "",
    ]
    print("\n".join(banner))


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import fastapi_poe

        return True
    except ImportError:
        print("Dependencies not installed. Installing...")
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        return True


def activate_venv():
    """Try to activate virtual environment if needed and possible."""
    if (
        getattr(sys, "real_prefix", None)
        or hasattr(sys, "base_prefix")
        and sys.base_prefix != sys.prefix
    ):
        # Already in a virtual environment
        return

    # Check for virtual environment
    venv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "venv")
    if os.path.exists(venv_path):
        print("Note: Running outside virtual environment. Consider activating it with:")
        if os.name == "nt":  # Windows
            print(f"    {venv_path}\\Scripts\\activate")
        else:  # Unix-like
            print(f"    source {venv_path}/bin/activate")


def main():
    """Run the Poe bots locally with command-line options."""
    parser = argparse.ArgumentParser(
        description="Run Poe bots locally for testing",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--host", default=settings.API_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=settings.API_PORT, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--debug", action="store_true", default=True, help="Enable debug mode")
    parser.add_argument(
        "--log-level",
        default=settings.LOG_LEVEL,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    parser.add_argument("--no-banner", action="store_true", help="Don't display the banner")

    args = parser.parse_args()

    # Set environment variables
    if args.debug:
        os.environ["DEBUG"] = "true"
        os.environ["LOG_LEVEL"] = "DEBUG"
    else:
        os.environ["LOG_LEVEL"] = args.log_level

    # Export environment variables for child processes
    os.environ["POE_API_HOST"] = args.host
    os.environ["POE_API_PORT"] = str(args.port)

    # Check dependencies and environment
    activate_venv()
    check_dependencies()

    # Display banner
    if not args.no_banner:
        print_banner(args)

    # Log the runtime configuration
    logger.info(f"Starting Poe bots on {args.host}:{args.port}")
    logger.info(f"Debug mode: {args.debug}")
    logger.info(f"Log level: {args.log_level}")
    logger.info(f"Auto-reload: {args.reload}")

    # Start the server
    uvicorn.run(
        "app:api",
        host=args.host,
        port=args.port,
        reload=args.reload,
        access_log=args.debug,
        log_level=args.log_level.lower(),
    )


if __name__ == "__main__":
    main()

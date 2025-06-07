#!/usr/bin/env python3
"""
Access Key Configuration Utility

This script helps configure and test access key resolution for bots.
"""

import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.auth import APIKeyResolver, debug_access_key_resolution  # noqa: E402
from utils.bot_config import get_bot_config  # noqa: E402


def main():
    """Main function to run access key configuration utilities."""
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()

    if command == "debug":
        if len(sys.argv) < 3:
            print("Usage: python access_key_config.py debug <bot_name>")
            return
        debug_bot(sys.argv[2])

    elif command == "patterns":
        if len(sys.argv) < 3:
            print("Usage: python access_key_config.py patterns <bot_name>")
            return
        show_patterns(sys.argv[2])

    elif command == "list":
        list_available_keys()

    elif command == "config":
        show_config()

    elif command == "test":
        if len(sys.argv) < 3:
            print("Usage: python access_key_config.py test <bot_name>")
            return
        test_resolution(sys.argv[2])

    else:
        print_usage()


def print_usage():
    """Print usage information."""
    print("Access Key Configuration Utility")
    print("\nUsage:")
    print("  python access_key_config.py debug <bot_name>     - Debug access key resolution")
    print("  python access_key_config.py patterns <bot_name>  - Show patterns for bot")
    print("  python access_key_config.py list                 - List available access keys")
    print("  python access_key_config.py config               - Show bot configuration")
    print("  python access_key_config.py test <bot_name>      - Test access key resolution")
    print("\nExamples:")
    print("  python access_key_config.py debug GeminiImageGenerationBot")
    print("  python access_key_config.py patterns EchoBot")
    print("  python access_key_config.py list")


def debug_bot(bot_name: str):
    """Debug access key resolution for a specific bot."""
    print(f"🐛 Debugging access key resolution for: {bot_name}")
    print("=" * 60)

    debug_info = debug_access_key_resolution(bot_name)

    print(f"Bot Name: {debug_info['bot_name']}")
    print(
        f"Resolution Status: {'✅ Success' if debug_info['resolution_successful'] else '❌ Failed'}"
    )
    print(f"Found Key: {debug_info['resolved_key'] or 'None'}")
    print()

    print(f"Patterns Tried ({len(debug_info['patterns_tried'])}):")
    for i, pattern in enumerate(debug_info["patterns_tried"], 1):
        env_value = os.environ.get(pattern)
        status = "✅ FOUND" if env_value else "❌ not found"
        print(f"  {i:2d}. {pattern:<40} {status}")

    print()
    print(f"Available Keys in Environment ({len(debug_info['available_env_keys'])}):")
    for key in debug_info["available_env_keys"]:
        print(f"  - {key}")

    if not debug_info["available_env_keys"]:
        print("  None found")


def show_patterns(bot_name: str):
    """Show access key patterns for a specific bot."""
    from utils.bot_config import get_access_key_patterns

    print(f"🔍 Access key patterns for: {bot_name}")
    print("=" * 60)

    patterns = get_access_key_patterns(bot_name)

    for i, pattern in enumerate(patterns, 1):
        print(f"  {i:2d}. {pattern}")

    print(f"\nTotal patterns: {len(patterns)}")


def list_available_keys():
    """List all available access keys in environment."""
    print("🔑 Available access keys in environment:")
    print("=" * 60)

    resolver = APIKeyResolver("dummy")
    available_keys = resolver.get_available_keys()

    if available_keys:
        for key in sorted(available_keys):
            value = os.environ.get(key, "")
            masked_value = value[:10] + "..." if len(value) > 10 else value
            print(f"  {key:<40} = {masked_value}")
    else:
        print("  No access keys found in environment")

    print(f"\nTotal keys: {len(available_keys)}")


def show_config():
    """Show bot configuration."""
    print("⚙️  Bot Configuration:")
    print("=" * 60)

    config = get_bot_config()

    print("Access Key Patterns:")
    patterns = config._config.get("access_key_patterns", {})
    for key, value in patterns.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for pattern in value[:3]:  # Show first 3
                print(f"    - {pattern}")
            if len(value) > 3:
                print(f"    ... and {len(value) - 3} more")
        else:
            print(f"  {key}: {value}")

    print("\nBot Metadata:")
    metadata = config._config.get("bot_metadata", {})
    for bot_name, data in metadata.items():
        print(f"  {bot_name}:")
        for key, value in data.items():
            print(f"    {key}: {value}")


def test_resolution(bot_name: str):
    """Test access key resolution for a bot."""
    print(f"🧪 Testing access key resolution for: {bot_name}")
    print("=" * 60)

    resolver = APIKeyResolver(bot_name)

    print("Step 1: Get patterns...")
    patterns = resolver.get_tried_patterns()
    print(f"  Found {len(patterns)} patterns to try")

    print("\nStep 2: Test resolution...")
    result = resolver.resolve()

    if result:
        print(f"  ✅ Success! Found key: {result[:10]}...")
    else:
        print("  ❌ Failed to resolve access key")

    print("\nStep 3: Check environment...")
    available = resolver.get_available_keys()
    print(f"  Found {len(available)} access keys in environment")

    print("\nRecommendations:")
    if not result and available:
        print("  - You have access keys but none match the expected patterns")
        print("  - Consider setting one of these variables:")
        for pattern in patterns[:3]:
            print(f"    export {pattern}=<your_key>")
    elif not available:
        print("  - No access keys found in environment")
        print("  - Set an access key environment variable")
    else:
        print("  - Access key resolution working correctly!")


if __name__ == "__main__":
    main()

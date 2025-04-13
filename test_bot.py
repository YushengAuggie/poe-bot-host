#!/usr/bin/env python
"""
Test script for Poe bots running locally.

This script provides a command-line interface for testing bots running
in the Poe Bots Framework. It can test specific bots, check API health,
and show OpenAPI schema information.
"""

import requests
import json
import argparse
import sys
import logging
import os
from typing import Dict, Any, Optional, List
from utils.config import settings

# Set up logger
logger = logging.getLogger("poe_bots.test_bot")

def get_available_bots(base_url: str) -> Dict[str, str]:
    """Get list of available bots from the API.
    
    Args:
        base_url: Base URL of the API
        
    Returns:
        Dictionary of bot names and descriptions
    """
    try:
        response = requests.get(f"{base_url}/bots")
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to get bot list: {response.status_code}")
            return {}
    except Exception as e:
        logger.error(f"Error getting bot list: {str(e)}")
        return {}

def check_health(base_url: str) -> Dict[str, Any]:
    """Check API health status.
    
    Args:
        base_url: Base URL of the API
        
    Returns:
        Health check response
    """
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Health check failed: {response.status_code}")
            return {"status": "error", "code": response.status_code}
    except Exception as e:
        logger.error(f"Error checking health: {str(e)}")
        return {"status": "error", "message": str(e)}

def fetch_openapi_schema(base_url: str) -> Optional[Dict[str, Any]]:
    """Fetch the OpenAPI schema from the API.
    
    Args:
        base_url: Base URL of the API
        
    Returns:
        OpenAPI schema
    """
    try:
        response = requests.get(f"{base_url}/openapi.json")
        if response.status_code == 200:
            return response.json()
        else:
            logger.warning(f"Failed to fetch OpenAPI schema: {response.status_code}")
            return None
    except Exception as e:
        logger.error(f"Error fetching OpenAPI schema: {str(e)}")
        return None

def test_bot(base_url: str, bot_name: str, message: str) -> Dict[str, Any]:
    """Test a bot with a sample message.
    
    Args:
        base_url: Base URL of the API
        bot_name: Name of the bot to test
        message: Message to send to the bot
        
    Returns:
        Bot response
    """
    url = f"{base_url}/{bot_name.lower()}"
    
    try:
        response = requests.post(
            url,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer dummytoken"  # Since allow_without_key=True in our bot
            },
            json={
                "version": "1.0",
                "type": "query",
                "query": [
                    {"role": "user", "content": message}
                ],
                "user_id": "test_user",
                "conversation_id": "test_convo_123",
                "message_id": "test_msg_123",
                "protocol": "poe"
            }
        )
        
        status_code = response.status_code
        headers = dict(response.headers)
        
        # Try to parse response as JSON
        try:
            content = response.json()
            content_type = "json"
        except ValueError:
            # For event stream responses
            content = response.text
            content_type = "text"
            
        return {
            "status_code": status_code,
            "headers": headers,
            "content": content,
            "content_type": content_type
        }
    except Exception as e:
        logger.error(f"Error testing bot: {str(e)}")
        return {
            "status_code": 0,
            "error": str(e),
            "content": None,
            "content_type": "error"
        }

def print_banner():
    """Print a banner for the test script."""
    print("\n======================================================")
    print("                 POE BOTS TEST TOOL                   ")
    print("======================================================\n")

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Poe bots running locally",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", default=settings.API_HOST, help="Host where the API is running")
    parser.add_argument("--port", type=int, default=settings.API_PORT, help="Port where the API is running")
    parser.add_argument("--bot", help="Specific bot to test (omit for auto-select)")
    parser.add_argument("--message", default="Hello, world!", help="Message to send to the bot")
    parser.add_argument("--schema", action="store_true", help="Show OpenAPI schema")
    parser.add_argument("--health", action="store_true", help="Show health check")
    parser.add_argument("--all", action="store_true", help="Test all available bots")
    parser.add_argument("--format", choices=["text", "json"], default="text", 
                        help="Output format (text or json)")
    
    args = parser.parse_args()
    
    # Use environment variables if specified
    host = os.environ.get("POE_API_HOST", args.host)
    port = int(os.environ.get("POE_API_PORT", args.port))
    
    base_url = f"http://{host}:{port}"
    
    # Print banner for non-JSON output
    if args.format == "text":
        print_banner()
        logger.info(f"Using API at {base_url}")
    
    # Check health if requested
    if args.health:
        health = check_health(base_url)
        if args.format == "json":
            print(json.dumps({"health": health}, indent=2))
        else:
            print("\nHealth Check:")
            print(json.dumps(health, indent=2))
    
    # Get list of available bots
    bots = get_available_bots(base_url)
    if bots:
        if args.format == "json":
            if not args.health and not args.schema:
                print(json.dumps({"bots": bots}, indent=2))
        else:
            print("\nAvailable Bots:")
            for bot, description in bots.items():
                print(f"  - {bot}: {description}")
    else:
        if args.format == "text":
            print("\nNo bots available!")
        else:
            print(json.dumps({"error": "No bots available"}, indent=2))
        return
        
    # Show OpenAPI schema if requested
    if args.schema:
        schema = fetch_openapi_schema(base_url)
        if schema:
            if args.format == "json":
                print(json.dumps({"schema": schema}, indent=2))
            else:
                print("\nAvailable Endpoints:")
                for path, methods in schema.get("paths", {}).items():
                    print(f"  {path}:")
                    for method in methods:
                        print(f"    - {method.upper()}")
    
    # Test a specific bot or all bots
    if args.all:
        # Test all bots
        results = {}
        for bot in bots.keys():
            if args.format == "text":
                print(f"\nTesting bot: {bot}")
                print(f"Message: {args.message}")
            
            result = test_bot(base_url, bot, args.message)
            results[bot] = result
            
            if args.format == "text":
                print(f"Status Code: {result['status_code']}")
                
                if result['content_type'] == "json":
                    print("Response JSON:")
                    print(json.dumps(result['content'], indent=2))
                elif result['content_type'] == "text":
                    print("Response Content:")
                    for line in result['content'].split('\n'):
                        if line.strip():
                            print(f"  {line}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
        
        if args.format == "json":
            print(json.dumps({"results": results}, indent=2))
    else:
        # Test a specific bot
        bot_to_test = args.bot
        if not bot_to_test and bots:
            bot_to_test = list(bots.keys())[0]
            if args.format == "text":
                print(f"\nNo bot specified, using first available bot: {bot_to_test}")
        
        if bot_to_test:
            if args.format == "text":
                print(f"\nTesting bot: {bot_to_test}")
                print(f"Message: {args.message}")
            
            result = test_bot(base_url, bot_to_test, args.message)
            
            if args.format == "json":
                print(json.dumps({"result": result}, indent=2))
            else:
                print(f"Status Code: {result['status_code']}")
                
                if result['content_type'] == "json":
                    print("Response JSON:")
                    print(json.dumps(result['content'], indent=2))
                elif result['content_type'] == "text":
                    print("Response Content:")
                    for line in result['content'].split('\n'):
                        if line.strip():
                            print(f"  {line}")
                else:
                    print(f"Error: {result.get('error', 'Unknown error')}")
    
if __name__ == "__main__":
    main()
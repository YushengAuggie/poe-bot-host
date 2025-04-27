#!/usr/bin/env python
"""
Modal Deployment Debug Tool

This script queries and pretty-prints the debug info from a Modal deployment.
"""

import json
import requests
import sys

def get_debug_info(url):
    """Get debug info from the specified URL."""
    response = requests.get(f"{url}/debug_info")
    if response.status_code == 200:
        return response.json()
    else:
        return {
            "error": f"Failed to get debug info. Status code: {response.status_code}",
            "text": response.text
        }

if __name__ == "__main__":
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://aibot2025us--aibot2025us-poe-bots-fastapi-app.modal.run"
        
    print(f"Getting debug info from: {url}")
    debug_info = get_debug_info(url)
    print(json.dumps(debug_info, indent=2, sort_keys=True))
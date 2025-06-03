#!/bin/bash
# Run the Poe bots locally with the correct access key for Gemini image generation

# Change to script directory
cd "$(dirname "$0")"

# Set environment variables for the image generation bot
export GEMINIIMAGEGENERATIONBOT_ACCESS_KEY="7NvqYURsx4UGQsdr7AAFgkJmLIlnZXlo"
export POE_ACCESS_KEY="7NvqYURsx4UGQsdr7AAFgkJmLIlnZXlo"

echo "=== Poe Bots Server with Image Generation Access Key ==="
echo "GEMINIIMAGEGENERATIONBOT_ACCESS_KEY: ${GEMINIIMAGEGENERATIONBOT_ACCESS_KEY:0:10}..."
echo "POE_ACCESS_KEY: ${POE_ACCESS_KEY:0:10}..."
echo ""

# Activate virtual environment if it exists and is not already activated
if [ -d "venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run the Python script
python run_local.py "$@"

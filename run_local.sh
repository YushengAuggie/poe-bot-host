#!/bin/bash
# Simple wrapper script to run the Poe bots locally
# This activates the virtual environment first, then delegates to run_local.py

# Change to script directory
cd "$(dirname "$0")"

# Activate virtual environment if it exists and is not already activated
if [ -d "venv" ] && [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run the Python script with all arguments passed through
python run_local.py "$@"

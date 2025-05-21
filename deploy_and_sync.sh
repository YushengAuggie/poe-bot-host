#!/bin/bash
# Script to deploy all bots and sync their settings

# Stop on any error
set -e

echo "===== Starting bot deployment and settings sync ====="

# Deploy the bots to Modal
echo "Deploying bots to Modal..."
modal deploy deploy_bots.py

# Wait for the deployment to complete
echo "Waiting for deployment to stabilize..."
sleep 5

# Sync all bot settings with Poe
echo "Syncing bot settings with Poe..."
/usr/local/bin/python3.11 sync_bot_settings.py --all

echo "===== Deployment and sync completed ====="
echo "✅ Bots are now deployed and settings are synced with Poe"
echo "✅ Attachment support should now be working for Gemini bots"
echo ""
echo "To verify the deployment:"
echo "1. Check your bots on Poe"
echo "2. Try uploading an image to a Gemini bot"
echo ""
echo "If you still encounter issues, try running:"
echo "python sync_bot_settings.py --bot YourSpecificBot --verbose"

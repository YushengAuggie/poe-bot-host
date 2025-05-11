#!/usr/bin/env python
"""
Manual test script for Gemini bot image input.
This script starts a local Poe server and provides instructions for testing.
"""

import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Set Google API key if not already set
if "GOOGLE_API_KEY" not in os.environ:
    print("❌ GOOGLE_API_KEY not found in environment.")
    api_key = input("Enter your Google API key: ").strip()
    if api_key:
        os.environ["GOOGLE_API_KEY"] = api_key
        print("✅ Set Google API key.")
    else:
        print("❌ No API key provided. Test will fail.")
        sys.exit(1)
else:
    print(f"✅ Found Google API key (starts with {os.environ['GOOGLE_API_KEY'][:5]}...)")


# Create a test image if needed
def create_test_image():
    """Create a simple test image if PIL is installed."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create a directory for test assets if it doesn't exist
        assets_dir = Path("test_assets")
        assets_dir.mkdir(exist_ok=True)

        # Create a colored square with text
        img = Image.new("RGB", (500, 500), color=(25, 25, 112))  # Midnight blue

        # Add some text
        draw = ImageDraw.Draw(img)

        # Try to use a font if available, otherwise use default
        try:
            font = ImageFont.truetype("Arial", 36)
        except IOError:
            font = None

        draw.text((100, 200), "Gemini Test Image", fill=(255, 255, 255), font=font)
        draw.text((100, 250), "This is a test", fill=(255, 255, 255), font=font)

        # Save the image
        img_path = assets_dir / "gemini_test_image.png"
        img.save(img_path)

        print(f"✅ Created test image at: {img_path}")
        return img_path

    except ImportError:
        print("⚠️ PIL not installed. Cannot create test image.")
        return None


# Check if run_local.py exists
if not Path("run_local.py").exists():
    print("❌ run_local.py not found. Make sure you're in the right directory.")
    sys.exit(1)

# Create test image
test_image = create_test_image()

print("\n===== MANUAL TESTING INSTRUCTIONS =====")
print("1. The local Poe server will start in a moment.")
print("2. Once started, the server address will be shown (usually http://localhost:8000).")
print("3. Open that URL in your browser.")
print("4. Send a message with an image to the Gemini20FlashBot.")
print("   - First send a text message like 'Hello'")
print("   - Then send an image with text like 'What's in this image?'")
print("5. Verify that the bot responds with a description of the image.")
print("\nPress Ctrl+C to stop the server when you're done testing.")
print("=================================\n")

# Start the server
try:
    time.sleep(2)  # Give time to read instructions
    print("Starting local Poe server...")
    # Run with Python directly to avoid shell variations
    subprocess.run([sys.executable, "run_local.py"])
except KeyboardInterrupt:
    print("\nServer stopped.")

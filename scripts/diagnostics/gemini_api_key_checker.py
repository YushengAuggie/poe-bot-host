#!/usr/bin/env python
"""
Gemini API Key Checker

This script specifically tests the Gemini API key configuration in both
local and Modal environments. It attempts to initialize the Gemini client
and make a test API call to verify that the key is working properly.

Usage:
    # Test locally
    python scripts/diagnostics/gemini_api_key_checker.py

    # Deploy and test on Modal
    modal deploy scripts/diagnostics/gemini_api_key_checker.py
"""

import os
import sys

# Add the project root to path so utils module can be found
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import modal
except ImportError:
    modal = None


def test_google_api_key_local():
    """Test if the GOOGLE_API_KEY is properly configured locally"""
    # Now utils can be imported
    from utils.api_keys import get_api_key

    try:
        # This will check environment variables
        api_key = get_api_key("GOOGLE_API_KEY")
        key_preview = api_key[:3] + "..." if api_key else "NOT FOUND"
        print(f"GOOGLE_API_KEY: {key_preview}")

        # Check if google-generativeai is installed
        try:
            import google.generativeai as genai

            print("google-generativeai package: INSTALLED")

            # Try to initialize the client
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name="gemini-2.0-flash")
            print("Gemini client initialization: SUCCESS")

            # Test a simple query
            response = model.generate_content("Say hello")
            print(f"Gemini API test response: {response.text[:20]}...")

        except ImportError:
            print("google-generativeai package: NOT INSTALLED")
        except Exception as e:
            print(f"Gemini client error: {str(e)}")

    except ValueError as e:
        print(f"API key error: {str(e)}")


# For Modal deployment
if modal:
    app = modal.App("gemini-key-test")

    # Create a Modal image that includes all necessary packages and our code
    image = (
        modal.Image.debian_slim()
        .pip_install(
            "google-generativeai>=0.8.5",
            "fastapi-poe>=0.0.21",
            "python-dotenv>=1.0.0",
            "fastapi>=0.105.0",
            "pydantic>=2.0.0",
            "requests>=2.27.1",
        )
        .add_local_dir(".", "/root")  # Using add_local_dir instead of copy_local_dir
    )

    @app.function(
        image=image,
        secrets=[modal.Secret.from_name("GOOGLE_API_KEY")],
    )
    def test_google_api_key_modal():
        """Test if the GOOGLE_API_KEY is properly configured in Modal"""
        # Add the project directory to the Python path
        import os
        import sys

        sys.path.append("/root")

        # Now utils can be imported
        from utils.api_keys import get_api_key

        try:
            # This will check both environment variables and Modal secrets
            api_key = get_api_key("GOOGLE_API_KEY")
            key_preview = api_key[:3] + "..." if api_key else "NOT FOUND"
            print(f"GOOGLE_API_KEY: {key_preview}")

            # Import and initialize genai
            import google.generativeai as genai

            print("google-generativeai package: INSTALLED")

            # Try to initialize the client
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name="gemini-2.0-flash")
            print("Gemini client initialization: SUCCESS")

            # Test a simple query
            response = model.generate_content("Say hello")
            print(f"Gemini API test response: {response.text[:20]}...")

            return "Success! Gemini API key is working correctly."

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(error_msg)
            return error_msg


if __name__ == "__main__":
    print("Running test for Gemini API key...")

    # Check if Modal is available
    if modal and not modal.is_local():
        print("Running in Modal cloud...")
        result = test_google_api_key_modal.remote()
        print(f"Result: {result}")
    else:
        print("Running in local mode...")

        # Check if key exists in environment
        if "GOOGLE_API_KEY" not in os.environ:
            print("Warning: GOOGLE_API_KEY not found in environment variables")
            print("Please set the GOOGLE_API_KEY environment variable and try again.")
            print(
                "Example: GOOGLE_API_KEY=your_key_here python scripts/diagnostics/gemini_api_key_checker.py"
            )
            sys.exit(1)

        # Execute the local test
        test_google_api_key_local()

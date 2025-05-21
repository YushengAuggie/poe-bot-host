import os

import google.generativeai as genai
from modal import App, Image, Secret
from openai import OpenAI

from utils.api_keys import get_api_key

# Create a Modal app for the chatbot
app = App("chatbot-example")
image = Image.debian_slim().pip_install(["openai", "google-generativeai"])
app.image = image


@app.function(secrets=[Secret.from_name("OPENAI_API_KEY")])
def openai_chat(prompt):
    """
    Function that uses OpenAI's API to generate a chat response.
    Checks local environment first, then falls back to Modal secrets.
    """
    # This will check local env first, then Modal secrets
    client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


@app.function(secrets=[Secret.from_name("GOOGLE_API_KEY")])
def gemini_chat(prompt):
    """
    Function that uses Google's Gemini API to generate a chat response.
    Checks local environment first, then falls back to Modal secrets.
    """
    # This will check local env first, then Modal secrets
    api_key = get_api_key("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)

    # Create a model and generate content
    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    response = model.generate_content(prompt)

    return response.text


# Function that can be used for local testing without Modal
def run_local(prompt, use_gemini=False):
    """Run the chat function locally without Modal."""
    if use_gemini:
        print("Using Gemini locally:")
        api_key = get_api_key("GOOGLE_API_KEY")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name="gemini-2.0-flash")
        response = model.generate_content(prompt)
        return response.text
    else:
        print("Using OpenAI locally:")
        client = OpenAI(api_key=get_api_key("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content


@app.local_entrypoint()
def main(
    prompt: str = "Explain quantum computing in simple terms",
    use_gemini: bool = False,
    local: bool = False,
):
    """
    Entry point for the Modal app.

    Args:
        prompt: The text prompt to send to the AI
        use_gemini: Whether to use Gemini instead of OpenAI
        local: Whether to run locally instead of using Modal
    """
    # First check if we should run locally
    if local:
        response = run_local(prompt, use_gemini)
    else:
        # Otherwise use Modal
        if use_gemini:
            print("Using Gemini via Modal:")
            response = gemini_chat.remote(prompt)
        else:
            print("Using OpenAI via Modal:")
            response = openai_chat.remote(prompt)

    print(f"\nPrompt: {prompt}")
    print(f"\nResponse: {response}")

"""
Configuration for Gemini model capabilities.

This module defines which features are supported by each Gemini model.
"""

from typing import Any, Dict

# Model capability configuration
# Format: "model_name": {capabilities}
MODEL_CAPABILITIES: Dict[str, Dict[str, Any]] = {
    # Gemini 2.0 Series
    "gemini-2.0-flash": {
        "supports_grounding": False,
        "supports_thinking": False,
        "supports_image_generation": False,
        "free_tier": True,
    },
    "gemini-2.0-flash-exp": {
        "supports_grounding": False,
        "supports_thinking": False,
        "supports_image_generation": True,  # Experimental features
        "free_tier": True,
    },
    "gemini-2.0-flash-thinking-exp-01-21": {
        "supports_grounding": False,
        "supports_thinking": True,
        "supports_image_generation": False,
        "free_tier": True,
    },
    "gemini-2.0-pro": {
        "supports_grounding": False,  # Disabled due to API errors
        "supports_thinking": False,
        "supports_image_generation": False,
        "free_tier": True,
    },
    "gemini-2.0-pro-exp-02-05": {
        "supports_grounding": False,  # Disabled due to API errors
        "supports_thinking": False,
        "supports_image_generation": False,
        "free_tier": True,
    },
    # Gemini 2.5 Series
    "gemini-2.5-flash-preview-04-17": {
        "supports_grounding": False,  # Disabled due to API errors
        "supports_thinking": False,  # Disabled due to API errors
        "supports_image_generation": False,
        "free_tier": False,  # May require paid plan
    },
    "gemini-2.5-pro-preview-06-05": {
        "supports_grounding": False,  # Disabled due to API errors
        "supports_thinking": False,  # Disabled due to API errors
        "supports_image_generation": False,
        "free_tier": False,  # Requires paid plan
    },
    # Image Generation Models
    "gemini-2.0-flash-preview-image-generation": {
        "supports_grounding": False,
        "supports_thinking": False,
        "supports_image_generation": True,
        "free_tier": True,
    },
    # Legacy Models (for fallback)
    "gemini-1.5-flash-latest": {
        "supports_grounding": False,
        "supports_thinking": False,
        "supports_image_generation": False,
        "free_tier": True,
    },
}

# Default capabilities for unknown models
DEFAULT_CAPABILITIES = {
    "supports_grounding": False,
    "supports_thinking": False,
    "supports_image_generation": False,
    "free_tier": True,
}


def get_model_capabilities(model_name: str) -> Dict[str, Any]:
    """Get capabilities for a specific model.

    Args:
        model_name: The Gemini model name

    Returns:
        Dictionary of model capabilities
    """
    return MODEL_CAPABILITIES.get(model_name, DEFAULT_CAPABILITIES.copy())


def supports_grounding(model_name: str) -> bool:
    """Check if a model supports Google Search grounding.

    Args:
        model_name: The Gemini model name

    Returns:
        True if the model supports grounding
    """
    return get_model_capabilities(model_name)["supports_grounding"]


def supports_thinking(model_name: str) -> bool:
    """Check if a model supports thinking budget.

    Args:
        model_name: The Gemini model name

    Returns:
        True if the model supports thinking budget
    """
    return get_model_capabilities(model_name)["supports_thinking"]


def supports_image_generation(model_name: str) -> bool:
    """Check if a model supports image generation.

    Args:
        model_name: The Gemini model name

    Returns:
        True if the model supports image generation
    """
    return get_model_capabilities(model_name)["supports_image_generation"]


def has_free_tier(model_name: str) -> bool:
    """Check if a model has free tier access.

    Args:
        model_name: The Gemini model name

    Returns:
        True if the model has free tier access
    """
    return get_model_capabilities(model_name)["free_tier"]


if __name__ == "__main__":
    # Test the configuration
    print("Testing model configuration:")
    print(f"Gemini 2.0 Flash: {get_model_capabilities('gemini-2.0-flash')}")
    print(f"Gemini 2.5 Flash: {get_model_capabilities('gemini-2.5-flash-preview-04-17')}")
    print(f"Unknown model: {get_model_capabilities('unknown-model')}")
    print(f"2.0 Flash supports grounding: {supports_grounding('gemini-2.0-flash')}")
    print(f"2.5 Flash supports grounding: {supports_grounding('gemini-2.5-flash-preview-04-17')}")

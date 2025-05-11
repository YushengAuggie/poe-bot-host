"""
Helper module for mocking Google Generative AI in tests.
"""

from unittest.mock import MagicMock


def create_google_genai_mock():
    """
    Create a properly structured mock of the google.generativeai module.

    Returns:
        A dictionary of mocks to be used with patch.dict('sys.modules', ...)
    """
    # Create the nested structure needed for google.generativeai
    google_mock = MagicMock()
    genai_mock = MagicMock()
    types_mock = MagicMock()

    # Create Part class with from_bytes method
    part_class_mock = MagicMock()
    part_class_mock.from_bytes = MagicMock(return_value=MagicMock())

    # Attach types to genai
    types_mock.Part = part_class_mock
    genai_mock.types = types_mock

    # Attach genai to google
    google_mock.generativeai = genai_mock

    # Set version attribute
    genai_mock.__version__ = "0.1.0"

    # Add GenerativeModel class
    generative_model_mock = MagicMock()
    genai_mock.GenerativeModel = generative_model_mock

    # Add configure method
    genai_mock.configure = MagicMock()

    return {
        "google": google_mock,
        "google.generativeai": genai_mock,
        "google.generativeai.types": types_mock,
    }
